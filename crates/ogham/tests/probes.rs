//! Needle-in-haystack probes: after agent compression and budget
//! enforcement, key facts must survive — either directly in the final
//! messages (errors, decisions) or retrievably via CCR (tool-result detail).

use ogham::agent::{AgentPolicy, apply_agent_compression};
use ogham::budget::{ContextBudget, enforce_budget};
use ogham::ccr::CcrStore;
use ogham::ccr::in_memory::InMemoryCcrStore;
use ogham::pipeline::DefaultCompressionPipeline;
use ogham::token_counter::HeuristicCounter;
use ogham_core::{Message, TokenCounter, meta_keys};
use std::sync::Arc;

/// Needles are scattered across the conversation, NOT in the protected
/// final user message — otherwise the probe can never fail:
/// - file path + TODO live in OLD tool results (will be cleared to CCR)
/// - the error trace is a tool message near the end (must survive verbatim)
/// - the decision is in a recent assistant message (must survive directly)
fn make_probe_conversation() -> Vec<Message> {
    let mut msgs = Vec::with_capacity(30);
    msgs.push(Message::new("system", "You are a helpful assistant."));

    for i in 1..=24 {
        if i % 2 == 0 {
            let content = match i {
                4 => format!("read file src/billing/invoice.rs:42\n{}", "x".repeat(2000)),
                8 => format!("notes: TODO: add retry with backoff\n{}", "x".repeat(2000)),
                _ => "x".repeat(2000),
            };
            let mut m = Message::new("tool", content);
            m.metadata
                .insert(meta_keys::TOOL_NAME.to_string(), format!("tool_{}", i));
            msgs.push(m);
        } else {
            msgs.push(Message::new("assistant", "working on it".to_string()));
        }
    }

    // Recent band: these must survive untouched.
    let mut err = Message::new("tool", "Error: ECONNREFUSED 127.0.0.1:5432");
    err.metadata
        .insert(meta_keys::TOOL_NAME.to_string(), "db".to_string());
    msgs.push(err);
    msgs.push(Message::new(
        "assistant",
        "Decision: use sqlite for the cache",
    ));
    msgs.push(Message::new("user", "Final question?"));
    msgs
}

#[test]
fn probes_present_after_compression() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut msgs = make_probe_conversation();
        let policy = AgentPolicy::default();
        let ccr_store = Arc::new(InMemoryCcrStore::new());
        let counter = HeuristicCounter::new();

        let initial_count = counter.count_messages(&msgs);
        let target = initial_count / 2;

        apply_agent_compression(
            &mut msgs,
            &policy,
            Some(ccr_store.clone() as Arc<dyn CcrStore>),
        )
        .await
        .expect("agent compression failed");

        let budget = ContextBudget {
            total_limit: target,
            safety_margin: Some(0.0),
        };
        let pipeline = DefaultCompressionPipeline::new(None, None);

        let budget_result = enforce_budget(
            &mut msgs,
            &budget,
            &counter,
            &pipeline,
            &policy,
            Some(ccr_store.clone() as Arc<dyn CcrStore>),
        )
        .await;
        assert!(
            budget_result.is_ok(),
            "budget enforcement failed: {:?}",
            budget_result.err()
        );

        let final_text = msgs
            .iter()
            .map(|m| m.content.clone())
            .collect::<Vec<_>>()
            .join("\n");

        // Errors and decisions must survive DIRECTLY in the message list.
        assert!(
            final_text.contains("Error: ECONNREFUSED"),
            "error trace must be present directly"
        );
        assert!(
            final_text.contains("Decision: use sqlite for the cache"),
            "decision must be present directly"
        );

        // Tool-result detail may survive directly or via CCR retrieval.
        let in_ccr = |needle: &str| ccr_store.get_all().iter().any(|(_, v)| v.contains(needle));
        assert!(
            final_text.contains("src/billing/invoice.rs") || in_ccr("src/billing/invoice.rs"),
            "file path must be present directly or retrievable via CCR"
        );
        assert!(
            final_text.contains("TODO: add retry with backoff")
                || in_ccr("TODO: add retry with backoff"),
            "next step must be present directly or retrievable via CCR"
        );
    });
}
