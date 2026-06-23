//! Focus / question-hint steering is honored end to end: a hint pulls matching
//! records into the kept set both on the direct pipeline path and through the
//! high-level `compact_conversation` budget cascade.

use ogham::agent::AgentPolicy;
use ogham::budget::ContextBudget;
use ogham::pipeline::{DEFAULT_COMPRESSORS, DefaultCompressionPipeline};
use ogham::{
    CcrPolicy, CompactConfig, CompressionPipeline, CompressionPolicy, Message, compact_conversation,
};

/// A JSON array of uniform records with a single low-salience "needle" that
/// length-based sampling discards unless a focus hint rescues it.
fn needle_json(n: usize, needle_at: usize) -> String {
    let items: Vec<serde_json::Value> = (0..n)
        .map(|i| {
            let tag = if i == needle_at { "zqxjv" } else { "aaaaa" };
            serde_json::json!({ "id": format!("{:03}", i), "tag": tag })
        })
        .collect();
    serde_json::to_string(&items).expect("serialize")
}

#[tokio::test]
async fn focus_hint_reaches_smart_crusher_via_pipeline() {
    let msgs = vec![Message::new("tool", needle_json(60, 30))];

    let plain = DefaultCompressionPipeline::with_builtin_compressors(None, DEFAULT_COMPRESSORS)
        .expect("pipeline");
    let focused = DefaultCompressionPipeline::with_builtin_compressors(None, DEFAULT_COMPRESSORS)
        .expect("pipeline")
        .with_question_hint(Some("zqxjv".to_string()));

    let out_plain = plain.run(&msgs).await.expect("plain run");
    let out_focused = focused.run(&msgs).await.expect("focused run");

    assert!(
        !out_plain.messages[0].content.contains("zqxjv"),
        "needle should be dropped without a focus hint"
    );
    assert!(
        out_focused.messages[0].content.contains("zqxjv"),
        "focus hint should reach SmartCrusher through the pipeline"
    );
}

#[tokio::test]
async fn compact_conversation_focus_biases_compression() {
    // The needle record lives in a large tool output that the budget cascade
    // must compress (not clear, not drop). A focus hint must steer which
    // records survive that compression.
    let conversation = || {
        vec![
            Message::new("system", "you are a helpful assistant"),
            Message::new("tool", needle_json(200, 100)),
            Message::new("user", "filler one"),
            Message::new("assistant", "filler two"),
            Message::new("user", "filler three"),
            Message::new("assistant", "filler four"),
            Message::new("user", "the latest question"),
        ]
    };

    let config = |focus: Option<&str>| CompactConfig {
        budget: Some(ContextBudget {
            total_limit: 400,
            safety_margin: Some(0.0),
        }),
        agent_policy: AgentPolicy {
            keep_recent_tool_results: 3,
            clear_old_tool_results: false,
            keep_recent_assistant: 2,
            protected_tail_tokens: None,
        },
        compression: CompressionPolicy::default(),
        ccr: CcrPolicy::Disabled,
        focus: focus.map(str::to_string),
        ..CompactConfig::default()
    };

    let plain = compact_conversation(conversation(), config(None))
        .await
        .expect("plain compact");
    let focused = compact_conversation(conversation(), config(Some("zqxjv")))
        .await
        .expect("focused compact");

    let joined = |result: &ogham::CompactResult| {
        result
            .messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    };

    assert!(
        !joined(&plain).contains("zqxjv"),
        "needle should be dropped by compaction without a focus hint"
    );
    assert!(
        joined(&focused).contains("zqxjv"),
        "focus must steer compaction to keep the matching record"
    );
}
