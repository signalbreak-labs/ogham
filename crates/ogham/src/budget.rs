use crate::agent::{self, AgentPolicy};
use crate::ccr::CcrStore;
use crate::conversation::{self, ConversationConfig};
use crate::pipeline::DefaultCompressionPipeline;
use ogham_core::{Message, OghamError, Result, TokenCountKind, TokenCounter};
use std::sync::Arc;

/// Token budget for one LLM call.
#[derive(Debug, Clone)]
pub struct ContextBudget {
    /// Hard cap for the whole message list (prompt side), e.g. 180_000.
    pub total_limit: usize,
    /// Fraction of total_limit held back as safety margin for tokenizer
    /// inexactness. Default 0.05 when the counter is not exact, 0.0 when it is.
    /// effective_limit = total_limit - ceil(total_limit * margin).
    pub safety_margin: Option<f64>,
}

impl Default for ContextBudget {
    fn default() -> Self {
        Self {
            total_limit: 180_000,
            safety_margin: None,
        }
    }
}

/// What the cascade did.
#[derive(Debug, Clone, Default)]
pub struct BudgetReport {
    pub tokens_initial: usize,
    pub tokens_final: usize,
    pub effective_limit: usize,
    /// Names of the steps that ran, in order: subset of
    /// ["agent_rules", "compress_middle", "summarize_old", "drop_old"].
    pub steps_applied: Vec<String>,
    /// How the token counts in this report were produced. Estimated counts mean
    /// `tokens_initial`/`tokens_final` carry the counter's safety margin.
    pub count_kind: TokenCountKind,
    /// The fractional safety margin actually applied to derive `effective_limit`
    /// from `ContextBudget::total_limit`.
    pub safety_margin: f64,
}

/// Make `messages` fit `budget`. Mutates in place. Fail-closed per step;
/// fails CLOSED overall: if after all steps the list still exceeds the
/// effective limit, returns Err(BudgetExceeded) and `messages` is left in
/// its (partially compacted, still valid) state — the caller must not send it.
#[allow(clippy::ptr_arg)]
pub async fn enforce_budget(
    messages: &mut Vec<Message>,
    budget: &ContextBudget,
    counter: &dyn TokenCounter,
    pipeline: &DefaultCompressionPipeline,
    agent_policy: &AgentPolicy,
    ccr: Option<Arc<dyn CcrStore>>,
) -> Result<BudgetReport> {
    let count_kind = counter.count_kind();
    let margin = budget
        .safety_margin
        .unwrap_or_else(|| count_kind.safety_margin());
    let effective_limit = budget
        .total_limit
        .saturating_sub(((budget.total_limit as f64) * margin).ceil() as usize);

    let mut report = BudgetReport {
        effective_limit,
        count_kind,
        safety_margin: margin,
        ..BudgetReport::default()
    };

    let current = counter.count_messages(messages);
    report.tokens_initial = current;

    if current <= effective_limit {
        report.tokens_final = current;
        return Ok(report);
    }

    // Step 1: agent_rules
    agent::apply_agent_compression(messages, agent_policy, ccr.clone()).await?;
    report.steps_applied.push("agent_rules".to_string());
    let current = counter.count_messages(messages);
    if current <= effective_limit {
        report.tokens_final = current;
        return Ok(report);
    }

    // Step 2: compress_middle
    let preserve_recent = 4
        .max(agent::protected_tail_message_count(
            messages,
            agent_policy.protected_tail_tokens,
        ))
        .max(agent::recent_assistant_preserve_count(
            messages,
            agent_policy.keep_recent_assistant,
        ));
    let cfg2 = ConversationConfig {
        preserve_recent,
        compress_middle: messages.len().saturating_sub(preserve_recent),
        summary_old: false,
        bias_system: 0.8,
    };
    let ctx = ogham_core::CompressionContext {
        model: "default".to_string(),
        question_hint: None,
        max_tokens: None,
        reversible: false,
    };
    conversation::compress_conversation_history(messages, &cfg2, pipeline, &ctx).await?;
    report.steps_applied.push("compress_middle".to_string());
    let current = counter.count_messages(messages);
    if current <= effective_limit {
        report.tokens_final = current;
        return Ok(report);
    }

    // Step 3: summarize_old
    let preserve_recent = 4
        .max(agent::protected_tail_message_count(
            messages,
            agent_policy.protected_tail_tokens,
        ))
        .max(agent::recent_assistant_preserve_count(
            messages,
            agent_policy.keep_recent_assistant,
        ));
    let cfg3 = ConversationConfig {
        preserve_recent,
        compress_middle: 4,
        summary_old: true,
        bias_system: 0.8,
    };
    conversation::compress_conversation_history(messages, &cfg3, pipeline, &ctx).await?;
    report.steps_applied.push("summarize_old".to_string());
    let current = counter.count_messages(messages);
    if current <= effective_limit {
        report.tokens_final = current;
        return Ok(report);
    }

    // Step 4: drop_old.
    //
    // Messages are dropped in PAIR-SAFE GROUPS: an assistant message followed
    // by consecutive tool-role messages forms one atomic unit (the tool call
    // and its results). Provider APIs reject orphaned tool results, and models
    // are confused by calls whose results vanished — so a group is only
    // droppable if every member is, and it is removed whole. A protected tool
    // result (an error, or pinned) therefore protects its calling assistant
    // message too.
    //
    // Track the running total instead of recounting every message per drop;
    // any exit on "now under limit" is re-verified with a full recount below.
    let mut running = counter.count_messages(messages);
    while running > effective_limit && messages.len() > 4 {
        let tail_protected =
            agent::protected_tail_mask(messages, agent_policy.protected_tail_tokens);
        let latest_user_idx = messages
            .iter()
            .rposition(|m| agent::classify(m) == agent::AgentContentType::UserQuery);

        let member_droppable = |i: usize, m: &Message| {
            if tail_protected.get(i).copied().unwrap_or(false) {
                return false;
            }
            let kind = agent::classify(m);
            if matches!(
                kind,
                agent::AgentContentType::SystemInstruction
                    | agent::AgentContentType::ToolResultError
            ) {
                return false;
            }
            if let Some(idx) = latest_user_idx
                && i == idx
            {
                return false;
            }
            if m.metadata.get(ogham_core::meta_keys::PINNED) == Some(&"true".to_string()) {
                return false;
            }
            true
        };
        let is_tool_role = |m: &Message| m.role == "tool" || m.role == "function";

        // Find the oldest droppable group: range [start, end).
        let mut droppable_group: Option<(usize, usize)> = None;
        let mut i = 0;
        while i < messages.len() {
            // Group = one message, plus its trailing tool results if it is
            // an assistant message that initiated tool calls.
            let mut end = i + 1;
            if messages[i].role == "assistant" {
                while end < messages.len() && is_tool_role(&messages[end]) {
                    end += 1;
                }
            }
            let all_droppable = (i..end).all(|j| member_droppable(j, &messages[j]));
            // Never shrink below 4 messages.
            if all_droppable && messages.len() - (end - i) >= 4 {
                droppable_group = Some((i, end));
                break;
            }
            i = end;
        }

        if let Some((start, end)) = droppable_group {
            let removed: Vec<Message> = messages.drain(start..end).collect();
            running = running.saturating_sub(counter.count_messages(&removed));
            if running <= effective_limit {
                // Authoritative recount before exiting the loop, in case the
                // counter is not strictly additive across messages.
                running = counter.count_messages(messages);
            }
            if report.steps_applied.last().map(String::as_str) != Some("drop_old") {
                report.steps_applied.push("drop_old".to_string());
            }
        } else {
            break;
        }
    }

    let current = counter.count_messages(messages);
    report.tokens_final = current;

    if current > effective_limit {
        return Err(OghamError::BudgetExceeded {
            needed: current,
            limit: effective_limit,
        });
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentPolicy;
    use crate::ccr::in_memory::InMemoryCcrStore;
    use crate::pipeline::DefaultCompressionPipeline;
    use crate::token_counter::HeuristicCounter;
    use ogham_core::Message;
    use ogham_core::meta_keys;

    fn huge_budget() -> ContextBudget {
        ContextBudget {
            total_limit: 1_000_000,
            safety_margin: Some(0.0),
        }
    }

    fn tiny_budget() -> ContextBudget {
        ContextBudget {
            total_limit: 1,
            safety_margin: Some(0.0),
        }
    }

    fn make_text_msgs(n: usize, text_len: usize) -> Vec<Message> {
        (0..n)
            .map(|i| {
                Message::new(
                    if i % 2 == 0 { "user" } else { "assistant" },
                    "x".repeat(text_len),
                )
            })
            .collect()
    }

    #[tokio::test]
    async fn under_budget_is_noop() {
        let mut msgs = make_text_msgs(3, 10);
        let original: Vec<_> = msgs.iter().map(|m| m.content.clone()).collect();
        let counter = HeuristicCounter::new();
        let pipeline = DefaultCompressionPipeline::new(None, None);
        let policy = AgentPolicy::default();
        let report = enforce_budget(
            &mut msgs,
            &huge_budget(),
            &counter,
            &pipeline,
            &policy,
            None,
        )
        .await
        .unwrap();
        assert!(report.steps_applied.is_empty());
        for (orig, m) in original.iter().zip(msgs.iter()) {
            assert_eq!(orig, &m.content);
        }
    }

    #[tokio::test]
    async fn agent_rules_sufficient() {
        let mut msgs: Vec<Message> = (0..10)
            .map(|i| {
                let mut m = Message::new("tool", "a".repeat(200));
                m.metadata
                    .insert(meta_keys::TOOL_NAME.to_string(), format!("t{}", i));
                m
            })
            .collect();
        // Limit chosen so that clearing 7 old tool results (3 kept) fits under budget.
        let budget = ContextBudget {
            total_limit: 400,
            safety_margin: Some(0.0),
        };
        let counter = HeuristicCounter::new();
        let pipeline = DefaultCompressionPipeline::new(None, None);
        let policy = AgentPolicy::default();
        let ccr = Some(Arc::new(InMemoryCcrStore::new()) as Arc<dyn CcrStore>);
        let report = enforce_budget(&mut msgs, &budget, &counter, &pipeline, &policy, ccr)
            .await
            .unwrap();
        assert_eq!(report.steps_applied, vec!["agent_rules"]);
    }

    #[tokio::test]
    async fn cascade_reaches_drop() {
        let mut msgs = make_text_msgs(15, 5);
        let budget = ContextBudget {
            total_limit: 60,
            safety_margin: Some(0.0),
        };
        let counter = HeuristicCounter::new();
        let pipeline = DefaultCompressionPipeline::new(None, None);
        let policy = AgentPolicy::default();
        let report = enforce_budget(&mut msgs, &budget, &counter, &pipeline, &policy, None)
            .await
            .unwrap();
        assert!(
            report.steps_applied.contains(&"drop_old".to_string()),
            "expected drop_old in {:?}",
            report.steps_applied
        );
        assert!(report.tokens_final <= report.effective_limit);
    }

    #[tokio::test]
    async fn budget_exceeded_errors() {
        let mut msgs = vec![
            {
                let mut m = Message::new("system", "sys");
                m.metadata
                    .insert(meta_keys::PINNED.to_string(), "true".to_string());
                m
            },
            {
                let mut m = Message::new("user", "hi");
                m.metadata
                    .insert(meta_keys::PINNED.to_string(), "true".to_string());
                m
            },
        ];
        let counter = HeuristicCounter::new();
        let pipeline = DefaultCompressionPipeline::new(None, None);
        let policy = AgentPolicy::default();
        let result = enforce_budget(
            &mut msgs,
            &tiny_budget(),
            &counter,
            &pipeline,
            &policy,
            None,
        )
        .await;
        assert!(matches!(result, Err(OghamError::BudgetExceeded { .. })));
    }

    #[tokio::test]
    async fn errors_survive_cascade() {
        let mut msgs: Vec<Message> = (0..12)
            .map(|i| {
                if i == 5 {
                    let mut m = Message::new("tool", "Error: connection refused");
                    m.metadata
                        .insert(meta_keys::TOOL_NAME.to_string(), "db".to_string());
                    m
                } else {
                    Message::new(if i % 3 == 0 { "user" } else { "assistant" }, "x".repeat(5))
                }
            })
            .collect();
        let budget = ContextBudget {
            total_limit: 50,
            safety_margin: Some(0.0),
        };
        let counter = HeuristicCounter::new();
        let pipeline = DefaultCompressionPipeline::new(None, None);
        let policy = AgentPolicy::default();
        enforce_budget(&mut msgs, &budget, &counter, &pipeline, &policy, None)
            .await
            .unwrap();
        assert!(
            msgs.iter()
                .any(|m| m.content.contains("Error: connection refused"))
        );
    }

    /// 1 system + 10 (assistant call, tool result) pairs + 1 user, with a
    /// budget that forces heavy dropping. No tool result may survive without
    /// its assistant call, and vice versa.
    #[tokio::test]
    async fn drop_old_never_orphans_tool_results() {
        let mut msgs = vec![Message::new("system", "sys")];
        for i in 0..10 {
            let mut call = Message::new("assistant", format!("calling tool {}", i));
            call.metadata
                .insert(meta_keys::TOOL_CALL_ID.to_string(), format!("c{}", i));
            let mut result = Message::new("tool", "x".repeat(400));
            result
                .metadata
                .insert(meta_keys::TOOL_CALL_ID.to_string(), format!("c{}", i));
            msgs.push(call);
            msgs.push(result);
        }
        msgs.push(Message::new("user", "latest question"));

        let budget = ContextBudget {
            total_limit: 300,
            safety_margin: Some(0.0),
        };
        let counter = HeuristicCounter::new();
        let pipeline = DefaultCompressionPipeline::new(None, None);
        let policy = AgentPolicy::default();
        // No CCR store: agent rules can't clear, so the cascade must drop.
        let _ = enforce_budget(&mut msgs, &budget, &counter, &pipeline, &policy, None).await;

        let ids_for = |role: &str| -> Vec<String> {
            msgs.iter()
                .filter(|m| m.role == role)
                .filter_map(|m| m.metadata.get(meta_keys::TOOL_CALL_ID).cloned())
                .collect()
        };
        let call_ids = ids_for("assistant");
        let result_ids = ids_for("tool");
        for id in &result_ids {
            assert!(call_ids.contains(id), "tool result {id} lost its call");
        }
        for id in &call_ids {
            assert!(result_ids.contains(id), "tool call {id} lost its result");
        }
    }

    /// An error tool result is protected — and must protect the assistant
    /// message that invoked the tool, even under heavy pressure.
    #[tokio::test]
    async fn protected_tool_result_protects_its_call() {
        // 8 messages total so the summarize_old band stays empty and the
        // pair is exposed directly to drop_old (the step under test).
        let mut msgs = vec![Message::new("system", "sys")];
        msgs.push(Message::new("assistant", "calling the db tool"));
        msgs.push(Message::new("tool", "Error: connection refused"));
        for _ in 0..4 {
            msgs.push(Message::new("assistant", "x".repeat(200)));
        }
        msgs.push(Message::new("user", "latest"));

        let budget = ContextBudget {
            total_limit: 80,
            safety_margin: Some(0.0),
        };
        let counter = HeuristicCounter::new();
        let pipeline = DefaultCompressionPipeline::new(None, None);
        let policy = AgentPolicy::default();
        let _ = enforce_budget(&mut msgs, &budget, &counter, &pipeline, &policy, None).await;

        let err_idx = msgs
            .iter()
            .position(|m| m.content.contains("Error: connection refused"))
            .expect("error tool result must survive");
        assert!(err_idx > 0, "error result must not be first");
        assert_eq!(
            msgs[err_idx - 1].role,
            "assistant",
            "the assistant call preceding the protected error must survive"
        );
        assert!(msgs[err_idx - 1].content.contains("calling the db tool"));
    }

    #[tokio::test]
    async fn protected_tail_survives_budget_cascade() {
        let mut msgs = vec![Message::new("system", "sys")];
        for i in 0..12 {
            msgs.push(Message::new(
                "assistant",
                format!("old turn {i} {}", "x".repeat(400)),
            ));
        }
        msgs.push(Message::new("assistant", "TAIL_DECISION_MUST_SURVIVE"));
        msgs.push(Message::new("user", "latest question"));

        let counter = HeuristicCounter::new();
        let tail_budget = counter.count_messages(&msgs[msgs.len() - 2..]);
        let protected = [
            msgs[msgs.len() - 2].content.clone(),
            msgs[msgs.len() - 1].content.clone(),
        ];
        let budget = ContextBudget {
            total_limit: tail_budget + 32,
            safety_margin: Some(0.0),
        };
        let pipeline = DefaultCompressionPipeline::new(None, None);
        let policy = AgentPolicy {
            protected_tail_tokens: Some(tail_budget),
            ..Default::default()
        };

        let _ = enforce_budget(&mut msgs, &budget, &counter, &pipeline, &policy, None).await;

        let final_text = msgs
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(final_text.contains(&protected[0]));
        assert!(final_text.contains(&protected[1]));
    }

    #[tokio::test]
    async fn safety_margin_applied() {
        let mut msgs = vec![Message::new("user", "hi")];
        let budget = ContextBudget {
            total_limit: 100,
            safety_margin: None,
        };
        // Use a counter that is NOT exact.
        struct FakeCounter;
        impl TokenCounter for FakeCounter {
            fn count(&self, _text: &str) -> usize {
                1
            }
            fn is_exact(&self) -> bool {
                false
            }
        }
        let counter = FakeCounter;
        let pipeline = DefaultCompressionPipeline::new(None, None);
        let policy = AgentPolicy::default();
        let report = enforce_budget(&mut msgs, &budget, &counter, &pipeline, &policy, None)
            .await
            .unwrap();
        assert_eq!(report.effective_limit, 95);
    }

    #[tokio::test]
    async fn report_surfaces_count_kind_and_margin() {
        let pipeline = DefaultCompressionPipeline::new(None, None);
        let policy = AgentPolicy::default();
        let budget = ContextBudget {
            total_limit: 1_000_000,
            safety_margin: None,
        };

        // Estimated counter (no explicit margin) -> nonzero applied margin.
        let mut msgs = make_text_msgs(3, 10);
        let report = enforce_budget(
            &mut msgs,
            &budget,
            &HeuristicCounter::new(),
            &pipeline,
            &policy,
            None,
        )
        .await
        .unwrap();
        assert!(!report.count_kind.is_exact());
        assert!(report.safety_margin > 0.0);

        // Exact counter -> Exact kind, zero margin.
        struct ExactCounter;
        impl TokenCounter for ExactCounter {
            fn count(&self, t: &str) -> usize {
                t.len()
            }
            fn is_exact(&self) -> bool {
                true
            }
        }
        let mut msgs2 = make_text_msgs(3, 10);
        let report2 = enforce_budget(&mut msgs2, &budget, &ExactCounter, &pipeline, &policy, None)
            .await
            .unwrap();
        assert_eq!(report2.count_kind, TokenCountKind::Exact);
        assert_eq!(report2.safety_margin, 0.0);
    }

    /// A recent assistant reply pushed out of the default preserve window by
    /// trailing tool messages is compressed by default, but kept raw when
    /// `keep_recent_assistant` covers it. A separate large tool output supplies
    /// the compressible bulk so the budget is met either way (no drop).
    #[tokio::test]
    async fn keep_recent_assistant_preserves_recent_assistant_raw() {
        fn json_array(n: usize, needle_at: usize, needle: &str) -> String {
            let items: Vec<serde_json::Value> = (0..n)
                .map(|i| {
                    let tag = if i == needle_at { needle } else { "aaaaa" };
                    serde_json::json!({ "id": format!("{:03}", i), "tag": tag })
                })
                .collect();
            serde_json::to_string(&items).expect("serialize")
        }

        // assistant (idx 2) sits 4 messages from the end — outside preserve=4.
        let build = || {
            vec![
                Message::new("system", "sys"),
                Message::new("tool", json_array(200, 100, "aaaaa")),
                Message::new("assistant", json_array(60, 30, "zzzzz")),
                Message::new("tool", "t3"),
                Message::new("tool", "t4"),
                Message::new("tool", "t5"),
                Message::new("user", "latest"),
            ]
        };

        let budget = ContextBudget {
            total_limit: 800,
            safety_margin: Some(0.0),
        };
        let counter = HeuristicCounter::new();
        let pipeline = DefaultCompressionPipeline::with_builtin_compressors(
            None,
            crate::pipeline::DEFAULT_COMPRESSORS,
        )
        .expect("pipeline");
        let policy = |keep: usize| AgentPolicy {
            keep_recent_tool_results: 8,
            clear_old_tool_results: false,
            keep_recent_assistant: keep,
            protected_tail_tokens: None,
        };
        let joined = |msgs: &[Message]| {
            msgs.iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        };

        let mut without = build();
        enforce_budget(&mut without, &budget, &counter, &pipeline, &policy(0), None)
            .await
            .unwrap();
        assert!(
            !joined(&without).contains("zzzzz"),
            "without keep_recent_assistant the recent assistant is compressed away"
        );

        let mut with = build();
        enforce_budget(&mut with, &budget, &counter, &pipeline, &policy(1), None)
            .await
            .unwrap();
        assert!(
            joined(&with).contains("zzzzz"),
            "keep_recent_assistant must preserve the recent assistant raw"
        );
    }
}
