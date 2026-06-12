use crate::ccr::CcrStore;
use ogham_core::{Message, OVERHEAD_PER_MESSAGE, Result, TokenCounter, meta_keys};
use std::sync::Arc;

/// Agent-semantic classification of a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentContentType {
    SystemInstruction,
    UserQuery,
    AssistantReply,
    ToolResultSuccess,
    ToolResultError,
    Unknown,
}

impl AgentContentType {
    pub fn as_str(&self) -> &'static str {
        match self {
            AgentContentType::SystemInstruction => "system_instruction",
            AgentContentType::UserQuery => "user_query",
            AgentContentType::AssistantReply => "assistant_reply",
            AgentContentType::ToolResultSuccess => "tool_result_success",
            AgentContentType::ToolResultError => "tool_result_error",
            AgentContentType::Unknown => "unknown",
        }
    }

    pub fn from_str_loose(s: &str) -> Self {
        match s {
            "system_instruction" => AgentContentType::SystemInstruction,
            "user_query" => AgentContentType::UserQuery,
            "assistant_reply" => AgentContentType::AssistantReply,
            "tool_result_success" => AgentContentType::ToolResultSuccess,
            "tool_result_error" => AgentContentType::ToolResultError,
            _ => AgentContentType::Unknown,
        }
    }
}

/// Substrings that mark content as an error trace. Shared with the
/// extractive summarizer so both features agree on what an error is.
pub const ERROR_PATTERNS: &[&str] = &[
    "Error:",
    "error:",
    "ERROR",
    "Traceback (most recent call last)",
    "panicked at",
    "Exception",
    "FAILED",
    "stderr:",
];

/// Truncate to at most `max` bytes without splitting a UTF-8 char.
fn prefix_to_char_boundary(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Classify one message. Precedence (first match wins):
/// 1. `metadata[AGENT_CONTENT_TYPE]` if present and parseable
/// 2. role == "system" -> SystemInstruction
/// 3. role == "tool" or role == "function":
///    `metadata[TOOL_STATUS] == "error"` -> ToolResultError
///    else if content matches ERROR_PATTERN -> ToolResultError
///    else -> ToolResultSuccess
/// 4. role == "user" -> UserQuery
/// 5. role == "assistant" -> AssistantReply
/// 6. anything else -> Unknown
///
/// ERROR_PATTERN (case-sensitive, any-of, checked against the first 512 bytes only):
///   "Error:", "error:", "ERROR", "Traceback (most recent call last)",
///   "panicked at", "Exception", "FAILED", "stderr:"
pub fn classify(msg: &Message) -> AgentContentType {
    if let Some(tag) = msg.metadata.get(meta_keys::AGENT_CONTENT_TYPE) {
        let parsed = AgentContentType::from_str_loose(tag);
        if !matches!(parsed, AgentContentType::Unknown) {
            return parsed;
        }
    }

    match msg.role.as_str() {
        "system" => AgentContentType::SystemInstruction,
        "tool" | "function" => {
            if msg.metadata.get(meta_keys::TOOL_STATUS) == Some(&"error".to_string()) {
                return AgentContentType::ToolResultError;
            }
            let window = prefix_to_char_boundary(&msg.content, 512);
            if ERROR_PATTERNS.iter().any(|p| window.contains(p)) {
                AgentContentType::ToolResultError
            } else {
                AgentContentType::ToolResultSuccess
            }
        }
        "user" => AgentContentType::UserQuery,
        "assistant" => AgentContentType::AssistantReply,
        _ => AgentContentType::Unknown,
    }
}

/// Policy knobs. All counts are in *messages of that kind*, newest-first.
#[derive(Debug, Clone)]
pub struct AgentPolicy {
    /// Keep this many most-recent ToolResultSuccess messages raw. Default 3.
    pub keep_recent_tool_results: usize,
    /// Tool results older than the kept window are CLEARED (replaced by a stub +
    /// CCR marker) if true, otherwise compressed via the pipeline. Default true.
    pub clear_old_tool_results: bool,
    /// Keep this many most-recent AssistantReply messages raw. Default 2.
    pub keep_recent_assistant: usize,
    /// Keep any message overlapping this many estimated trailing tokens raw.
    /// Default None preserves the 0.2.x count-only behavior.
    pub protected_tail_tokens: Option<usize>,
}

impl Default for AgentPolicy {
    fn default() -> Self {
        Self {
            keep_recent_tool_results: 3,
            clear_old_tool_results: true,
            keep_recent_assistant: 2,
            protected_tail_tokens: None,
        }
    }
}

fn estimated_message_tokens(
    counter: &crate::token_counter::HeuristicCounter,
    msg: &Message,
) -> usize {
    counter.count(&msg.content) + OVERHEAD_PER_MESSAGE
}

/// Return a message mask for `protected_tail_tokens`.
///
/// Protection is message-granular. If the requested suffix boundary falls in
/// the middle of a message, the whole message is protected so no protected
/// token can be rewritten by a later clearing, compression, summary, or drop.
pub fn protected_tail_mask(
    messages: &[Message],
    protected_tail_tokens: Option<usize>,
) -> Vec<bool> {
    let mut protected = vec![false; messages.len()];
    let Some(budget) = protected_tail_tokens else {
        return protected;
    };
    if budget == 0 {
        return protected;
    }

    let counter = crate::token_counter::HeuristicCounter::new();
    let mut tail_tokens = 0usize;
    for (idx, msg) in messages.iter().enumerate().rev() {
        if tail_tokens >= budget {
            break;
        }
        protected[idx] = true;
        tail_tokens = tail_tokens.saturating_add(estimated_message_tokens(&counter, msg));
    }
    protected
}

/// Number of newest messages protected by `protected_tail_tokens`.
pub fn protected_tail_message_count(
    messages: &[Message],
    protected_tail_tokens: Option<usize>,
) -> usize {
    protected_tail_mask(messages, protected_tail_tokens)
        .iter()
        .filter(|&&is_protected| is_protected)
        .count()
}

/// What was done, for observability.
#[derive(Debug, Clone, Default)]
pub struct AgentCompressionStats {
    pub tool_results_cleared: usize,
    pub tool_results_kept_raw: usize,
    pub errors_preserved: usize,
    pub protected_tail_messages: usize,
    pub protected_tail_tokens: usize,
    pub tokens_before: usize,
    pub tokens_after: usize,
}

/// Apply agent-aware rules in place. Fail-closed: any per-message failure
/// (e.g. CCR save error) leaves that message unchanged.
#[allow(clippy::ptr_arg)]
pub async fn apply_agent_compression(
    messages: &mut Vec<Message>,
    policy: &AgentPolicy,
    ccr: Option<Arc<dyn CcrStore>>,
) -> Result<AgentCompressionStats> {
    let counter = crate::token_counter::HeuristicCounter::new();
    let tokens_before = counter.count_messages(messages);
    let tail_protected = protected_tail_mask(messages, policy.protected_tail_tokens);

    let mut seen_tool_success = 0usize;
    let mut stats = AgentCompressionStats {
        protected_tail_messages: tail_protected
            .iter()
            .filter(|&&is_protected| is_protected)
            .count(),
        protected_tail_tokens: messages
            .iter()
            .zip(tail_protected.iter())
            .filter(|(_, is_protected)| **is_protected)
            .map(|(msg, _)| estimated_message_tokens(&counter, msg))
            .sum(),
        ..AgentCompressionStats::default()
    };

    // Walk newest to oldest.
    for (idx, msg) in messages.iter_mut().enumerate().rev() {
        let kind = classify(msg);
        let protected_by_tail = tail_protected[idx];
        match kind {
            AgentContentType::SystemInstruction
            | AgentContentType::UserQuery
            | AgentContentType::AssistantReply
            | AgentContentType::Unknown => {
                // Never modified.
            }
            AgentContentType::ToolResultError => {
                stats.errors_preserved += 1;
            }
            AgentContentType::ToolResultSuccess => {
                if msg.metadata.get(meta_keys::PINNED) == Some(&"true".to_string()) {
                    continue;
                }
                let protected_by_count = seen_tool_success < policy.keep_recent_tool_results;
                seen_tool_success += 1;
                if protected_by_tail || protected_by_count {
                    stats.tool_results_kept_raw += 1;
                    continue;
                }
                if policy.clear_old_tool_results
                    && let Some(ref store) = ccr
                {
                    let hash = crate::ccr::compute_key(msg.content.as_bytes());
                    if store.save(&hash, &msg.content, None).await.is_ok() {
                        let tool_name = msg
                            .metadata
                            .get(meta_keys::TOOL_NAME)
                            .cloned()
                            .unwrap_or_else(|| "unknown".to_string());
                        let n = counter.count(&msg.content);
                        msg.content = format!(
                            "[tool:{tool_name}] result cleared ({n} tokens) — original retrievable via <<ccr:{hash}>>"
                        );
                        msg.metadata.insert(meta_keys::CCR_ID.to_string(), hash);
                        msg.metadata.insert(
                            meta_keys::AGENT_CONTENT_TYPE.to_string(),
                            "tool_result_success".to_string(),
                        );
                        stats.tool_results_cleared += 1;
                    }
                    // On Err, leave message unchanged (fail-closed).
                }
                // If ccr is None, clearing is skipped (fail-closed).
            }
        }
    }

    stats.tokens_before = tokens_before;
    stats.tokens_after = counter.count_messages(messages);
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccr::in_memory::InMemoryCcrStore;

    fn tool_msg(name: &str, content: &str) -> Message {
        let mut msg = Message::new("tool", content);
        msg.metadata
            .insert(meta_keys::TOOL_NAME.to_string(), name.to_string());
        msg
    }

    #[test]
    fn classify_precedence_metadata_wins() {
        let mut msg = Message::new("user", "hi");
        msg.metadata.insert(
            meta_keys::AGENT_CONTENT_TYPE.to_string(),
            "tool_result_error".to_string(),
        );
        assert_eq!(classify(&msg), AgentContentType::ToolResultError);
    }

    #[test]
    fn classify_tool_error_by_content() {
        let msg = Message::new(
            "tool",
            "Traceback (most recent call last):\n  File \"x.py\"",
        );
        assert_eq!(classify(&msg), AgentContentType::ToolResultError);
    }

    #[test]
    fn classify_multibyte_content_no_panic() {
        // Byte 512 must fall inside a multibyte char to hit the old panic.
        let msg = Message::new("tool", "日本語のテキスト".repeat(60));
        let _ = classify(&msg);
    }

    #[tokio::test]
    async fn errors_never_cleared() {
        let mut msgs: Vec<Message> = (0..10)
            .map(|i| {
                let mut m = Message::new(
                    "tool",
                    if i % 2 == 0 {
                        "ok output".to_string()
                    } else {
                        "Error: something failed".to_string()
                    },
                );
                m.metadata
                    .insert(meta_keys::TOOL_NAME.to_string(), format!("tool_{}", i));
                m
            })
            .collect();

        let original_errors: Vec<String> = msgs
            .iter()
            .filter(|m| classify(m) == AgentContentType::ToolResultError)
            .map(|m| m.content.clone())
            .collect();

        let policy = AgentPolicy {
            keep_recent_tool_results: 0,
            clear_old_tool_results: true,
            ..Default::default()
        };
        let ccr = Some(Arc::new(InMemoryCcrStore::new()) as Arc<dyn CcrStore>);
        apply_agent_compression(&mut msgs, &policy, ccr)
            .await
            .unwrap();

        let current_errors: Vec<String> = msgs
            .iter()
            .filter(|m| m.content.contains("Error:"))
            .map(|m| m.content.clone())
            .collect();
        assert_eq!(current_errors, original_errors);
    }

    #[tokio::test]
    async fn recent_tool_results_kept_raw() {
        let mut msgs: Vec<Message> = (0..6)
            .map(|i| tool_msg(&format!("tool_{}", i), &format!("output {}", i)))
            .collect();

        let policy = AgentPolicy::default();
        let ccr = Some(Arc::new(InMemoryCcrStore::new()) as Arc<dyn CcrStore>);
        apply_agent_compression(&mut msgs, &policy, ccr)
            .await
            .unwrap();

        let unchanged: Vec<_> = msgs
            .iter()
            .filter(|m| m.content.starts_with("output "))
            .collect();
        assert_eq!(unchanged.len(), 3, "exactly 3 newest kept raw");

        let cleared: Vec<_> = msgs
            .iter()
            .filter(|m| m.content.starts_with("[tool:"))
            .collect();
        assert_eq!(cleared.len(), 3, "exactly 3 oldest cleared");

        for c in &cleared {
            assert!(c.content.contains("<<ccr:"));
        }
    }

    #[tokio::test]
    async fn protected_tail_keeps_suffix_and_clears_older_tools() {
        let tail_assistant = Message::new("assistant", "Decision: keep the live edit plan");
        let tail_tool = tool_msg("tail", "TAIL_TOOL_OUTPUT_MUST_SURVIVE");
        let counter = crate::token_counter::HeuristicCounter::new();
        let tail_budget = estimated_message_tokens(&counter, &tail_assistant)
            + estimated_message_tokens(&counter, &tail_tool);
        let mut msgs = vec![
            tool_msg("old_0", "old output 0"),
            tool_msg("old_1", "old output 1"),
            tool_msg("old_2", "old output 2"),
            tail_assistant.clone(),
            tail_tool.clone(),
        ];

        let policy = AgentPolicy {
            keep_recent_tool_results: 0,
            clear_old_tool_results: true,
            protected_tail_tokens: Some(tail_budget),
            ..Default::default()
        };
        let ccr = Some(Arc::new(InMemoryCcrStore::new()) as Arc<dyn CcrStore>);
        let stats = apply_agent_compression(&mut msgs, &policy, ccr)
            .await
            .unwrap();

        assert!(msgs[0].content.starts_with("[tool:old_0] result cleared"));
        assert!(msgs[1].content.starts_with("[tool:old_1] result cleared"));
        assert!(msgs[2].content.starts_with("[tool:old_2] result cleared"));
        assert_eq!(msgs[3].content, tail_assistant.content);
        assert_eq!(msgs[4].content, tail_tool.content);
        assert_eq!(stats.tool_results_cleared, 3);
        assert_eq!(stats.tool_results_kept_raw, 1);
        assert_eq!(stats.protected_tail_messages, 2);
        assert_eq!(stats.protected_tail_tokens, tail_budget);
    }

    #[tokio::test]
    async fn protected_tail_unions_with_recent_tool_count() {
        let base_msgs: Vec<Message> = (0..5)
            .map(|i| tool_msg(&format!("tool_{i}"), &format!("result {i}")))
            .collect();
        let counter = crate::token_counter::HeuristicCounter::new();
        let one_msg = estimated_message_tokens(&counter, &base_msgs[0]);
        let newest_three_budget = one_msg * 3;

        let mut outside_tail = base_msgs.clone();
        let policy = AgentPolicy {
            keep_recent_tool_results: 3,
            clear_old_tool_results: true,
            protected_tail_tokens: Some(newest_three_budget),
            ..Default::default()
        };
        apply_agent_compression(
            &mut outside_tail,
            &policy,
            Some(Arc::new(InMemoryCcrStore::new()) as Arc<dyn CcrStore>),
        )
        .await
        .unwrap();
        assert!(
            outside_tail[1].content.starts_with("[tool:tool_1]"),
            "4th-newest tool result outside the token tail must still clear"
        );

        let mut inside_tail = base_msgs;
        let policy = AgentPolicy {
            protected_tail_tokens: Some(newest_three_budget + 1),
            ..policy
        };
        apply_agent_compression(
            &mut inside_tail,
            &policy,
            Some(Arc::new(InMemoryCcrStore::new()) as Arc<dyn CcrStore>),
        )
        .await
        .unwrap();
        assert_eq!(
            inside_tail[1].content, "result 1",
            "4th-newest tool result inside the token tail must stay raw"
        );
        assert!(
            inside_tail[0].content.starts_with("[tool:tool_0]"),
            "older tool result outside both protections should still clear"
        );
    }

    #[tokio::test]
    async fn cleared_content_roundtrip() {
        let mut msg = Message::new("tool", "original content here");
        msg.metadata
            .insert(meta_keys::TOOL_NAME.to_string(), "my_tool".to_string());
        let mut msgs = vec![msg];

        let ccr = Arc::new(InMemoryCcrStore::new());
        let policy = AgentPolicy {
            keep_recent_tool_results: 0,
            clear_old_tool_results: true,
            ..Default::default()
        };
        apply_agent_compression(&mut msgs, &policy, Some(ccr.clone()))
            .await
            .unwrap();

        let hash = msgs[0].metadata.get(meta_keys::CCR_ID).unwrap();
        let retrieved = ccr.retrieve(hash).await.unwrap();
        assert_eq!(retrieved, Some("original content here".to_string()));
    }

    #[tokio::test]
    async fn no_ccr_store_means_no_clearing() {
        let mut msgs: Vec<Message> = (0..6)
            .map(|i| Message::new("tool", format!("output {}", i)))
            .collect();

        let policy = AgentPolicy {
            keep_recent_tool_results: 0,
            clear_old_tool_results: true,
            ..Default::default()
        };
        apply_agent_compression(&mut msgs, &policy, None)
            .await
            .unwrap();

        for (i, m) in msgs.iter().enumerate() {
            assert_eq!(m.content, format!("output {}", i));
        }
    }

    #[tokio::test]
    async fn pinned_never_cleared() {
        let mut msg = Message::new("tool", "secret output");
        msg.metadata
            .insert(meta_keys::PINNED.to_string(), "true".to_string());
        let mut msgs = vec![msg];

        let policy = AgentPolicy {
            keep_recent_tool_results: 0,
            clear_old_tool_results: true,
            ..Default::default()
        };
        let ccr = Some(Arc::new(InMemoryCcrStore::new()) as Arc<dyn CcrStore>);
        apply_agent_compression(&mut msgs, &policy, ccr)
            .await
            .unwrap();

        assert_eq!(msgs[0].content, "secret output");
    }

    #[tokio::test]
    async fn system_and_user_untouched() {
        let mut msgs = vec![
            Message::new("system", "You are helpful."),
            Message::new("user", "What is 2+2?"),
            Message::new("tool", "output 1"),
            Message::new("tool", "output 2"),
        ];

        let policy = AgentPolicy {
            keep_recent_tool_results: 0,
            clear_old_tool_results: true,
            ..Default::default()
        };
        let ccr = Some(Arc::new(InMemoryCcrStore::new()) as Arc<dyn CcrStore>);
        apply_agent_compression(&mut msgs, &policy, ccr)
            .await
            .unwrap();

        assert_eq!(msgs[0].content, "You are helpful.");
        assert_eq!(msgs[1].content, "What is 2+2?");
    }
}
