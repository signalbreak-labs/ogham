//! Conversation-history compression.
//!
//! Multi-turn conversations grow without bound.  This module applies
//! age-based compression: recent turns are preserved verbatim, middle
//! turns are lightly compressed, and old turns are aggressively crushed.
//!
//! The design follows the insight from headroom: the model only needs
//! *context* from older turns, not exact token sequences.  By compressing
//! old turns we keep the total prompt under `max_tokens` while
//! retaining semantic continuity.

use crate::pipeline::DefaultCompressionPipeline;
use ogham_core::{CompressionContext, CompressionPipeline, Message, Result, meta_keys};

/// How aggressively to compress different age bands.
#[derive(Debug, Clone, Copy)]
pub struct ConversationConfig {
    /// Number of most-recent turns to preserve verbatim.
    pub preserve_recent: usize,
    /// Number of middle turns to compress with the default pipeline.
    pub compress_middle: usize,
    /// Whether to replace very old turns with a single summary message.
    pub summary_old: bool,
    /// Bias toward preserving system/tool messages.
    pub bias_system: f64,
}

impl Default for ConversationConfig {
    fn default() -> Self {
        Self {
            preserve_recent: 4,
            compress_middle: 8,
            summary_old: true,
            bias_system: 0.8,
        }
    }
}

/// Compress a conversation in place.
///
/// 1. **Preserve** the `preserve_recent` most recent turns.  
/// 2. **Compress** the next `compress_middle` turns via the default
///    pipeline (SmartCrusher for JSON, LogStripper for logs, etc.).  
/// 3. **Summarise** any older turns into a single system message if
///    `summary_old` is enabled, otherwise drop them.
///
/// Returns stats for the compression that was actually applied.
pub async fn compress_conversation_history(
    messages: &mut Vec<Message>,
    config: &ConversationConfig,
    pipeline: &DefaultCompressionPipeline,
    _ctx: &CompressionContext,
) -> Result<ConversationStats> {
    if messages.len() <= config.preserve_recent {
        return Ok(ConversationStats {
            preserved_recent: messages.len(),
            total_turns_after: messages.len(),
            ..ConversationStats::default()
        });
    }

    // Split into bands: [old..middle..recent]
    let recent_start = messages.len().saturating_sub(config.preserve_recent);
    let middle_start = align_band_start(
        messages,
        recent_start.saturating_sub(config.compress_middle),
    );
    // Never summarise or drop across a pinned message: shrink the old band so a
    // pinned message falls into the compress band, which preserves it verbatim.
    let middle_start = clamp_band_start_before_pinned(messages, middle_start);

    let mut stats = ConversationStats::default();

    // --- Band 1: Old turns (optional summary) ---
    if middle_start > 0 && config.summary_old {
        let old_turns = messages.drain(..middle_start).collect::<Vec<_>>();
        let summary = summarise_turns(&old_turns);
        messages.insert(
            0,
            Message {
                role: "system".into(),
                content: format!("[Earlier conversation context]: {}", summary),
                metadata: Default::default(),
            },
        );
        stats.old_turns_summarised = old_turns.len();
        stats.old_summary_tokens = crate::token_est::count_tokens(&messages[0].content);
    } else if middle_start > 0 {
        stats.old_turns_dropped = middle_start;
        messages.drain(..middle_start);
    }

    // --- Band 2: Middle turns (compress via pipeline) ---
    let compress_end = messages.len().saturating_sub(config.preserve_recent);
    if compress_end > 0 {
        let middle = messages.drain(..compress_end).collect::<Vec<_>>();
        let compressed = pipeline.run(&middle).await?;
        stats.middle_turns_compressed = compressed.messages.len();
        stats.original_middle_tokens = compressed.stats.original_tokens;
        stats.compressed_middle_tokens = compressed.stats.compressed_tokens;

        // Honor the PINNED contract: a pinned message is never rewritten.
        let mut out_msgs = compressed.messages;
        for (i, original) in middle.iter().enumerate() {
            if original.metadata.get(meta_keys::PINNED).map(String::as_str) == Some("true") {
                out_msgs[i] = original.clone();
            }
        }
        for msg in out_msgs.into_iter().rev() {
            messages.insert(0, msg);
        }
    }

    // --- Band 3: Recent turns (preserve verbatim) ---
    stats.preserved_recent = config.preserve_recent.min(messages.len());
    stats.total_turns_after = messages.len();

    Ok(stats)
}

/// Like compress_conversation_history, but old turns are summarized with the
/// given Summarizer into a StructuredSummary rendered as markdown, inserted as
/// one system message prefixed "[Earlier conversation summary]\n".
pub async fn compress_conversation_history_with_summarizer(
    messages: &mut Vec<Message>,
    config: &ConversationConfig,
    pipeline: &DefaultCompressionPipeline,
    summarizer: &dyn crate::memory::Summarizer,
) -> Result<ConversationStats> {
    if messages.len() <= config.preserve_recent {
        return Ok(ConversationStats {
            preserved_recent: messages.len(),
            total_turns_after: messages.len(),
            ..ConversationStats::default()
        });
    }

    let recent_start = messages.len().saturating_sub(config.preserve_recent);
    let middle_start = align_band_start(
        messages,
        recent_start.saturating_sub(config.compress_middle),
    );
    // Never summarise or drop across a pinned message: shrink the old band so a
    // pinned message falls into the compress band, which preserves it verbatim.
    let middle_start = clamp_band_start_before_pinned(messages, middle_start);

    let mut stats = ConversationStats::default();

    // --- Band 1: Old turns (summarizer) ---
    if middle_start > 0 && config.summary_old {
        let old_turns = messages.drain(..middle_start).collect::<Vec<_>>();
        match summarizer.summarize(&old_turns, None).await {
            Ok(summary) => {
                messages.insert(
                    0,
                    Message::new(
                        "system",
                        format!(
                            "[Earlier conversation summary]\n{}",
                            summary.render_markdown()
                        ),
                    ),
                );
                stats.old_turns_summarised = old_turns.len();
                stats.old_summary_tokens = crate::token_est::count_tokens(&messages[0].content);
            }
            Err(_) => {
                // Fail-closed: fall back to the existing summarise_turns output.
                let fallback = summarise_turns(&old_turns);
                messages.insert(
                    0,
                    Message {
                        role: "system".into(),
                        content: format!("[Earlier conversation context]: {}", fallback),
                        metadata: Default::default(),
                    },
                );
                stats.old_turns_summarised = old_turns.len();
                stats.old_summary_tokens = crate::token_est::count_tokens(&messages[0].content);
            }
        }
    } else if middle_start > 0 {
        stats.old_turns_dropped = middle_start;
        messages.drain(..middle_start);
    }

    // --- Band 2: Middle turns (compress via pipeline) ---
    let compress_end = messages.len().saturating_sub(config.preserve_recent);
    if compress_end > 0 {
        let middle = messages.drain(..compress_end).collect::<Vec<_>>();
        let compressed = pipeline.run(&middle).await?;
        stats.middle_turns_compressed = compressed.messages.len();
        stats.original_middle_tokens = compressed.stats.original_tokens;
        stats.compressed_middle_tokens = compressed.stats.compressed_tokens;

        // Honor the PINNED contract: a pinned message is never rewritten.
        let mut out_msgs = compressed.messages;
        for (i, original) in middle.iter().enumerate() {
            if original.metadata.get(meta_keys::PINNED).map(String::as_str) == Some("true") {
                out_msgs[i] = original.clone();
            }
        }
        for msg in out_msgs.into_iter().rev() {
            messages.insert(0, msg);
        }
    }

    // --- Band 3: Recent turns (preserve verbatim) ---
    stats.preserved_recent = config.preserve_recent.min(messages.len());
    stats.total_turns_after = messages.len();

    Ok(stats)
}

/// Move a band-start index backward so it never lands on a tool-role
/// message. Draining a prefix that ends between an assistant tool call and
/// its results would orphan the results — provider APIs reject that.
fn align_band_start(messages: &[Message], mut idx: usize) -> usize {
    while idx > 0
        && idx < messages.len()
        && (messages[idx].role == "tool" || messages[idx].role == "function")
    {
        idx -= 1;
    }
    idx
}

/// Shrink a band-start so the summarised/dropped prefix never includes a pinned
/// message — honoring the `PINNED` contract ("never compressed, cleared, or
/// summarized"). Pinned messages fall into the compress band instead, which
/// passes them through verbatim.
fn clamp_band_start_before_pinned(messages: &[Message], idx: usize) -> usize {
    messages
        .iter()
        .take(idx)
        .position(|m| m.metadata.get(meta_keys::PINNED).map(String::as_str) == Some("true"))
        .unwrap_or(idx)
}

/// Truncate to at most `max` bytes without splitting a UTF-8 char.
fn truncate_to_char_boundary(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Simple extractive summary: concatenate the first sentence of each
/// old turn, truncated to ~200 tokens.
fn summarise_turns(turns: &[Message]) -> String {
    let mut parts = Vec::new();
    for turn in turns {
        let sentence = turn
            .content
            .split(['.', '!', '?'])
            .next()
            .unwrap_or(&turn.content)
            .trim();
        if !sentence.is_empty() {
            parts.push(sentence.to_string());
        }
    }
    let joined = parts.join(" | ");
    // Hard cap to avoid bloating the prompt.
    let cap = 800;
    if joined.len() > cap {
        format!("{}…", truncate_to_char_boundary(&joined, cap))
    } else {
        joined
    }
}

/// Per-band statistics for conversation compression.
#[derive(Debug, Clone, Default)]
pub struct ConversationStats {
    pub preserved_recent: usize,
    pub middle_turns_compressed: usize,
    pub original_middle_tokens: usize,
    pub compressed_middle_tokens: usize,
    pub old_turns_summarised: usize,
    pub old_turns_dropped: usize,
    pub old_summary_tokens: usize,
    pub total_turns_after: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::DefaultCompressionPipeline;

    fn make_msgs(n: usize) -> Vec<Message> {
        (0..n)
            .map(|i| {
                Message::new(
                    if i % 2 == 0 { "user" } else { "assistant" },
                    format!("Message number {} with some content here.", i),
                )
            })
            .collect()
    }

    #[tokio::test]
    async fn test_no_compression_needed() {
        let mut msgs = make_msgs(3);
        let pipeline = DefaultCompressionPipeline::new(None, None);
        let ctx = CompressionContext {
            model: "gpt-4".into(),
            question_hint: None,
            max_tokens: None,
            reversible: true,
        };
        let stats = compress_conversation_history(
            &mut msgs,
            &ConversationConfig::default(),
            &pipeline,
            &ctx,
        )
        .await
        .unwrap();
        assert_eq!(stats.preserved_recent, 3);
        assert_eq!(msgs.len(), 3);
    }

    #[tokio::test]
    async fn test_middle_compression() {
        let mut msgs = make_msgs(12);
        let pipeline = DefaultCompressionPipeline::new(None, None);
        let ctx = CompressionContext {
            model: "gpt-4".into(),
            question_hint: None,
            max_tokens: None,
            reversible: true,
        };
        let config = ConversationConfig {
            preserve_recent: 4,
            compress_middle: 4,
            summary_old: false,
            bias_system: 0.8,
        };
        let stats = compress_conversation_history(&mut msgs, &config, &pipeline, &ctx)
            .await
            .unwrap();
        assert_eq!(stats.preserved_recent, 4);
        assert_eq!(stats.middle_turns_compressed, 4);
        assert_eq!(stats.old_turns_dropped, 4);
        assert_eq!(msgs.len(), 8); // 4 compressed + 4 preserved
    }

    #[tokio::test]
    async fn test_old_summary() {
        let mut msgs = make_msgs(12);
        let pipeline = DefaultCompressionPipeline::new(None, None);
        let ctx = CompressionContext {
            model: "gpt-4".into(),
            question_hint: None,
            max_tokens: None,
            reversible: true,
        };
        let config = ConversationConfig {
            preserve_recent: 4,
            compress_middle: 4,
            summary_old: true,
            bias_system: 0.8,
        };
        let stats = compress_conversation_history(&mut msgs, &config, &pipeline, &ctx)
            .await
            .unwrap();
        assert_eq!(stats.old_turns_summarised, 4);
        assert_eq!(msgs.len(), 9); // 1 summary + 4 compressed + 4 preserved
        assert!(msgs[0].role == "system");
        assert!(
            msgs[0]
                .content
                .starts_with("[Earlier conversation context]")
        );
    }

    #[tokio::test]
    async fn test_summary_multibyte_no_panic() {
        // 12 turns of multibyte text long enough to force summary truncation at byte 800.
        let mut msgs: Vec<Message> = (0..12)
            .map(|i| {
                Message::new(
                    if i % 2 == 0 { "user" } else { "assistant" },
                    "héllo wörld 日本語のテキストです。".repeat(20),
                )
            })
            .collect();
        let pipeline = DefaultCompressionPipeline::new(None, None);
        let ctx = CompressionContext {
            model: "m".into(),
            question_hint: None,
            max_tokens: None,
            reversible: true,
        };
        let cfg = ConversationConfig {
            preserve_recent: 2,
            compress_middle: 2,
            summary_old: true,
            bias_system: 0.8,
        };
        let stats = compress_conversation_history(&mut msgs, &cfg, &pipeline, &ctx)
            .await
            .unwrap();
        assert!(stats.old_turns_summarised > 0);
        assert!(msgs[0].content.is_char_boundary(msgs[0].content.len()));
    }

    #[tokio::test]
    async fn with_summarizer_fail_closed() {
        use crate::memory::Summarizer;
        use async_trait::async_trait;
        use ogham_core::{OghamError, Result};

        struct FailingSummarizer;

        #[async_trait]
        impl Summarizer for FailingSummarizer {
            async fn summarize(
                &self,
                _turns: &[Message],
                _existing: Option<&crate::memory::StructuredSummary>,
            ) -> Result<crate::memory::StructuredSummary> {
                Err(OghamError::CompressionFailed("fail".into()))
            }
        }

        let mut msgs = make_msgs(12);
        let pipeline = DefaultCompressionPipeline::new(None, None);
        let config = ConversationConfig {
            preserve_recent: 4,
            compress_middle: 4,
            summary_old: true,
            bias_system: 0.8,
        };
        let stats = compress_conversation_history_with_summarizer(
            &mut msgs,
            &config,
            &pipeline,
            &FailingSummarizer,
        )
        .await
        .unwrap();
        assert_eq!(stats.old_turns_summarised, 4);
        assert_eq!(msgs.len(), 9); // 1 summary + 4 compressed + 4 preserved
        assert!(msgs[0].role == "system");
        assert!(
            msgs[0]
                .content
                .starts_with("[Earlier conversation context]")
                || msgs[0]
                    .content
                    .starts_with("[Earlier conversation summary]")
        );
    }

    #[tokio::test]
    async fn pinned_middle_message_is_not_compressed() {
        // A pinned, highly-compressible message sitting in the middle band must
        // pass through byte-for-byte (the documented PINNED contract).
        let big: Vec<_> = (0..60)
            .map(|i| serde_json::json!({ "id": i, "name": format!("item_{i}"), "score": i }))
            .collect();
        let big_json = serde_json::to_string(&big).unwrap();
        let mut pinned = Message::new("tool", big_json.clone());
        pinned
            .metadata
            .insert(meta_keys::PINNED.to_string(), "true".to_string());

        let mut msgs = vec![
            Message::new("system", "sys"),
            pinned,
            Message::new("user", "u1"),
            Message::new("assistant", "a1"),
            Message::new("user", "recent"),
        ];
        let pipeline = DefaultCompressionPipeline::with_builtin_compressors(
            None,
            crate::pipeline::DEFAULT_COMPRESSORS,
        )
        .unwrap();
        let ctx = CompressionContext {
            model: "default".into(),
            question_hint: None,
            max_tokens: None,
            reversible: false,
        };
        let config = ConversationConfig {
            preserve_recent: 2,
            compress_middle: 10,
            summary_old: false,
            bias_system: 0.8,
        };
        compress_conversation_history(&mut msgs, &config, &pipeline, &ctx)
            .await
            .unwrap();

        assert!(
            msgs.iter().any(|m| m.content == big_json
                && m.metadata.get(meta_keys::PINNED).map(String::as_str) == Some("true")),
            "a pinned message must never be compressed"
        );
    }

    #[tokio::test]
    async fn pinned_old_message_is_not_summarized() {
        let mut pinned = Message::new("tool", "PINNED_KEEP_VERBATIM");
        pinned
            .metadata
            .insert(meta_keys::PINNED.to_string(), "true".to_string());

        let mut msgs = vec![
            Message::new("user", "old turn to summarize"),
            pinned,
            Message::new("assistant", "m1"),
            Message::new("user", "m2"),
            Message::new("assistant", "recent1"),
            Message::new("user", "recent2"),
        ];
        let pipeline = DefaultCompressionPipeline::new(None, None);
        let ctx = CompressionContext {
            model: "default".into(),
            question_hint: None,
            max_tokens: None,
            reversible: false,
        };
        let config = ConversationConfig {
            preserve_recent: 2,
            compress_middle: 1,
            summary_old: true,
            bias_system: 0.8,
        };
        compress_conversation_history(&mut msgs, &config, &pipeline, &ctx)
            .await
            .unwrap();

        // The pinned message survives verbatim, never folded into the summary.
        assert!(
            msgs.iter().any(|m| m.content == "PINNED_KEEP_VERBATIM"
                && m.metadata.get(meta_keys::PINNED).map(String::as_str) == Some("true")),
            "a pinned message must never be summarized"
        );
        // No nested summary-of-a-summary.
        assert!(
            !msgs.iter().any(|m| m
                .content
                .contains("[Earlier conversation context]: [Earlier")),
            "summaries must not nest"
        );
    }
}
