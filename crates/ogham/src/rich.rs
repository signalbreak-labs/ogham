//! Block-aware compression.
//!
//! [`compress_rich_messages`] compresses the bulky text *inside*
//! [`RichMessage`] blocks — routing each text payload through the same
//! content-type-aware compressors the flat pipeline uses — while preserving
//! tool-call ids, non-text blocks (images, references), and role semantics. A
//! host can therefore keep structured content structured instead of flattening
//! it to a JSON string before compression.
//!
//! Reversibility is handled at the message level: when `policy.reversible` and
//! a CCR store are provided, each rewritten message's *original* blocks are
//! stored as a [`CcrPayload`] and the message is tagged with
//! `metadata[meta_keys::CCR_ID]`, so [`restore_rich_message`] can return the
//! exact original. Saves are awaited and fail closed — a store error keeps the
//! original message rather than emitting an unretrievable one.

use crate::ccr::{CcrPayload, CcrStore, compute_key};
use crate::pipeline::{DEFAULT_COMPRESSORS, DefaultCompressionPipeline};
use ogham_core::{
    CompressionPipeline, ContentBlock, Message, MessageContent, Result, RichMessage, meta_keys,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Media type used for CCR payloads holding a serialized [`RichMessage`].
pub const RICH_MESSAGE_MEDIA_TYPE: &str = "application/vnd.ogham.rich-message+json";

/// What a rich compression pass may rewrite.
///
/// Tool-call ids, tool inputs, images, and references are always preserved.
#[derive(Debug, Clone)]
pub struct RichCompressionPolicy {
    /// Compress standalone `Text` blocks. Default true.
    pub compress_text_blocks: bool,
    /// Compress the text inside `ToolResult` blocks — usually the bulk of an
    /// agent transcript. Default true.
    pub compress_tool_results: bool,
    /// Compress `Thinking` blocks. Default false: reasoning is often wanted raw.
    pub compress_thinking: bool,
    /// Store each rewritten message's original blocks in CCR for exact undo, and
    /// tag the message with its id. Default true.
    pub reversible: bool,
}

impl Default for RichCompressionPolicy {
    fn default() -> Self {
        Self {
            compress_text_blocks: true,
            compress_tool_results: true,
            compress_thinking: false,
            reversible: true,
        }
    }
}

/// Compress the text inside rich messages, preserving structure and tool ids.
///
/// Each message whose content actually changes and `policy.reversible` with a
/// store: its original blocks are saved to `ccr` and the returned message is
/// tagged with `metadata[meta_keys::CCR_ID]`. If the save fails, the original
/// message is kept unchanged (fail-closed). Messages that do not change, and
/// non-text blocks, pass through untouched.
pub async fn compress_rich_messages(
    messages: Vec<RichMessage>,
    ccr: Option<&Arc<dyn CcrStore>>,
    policy: &RichCompressionPolicy,
) -> Result<Vec<RichMessage>> {
    // Block text is compressed without per-block CCR markers; reversibility is
    // provided at the message level by the original-blocks payload below.
    let pipeline = DefaultCompressionPipeline::with_builtin_compressors(None, DEFAULT_COMPRESSORS)?;

    let mut out = Vec::with_capacity(messages.len());
    for original in messages {
        let content =
            compress_content(&original.content, &original.role, &pipeline, policy).await?;
        if content == original.content {
            out.push(original);
            continue;
        }
        let mut compressed = RichMessage {
            role: original.role.clone(),
            content,
            metadata: original.metadata.clone(),
        };
        match (policy.reversible, ccr) {
            (true, Some(store)) => {
                let Ok(json) = serde_json::to_string(&original) else {
                    out.push(original);
                    continue;
                };
                let id = format!("rich-{}", compute_key(json.as_bytes()));
                let payload = CcrPayload::text(RICH_MESSAGE_MEDIA_TYPE, json);
                match store.save_payload(&id, &payload).await {
                    Ok(()) => {
                        compressed
                            .metadata
                            .insert(meta_keys::CCR_ID.to_string(), id);
                        out.push(compressed);
                    }
                    // Fail-closed: never emit a compressed message whose original
                    // was not durably stored.
                    Err(_) => out.push(original),
                }
            }
            // Reversible requested but no store: do not lose data.
            (true, None) => out.push(original),
            // Explicitly irreversible compression.
            (false, _) => out.push(compressed),
        }
    }
    Ok(out)
}

/// Restore the exact original [`RichMessage`] for a message previously rewritten
/// by [`compress_rich_messages`]. Returns `None` when the message carries no CCR
/// id or the original is no longer stored.
pub async fn restore_rich_message(
    message: &RichMessage,
    ccr: &dyn CcrStore,
) -> Result<Option<RichMessage>> {
    let Some(id) = message.metadata.get(meta_keys::CCR_ID) else {
        return Ok(None);
    };
    let Some(payload) = ccr.retrieve_payload(id).await? else {
        return Ok(None);
    };
    let text = String::from_utf8_lossy(&payload.bytes);
    Ok(serde_json::from_str(&text).ok())
}

async fn compress_content(
    content: &MessageContent,
    role: &str,
    pipeline: &DefaultCompressionPipeline,
    policy: &RichCompressionPolicy,
) -> Result<MessageContent> {
    match content {
        MessageContent::Text(text) if policy.compress_text_blocks => Ok(MessageContent::Text(
            compress_text(pipeline, role, text).await?,
        )),
        MessageContent::Text(_) => Ok(content.clone()),
        MessageContent::Blocks(blocks) => {
            let mut out = Vec::with_capacity(blocks.len());
            for block in blocks {
                out.push(compress_block(block, role, pipeline, policy).await?);
            }
            Ok(MessageContent::Blocks(out))
        }
    }
}

type BlockFuture<'a> = Pin<Box<dyn Future<Output = Result<ContentBlock>> + Send + 'a>>;

fn compress_block<'a>(
    block: &'a ContentBlock,
    role: &'a str,
    pipeline: &'a DefaultCompressionPipeline,
    policy: &'a RichCompressionPolicy,
) -> BlockFuture<'a> {
    Box::pin(async move {
        match block {
            ContentBlock::Text { text } if policy.compress_text_blocks => Ok(ContentBlock::Text {
                text: compress_text(pipeline, role, text).await?,
            }),
            ContentBlock::Thinking { text } if policy.compress_thinking => {
                Ok(ContentBlock::Thinking {
                    text: compress_text(pipeline, role, text).await?,
                })
            }
            ContentBlock::ToolResult {
                tool_use_id,
                is_error,
                content,
            } if policy.compress_tool_results => {
                let mut new_content = Vec::with_capacity(content.len());
                for inner in content {
                    new_content.push(compress_block(inner, role, pipeline, policy).await?);
                }
                Ok(ContentBlock::ToolResult {
                    tool_use_id: tool_use_id.clone(),
                    is_error: *is_error,
                    content: new_content,
                })
            }
            // ToolUse, Image, Reference, and opted-out blocks pass through exactly.
            other => Ok(other.clone()),
        }
    })
}

async fn compress_text(
    pipeline: &DefaultCompressionPipeline,
    role: &str,
    text: &str,
) -> Result<String> {
    if text.is_empty() {
        return Ok(String::new());
    }
    let msg = Message::new(role, text);
    let out = pipeline.run(std::slice::from_ref(&msg)).await?;
    Ok(out
        .messages
        .into_iter()
        .next()
        .map(|m| m.content)
        .unwrap_or_else(|| text.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccr::in_memory::InMemoryCcrStore;
    use ogham_core::ImageSource;

    fn big_json_array() -> String {
        let items: Vec<serde_json::Value> = (0..200)
            .map(|i| serde_json::json!({ "id": format!("{:03}", i), "tag": "aaaaa" }))
            .collect();
        serde_json::to_string(&items).unwrap()
    }

    fn agent_turn() -> RichMessage {
        RichMessage::blocks(
            "assistant",
            vec![
                ContentBlock::ToolUse {
                    id: "call_42".into(),
                    name: "shell".into(),
                    input: serde_json::json!({ "cmd": "cat data.json" }),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "call_42".into(),
                    is_error: false,
                    content: vec![ContentBlock::Text {
                        text: big_json_array(),
                    }],
                },
                ContentBlock::Image {
                    source: ImageSource::Url {
                        url: "https://example.com/x.png".into(),
                    },
                    alt: None,
                },
            ],
        )
    }

    #[tokio::test]
    async fn compresses_tool_result_text_but_preserves_ids_and_nontext() {
        let original = agent_turn();
        let out = compress_rich_messages(
            vec![original.clone()],
            None,
            &RichCompressionPolicy {
                reversible: false,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let MessageContent::Blocks(blocks) = &out[0].content else {
            panic!("expected blocks");
        };
        // Tool-use id and input preserved exactly.
        assert!(matches!(&blocks[0],
            ContentBlock::ToolUse { id, name, .. } if id == "call_42" && name == "shell"));
        // Tool-result id preserved; its text compressed (smaller than original).
        let ContentBlock::ToolResult {
            tool_use_id,
            content,
            ..
        } = &blocks[1]
        else {
            panic!("expected tool result");
        };
        assert_eq!(tool_use_id, "call_42");
        let ContentBlock::Text { text } = &content[0] else {
            panic!("expected text");
        };
        assert!(
            text.len() < big_json_array().len(),
            "tool-result text must shrink"
        );
        // Image preserved byte-exact.
        assert_eq!(blocks[2], agent_turn_image());
    }

    fn agent_turn_image() -> ContentBlock {
        ContentBlock::Image {
            source: ImageSource::Url {
                url: "https://example.com/x.png".into(),
            },
            alt: None,
        }
    }

    #[tokio::test]
    async fn reversible_round_trip_restores_exact_blocks() {
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        let original = agent_turn();

        let out = compress_rich_messages(
            vec![original.clone()],
            Some(&store),
            &RichCompressionPolicy::default(),
        )
        .await
        .unwrap();

        // Rewritten message is tagged and differs from the original.
        assert_ne!(out[0].content, original.content);
        assert!(out[0].metadata.contains_key(meta_keys::CCR_ID));

        let restored = restore_rich_message(&out[0], store.as_ref())
            .await
            .unwrap()
            .expect("original must be restorable");
        assert_eq!(restored, original);
    }

    #[tokio::test]
    async fn unchanged_message_is_not_tagged() {
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        // Small text that the compressors leave alone.
        let msg = RichMessage::text("user", "hello");
        let out =
            compress_rich_messages(vec![msg], Some(&store), &RichCompressionPolicy::default())
                .await
                .unwrap();
        assert!(!out[0].metadata.contains_key(meta_keys::CCR_ID));
    }

    #[tokio::test]
    async fn thinking_is_preserved_by_default() {
        let msg = RichMessage::blocks(
            "assistant",
            vec![ContentBlock::Thinking {
                text: big_json_array(),
            }],
        );
        let out = compress_rich_messages(
            vec![msg.clone()],
            None,
            &RichCompressionPolicy {
                reversible: false,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(
            out[0].content, msg.content,
            "thinking must be untouched by default"
        );
    }
}
