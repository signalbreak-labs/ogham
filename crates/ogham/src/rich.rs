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

use crate::agent::{self, AgentCompressionStats, AgentPolicy};
use crate::budget::{self, BudgetReport, ContextBudget};
use crate::ccr::{CcrPayload, CcrStore, compute_key};
use crate::compact::{
    CachePlan, CachePolicy, CcrPolicy, FoldKind, FoldRecord, ORIGINAL_INDEX_KEY, ProtectedReport,
    apply_cache_policy, build_fold_records, protected_report, tag_original_indices,
};
use crate::pipeline::{DEFAULT_COMPRESSORS, DefaultCompressionPipeline};
use crate::token_counter::counter_for_model;
use ogham_core::{
    CompressionPipeline, ContentBlock, ImageSource, Message, MessageContent, OVERHEAD_PER_MESSAGE,
    OghamError, Result, RichMessage, TokenCountKind, TokenCounter, meta_keys,
};
use std::collections::{BTreeSet, HashMap};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Media type used for CCR payloads holding a serialized [`RichMessage`].
pub const RICH_MESSAGE_MEDIA_TYPE: &str = "application/vnd.ogham.rich-message+json";

/// What a rich compression pass may rewrite.
///
/// Tool-call ids, tool inputs, images, references, and error tool results are
/// always preserved.
#[derive(Debug, Clone)]
pub struct RichCompressionPolicy {
    /// Compress standalone `Text` blocks. Default true.
    pub compress_text_blocks: bool,
    /// Compress the text inside non-error `ToolResult` blocks — usually the
    /// bulk of an agent transcript. Error tool results are always preserved.
    /// Default true.
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
    if payload.media_type != RICH_MESSAGE_MEDIA_TYPE {
        return Ok(None);
    }
    let Ok(text) = String::from_utf8(payload.bytes) else {
        return Ok(None);
    };
    Ok(serde_json::from_str(&text).ok())
}

/// High-level block-aware conversation compaction configuration — the rich
/// analogue of [`crate::compact::CompactConfig`].
#[derive(Clone, Default)]
pub struct CompactRichConfig {
    /// Optional token budget. The cascade enforces it on the flat projection and
    /// the rich output is then recounted at its true size and fails closed.
    pub budget: Option<ContextBudget>,
    /// Agent rules for clearing/protecting messages.
    pub agent_policy: AgentPolicy,
    /// Block-aware text-compression policy, applied to kept, non-protected
    /// messages after the cascade.
    pub rich: RichCompressionPolicy,
    /// CCR storage policy shared by block compression and cascade folds. Block
    /// compression uses it only when `rich.reversible` is enabled.
    pub ccr: CcrPolicy,
    /// Provider cache policy applied to the output.
    pub cache: CachePolicy,
    /// Model id used for token counting and compressor routing.
    pub model: Option<String>,
    /// Optional focus/question hint for the cascade's middle-band compression.
    pub focus: Option<String>,
}

/// Block-aware compaction result: structure-preserving messages plus the same
/// audit records as [`crate::compact::compact_conversation`].
#[derive(Debug, Clone)]
pub struct CompactRichResult {
    /// The compacted messages. Messages the cascade kept retain their block
    /// structure (with block text compressed); messages it folded
    /// (cleared/summarized/dropped) appear as reversible flat text.
    pub messages: Vec<RichMessage>,
    /// Audit records covering both block compression and the cascade.
    pub folds: Vec<FoldRecord>,
    /// Protected-tail evidence for the input.
    pub protected: ProtectedReport,
    /// Budget cascade report, when a budget was enforced.
    pub budget_report: Option<BudgetReport>,
    /// Agent-rules report, when no budget was enforced.
    pub agent_report: Option<AgentCompressionStats>,
    /// Provider cache plan for the output.
    pub cache_plan: CachePlan,
    /// Non-fatal warnings (e.g. reversibility requested without a store).
    pub warnings: Vec<String>,
}

/// Compact a rich conversation — the block-aware analogue of
/// [`crate::compact::compact_conversation`]. Returns structure-preserving
/// messages plus fold records, protected-tail evidence, optional budget/agent
/// reports, and a cache plan.
///
/// The agent/budget cascade runs FIRST, on the verbatim flat projection, so it
/// clears/drops/summarizes the original content and stores verbatim CCR
/// originals with correct fold records. Only AFTER the cascade are the kept,
/// non-protected messages block-compressed (structure preserving, with their
/// own clean rich CCR ids), so the cascade can never clobber a block-compressed
/// message's reversibility.
///
/// Messages the cascade keeps retain their exact block structure (tool ids,
/// images, references, error tool results); messages it folds become reversible
/// flat text. Block-compressed kept messages restore to exact blocks via
/// [`restore_rich_message`]; cascade-folded messages restore to their verbatim
/// flat text via the CCR store. Because the cascade counts the lossy flat
/// projection (which under-represents opaque blocks like images), the actual
/// rich output is recounted at its true size and fails closed
/// ([`OghamError::BudgetExceeded`]) if it exceeds the budget. `focus` biases the
/// cascade's middle-band compression.
pub async fn compact_rich(
    messages: Vec<RichMessage>,
    config: CompactRichConfig,
) -> Result<CompactRichResult> {
    let model = config.model.as_deref().unwrap_or("default");
    let counter = counter_for_model(model);
    let mut warnings = Vec::new();

    let ccr_store = match &config.ccr {
        CcrPolicy::Store(store) => Some(store.clone()),
        CcrPolicy::Disabled => {
            if config.rich.reversible {
                warnings.push(
                    "rich.reversible requested but ccr is disabled; reversible undo is disabled"
                        .to_string(),
                );
            }
            None
        }
    };

    let rich_ccr = if config.rich.reversible {
        ccr_store.as_ref()
    } else {
        None
    };

    // The conversation cascade runs FIRST, on the verbatim flat projection, so
    // it clears/drops/summarizes the ORIGINAL (uncompressed) content — storing
    // verbatim CCR originals and producing correct fold records and token
    // counts. Block-aware text compression then runs only on kept messages, so
    // the cascade can never clobber a block-compressed message's CCR id.
    let input_flats: Vec<Message> = messages.iter().map(rich_to_agent_flat_lossy).collect();
    let protected = protected_report(&input_flats, &config.agent_policy, counter.as_ref());

    let mut working = input_flats.clone();
    let saved = tag_original_indices(&mut working);

    let pipeline = DefaultCompressionPipeline::with_builtin_compressors(
        ccr_store.clone(),
        DEFAULT_COMPRESSORS,
    )?
    .with_model(model.to_string())
    .with_question_hint(config.focus.clone())
    .with_reversible(ccr_store.is_some())
    .with_max_tokens(config.budget.as_ref().map(|b| b.total_limit));

    let mut budget_report = None;
    let mut agent_report = None;
    if let Some(budget) = &config.budget {
        budget_report = Some(
            budget::enforce_budget(
                &mut working,
                budget,
                counter.as_ref(),
                &pipeline,
                &config.agent_policy,
                ccr_store.clone(),
            )
            .await?,
        );
    } else {
        agent_report = Some(
            agent::apply_agent_compression(&mut working, &config.agent_policy, ccr_store.clone())
                .await?,
        );
    }

    // Capture mapping + cascade-kept flags before fold building cleans the tags.
    let mapping: Vec<Option<usize>> = working
        .iter()
        .map(|m| {
            m.metadata
                .get(ORIGINAL_INDEX_KEY)
                .and_then(|raw| raw.parse().ok())
        })
        .collect();
    let kept_by_cascade: Vec<bool> = working
        .iter()
        .zip(&mapping)
        .map(|(m, idx)| {
            idx.is_some_and(|i| i < input_flats.len() && m.content == input_flats[i].content)
        })
        .collect();

    let mut folds = build_fold_records(&input_flats, &mut working, &saved, counter.as_ref());
    let cache_plan =
        apply_cache_policy(&mut working, config.cache, counter.as_ref(), &mut warnings);

    // Block-compress kept, non-protected messages (structure preserving, with
    // clean rich CCR ids and their own Compressed fold records).
    let protected_set: BTreeSet<usize> = protected
        .protected_message_indices
        .iter()
        .copied()
        .collect();
    let (out, block_folds) = reconstruct_and_compress(
        &working,
        &messages,
        &mapping,
        &kept_by_cascade,
        &protected_set,
        rich_ccr,
        &config.rich,
        counter.as_ref(),
    )
    .await?;
    folds.extend(block_folds);

    // Fail-closed budget guard: the flat projection under-counts opaque blocks
    // (image bytes, tool inputs), so recount the ACTUAL emitted rich messages
    // and fail closed if they exceed the budget.
    if let Some(budget) = &config.budget {
        let limit = effective_limit(budget, counter.count_kind());
        let actual: usize = out
            .iter()
            .map(|m| rich_message_tokens(m, counter.as_ref()))
            .sum();
        if actual > limit {
            return Err(OghamError::BudgetExceeded {
                needed: actual,
                limit,
            });
        }
        if let Some(report) = budget_report.as_mut() {
            report.tokens_final = actual;
        }
    }

    Ok(CompactRichResult {
        messages: out,
        folds,
        protected,
        budget_report,
        agent_report,
        cache_plan,
        warnings,
    })
}

/// Rebuild rich output from the compacted flat projection, block-compressing
/// the kept, non-protected messages.
///
/// A message the cascade kept (working content unchanged from its verbatim
/// projection) recovers its original block structure; if it is not protected it
/// is then block-compressed (structure preserving, CCR-backed) and earns a
/// `Compressed` fold. A message the cascade folded stays as flat text carrying
/// the cascade's own (correct, verbatim) CCR id. Every emitted message mirrors
/// the final flat cache annotation exactly.
#[allow(clippy::too_many_arguments)]
async fn reconstruct_and_compress(
    working: &[Message],
    originals: &[RichMessage],
    mapping: &[Option<usize>],
    kept_by_cascade: &[bool],
    protected: &BTreeSet<usize>,
    rich_ccr: Option<&Arc<dyn CcrStore>>,
    policy: &RichCompressionPolicy,
    counter: &dyn TokenCounter,
) -> Result<(Vec<RichMessage>, Vec<FoldRecord>)> {
    // Collect kept, non-protected originals and block-compress them in one pass.
    let mut compress_positions: Vec<(usize, usize)> = Vec::new();
    for (pos, kept) in kept_by_cascade.iter().enumerate() {
        if *kept
            && let Some(idx) = mapping[pos]
            && !protected.contains(&idx)
        {
            compress_positions.push((pos, idx));
        }
    }
    let batch: Vec<RichMessage> = compress_positions
        .iter()
        .map(|(_, idx)| originals[*idx].clone())
        .collect();
    let compressed = compress_rich_messages(batch, rich_ccr, policy).await?;
    let mut compressed_by_pos: HashMap<usize, RichMessage> = compress_positions
        .iter()
        .map(|(pos, _)| *pos)
        .zip(compressed)
        .collect();

    let mut out = Vec::with_capacity(working.len());
    let mut folds = Vec::new();
    for (pos, flat) in working.iter().enumerate() {
        let mut rich = if let Some(comp) = compressed_by_pos.remove(&pos) {
            let idx = mapping[pos].expect("compressed positions carry an original index");
            if comp.content != originals[idx].content {
                folds.push(block_compression_fold(
                    idx,
                    pos,
                    &originals[idx],
                    &comp,
                    counter,
                ));
            }
            comp
        } else if kept_by_cascade[pos]
            && let Some(idx) = mapping[pos]
        {
            originals[idx].clone()
        } else {
            let mut metadata = flat.metadata.clone();
            metadata.remove(ORIGINAL_INDEX_KEY);
            RichMessage {
                role: flat.role.clone(),
                content: MessageContent::Text(flat.content.clone()),
                metadata,
            }
        };
        match flat.metadata.get(meta_keys::CACHE_CONTROL) {
            Some(cache) => {
                rich.metadata
                    .insert(meta_keys::CACHE_CONTROL.to_string(), cache.clone());
            }
            None => {
                rich.metadata.remove(meta_keys::CACHE_CONTROL);
            }
        }
        out.push(rich);
    }
    Ok((out, folds))
}

fn block_compression_fold(
    original_index: usize,
    replacement_index: usize,
    original: &RichMessage,
    compressed: &RichMessage,
    counter: &dyn TokenCounter,
) -> FoldRecord {
    let ccr_id = compressed.metadata.get(meta_keys::CCR_ID).cloned();
    let id = ccr_id.clone().unwrap_or_else(|| {
        let payload = serde_json::to_string(original).unwrap_or_default();
        format!("fold-{}", compute_key(payload.as_bytes()))
    });
    FoldRecord {
        id,
        kind: FoldKind::Compressed,
        original_range: original_index..original_index + 1,
        replacement_index: Some(replacement_index),
        original_roles: vec![original.role.clone()],
        original_tokens: rich_message_tokens(original, counter),
        replacement_tokens: rich_message_tokens(compressed, counter),
        ccr_id,
        marker: None,
    }
}

/// Token estimate for a rich message that counts opaque block payloads (image
/// bytes, tool inputs, references) at their real size — not the lossy
/// flat-render placeholder the budget cascade sees.
fn rich_message_tokens(message: &RichMessage, counter: &dyn TokenCounter) -> usize {
    let body = match &message.content {
        MessageContent::Text(text) => counter.count(text),
        MessageContent::Blocks(blocks) => blocks.iter().map(|b| block_tokens(b, counter)).sum(),
    };
    body + OVERHEAD_PER_MESSAGE
}

fn block_tokens(block: &ContentBlock, counter: &dyn TokenCounter) -> usize {
    match block {
        ContentBlock::Text { text } | ContentBlock::Thinking { text } => counter.count(text),
        ContentBlock::Image { source, alt } => {
            let body = match source {
                ImageSource::Base64 { data, .. } => counter.count(data),
                ImageSource::Url { url } => counter.count(url),
            };
            body + alt.as_deref().map_or(0, |a| counter.count(a))
        }
        ContentBlock::ToolUse { name, input, .. } => {
            counter.count(name) + counter.count(&input.to_string())
        }
        ContentBlock::ToolResult { content, .. } => {
            content.iter().map(|b| block_tokens(b, counter)).sum()
        }
        ContentBlock::Reference {
            id_or_path,
            metadata,
            ..
        } => counter.count(id_or_path) + metadata.values().map(|v| counter.count(v)).sum::<usize>(),
    }
}

fn effective_limit(budget: &ContextBudget, kind: TokenCountKind) -> usize {
    let margin = budget.safety_margin.unwrap_or_else(|| kind.safety_margin());
    budget
        .total_limit
        .saturating_sub(((budget.total_limit as f64) * margin).ceil() as usize)
}

fn rich_to_agent_flat_lossy(message: &RichMessage) -> Message {
    let mut flat = message.to_flat_lossy();
    if rich_content_has_error_tool_result(&message.content) {
        flat.metadata.insert(
            meta_keys::AGENT_CONTENT_TYPE.to_string(),
            agent::AgentContentType::ToolResultError
                .as_str()
                .to_string(),
        );
        flat.metadata
            .insert(meta_keys::TOOL_STATUS.to_string(), "error".to_string());
    }
    flat
}

fn rich_content_has_error_tool_result(content: &MessageContent) -> bool {
    match content {
        MessageContent::Text(_) => false,
        MessageContent::Blocks(blocks) => blocks.iter().any(block_has_error_tool_result),
    }
}

fn block_has_error_tool_result(block: &ContentBlock) -> bool {
    match block {
        ContentBlock::ToolResult {
            is_error, content, ..
        } => *is_error || content.iter().any(block_has_error_tool_result),
        _ => false,
    }
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
            ContentBlock::ToolResult { is_error: true, .. } => Ok(block.clone()),
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
    use crate::compact::FoldKind;
    use ogham_core::ImageSource;

    fn big_json_array() -> String {
        let items: Vec<serde_json::Value> = (0..200)
            .map(|i| serde_json::json!({ "id": format!("{:03}", i), "tag": "aaaaa" }))
            .collect();
        serde_json::to_string(&items).unwrap()
    }

    fn large_tool_output(seed: usize) -> String {
        (0usize..300)
            .map(|i| {
                format!(
                    "record={seed:02}-{i:03} path=/tmp/ogham/{seed}/{i} status=ok value={} digest={:x}",
                    i * 37,
                    seed.wrapping_mul(1_000_003) ^ i.wrapping_mul(97)
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
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
    async fn restore_ignores_non_rich_payloads() {
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        store
            .save_payload(
                "not-rich",
                &CcrPayload::text(
                    "application/json",
                    r#"{"role":"assistant","content":"looks rich"}"#,
                ),
            )
            .await
            .unwrap();
        let mut msg = RichMessage::text("assistant", "compressed");
        msg.metadata
            .insert(meta_keys::CCR_ID.to_string(), "not-rich".to_string());

        let restored = restore_rich_message(&msg, store.as_ref()).await.unwrap();

        assert!(restored.is_none());
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

    #[tokio::test]
    async fn error_tool_results_are_preserved() {
        let msg = RichMessage::blocks(
            "assistant",
            vec![ContentBlock::ToolResult {
                tool_use_id: "call_99".into(),
                is_error: true,
                content: vec![ContentBlock::Text {
                    text: big_json_array(),
                }],
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
            "error tool results must stay byte-for-byte visible"
        );
        assert!(!out[0].metadata.contains_key(meta_keys::CCR_ID));
    }

    #[test]
    fn rich_projection_marks_error_tool_results_for_agent_policy() {
        let msg = RichMessage::blocks(
            "assistant",
            vec![ContentBlock::ToolResult {
                tool_use_id: "call_99".into(),
                is_error: true,
                content: vec![ContentBlock::Text {
                    text: "permission denied".into(),
                }],
            }],
        );

        let flat = rich_to_agent_flat_lossy(&msg);

        assert_eq!(
            crate::agent::classify(&flat),
            crate::agent::AgentContentType::ToolResultError
        );
        assert_eq!(
            flat.metadata
                .get(meta_keys::TOOL_STATUS)
                .map(String::as_str),
            Some("error")
        );
    }

    #[tokio::test]
    async fn compact_rich_preserves_structure_for_kept_messages() {
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        let result = compact_rich(
            vec![agent_turn(), RichMessage::text("user", "latest")],
            CompactRichConfig {
                ccr: CcrPolicy::Store(store),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        // No budget and the turn is an assistant message (never cleared), so it
        // is kept with full block structure — tool id and image intact.
        let MessageContent::Blocks(blocks) = &result.messages[0].content else {
            panic!("structure lost for a kept message");
        };
        assert!(matches!(&blocks[0], ContentBlock::ToolUse { id, .. } if id == "call_42"));
        assert_eq!(blocks[2], agent_turn_image());
    }

    #[tokio::test]
    async fn compact_rich_kept_block_compressed_message_is_reversible() {
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        let original = agent_turn();
        let result = compact_rich(
            vec![original.clone(), RichMessage::text("user", "latest")],
            CompactRichConfig {
                ccr: CcrPolicy::Store(store.clone()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        // The turn's tool-result text was compressed (so it carries a CCR id),
        // but it restores to the exact original blocks.
        assert!(result.messages[0].metadata.contains_key(meta_keys::CCR_ID));
        let restored = restore_rich_message(&result.messages[0], store.as_ref())
            .await
            .unwrap()
            .expect("kept, block-compressed message must be restorable");
        assert_eq!(restored, original);
    }

    #[tokio::test]
    async fn compact_rich_removes_stale_cache_metadata_from_kept_messages() {
        let mut cached = RichMessage::text("system", "sys");
        cached.metadata.insert(
            meta_keys::CACHE_CONTROL.to_string(),
            "ephemeral".to_string(),
        );

        let result = compact_rich(
            vec![cached, RichMessage::text("user", "latest")],
            CompactRichConfig {
                cache: CachePolicy::None,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert!(
            !result.messages[0]
                .metadata
                .contains_key(meta_keys::CACHE_CONTROL),
            "kept rich messages must mirror the final flat cache annotations"
        );
        assert!(result.cache_plan.annotated_message_indices.is_empty());
    }

    #[tokio::test]
    async fn compact_rich_enforces_budget_and_reports_folds() {
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        let mut msgs = vec![RichMessage::text("system", "sys")];
        for i in 0..8 {
            let mut m = RichMessage::text("tool", large_tool_output(i));
            m.metadata
                .insert(meta_keys::TOOL_NAME.to_string(), format!("t{i}"));
            msgs.push(m);
        }
        msgs.push(RichMessage::text("user", "latest"));

        let result = compact_rich(
            msgs,
            CompactRichConfig {
                budget: Some(ContextBudget {
                    total_limit: 1_000,
                    safety_margin: Some(0.0),
                }),
                agent_policy: AgentPolicy {
                    keep_recent_tool_results: 0,
                    clear_old_tool_results: true,
                    ..Default::default()
                },
                ccr: CcrPolicy::Store(store),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let report = result.budget_report.expect("budget enforced");
        assert!(report.tokens_final <= report.effective_limit);
        assert!(
            result
                .folds
                .iter()
                .any(|fold| fold.kind == FoldKind::Cleared),
            "clearing old tool results must produce fold records"
        );
    }

    #[tokio::test]
    async fn compact_rich_irreversible_blocks_still_allow_cascade_ccr() {
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        let mut tool = RichMessage::text("tool", large_tool_output(42));
        tool.metadata
            .insert(meta_keys::TOOL_NAME.to_string(), "shell".to_string());

        let result = compact_rich(
            vec![tool, RichMessage::text("user", "latest")],
            CompactRichConfig {
                agent_policy: AgentPolicy {
                    keep_recent_tool_results: 0,
                    clear_old_tool_results: true,
                    ..Default::default()
                },
                rich: RichCompressionPolicy {
                    reversible: false,
                    ..Default::default()
                },
                ccr: CcrPolicy::Store(store),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert!(
            result
                .folds
                .iter()
                .any(|fold| fold.kind == FoldKind::Cleared && fold.ccr_id.is_some()),
            "cascade folds should remain CCR-backed when block compression is irreversible"
        );
    }

    #[tokio::test]
    async fn compact_rich_emits_cache_plan() {
        let result = compact_rich(
            vec![
                RichMessage::text("system", "sys"),
                RichMessage::text("user", "a"),
                RichMessage::text("assistant", "b"),
                RichMessage::text("user", "latest"),
            ],
            CompactRichConfig {
                cache: CachePolicy::OpenAi {
                    stable_suffix_messages: 1,
                },
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(result.cache_plan.policy, "openai");
        assert_eq!(result.cache_plan.stable_prefix_messages, 3);
    }

    #[tokio::test]
    async fn compact_rich_fails_closed_on_oversize_opaque_blocks() {
        // Image base64 renders to a tiny flat placeholder, so the cascade
        // under-counts it. The post-reconstruction recount must catch the real
        // size and fail closed instead of returning an over-budget payload.
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        let image = || {
            RichMessage::blocks(
                "assistant",
                vec![ContentBlock::Image {
                    source: ImageSource::Base64 {
                        media_type: "image/png".into(),
                        data: "A".repeat(200_000),
                    },
                    alt: None,
                }],
            )
        };
        let result = compact_rich(
            vec![
                RichMessage::text("system", "sys"),
                image(),
                image(),
                image(),
                RichMessage::text("user", "latest"),
            ],
            CompactRichConfig {
                budget: Some(ContextBudget {
                    total_limit: 200,
                    safety_margin: Some(0.0),
                }),
                ccr: CcrPolicy::Store(store),
                ..Default::default()
            },
        )
        .await;

        assert!(
            matches!(result, Err(OghamError::BudgetExceeded { .. })),
            "oversize opaque blocks must fail closed, got {result:?}"
        );
    }

    #[tokio::test]
    async fn compact_rich_cleared_tool_message_restores_verbatim_original() {
        // The cascade runs on the verbatim projection, so a cleared tool result
        // must store and advertise the EXACT original — never a lossy projection.
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        let original_text = large_tool_output(7);
        let mut tool = RichMessage::text("tool", original_text.clone());
        tool.metadata
            .insert(meta_keys::TOOL_NAME.to_string(), "shell".into());

        let result = compact_rich(
            vec![tool, RichMessage::text("user", "latest")],
            CompactRichConfig {
                agent_policy: AgentPolicy {
                    keep_recent_tool_results: 0,
                    clear_old_tool_results: true,
                    ..Default::default()
                },
                ccr: CcrPolicy::Store(store.clone()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let cleared = result
            .folds
            .iter()
            .find(|f| f.kind == FoldKind::Cleared)
            .expect("a Cleared fold");
        let ccr_id = cleared
            .ccr_id
            .as_ref()
            .expect("cleared fold carries a ccr id");
        let stored = store
            .retrieve(ccr_id)
            .await
            .unwrap()
            .expect("cleared original must be retrievable");
        assert_eq!(
            stored, original_text,
            "cleared fold must resolve to the verbatim original, not a lossy projection"
        );
        assert_eq!(
            cleared.original_tokens,
            counter_for_model("default").count(&original_text),
            "cleared fold original_tokens must describe the verbatim original"
        );
    }
}
