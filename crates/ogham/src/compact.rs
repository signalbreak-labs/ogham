use crate::agent::{self, AgentCompressionStats, AgentPolicy};
use crate::budget::{self, BudgetReport, ContextBudget};
use crate::cache_strategy::{self, CacheStrategy};
use crate::ccr::CcrStore;
use crate::pipeline::{DEFAULT_COMPRESSORS, DefaultCompressionPipeline};
use ogham_core::{Message, Result, TokenCounter, meta_keys};
use std::collections::BTreeSet;
use std::sync::Arc;

const ORIGINAL_INDEX_KEY: &str = "ogham.compact.original_index";

/// Compression behavior used by `compact_conversation`.
#[derive(Debug, Clone)]
pub struct CompressionPolicy {
    /// Built-in compressor names to register in the pipeline.
    pub enabled_compressors: Vec<String>,
    /// Whether eligible replacements must be backed by the configured CCR store.
    pub reversible: bool,
}

impl CompressionPolicy {
    pub fn safe_agent_default() -> Self {
        Self::default()
    }
}

impl Default for CompressionPolicy {
    fn default() -> Self {
        Self {
            enabled_compressors: DEFAULT_COMPRESSORS
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
            reversible: true,
        }
    }
}

/// CCR storage policy for compaction.
#[derive(Clone, Default)]
pub enum CcrPolicy {
    #[default]
    Disabled,
    Store(Arc<dyn CcrStore>),
}

/// Provider cache behavior requested by the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CachePolicy {
    #[default]
    None,
    Generic {
        stable_suffix_messages: usize,
    },
    Anthropic {
        stable_suffix_messages: usize,
    },
    OpenAi {
        stable_suffix_messages: usize,
    },
    Gemini {
        stable_suffix_messages: usize,
    },
}

/// High-level conversation compaction configuration.
#[derive(Clone, Default)]
pub struct CompactConfig {
    pub budget: Option<ContextBudget>,
    pub agent_policy: AgentPolicy,
    pub compression: CompressionPolicy,
    pub ccr: CcrPolicy,
    pub cache: CachePolicy,
    pub model: Option<String>,
    /// Optional focus/question hint forwarded to compressors via
    /// `CompressionContext::question_hint`.
    ///
    /// Status 2026-06-23: plumbed end to end but not yet consumed by any
    /// built-in compressor, so it currently does not change output. See
    /// `docs/remediation-strategy-2026-06-23.md` (Patch Boundary Status).
    pub focus: Option<String>,
}

/// What kind of fold or rewrite happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldKind {
    Cleared,
    Compressed,
    Summarized,
    Dropped,
}

/// Durable record for a message rewrite, summary, or drop.
#[derive(Debug, Clone)]
pub struct FoldRecord {
    pub id: String,
    pub kind: FoldKind,
    pub original_range: std::ops::Range<usize>,
    pub replacement_index: Option<usize>,
    pub original_roles: Vec<String>,
    pub original_tokens: usize,
    pub replacement_tokens: usize,
    pub ccr_id: Option<String>,
    pub marker: Option<String>,
}

/// Message-granular protected-tail evidence.
#[derive(Debug, Clone, Default)]
pub struct ProtectedReport {
    pub protected_tail_tokens: usize,
    pub protected_message_indices: Vec<usize>,
}

/// Provider-cache annotations applied to the returned messages.
#[derive(Debug, Clone, Default)]
pub struct CachePlan {
    pub policy: String,
    pub stable_suffix_messages: usize,
    pub annotated_message_indices: Vec<usize>,
}

/// Fully compacted messages plus audit records for downstream ledgers/UIs.
#[derive(Debug, Clone)]
pub struct CompactResult {
    pub messages: Vec<Message>,
    pub folds: Vec<FoldRecord>,
    pub protected: ProtectedReport,
    pub budget_report: Option<BudgetReport>,
    pub agent_report: Option<AgentCompressionStats>,
    pub cache_plan: CachePlan,
    pub warnings: Vec<String>,
}

/// Compact a conversation with agent rules, optional budget enforcement, fold records, and cache annotations.
pub async fn compact_conversation(
    messages: Vec<Message>,
    config: CompactConfig,
) -> Result<CompactResult> {
    let model = config.model.as_deref().unwrap_or("default");
    let counter = crate::token_counter::HeuristicCounter::for_model(model);
    let original = messages.clone();
    let protected = protected_report(&original, &config.agent_policy, &counter);
    let mut warnings = Vec::new();

    let ccr_store = match (&config.ccr, config.compression.reversible) {
        (CcrPolicy::Store(store), true) => Some(store.clone()),
        (CcrPolicy::Store(_), false) => None,
        (CcrPolicy::Disabled, true) => {
            warnings.push("compression.reversible requested but ccr is disabled; reversible markers are disabled".to_string());
            None
        }
        (CcrPolicy::Disabled, false) => None,
    };

    let mut pipeline = DefaultCompressionPipeline::with_builtin_compressors(
        ccr_store.clone(),
        &config.compression.enabled_compressors,
    )?
    .with_model(model.to_string())
    .with_question_hint(config.focus.clone())
    .with_reversible(config.compression.reversible && ccr_store.is_some());

    if let Some(budget) = &config.budget {
        pipeline = pipeline.with_max_tokens(Some(budget.total_limit));
    }

    let mut working = messages;
    let saved_internal_values = tag_original_indices(&mut working);

    let mut budget_report = None;
    let mut agent_report = None;
    if let Some(budget) = &config.budget {
        budget_report = Some(
            budget::enforce_budget(
                &mut working,
                budget,
                &counter,
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

    let folds = build_fold_records(&original, &mut working, &saved_internal_values, &counter);
    let cache_plan = apply_cache_policy(&mut working, config.cache, &mut warnings);

    Ok(CompactResult {
        messages: working,
        folds,
        protected,
        budget_report,
        agent_report,
        cache_plan,
        warnings,
    })
}

fn protected_report(
    messages: &[Message],
    policy: &AgentPolicy,
    counter: &dyn TokenCounter,
) -> ProtectedReport {
    let mask = agent::protected_tail_mask(messages, policy.protected_tail_tokens);
    let protected_message_indices: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(idx, protected)| protected.then_some(idx))
        .collect();
    let protected_tail_tokens = protected_message_indices
        .iter()
        .map(|idx| counter.count(&messages[*idx].content) + ogham_core::OVERHEAD_PER_MESSAGE)
        .sum();
    ProtectedReport {
        protected_tail_tokens,
        protected_message_indices,
    }
}

fn tag_original_indices(messages: &mut [Message]) -> Vec<Option<String>> {
    let mut saved = Vec::with_capacity(messages.len());
    for (idx, msg) in messages.iter_mut().enumerate() {
        saved.push(
            msg.metadata
                .insert(ORIGINAL_INDEX_KEY.to_string(), idx.to_string()),
        );
    }
    saved
}

fn restore_internal_index(
    msg: &mut Message,
    saved_internal_values: &[Option<String>],
) -> Option<usize> {
    let original_index = msg
        .metadata
        .get(ORIGINAL_INDEX_KEY)
        .and_then(|raw| raw.parse::<usize>().ok());
    if let Some(idx) = original_index {
        match saved_internal_values
            .get(idx)
            .and_then(|value| value.clone())
        {
            Some(value) => {
                msg.metadata.insert(ORIGINAL_INDEX_KEY.to_string(), value);
            }
            None => {
                msg.metadata.remove(ORIGINAL_INDEX_KEY);
            }
        }
    }
    original_index
}

fn build_fold_records(
    original: &[Message],
    compacted: &mut [Message],
    saved_internal_values: &[Option<String>],
    counter: &dyn TokenCounter,
) -> Vec<FoldRecord> {
    let mut folds = Vec::new();
    let mut present = BTreeSet::new();
    let mut summary_replacements = Vec::new();

    for (replacement_index, msg) in compacted.iter_mut().enumerate() {
        let original_index = restore_internal_index(msg, saved_internal_values);
        if let Some(idx) = original_index.filter(|idx| *idx < original.len()) {
            present.insert(idx);
            if msg.content != original[idx].content || msg.metadata != original[idx].metadata {
                folds.push(fold_for_replacement(
                    idx,
                    Some(replacement_index),
                    &original[idx..idx + 1],
                    Some(msg),
                    counter,
                ));
            }
        } else if msg.content.starts_with("[Earlier conversation context]")
            || msg.content.starts_with("[Earlier conversation summary]")
        {
            summary_replacements.push(replacement_index);
        }
    }

    let mut missing: Vec<usize> = (0..original.len())
        .filter(|idx| !present.contains(idx))
        .collect();
    if let Some(replacement_index) = summary_replacements.first().copied()
        && !missing.is_empty()
    {
        let start = missing[0];
        let mut end = start;
        while missing.contains(&end) {
            end += 1;
        }
        folds.push(fold_for_span(
            FoldKind::Summarized,
            start..end,
            Some(replacement_index),
            &original[start..end],
            compacted.get(replacement_index),
            counter,
        ));
        missing.retain(|idx| !(*idx >= start && *idx < end));
    }

    for idx in missing {
        folds.push(fold_for_span(
            FoldKind::Dropped,
            idx..idx + 1,
            None,
            &original[idx..idx + 1],
            None,
            counter,
        ));
    }

    folds.sort_by_key(|fold| {
        (
            fold.original_range.start,
            fold.replacement_index.unwrap_or(usize::MAX),
        )
    });
    folds
}

fn fold_for_replacement(
    original_index: usize,
    replacement_index: Option<usize>,
    originals: &[Message],
    replacement: Option<&Message>,
    counter: &dyn TokenCounter,
) -> FoldRecord {
    let kind = replacement
        .map(classify_replacement)
        .unwrap_or(FoldKind::Dropped);
    fold_for_span(
        kind,
        original_index..original_index + originals.len(),
        replacement_index,
        originals,
        replacement,
        counter,
    )
}

fn fold_for_span(
    kind: FoldKind,
    original_range: std::ops::Range<usize>,
    replacement_index: Option<usize>,
    originals: &[Message],
    replacement: Option<&Message>,
    counter: &dyn TokenCounter,
) -> FoldRecord {
    let original_roles = originals.iter().map(|msg| msg.role.clone()).collect();
    let original_tokens = originals
        .iter()
        .map(|msg| counter.count(&msg.content))
        .sum();
    let replacement_tokens = replacement
        .map(|msg| counter.count(&msg.content))
        .unwrap_or(0);
    let marker = replacement.and_then(|msg| extract_ccr_marker(&msg.content));
    let ccr_id = replacement
        .and_then(|msg| msg.metadata.get(meta_keys::CCR_ID).cloned())
        .or_else(|| {
            marker.as_ref().map(|marker| {
                marker
                    .trim_start_matches("<<ccr:")
                    .trim_end_matches(">>")
                    .to_string()
            })
        });
    let id = ccr_id.clone().unwrap_or_else(|| {
        let payload = originals
            .iter()
            .map(|msg| format!("{}\n{}", msg.role, msg.content))
            .collect::<Vec<_>>()
            .join("\n---\n");
        format!("fold-{}", crate::ccr::compute_key(payload.as_bytes()))
    });

    FoldRecord {
        id,
        kind,
        original_range,
        replacement_index,
        original_roles,
        original_tokens,
        replacement_tokens,
        ccr_id,
        marker,
    }
}

fn classify_replacement(msg: &Message) -> FoldKind {
    if msg.metadata.contains_key(meta_keys::CCR_ID)
        && msg.content.starts_with("[tool:")
        && msg.content.contains(" result cleared ")
    {
        FoldKind::Cleared
    } else if msg.content.starts_with("[Earlier conversation context]")
        || msg.content.starts_with("[Earlier conversation summary]")
    {
        FoldKind::Summarized
    } else {
        FoldKind::Compressed
    }
}

fn extract_ccr_marker(content: &str) -> Option<String> {
    let start = content.find("<<ccr:")?;
    let rest = &content[start..];
    let end = rest.find(">>")? + 2;
    Some(rest[..end].to_string())
}

fn apply_cache_policy(
    messages: &mut [Message],
    policy: CachePolicy,
    warnings: &mut Vec<String>,
) -> CachePlan {
    let (policy_name, stable_suffix_messages, strategy) = match policy {
        CachePolicy::None => ("none", 0, None),
        CachePolicy::Generic {
            stable_suffix_messages,
        } => (
            "generic",
            stable_suffix_messages,
            Some(CacheStrategy::Generic),
        ),
        CachePolicy::Anthropic {
            stable_suffix_messages,
        } => (
            "anthropic",
            stable_suffix_messages,
            Some(CacheStrategy::Anthropic),
        ),
        CachePolicy::OpenAi {
            stable_suffix_messages,
        } => (
            "openai",
            stable_suffix_messages,
            Some(CacheStrategy::OpenAi),
        ),
        CachePolicy::Gemini {
            stable_suffix_messages,
        } => {
            warnings.push(
                "gemini cache policy currently emits a generic stable-prefix plan only".to_string(),
            );
            (
                "gemini",
                stable_suffix_messages,
                Some(CacheStrategy::Generic),
            )
        }
    };

    if let Some(strategy) = strategy {
        cache_strategy::apply_cache_strategy(messages, strategy, stable_suffix_messages);
    } else {
        for msg in messages.iter_mut() {
            msg.metadata.remove(meta_keys::CACHE_CONTROL);
        }
    }

    let annotated_message_indices = messages
        .iter()
        .enumerate()
        .filter_map(|(idx, msg)| {
            msg.metadata
                .contains_key(meta_keys::CACHE_CONTROL)
                .then_some(idx)
        })
        .collect();

    CachePlan {
        policy: policy_name.to_string(),
        stable_suffix_messages,
        annotated_message_indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccr::in_memory::InMemoryCcrStore;

    #[tokio::test]
    async fn compact_conversation_reports_tool_clear_fold() {
        let store = Arc::new(InMemoryCcrStore::new());
        let mut old_tool = Message::new("tool", "old successful output".repeat(80));
        old_tool
            .metadata
            .insert(meta_keys::TOOL_NAME.to_string(), "shell".to_string());
        let result = compact_conversation(
            vec![
                Message::new("system", "sys"),
                old_tool,
                Message::new("tool", "recent output"),
                Message::new("user", "latest"),
            ],
            CompactConfig {
                agent_policy: AgentPolicy {
                    keep_recent_tool_results: 1,
                    clear_old_tool_results: true,
                    keep_recent_assistant: 2,
                    protected_tail_tokens: None,
                },
                ccr: CcrPolicy::Store(store),
                ..CompactConfig::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(result.folds.len(), 1);
        assert_eq!(result.folds[0].kind, FoldKind::Cleared);
        assert_eq!(result.folds[0].original_range, 1..2);
        assert!(result.folds[0].ccr_id.is_some());
    }

    #[tokio::test]
    async fn compact_conversation_reports_protected_tail() {
        let result = compact_conversation(
            vec![
                Message::new("user", "a".repeat(32)),
                Message::new("assistant", "b".repeat(32)),
            ],
            CompactConfig {
                agent_policy: AgentPolicy {
                    protected_tail_tokens: Some(8),
                    ..AgentPolicy::default()
                },
                ..CompactConfig::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(result.protected.protected_message_indices, vec![1]);
        assert!(result.protected.protected_tail_tokens > 0);
    }

    #[tokio::test]
    async fn compact_conversation_applies_anthropic_cache_plan() {
        let result = compact_conversation(
            vec![
                Message::new("system", "sys"),
                Message::new("user", "hi"),
                Message::new("assistant", "there"),
            ],
            CompactConfig {
                cache: CachePolicy::Anthropic {
                    stable_suffix_messages: 1,
                },
                ..CompactConfig::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(result.cache_plan.policy, "anthropic");
        assert!(!result.cache_plan.annotated_message_indices.is_empty());
    }

    #[test]
    fn compressed_marker_fold_is_not_reported_as_clear() {
        let original = vec![Message::new("tool", r#"[{"id":1},{"id":2}]"#)];
        let mut tagged = original.clone();
        let saved_internal_values = tag_original_indices(&mut tagged);
        tagged[0].content = r#"[{"id":1},{"_ccr_dropped":"<<ccr:abc123>>"}]"#.to_string();
        tagged[0]
            .metadata
            .insert(meta_keys::CCR_ID.to_string(), "top-level-id".to_string());
        let counter = crate::token_counter::HeuristicCounter::new();

        let folds = build_fold_records(&original, &mut tagged, &saved_internal_values, &counter);

        assert_eq!(folds.len(), 1);
        assert_eq!(folds[0].kind, FoldKind::Compressed);
        assert_eq!(folds[0].ccr_id.as_deref(), Some("top-level-id"));
    }

    #[test]
    fn summary_fold_does_not_swallow_later_drops() {
        let original = vec![
            Message::new("user", "old 0"),
            Message::new("assistant", "old 1"),
            Message::new("user", "kept 2"),
            Message::new("assistant", "kept 3"),
            Message::new("user", "dropped 4"),
        ];
        let mut tagged = original.clone();
        let saved_internal_values = tag_original_indices(&mut tagged);
        let mut compacted = vec![
            Message::new("system", "[Earlier conversation context]: summary"),
            tagged[2].clone(),
            tagged[3].clone(),
        ];
        let counter = crate::token_counter::HeuristicCounter::new();

        let folds = build_fold_records(&original, &mut compacted, &saved_internal_values, &counter);

        assert_eq!(folds.len(), 2);
        assert_eq!(folds[0].kind, FoldKind::Summarized);
        assert_eq!(folds[0].original_range, 0..2);
        assert_eq!(folds[1].kind, FoldKind::Dropped);
        assert_eq!(folds[1].original_range, 4..5);
    }
}
