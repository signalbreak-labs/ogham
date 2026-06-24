//! Stateful, incremental conversation compaction.
//!
//! [`compact_conversation`] recompacts a whole history from scratch every call.
//! An agent calls compaction every turn, so a [`ContextSession`] makes it
//! *append-only*: you [`push`](ContextSession::push) new turns and
//! [`compact`](ContextSession::compact) folds only the active tail, leaving
//! already-folded messages frozen.
//!
//! A message that has been folded (cleared / summarized / dropped) is marked
//! **finalized** and pinned, so the cascade never reprocesses it. This is both
//! the incrementality win and a correctness guard — re-running the cascade over
//! an already-cleared stub would re-hash and clobber its CCR id. The fold ledger
//! is append-only, so undo/UI references stay stable across turns, and the
//! finalized region stays byte-stable, which preserves provider prompt-cache
//! reuse.
//!
//! ## Durability and growth
//!
//! Finalized stubs accumulate. With the default [`RetentionPolicy::KeepAll`]
//! they are kept forever — use a non-evicting store
//! ([`crate::ccr::in_memory::InMemoryCcrStore::unbounded`]) so a referenced
//! original is never silently dropped. [`RetentionPolicy::EvictFinalized`]
//! bounds the prompt by evicting the oldest finalized stubs and garbage-collects
//! their CCR originals — but only ones no live marker still references, so
//! reversibility holds for everything still in the prompt.

use crate::agent::AgentPolicy;
use crate::budget::ContextBudget;
use crate::ccr::referenced_ccr_ids;
use crate::compact::{
    CachePlan, CachePolicy, CcrPolicy, CompactConfig, CompressionPolicy, FoldRecord,
    apply_cache_policy, compact_conversation,
};
use crate::recall::RecallIndex;
use crate::token_counter::counter_for_model;
use ogham_core::{Message, Result, meta_keys};
use std::collections::{BTreeSet, HashSet};

/// Metadata flag marking a message the session has finalized (folded). Finalized
/// messages are pinned and never reprocessed by a later [`ContextSession::compact`].
pub const SESSION_FINALIZED: &str = "ogham.session.finalized";

/// How a session bounds the growth of its finalized (folded) stubs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RetentionPolicy {
    /// Keep every finalized stub and CCR original forever (durable, unbounded).
    /// Use a non-evicting store ([`crate::ccr::in_memory::InMemoryCcrStore::unbounded`])
    /// so referenced originals are never silently dropped.
    #[default]
    KeepAll,
    /// Keep at most `max_finalized` finalized stubs in the prompt; evict the
    /// oldest beyond that. When `evict_originals`, also delete an evicted stub's
    /// CCR original — but only when no live marker still references it (so a
    /// referenced original is never deleted).
    EvictFinalized {
        /// Maximum number of finalized stubs to keep in the working prompt.
        max_finalized: usize,
        /// Garbage-collect the CCR originals of evicted, unreferenced stubs.
        evict_originals: bool,
    },
}

/// Configuration for a [`ContextSession`].
#[derive(Clone, Default)]
pub struct SessionConfig {
    /// Optional token budget enforced (fail-closed) on each compaction.
    pub budget: Option<ContextBudget>,
    /// Agent rules for clearing/protecting messages.
    pub agent_policy: AgentPolicy,
    /// CCR storage policy (shared across turns).
    pub ccr: CcrPolicy,
    /// Provider cache policy applied to the compacted output.
    pub cache: CachePolicy,
    /// Model id used for token counting and compressor routing.
    pub model: Option<String>,
    /// How to bound the growth of finalized stubs across turns.
    pub retention: RetentionPolicy,
}

/// What one [`ContextSession::compact`] produced.
#[derive(Debug, Clone)]
pub struct SessionStep {
    /// Fold records created by *this* compaction (the append-only delta).
    pub new_folds: Vec<FoldRecord>,
    /// Token count of the compacted messages after this step.
    pub tokens: usize,
    /// Provider cache plan for the compacted output (finalized region is stable).
    pub cache_plan: CachePlan,
    /// CCR ids successfully garbage-collected this step.
    pub evicted: Vec<String>,
}

/// A stateful, incremental view of a conversation that compacts append-only.
///
/// Push turns as they arrive and call [`compact`](Self::compact) when you need
/// the prompt to fit the budget. Already-folded messages are frozen, so each
/// compaction only does real work on the recent active tail.
pub struct ContextSession {
    config: SessionConfig,
    messages: Vec<Message>,
    folds: Vec<FoldRecord>,
    recall: RecallIndex,
}

impl ContextSession {
    /// Create an empty session.
    pub fn new(config: SessionConfig) -> Self {
        Self {
            config,
            messages: Vec::new(),
            folds: Vec::new(),
            recall: RecallIndex::new(),
        }
    }

    /// Append a message (a new turn). Does not compact.
    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Append several messages. Does not compact.
    pub fn extend(&mut self, messages: impl IntoIterator<Item = Message>) {
        self.messages.extend(messages);
    }

    /// The current compacted message list to send to the provider.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// The append-only fold ledger accumulated across all compactions.
    pub fn folds(&self) -> &[FoldRecord] {
        &self.folds
    }

    /// The searchable recall index over folded content. Search it by relevance
    /// to get CCR ids, then [`retrieve`](Self::retrieve) the originals.
    pub fn recall(&self) -> &crate::recall::RecallIndex {
        &self.recall
    }

    /// Number of messages currently held.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Whether the session holds no messages.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Compact the active tail to fit the budget, leaving finalized messages
    /// frozen. Idempotent: with nothing new to fold it returns an empty delta.
    ///
    /// Fails closed ([`OghamError::BudgetExceeded`](ogham_core::OghamError)) if
    /// the conversation cannot be made to fit; the session state is left
    /// unchanged so the caller can retry after pushing/removing content.
    pub async fn compact(&mut self) -> Result<SessionStep> {
        // Pin finalized messages so the cascade never reprocesses them.
        let mut working = self.messages.clone();
        for msg in working.iter_mut() {
            if is_finalized(msg) {
                msg.metadata
                    .insert(meta_keys::PINNED.to_string(), "true".to_string());
            }
        }

        let result = compact_conversation(working, self.compact_config()).await?;
        let mut next_messages = result.messages;

        // Mark this turn's fold replacements finalized. Finalized messages are
        // pinned and never reprocessed, so `result.folds` is exactly this turn's
        // new delta — append all of them (the ledger is append-only).
        let mut new_folds = result.folds;
        uniquify_fold_ids(&self.folds, &mut new_folds);
        for fold in &new_folds {
            if let Some(idx) = fold.replacement_index
                && let Some(msg) = next_messages.get_mut(idx)
            {
                mark_finalized(msg);
            }
        }

        // Index newly folded content for searchable recall, using the original
        // text still held in `self.messages` (reassigned below). Only folds with
        // a retrievable CCR id are indexed.
        for fold in &new_folds {
            let Some(ccr_id) = fold.ccr_id.clone() else {
                continue;
            };
            let text = match self.messages.get(fold.original_range.clone()) {
                Some(span) => span
                    .iter()
                    .map(|m| m.content.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
                None => continue,
            };
            if !text.is_empty() {
                self.recall
                    .index(ccr_id, fold.kind, fold.tags.clone(), &text);
            }
        }

        let evicted = self.apply_retention(&mut next_messages).await;
        for id in &evicted {
            self.recall.remove(id);
        }

        let model = self.config.model.as_deref().unwrap_or("default");
        let counter = counter_for_model(model);
        let cache_plan = apply_cache_policy(
            &mut next_messages,
            self.config.cache,
            model,
            counter.as_ref(),
        );
        let tokens = counter.count_messages(&next_messages);

        self.messages = next_messages;
        self.folds.extend(new_folds.iter().cloned());

        Ok(SessionStep {
            new_folds,
            tokens,
            cache_plan,
            evicted,
        })
    }

    /// Bound finalized-stub growth per the retention policy. Evicts the oldest
    /// finalized stubs beyond the cap from the prompt and, when configured,
    /// garbage-collects their CCR originals — but never an original a remaining
    /// live marker still references. Returns the successfully deleted CCR ids.
    async fn apply_retention(&self, messages: &mut Vec<Message>) -> Vec<String> {
        let RetentionPolicy::EvictFinalized {
            max_finalized,
            evict_originals,
        } = self.config.retention
        else {
            return Vec::new();
        };

        let finalized: Vec<usize> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| is_finalized(m))
            .map(|(i, _)| i)
            .collect();
        if finalized.len() <= max_finalized {
            return Vec::new();
        }

        // Evict the oldest (lowest-index) finalized stubs beyond the cap.
        let evict_count = finalized.len() - max_finalized;
        let to_remove: BTreeSet<usize> = finalized.into_iter().take(evict_count).collect();

        let mut removed = Vec::with_capacity(to_remove.len());
        let mut kept = Vec::with_capacity(messages.len() - to_remove.len());
        for (i, msg) in std::mem::take(messages).into_iter().enumerate() {
            if to_remove.contains(&i) {
                removed.push(msg);
            } else {
                kept.push(msg);
            }
        }
        *messages = kept;

        if !evict_originals {
            return Vec::new();
        }
        let CcrPolicy::Store(store) = &self.config.ccr else {
            return Vec::new();
        };
        // Delete a removed stub's original only if nothing still references it.
        let still_referenced = referenced_ccr_ids(messages);
        let mut evicted = Vec::new();
        for id in referenced_ccr_ids(&removed) {
            if !still_referenced.contains(&id) {
                match store.delete(&id).await {
                    Ok(()) => evicted.push(id),
                    Err(err) => {
                        tracing::warn!(ccr_id = %id, error = %err, "session_ccr_gc_failed");
                    }
                }
            }
        }
        evicted
    }

    /// Retrieve a CCR original by id (delegates to the configured store).
    pub async fn retrieve(&self, ccr_id: &str) -> Result<Option<String>> {
        match &self.config.ccr {
            CcrPolicy::Store(store) => store.retrieve(ccr_id).await,
            CcrPolicy::Disabled => Ok(None),
        }
    }

    fn compact_config(&self) -> CompactConfig {
        CompactConfig {
            budget: self.config.budget.clone(),
            agent_policy: self.config.agent_policy.clone(),
            compression: CompressionPolicy::default(),
            ccr: self.config.ccr.clone(),
            cache: self.config.cache,
            model: self.config.model.clone(),
            focus: None,
        }
    }
}

fn is_finalized(message: &Message) -> bool {
    message.metadata.get(SESSION_FINALIZED).map(String::as_str) == Some("true")
}

fn mark_finalized(message: &mut Message) {
    message
        .metadata
        .insert(SESSION_FINALIZED.to_string(), "true".to_string());
    message
        .metadata
        .insert(meta_keys::PINNED.to_string(), "true".to_string());
}

fn uniquify_fold_ids(existing: &[FoldRecord], new_folds: &mut [FoldRecord]) {
    let mut used: HashSet<String> = existing.iter().map(|fold| fold.id.clone()).collect();
    for fold in new_folds {
        if used.insert(fold.id.clone()) {
            continue;
        }
        let base = fold.id.clone();
        let mut suffix = 2usize;
        loop {
            let candidate = format!("{base}#{suffix}");
            if used.insert(candidate.clone()) {
                fold.id = candidate;
                break;
            }
            suffix += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccr::CcrStore;
    use crate::ccr::in_memory::InMemoryCcrStore;
    use crate::compact::FoldKind;
    use async_trait::async_trait;
    use std::sync::Arc;

    fn tool_msg(name: &str, content: impl Into<String>) -> Message {
        let mut m = Message::new("tool", content);
        m.metadata
            .insert(meta_keys::TOOL_NAME.to_string(), name.to_string());
        m
    }

    fn big_output(seed: usize) -> String {
        (0..120)
            .map(|i| {
                format!(
                    "line {seed}-{i}: status=ok path=/tmp/{seed}/{i} value={}",
                    i * 7
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn session_with_budget(store: Arc<dyn CcrStore>, limit: usize) -> ContextSession {
        ContextSession::new(SessionConfig {
            budget: Some(ContextBudget {
                total_limit: limit,
                safety_margin: Some(0.0),
            }),
            agent_policy: AgentPolicy {
                keep_recent_tool_results: 0,
                clear_old_tool_results: true,
                keep_recent_assistant: 1,
                protected_tail_tokens: None,
            },
            ccr: CcrPolicy::Store(store),
            ..Default::default()
        })
    }

    struct DeleteFailsStore {
        inner: InMemoryCcrStore,
    }

    impl DeleteFailsStore {
        fn new() -> Self {
            Self {
                inner: InMemoryCcrStore::unbounded(),
            }
        }
    }

    #[async_trait]
    impl CcrStore for DeleteFailsStore {
        async fn save(&self, id: &str, original: &str, metadata: Option<&str>) -> Result<()> {
            self.inner.save(id, original, metadata).await
        }

        async fn retrieve(&self, id: &str) -> Result<Option<String>> {
            self.inner.retrieve(id).await
        }

        async fn delete(&self, id: &str) -> Result<()> {
            Err(ogham_core::OghamError::StoreError(format!(
                "delete failed for {id}"
            )))
        }
    }

    #[tokio::test]
    async fn idempotent_when_under_budget() {
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        let mut s = session_with_budget(store, 1_000_000);
        s.push(Message::new("system", "sys"));
        s.push(Message::new("user", "hi"));
        let step1 = s.compact().await.unwrap();
        assert!(step1.new_folds.is_empty());
        let step2 = s.compact().await.unwrap();
        assert!(
            step2.new_folds.is_empty(),
            "second compact with no push is a no-op"
        );
        assert_eq!(s.messages().len(), 2);
    }

    #[tokio::test]
    async fn folds_are_append_only_and_old_originals_survive_many_turns() {
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        let mut s = session_with_budget(store.clone(), 300);

        s.push(Message::new("system", "sys"));
        // First bulky tool result that will get cleared as the session grows.
        let first = big_output(0);
        s.push(tool_msg("shell", first.clone()));
        s.push(Message::new("user", "q0"));
        let _ = s.compact().await.unwrap();

        // Drive several more turns, each adding a bulky tool result.
        let mut prior_ledger_len = s.folds().len();
        for turn in 1..6 {
            s.push(Message::new("assistant", format!("a{turn}")));
            s.push(tool_msg("shell", big_output(turn)));
            s.push(Message::new("user", format!("q{turn}")));
            let step = s.compact().await.unwrap();
            // Ledger only ever grows.
            assert!(s.folds().len() >= prior_ledger_len);
            // Every new fold is genuinely new (unique id).
            for f in &step.new_folds {
                assert!(
                    s.folds().iter().filter(|x| x.id == f.id).count() == 1,
                    "fold ids must be unique in the ledger"
                );
            }
            prior_ledger_len = s.folds().len();
        }

        // The very first tool output was cleared early; its verbatim original
        // must still be retrievable after all the later turns.
        let first_clear = s
            .folds()
            .iter()
            .find(|f| f.kind == FoldKind::Cleared && f.original_range.start <= 1)
            .expect("the first tool result was cleared");
        let id = first_clear
            .ccr_id
            .as_ref()
            .expect("cleared fold has a ccr id");
        let restored = s
            .retrieve(id)
            .await
            .unwrap()
            .expect("verbatim original survives");
        assert_eq!(
            restored, first,
            "an early-cleared original must stay byte-exact across many turns"
        );

        // Budget held.
        let final_tokens = s.compact().await.unwrap().tokens;
        assert!(final_tokens <= 300);
    }

    #[tokio::test]
    async fn finalized_stubs_are_not_recleared() {
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        let mut s = session_with_budget(store.clone(), 250);
        s.push(Message::new("system", "sys"));
        s.push(tool_msg("shell", big_output(1)));
        s.push(Message::new("user", "u1"));
        s.push(tool_msg("shell", big_output(2)));
        s.push(Message::new("user", "u2"));
        s.compact().await.unwrap();

        // Snapshot the cleared stub + its id.
        assert!(
            s.messages().iter().any(|m| {
                is_finalized(m)
                    && m.metadata.get(meta_keys::PINNED).map(String::as_str) == Some("true")
            }),
            "newly finalized replacements must be pinned immediately"
        );
        let cleared: Vec<_> = s
            .folds()
            .iter()
            .filter(|f| f.kind == FoldKind::Cleared)
            .map(|f| f.ccr_id.clone().unwrap())
            .collect();
        assert!(!cleared.is_empty());

        // Several more no-op-ish compactions must not re-clear (no new Cleared
        // folds for the same content, originals stay retrievable).
        for _ in 0..3 {
            s.push(Message::new("assistant", "ok"));
            s.compact().await.unwrap();
        }
        for id in &cleared {
            assert!(
                s.retrieve(id).await.unwrap().is_some(),
                "a finalized stub's original must remain retrievable"
            );
        }
    }

    #[tokio::test]
    async fn identical_content_cleared_twice_records_two_folds() {
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::new());
        // Budget tight enough to clear each bulky tool result, loose enough that
        // the resulting stubs fit under any counter (heuristic or tiktoken).
        let mut s = session_with_budget(store, 220);
        let same = "IDENTICAL_TOOL_OUTPUT_LINE\n".repeat(60);

        s.push(Message::new("system", "sys"));
        s.push(tool_msg("shell", same.clone()));
        s.push(Message::new("user", "a"));
        s.compact().await.unwrap();
        let after_first = s
            .folds()
            .iter()
            .filter(|f| f.kind == FoldKind::Cleared)
            .count();
        assert_eq!(after_first, 1);

        // The same command runs again, producing byte-identical output.
        s.push(tool_msg("shell", same.clone()));
        s.push(Message::new("user", "b"));
        let step = s.compact().await.unwrap();
        let after_second = s
            .folds()
            .iter()
            .filter(|f| f.kind == FoldKind::Cleared)
            .count();
        assert_eq!(
            after_second, 2,
            "a second clear of identical content is its own ledger event"
        );
        let clear_ids: Vec<_> = s
            .folds()
            .iter()
            .filter(|f| f.kind == FoldKind::Cleared)
            .map(|f| f.id.clone())
            .collect();
        let unique_clear_ids: HashSet<_> = clear_ids.iter().cloned().collect();
        assert_eq!(
            unique_clear_ids.len(),
            clear_ids.len(),
            "session fold ids must stay unique even when CCR ids collide"
        );
        assert!(
            step.new_folds.iter().any(|f| f.kind == FoldKind::Cleared),
            "the second clear must appear in this step's delta"
        );
    }

    #[tokio::test]
    async fn evict_finalized_bounds_stubs_and_gcs_unreferenced_originals() {
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::unbounded());
        let mut s = ContextSession::new(SessionConfig {
            budget: Some(ContextBudget {
                total_limit: 200,
                safety_margin: Some(0.0),
            }),
            agent_policy: AgentPolicy {
                keep_recent_tool_results: 0,
                clear_old_tool_results: true,
                keep_recent_assistant: 1,
                protected_tail_tokens: None,
            },
            ccr: CcrPolicy::Store(store.clone()),
            retention: RetentionPolicy::EvictFinalized {
                max_finalized: 2,
                evict_originals: true,
            },
            ..Default::default()
        });

        s.push(Message::new("system", "sys"));
        let mut cleared_ids = Vec::new();
        let mut evicted_total = Vec::new();
        for turn in 0..6 {
            s.push(tool_msg("shell", big_output(turn)));
            s.push(Message::new("user", format!("q{turn}")));
            let step = s.compact().await.unwrap();
            for fold in step
                .new_folds
                .iter()
                .filter(|f| f.kind == FoldKind::Cleared)
            {
                if let Some(id) = &fold.ccr_id {
                    cleared_ids.push(id.clone());
                }
            }
            evicted_total.extend(step.evicted);
        }

        // The prompt holds at most `max_finalized` finalized stubs.
        let finalized = s.messages().iter().filter(|m| is_finalized(m)).count();
        assert!(
            finalized <= 2,
            "finalized stubs must be bounded to max_finalized, got {finalized}"
        );
        // Old originals were garbage-collected.
        assert!(!evicted_total.is_empty(), "old originals must be evicted");
        // The earliest cleared original is gone (evicted, unreferenced).
        assert!(
            s.retrieve(&cleared_ids[0]).await.unwrap().is_none(),
            "the oldest cleared original must be GC'd"
        );
        // A recent cleared original is still referenced by a live stub → retained.
        let recent = cleared_ids.last().unwrap();
        assert!(
            s.retrieve(recent).await.unwrap().is_some(),
            "a referenced original must never be evicted (durability)"
        );
    }

    #[tokio::test]
    async fn keep_all_retention_evicts_nothing() {
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::unbounded());
        let mut s = session_with_budget(store, 200);
        s.push(Message::new("system", "sys"));
        for turn in 0..5 {
            s.push(tool_msg("shell", big_output(turn)));
            s.push(Message::new("user", format!("q{turn}")));
            let step = s.compact().await.unwrap();
            assert!(
                step.evicted.is_empty(),
                "KeepAll must never evict an original"
            );
        }
        // Every cleared original remains retrievable under the default policy.
        for fold in s.folds().iter().filter(|f| f.kind == FoldKind::Cleared) {
            let id = fold.ccr_id.as_ref().unwrap();
            assert!(s.retrieve(id).await.unwrap().is_some());
        }
    }

    #[tokio::test]
    async fn retention_recomputes_cache_plan_after_stub_eviction() {
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::unbounded());
        let mut s = ContextSession::new(SessionConfig {
            budget: Some(ContextBudget {
                total_limit: 220,
                safety_margin: Some(0.0),
            }),
            agent_policy: AgentPolicy {
                keep_recent_tool_results: 0,
                clear_old_tool_results: true,
                keep_recent_assistant: 1,
                protected_tail_tokens: None,
            },
            ccr: CcrPolicy::Store(store),
            cache: CachePolicy::OpenAi {
                stable_suffix_messages: 1,
            },
            retention: RetentionPolicy::EvictFinalized {
                max_finalized: 0,
                evict_originals: false,
            },
            ..Default::default()
        });

        s.push(Message::new("system", "sys"));
        s.push(tool_msg("shell", big_output(99)));
        s.push(Message::new("user", "latest"));

        let step = s.compact().await.unwrap();

        assert_eq!(s.messages().iter().filter(|m| is_finalized(m)).count(), 0);
        assert_eq!(step.cache_plan.policy, "openai");
        assert_eq!(
            step.cache_plan.stable_prefix_messages,
            s.messages().len().saturating_sub(1),
            "cache plan must describe the post-retention message list"
        );
    }

    #[tokio::test]
    async fn retention_delete_failure_does_not_fail_compaction() {
        let store: Arc<dyn CcrStore> = Arc::new(DeleteFailsStore::new());
        let mut s = ContextSession::new(SessionConfig {
            budget: Some(ContextBudget {
                total_limit: 220,
                safety_margin: Some(0.0),
            }),
            agent_policy: AgentPolicy {
                keep_recent_tool_results: 0,
                clear_old_tool_results: true,
                keep_recent_assistant: 1,
                protected_tail_tokens: None,
            },
            ccr: CcrPolicy::Store(store),
            retention: RetentionPolicy::EvictFinalized {
                max_finalized: 0,
                evict_originals: true,
            },
            ..Default::default()
        });

        s.push(Message::new("system", "sys"));
        s.push(tool_msg("shell", big_output(100)));
        s.push(Message::new("user", "latest"));

        let step = s.compact().await.unwrap();

        assert!(
            step.evicted.is_empty(),
            "failed deletes must not be reported as evictions"
        );
        assert_eq!(s.messages().iter().filter(|m| is_finalized(m)).count(), 0);
        let id = s
            .folds()
            .iter()
            .find(|f| f.kind == FoldKind::Cleared)
            .and_then(|f| f.ccr_id.as_ref())
            .expect("cleared fold must retain its ccr id");
        assert!(
            s.retrieve(id).await.unwrap().is_some(),
            "failed GC leaves the original available"
        );
    }

    #[tokio::test]
    async fn recall_finds_folded_content_by_relevance() {
        // No budget: agent rules clear every old tool result (keep_recent = 0),
        // so each tool output is folded and indexed regardless of size.
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::unbounded());
        let mut s = ContextSession::new(SessionConfig {
            agent_policy: AgentPolicy {
                keep_recent_tool_results: 0,
                clear_old_tool_results: true,
                keep_recent_assistant: 2,
                protected_tail_tokens: None,
            },
            ccr: CcrPolicy::Store(store),
            ..Default::default()
        });
        s.push(Message::new("system", "sys"));

        // A distinctive tool output that gets cleared and indexed.
        let auth = "authentication succeeded for user alice in src/auth/login.rs session handler "
            .repeat(8);
        s.push(Message::new("assistant", "checking auth"));
        s.push(tool_msg("shell", auth.clone()));
        s.push(Message::new("user", "ok"));
        s.compact().await.unwrap();

        // Several unrelated later turns.
        for turn in 0..3 {
            s.push(tool_msg(
                "shell",
                format!("database migration step {turn} ").repeat(20),
            ));
            s.push(Message::new("user", format!("q{turn}")));
            s.compact().await.unwrap();
        }

        let hits = s.recall().search("authentication login src/auth", 5);
        assert!(!hits.is_empty(), "recall must find the auth fold");
        let original = s
            .retrieve(&hits[0].ccr_id)
            .await
            .unwrap()
            .expect("the recalled original must be retrievable");
        assert!(
            original.contains("authentication succeeded for user alice"),
            "recall resolves to the verbatim original"
        );
    }

    #[tokio::test]
    async fn recall_finds_folded_content_by_tag() {
        use crate::fold_tags::FoldTagKind;

        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::unbounded());
        let mut s = ContextSession::new(SessionConfig {
            agent_policy: AgentPolicy {
                keep_recent_tool_results: 0,
                clear_old_tool_results: true,
                keep_recent_assistant: 2,
                protected_tail_tokens: None,
            },
            ccr: CcrPolicy::Store(store),
            ..Default::default()
        });
        s.push(Message::new("system", "sys"));
        s.push(Message::new("assistant", "checking auth"));
        s.push(tool_msg(
            "shell",
            "authentication succeeded for user alice in src/auth/login.rs handler ".repeat(8),
        ));
        s.push(Message::new("user", "ok"));
        s.compact().await.unwrap();

        // The fold carries its structured tags; the file path is unique to it.
        let by_path = s
            .recall()
            .find_by_tag(FoldTagKind::FilePath, "src/auth/login.rs");
        assert_eq!(by_path.len(), 1, "exactly one fold mentions that path");
        assert!(by_path[0].tags.tool_names.contains(&"shell".to_string()));

        let original = s
            .retrieve(&by_path[0].ccr_id)
            .await
            .unwrap()
            .expect("the tag-recalled original must be retrievable");
        assert!(original.contains("authentication succeeded for user alice"));

        // Tool-name filtering also finds it.
        assert!(
            s.recall()
                .find_by_tag(FoldTagKind::ToolName, "shell")
                .iter()
                .any(|h| h.ccr_id == by_path[0].ccr_id)
        );
    }

    #[tokio::test]
    async fn recall_merges_tags_for_repeated_identical_originals() {
        use crate::fold_tags::FoldTagKind;

        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::unbounded());
        let mut s = session_with_budget(store, 220);
        let same = "IDENTICAL_TOOL_OUTPUT_LINE\n".repeat(60);

        s.push(Message::new("system", "sys"));
        s.push(tool_msg("shell", same.clone()));
        s.push(Message::new("user", "a"));
        s.compact().await.unwrap();

        s.push(tool_msg("editor", same));
        s.push(Message::new("user", "b"));
        s.compact().await.unwrap();

        assert_eq!(
            s.folds()
                .iter()
                .filter_map(|f| f.ccr_id.as_ref())
                .collect::<HashSet<_>>()
                .len(),
            1,
            "identical originals share one CCR id"
        );
        assert!(
            !s.recall()
                .find_by_tag(FoldTagKind::ToolName, "shell")
                .is_empty(),
            "the first fold's tool tag remains queryable"
        );
        assert!(
            !s.recall()
                .find_by_tag(FoldTagKind::ToolName, "editor")
                .is_empty(),
            "the second fold's tool tag is queryable too"
        );
    }

    #[tokio::test]
    async fn eviction_removes_content_from_recall() {
        let store: Arc<dyn CcrStore> = Arc::new(InMemoryCcrStore::unbounded());
        let mut s = ContextSession::new(SessionConfig {
            budget: Some(ContextBudget {
                total_limit: 200,
                safety_margin: Some(0.0),
            }),
            agent_policy: AgentPolicy {
                keep_recent_tool_results: 0,
                clear_old_tool_results: true,
                keep_recent_assistant: 1,
                protected_tail_tokens: None,
            },
            ccr: CcrPolicy::Store(store),
            retention: RetentionPolicy::EvictFinalized {
                max_finalized: 1,
                evict_originals: true,
            },
            ..Default::default()
        });
        s.push(Message::new("system", "sys"));
        // The oldest distinctive content will be evicted as newer turns arrive.
        s.push(tool_msg(
            "shell",
            "ZEBRAFISH telemetry anomaly report ".repeat(20),
        ));
        s.push(Message::new("user", "u0"));
        s.compact().await.unwrap();
        for turn in 0..3 {
            s.push(tool_msg(
                "shell",
                format!("routine log line {turn} ").repeat(20),
            ));
            s.push(Message::new("user", format!("u{turn}")));
            s.compact().await.unwrap();
        }

        // The evicted content is gone from the recall index.
        assert!(
            s.recall()
                .search("zebrafish telemetry anomaly", 5)
                .is_empty(),
            "evicted content must be removed from recall"
        );
    }
}
