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

use crate::agent::AgentPolicy;
use crate::budget::ContextBudget;
use crate::compact::{
    CachePlan, CachePolicy, CcrPolicy, CompactConfig, CompressionPolicy, FoldRecord,
    compact_conversation,
};
use crate::token_counter::counter_for_model;
use ogham_core::{Message, Result, meta_keys};

/// Metadata flag marking a message the session has finalized (folded). Finalized
/// messages are pinned and never reprocessed by a later [`ContextSession::compact`].
pub const SESSION_FINALIZED: &str = "ogham.session.finalized";

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
}

impl ContextSession {
    /// Create an empty session.
    pub fn new(config: SessionConfig) -> Self {
        Self {
            config,
            messages: Vec::new(),
            folds: Vec::new(),
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
        self.messages = result.messages;

        // Mark this turn's fold replacements finalized. Finalized messages are
        // pinned and never reprocessed, so `result.folds` is exactly this turn's
        // new delta — append all of them (the ledger is append-only).
        let new_folds = result.folds;
        for fold in &new_folds {
            if let Some(idx) = fold.replacement_index
                && let Some(msg) = self.messages.get_mut(idx)
            {
                msg.metadata
                    .insert(SESSION_FINALIZED.to_string(), "true".to_string());
            }
        }
        self.folds.extend(new_folds.iter().cloned());

        let counter = counter_for_model(self.config.model.as_deref().unwrap_or("default"));
        let tokens = counter.count_messages(&self.messages);
        Ok(SessionStep {
            new_folds,
            tokens,
            cache_plan: result.cache_plan,
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccr::CcrStore;
    use crate::ccr::in_memory::InMemoryCcrStore;
    use crate::compact::FoldKind;
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
        assert!(
            step.new_folds.iter().any(|f| f.kind == FoldKind::Cleared),
            "the second clear must appear in this step's delta"
        );
    }
}
