//! OpenAI prompt-caching planning helpers.
//!
//! OpenAI prompt caching is **automatic**: the API caches the longest
//! previously seen prefix (roughly 1024 tokens or more) with no per-message
//! annotations and no request fields to opt in. Ogham therefore invents no
//! `cache_control` here. Instead it reports the stable-prefix boundary, whether
//! the prefix is large enough for caching to engage, and an optional
//! deterministic `prompt_cache_key` the host can pass to OpenAI to improve
//! cache routing. Like every adapter in [`crate::providers`], this is a pure
//! data-structure builder — Ogham never talks to OpenAI itself.

use crate::ccr::compute_key;
use ogham_core::{Message, TokenCounter};

/// OpenAI's approximate minimum prefix length, in tokens, before automatic
/// prompt caching engages.
pub const MIN_CACHEABLE_PREFIX_TOKENS: usize = 1024;

/// A report describing the cacheable stable prefix of an OpenAI request.
///
/// The report is advisory: it mutates nothing and adds no annotations, because
/// OpenAI caching is automatic. Hosts use it to avoid disturbing a cacheable
/// prefix and, optionally, to set `prompt_cache_key` for better cache routing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StablePrefixReport {
    /// Number of leading messages treated as the stable (cacheable) prefix.
    pub stable_prefix_messages: usize,
    /// Estimated tokens in the stable prefix.
    pub stable_prefix_tokens: usize,
    /// Whether the prefix is large enough for OpenAI auto-caching to engage.
    pub cacheable: bool,
    /// Deterministic key derived from the stable-prefix content, suitable for
    /// OpenAI's `prompt_cache_key` request field. `None` when there is no
    /// stable prefix to cache.
    pub prompt_cache_key: Option<String>,
    /// Human-readable notes about cache boundaries or risks.
    pub notes: Vec<String>,
}

/// Build a stable-prefix report for an OpenAI request.
///
/// `stable_suffix_messages` is the number of trailing messages expected to
/// change every turn; everything before them forms the stable prefix. The
/// report never mutates `messages` and emits no `cache_control` annotations.
pub fn stable_prefix_report(
    messages: &[Message],
    stable_suffix_messages: usize,
    counter: &dyn TokenCounter,
) -> StablePrefixReport {
    let prefix_len = messages.len().saturating_sub(stable_suffix_messages);
    let prefix = &messages[..prefix_len];
    let stable_prefix_tokens = counter.count_messages(prefix);
    let cacheable = stable_prefix_tokens >= MIN_CACHEABLE_PREFIX_TOKENS;

    let mut notes = Vec::new();
    let key = if prefix.is_empty() {
        notes.push("no stable prefix: every message is treated as volatile".to_string());
        None
    } else {
        if !cacheable {
            notes.push(format!(
                "stable prefix is {stable_prefix_tokens} tokens; OpenAI auto-caching engages around {MIN_CACHEABLE_PREFIX_TOKENS}"
            ));
        }
        Some(prompt_cache_key(prefix))
    };

    StablePrefixReport {
        stable_prefix_messages: prefix_len,
        stable_prefix_tokens,
        cacheable,
        prompt_cache_key: key,
        notes,
    }
}

/// Derive a deterministic `prompt_cache_key` from stable-prefix content.
///
/// Identical prefixes produce identical keys, so a host can route requests that
/// share a prefix to the same OpenAI cache bucket. Returns an empty-prefix key
/// for an empty slice; callers normally guard on a non-empty prefix.
pub fn prompt_cache_key(prefix: &[Message]) -> String {
    let joined = prefix
        .iter()
        .map(|m| format!("{}\n{}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n---\n");
    format!("ogham-{}", compute_key(joined.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ogham_core::meta_keys;

    /// Deterministic counter: one token per content byte, exact.
    struct ByteCounter;
    impl TokenCounter for ByteCounter {
        fn count(&self, text: &str) -> usize {
            text.len()
        }
        fn is_exact(&self) -> bool {
            true
        }
    }

    fn msgs(prefix_bytes: usize, suffix: usize) -> Vec<Message> {
        let mut v = vec![Message::new("system", "x".repeat(prefix_bytes))];
        for _ in 0..suffix {
            v.push(Message::new("user", "hi"));
        }
        v
    }

    #[test]
    fn large_prefix_is_cacheable_with_key() {
        let messages = msgs(2000, 2);
        let report = stable_prefix_report(&messages, 2, &ByteCounter);
        assert_eq!(report.stable_prefix_messages, 1);
        assert!(report.cacheable);
        assert!(report.prompt_cache_key.is_some());
        assert!(report.notes.is_empty());
    }

    #[test]
    fn small_prefix_is_not_cacheable_but_still_keyed() {
        let messages = msgs(100, 1);
        let report = stable_prefix_report(&messages, 1, &ByteCounter);
        assert!(!report.cacheable);
        // A key is still offered; the note explains why caching may not engage.
        assert!(report.prompt_cache_key.is_some());
        assert!(report.notes.iter().any(|n| n.contains("auto-caching")));
    }

    #[test]
    fn all_volatile_has_no_prefix_or_key() {
        let messages = msgs(2000, 0); // 1 message, all of it volatile
        let report = stable_prefix_report(&messages, 1, &ByteCounter);
        assert_eq!(report.stable_prefix_messages, 0);
        assert_eq!(report.stable_prefix_tokens, 0);
        assert!(!report.cacheable);
        assert!(report.prompt_cache_key.is_none());
        assert!(report.notes.iter().any(|n| n.contains("no stable prefix")));
    }

    #[test]
    fn key_is_deterministic_and_prefix_sensitive() {
        let a = msgs(2000, 2);
        let mut b = a.clone();
        b[0].content.push('!'); // change the prefix content
        let ka = stable_prefix_report(&a, 2, &ByteCounter).prompt_cache_key;
        let ka2 = stable_prefix_report(&a, 2, &ByteCounter).prompt_cache_key;
        let kb = stable_prefix_report(&b, 2, &ByteCounter).prompt_cache_key;
        assert_eq!(ka, ka2, "same prefix must yield the same key");
        assert_ne!(ka, kb, "different prefix must yield a different key");
    }

    #[test]
    fn report_is_pure_no_annotations() {
        let messages = msgs(2000, 2);
        let _ = stable_prefix_report(&messages, 2, &ByteCounter);
        assert!(
            messages
                .iter()
                .all(|m| !m.metadata.contains_key(meta_keys::CACHE_CONTROL)),
            "OpenAI planning must not write cache_control annotations"
        );
    }
}
