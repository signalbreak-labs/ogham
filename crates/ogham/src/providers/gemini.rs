//! Gemini context-caching planning helpers.
//!
//! Gemini supports explicit context caching: a host creates a `CachedContent`
//! resource holding reusable prefix content and references it by name on later
//! requests. Ogham creates no resource (no HTTP client); it reports the
//! cache-candidate span — the stable prefix — its token estimate, whether it is
//! likely large enough to cache, and a deterministic content identity the host
//! can use to detect when the cached content must be refreshed. Like every
//! adapter in [`crate::providers`], this is a pure data-structure builder.

use ogham_core::{Message, TokenCounter};

/// Conservative minimum prefix length, in tokens, for Gemini explicit caching.
///
/// Real minimums vary by model; treat this as a floor and confirm per model.
pub const MIN_CACHEABLE_PREFIX_TOKENS: usize = 1024;

/// A Gemini cache-candidate report for the stable prefix of a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheCandidate {
    /// Number of leading messages that form the cache candidate.
    pub prefix_messages: usize,
    /// Estimated tokens in the candidate prefix.
    pub prefix_tokens: usize,
    /// Whether the prefix is likely large enough to cache (model-dependent).
    pub cacheable: bool,
    /// Deterministic identity of the candidate content. Re-create or refresh the
    /// Gemini `CachedContent` when this changes. `None` when there is no prefix.
    pub content_id: Option<String>,
    /// Human-readable notes about cache boundaries or risks.
    pub notes: Vec<String>,
}

/// Build a cache-candidate report for a Gemini request.
///
/// `stable_suffix_messages` is the number of trailing messages expected to
/// change every turn; everything before them is the cache candidate. Mutates
/// nothing.
pub fn cache_candidate(
    messages: &[Message],
    stable_suffix_messages: usize,
    counter: &dyn TokenCounter,
) -> CacheCandidate {
    let prefix_len = messages.len().saturating_sub(stable_suffix_messages);
    let prefix = &messages[..prefix_len];
    let prefix_tokens = counter.count_messages(prefix);
    let cacheable = prefix_tokens >= MIN_CACHEABLE_PREFIX_TOKENS;

    let mut notes = Vec::new();
    let content_id = if prefix.is_empty() {
        notes.push("no stable prefix to cache".to_string());
        None
    } else {
        if !cacheable {
            notes.push(format!(
                "candidate prefix is {prefix_tokens} tokens; Gemini caching minimums are model-dependent (~{MIN_CACHEABLE_PREFIX_TOKENS}+)"
            ));
        }
        Some(crate::providers::content_key(prefix))
    };

    CacheCandidate {
        prefix_messages: prefix_len,
        prefix_tokens,
        cacheable,
        content_id,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn large_prefix_is_a_cache_candidate() {
        let m = msgs(2000, 2);
        let c = cache_candidate(&m, 2, &ByteCounter);
        assert_eq!(c.prefix_messages, 1);
        assert!(c.cacheable);
        assert!(c.content_id.is_some());
        assert!(c.notes.is_empty());
    }

    #[test]
    fn small_prefix_warns_but_still_identifies() {
        let m = msgs(100, 1);
        let c = cache_candidate(&m, 1, &ByteCounter);
        assert!(!c.cacheable);
        assert!(c.content_id.is_some());
        assert!(c.notes.iter().any(|n| n.contains("model-dependent")));
    }

    #[test]
    fn no_prefix_has_no_candidate() {
        let m = msgs(2000, 0);
        let c = cache_candidate(&m, 1, &ByteCounter);
        assert_eq!(c.prefix_messages, 0);
        assert!(!c.cacheable);
        assert!(c.content_id.is_none());
        assert!(c.notes.iter().any(|n| n.contains("no stable prefix")));
    }

    #[test]
    fn content_id_is_stable_and_prefix_sensitive() {
        let a = msgs(2000, 2);
        let mut b = a.clone();
        b[0].content.push('!');
        let ia = cache_candidate(&a, 2, &ByteCounter).content_id;
        let ib = cache_candidate(&b, 2, &ByteCounter).content_id;
        assert_eq!(ia, cache_candidate(&a, 2, &ByteCounter).content_id);
        assert_ne!(ia, ib);
    }
}
