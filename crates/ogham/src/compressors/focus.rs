//! Shared focus/question-hint steering for content compressors.
//!
//! `SmartCrusher` pioneered biasing retention toward content relevant to the
//! caller's `CompactConfig.focus` hint. These helpers let the log, code, and
//! semantic compressors apply the same deterministic steering. An empty or
//! noise-only hint yields no terms, which keeps unfocused compression
//! byte-for-byte identical to the no-hint path.

/// Boost added to a discretionary item's relevance per matched focus term —
/// large enough to dominate length-based ties so a matching item is retained.
const FOCUS_TERM_BOOST: f64 = 100.0;

/// High-frequency English function words dropped from focus hints. Without this,
/// a natural hint like "what is the error" would keep `is`/`the`, which match
/// almost any text and defeat compression by marking everything relevant. These
/// are pure function words — domain terms (`error`, `fail`, `test`, …) are kept.
const STOPWORDS: &[&str] = &[
    "the", "an", "and", "or", "but", "nor", "so", "as", "if", "then", "than", "is", "are", "was",
    "were", "be", "been", "being", "am", "to", "of", "in", "on", "at", "by", "for", "with", "from",
    "into", "onto", "up", "out", "it", "its", "this", "that", "these", "those", "there", "here",
    "do", "does", "did", "done", "can", "could", "would", "should", "shall", "will", "may",
    "might", "must", "have", "has", "had", "what", "which", "who", "whom", "whose", "when",
    "where", "why", "how", "my", "me", "we", "us", "our", "you", "your", "they", "them", "their",
    "he", "she", "his", "her", "him", "too", "very", "just", "also", "not", "no",
];

/// Split a focus hint into lowercased, deduplicated search terms (alphanumeric
/// runs of length >= 2, stopwords removed). Returns empty for an empty or
/// noise-only hint, which keeps unfocused compression byte-identical.
pub fn terms(query: &str) -> Vec<String> {
    let mut terms: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.chars().count() >= 2)
        .map(str::to_lowercase)
        .filter(|t| !STOPWORDS.contains(&t.as_str()))
        .collect();
    terms.sort();
    terms.dedup();
    terms
}

/// Whether `text` contains any focus term (case-insensitive). Always `false`
/// when there are no terms, so unfocused paths are unaffected.
pub fn matches(text: &str, focus: &[String]) -> bool {
    if focus.is_empty() {
        return false;
    }
    focus.iter().any(|t| contains_ignore_case(text, t))
}

/// Score how strongly `text` matches the focus terms: matched-term count times
/// a boost large enough to dominate length-based ties. Zero when there are no
/// terms.
pub fn relevance(text: &str, focus: &[String]) -> f64 {
    if focus.is_empty() {
        return 0.0;
    }
    let hits = focus
        .iter()
        .filter(|t| contains_ignore_case(text, t))
        .count();
    hits as f64 * FOCUS_TERM_BOOST
}

/// Case-insensitive substring test. `needle` is already lowercased by
/// [`terms`]. Avoids allocating a lowercased copy of `haystack` on the common
/// all-ASCII path (these compressors run per log line / code line), falling
/// back to a full lowercasing only when non-ASCII is involved — so the result
/// is identical to `haystack.to_lowercase().contains(needle)`.
fn contains_ignore_case(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    if haystack.is_ascii() && needle.is_ascii() {
        let (h, n) = (haystack.as_bytes(), needle.as_bytes());
        return n.len() <= h.len() && h.windows(n.len()).any(|w| w.eq_ignore_ascii_case(n));
    }
    haystack.to_lowercase().contains(needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terms_lowercases_drops_stopwords_and_dedups() {
        // Stopwords ("the", "in") removed; result lowercased, sorted, deduped.
        assert_eq!(
            terms("Fix the AuthBug in login login"),
            vec!["authbug", "fix", "login"]
        );
        assert!(terms("").is_empty());
        assert!(terms("  , . ;").is_empty());
        // A hint made only of stopwords yields no terms -> focus stays off.
        assert!(terms("what is the").is_empty());
    }

    #[test]
    fn matches_is_case_insensitive_and_empty_safe() {
        let f = terms("auth login");
        assert!(matches("checking AUTH flow", &f));
        assert!(!matches("unrelated database call", &f));
        assert!(!matches("anything", &[]));
    }

    #[test]
    fn relevance_counts_terms() {
        let f = terms("auth login");
        assert_eq!(relevance("auth and login here", &f), 200.0);
        assert_eq!(relevance("only auth", &f), 100.0);
        assert_eq!(relevance("nothing", &f), 0.0);
        assert_eq!(relevance("auth", &[]), 0.0);
    }

    #[test]
    fn contains_ignore_case_matches_lowercasing_semantics() {
        // ASCII fast path and the unicode fallback must agree with the
        // reference `to_lowercase().contains` behavior.
        for (hay, needle) in [
            ("checking AUTH flow", "auth"),
            ("no match here", "auth"),
            ("Visited the CAFÉ today", "café"), // non-ASCII -> fallback
            ("ASCII text", "ascii"),
        ] {
            let reference = hay.to_lowercase().contains(needle);
            assert_eq!(
                contains_ignore_case(hay, needle),
                reference,
                "mismatch for {hay:?} / {needle:?}"
            );
        }
        // Empty needle is vacuously contained; over-long needle is not.
        assert!(contains_ignore_case("x", ""));
        assert!(!contains_ignore_case("x", "xxxx"));
    }
}
