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

/// Split a focus hint into lowercased search terms (alphanumeric runs of length
/// >= 2). Returns empty for an empty or noise-only hint.
pub fn terms(query: &str) -> Vec<String> {
    query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.chars().count() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Whether `text` contains any focus term (case-insensitive). Always `false`
/// when there are no terms, so unfocused paths are unaffected.
pub fn matches(text: &str, focus: &[String]) -> bool {
    if focus.is_empty() {
        return false;
    }
    let lower = text.to_lowercase();
    focus.iter().any(|t| lower.contains(t.as_str()))
}

/// Score how strongly `text` matches the focus terms: matched-term count times
/// a boost large enough to dominate length-based ties. Zero when there are no
/// terms.
pub fn relevance(text: &str, focus: &[String]) -> f64 {
    if focus.is_empty() {
        return 0.0;
    }
    let lower = text.to_lowercase();
    let hits = focus.iter().filter(|t| lower.contains(t.as_str())).count();
    hits as f64 * FOCUS_TERM_BOOST
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terms_lowercases_and_drops_short_tokens() {
        assert_eq!(terms("Fix the AuthBug in login"), {
            // "the" and "in" are kept (>=2 chars); single chars dropped.
            vec!["fix", "the", "authbug", "in", "login"]
        });
        assert!(terms("").is_empty());
        assert!(terms("  , . ;").is_empty());
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
}
