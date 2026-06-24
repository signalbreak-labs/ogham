//! Searchable recall of folded content.
//!
//! Reversible CCR is exact-id-only: you can retrieve an original only if you
//! hold its `<<ccr:HASH>>` marker. An agent can't ask "pull back that earlier
//! tool output about the auth bug." A [`RecallIndex`] is a deterministic BM25
//! keyword index over folded content — index each fold's original text under
//! its CCR id, then [`search`](RecallIndex::search) by relevance to get the ids
//! to retrieve. It is pure and deterministic (no embeddings, no network), and
//! [`ContextSession`](crate::session::ContextSession) maintains one automatically
//! so its fold ledger becomes searchable memory.

use crate::compact::FoldKind;
use std::collections::HashMap;

const PREVIEW_CHARS: usize = 160;
// Standard BM25 parameters.
const K1: f64 = 1.5;
const B: f64 = 0.75;

/// One ranked search result.
#[derive(Debug, Clone, PartialEq)]
pub struct RecallHit {
    /// CCR id to retrieve the original content.
    pub ccr_id: String,
    /// What kind of fold produced this content.
    pub kind: FoldKind,
    /// A short preview of the original content.
    pub preview: String,
    /// BM25 relevance score (higher is more relevant).
    pub score: f64,
}

struct Doc {
    ccr_id: String,
    kind: FoldKind,
    preview: String,
    /// term -> frequency in this document.
    terms: HashMap<String, usize>,
    /// total number of (non-unique) terms.
    len: usize,
}

/// A deterministic BM25 index over folded content, addressable by CCR id.
#[derive(Default)]
pub struct RecallIndex {
    docs: Vec<Doc>,
}

impl RecallIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Whether the index holds no documents.
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Index a piece of folded content's original `text`, addressable by
    /// `ccr_id`. Re-indexing the same id replaces the prior entry.
    pub fn index(&mut self, ccr_id: impl Into<String>, kind: FoldKind, text: &str) {
        let ccr_id = ccr_id.into();
        self.remove(&ccr_id);

        let mut terms: HashMap<String, usize> = HashMap::new();
        let mut len = 0usize;
        for term in extract_terms(text) {
            *terms.entry(term).or_default() += 1;
            len += 1;
        }
        self.docs.push(Doc {
            ccr_id,
            kind,
            preview: preview(text),
            terms,
            len,
        });
    }

    /// Remove the document for `ccr_id`, if present (e.g. when its original was
    /// garbage-collected). Returns whether anything was removed.
    pub fn remove(&mut self, ccr_id: &str) -> bool {
        let before = self.docs.len();
        self.docs.retain(|d| d.ccr_id != ccr_id);
        self.docs.len() != before
    }

    /// Search by relevance, returning up to `limit` hits with a positive score,
    /// best first. Ties break by `ccr_id` for determinism.
    pub fn search(&self, query: &str, limit: usize) -> Vec<RecallHit> {
        if self.docs.is_empty() || limit == 0 {
            return Vec::new();
        }
        let n = self.docs.len() as f64;
        let avg_len = self.docs.iter().map(|d| d.len).sum::<usize>() as f64 / n;

        // Unique query terms + their document frequency.
        let mut query_terms: Vec<String> = extract_terms(query);
        query_terms.sort();
        query_terms.dedup();
        let idf: HashMap<&str, f64> = query_terms
            .iter()
            .map(|term| {
                let df = self
                    .docs
                    .iter()
                    .filter(|d| d.terms.contains_key(term))
                    .count() as f64;
                // Lucene/Elasticsearch BM25 IDF: ln(1 + (n - df + 0.5)/(df + 0.5)).
                // The `1 +` inside the log floors IDF at 0, avoiding the negative
                // IDF the classic Robertson–Spärck-Jones form produces for terms in
                // >50% of docs (which would make a common term subtract from a doc's
                // score — wrong for a small per-session recall index). Still strictly
                // decreasing in df, so rarer terms always outrank common ones.
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                (term.as_str(), idf)
            })
            .collect();

        let mut hits: Vec<RecallHit> = self
            .docs
            .iter()
            .filter_map(|doc| {
                let score: f64 = query_terms
                    .iter()
                    .map(|term| {
                        let tf = *doc.terms.get(term).unwrap_or(&0) as f64;
                        if tf == 0.0 {
                            return 0.0;
                        }
                        let idf = idf.get(term.as_str()).copied().unwrap_or(0.0);
                        let denom = tf + K1 * (1.0 - B + B * doc.len as f64 / avg_len);
                        idf * (tf * (K1 + 1.0)) / denom
                    })
                    .sum();
                (score > 0.0).then(|| RecallHit {
                    ccr_id: doc.ccr_id.clone(),
                    kind: doc.kind,
                    preview: doc.preview.clone(),
                    score,
                })
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.ccr_id.cmp(&b.ccr_id))
        });
        hits.truncate(limit);
        hits
    }
}

/// Tokenize `text` into lowercased recall terms: each word, plus the sub-parts
/// of path-like and identifier-like tokens, so `src/auth/login.rs` also matches
/// `auth`, `login`, and `rs`, and `parseToolResult` also matches `parse`,
/// `tool`, `result`.
pub fn extract_terms(text: &str) -> Vec<String> {
    let mut terms = Vec::new();
    for raw in text.split(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                '"' | '\''
                    | '`'
                    | '('
                    | ')'
                    | ','
                    | ';'
                    | ':'
                    | '['
                    | ']'
                    | '{'
                    | '}'
                    | '='
                    | '<'
                    | '>'
                    | '|'
                    | '!'
                    | '?'
                    | '*'
                    | '&'
                    | '@'
                    | '#'
                    | '%'
                    | '+'
                    | '~'
                    | '^'
                    | '$'
            )
    }) {
        let token = raw.trim_matches(|c: char| {
            !c.is_alphanumeric() && !matches!(c, '/' | '.' | '_' | '-' | '\\')
        });
        if token.chars().filter(|c| c.is_alphanumeric()).count() < 2 {
            continue;
        }
        let lower = token.to_lowercase();
        terms.push(lower.clone());
        for part in split_token(token) {
            if part != lower && part.chars().count() >= 2 {
                terms.push(part);
            }
        }
    }
    terms
}

/// Split a token on path/identifier separators and camelCase boundaries,
/// returning lowercased sub-parts.
fn split_token(token: &str) -> Vec<String> {
    let mut parts = Vec::new();
    for chunk in token.split(['/', '.', '_', '-', '\\']) {
        if chunk.is_empty() {
            continue;
        }
        let mut current = String::new();
        let mut prev_alnum_lower = false;
        for ch in chunk.chars() {
            if ch.is_uppercase() && prev_alnum_lower && !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
            current.push(ch.to_ascii_lowercase());
            prev_alnum_lower = ch.is_lowercase() || ch.is_numeric();
        }
        if !current.is_empty() {
            parts.push(current);
        }
    }
    parts
}

fn preview(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= PREVIEW_CHARS {
        return trimmed.to_string();
    }
    let mut out: String = trimmed.chars().take(PREVIEW_CHARS).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_terms_splits_paths_and_identifiers() {
        let terms = extract_terms("opened src/auth/login.rs in parseToolResult");
        for expected in [
            "opened",
            "src/auth/login.rs",
            "auth",
            "login",
            "rs",
            "parse",
            "tool",
            "result",
        ] {
            assert!(
                terms.contains(&expected.to_string()),
                "missing {expected}: {terms:?}"
            );
        }
    }

    #[test]
    fn search_ranks_relevant_fold_first() {
        let mut idx = RecallIndex::new();
        idx.index(
            "b3:auth",
            FoldKind::Cleared,
            "Error: authentication failed for user login in src/auth/login.rs",
        );
        idx.index(
            "b3:db",
            FoldKind::Cleared,
            "connected to the postgres database and ran a migration",
        );
        idx.index(
            "b3:ui",
            FoldKind::Dropped,
            "rendered the settings panel and the sidebar",
        );

        let hits = idx.search("auth login bug", 5);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].ccr_id, "b3:auth", "the auth fold must rank first");
        // The unrelated UI fold should not match these terms at all.
        assert!(hits.iter().all(|h| h.ccr_id != "b3:ui"));
    }

    #[test]
    fn search_returns_nothing_for_unmatched_query() {
        let mut idx = RecallIndex::new();
        idx.index("b3:x", FoldKind::Cleared, "the cat sat on the mat");
        assert!(idx.search("kubernetes deployment", 5).is_empty());
    }

    #[test]
    fn remove_drops_a_document() {
        let mut idx = RecallIndex::new();
        idx.index("b3:x", FoldKind::Cleared, "auth login token");
        assert_eq!(idx.len(), 1);
        assert!(idx.remove("b3:x"));
        assert!(idx.search("auth", 5).is_empty());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn reindexing_same_id_replaces() {
        let mut idx = RecallIndex::new();
        idx.index("b3:x", FoldKind::Cleared, "old content about apples");
        idx.index("b3:x", FoldKind::Cleared, "new content about oranges");
        assert_eq!(idx.len(), 1);
        assert!(idx.search("apples", 5).is_empty());
        assert_eq!(idx.search("oranges", 5).len(), 1);
    }

    #[test]
    fn search_is_deterministic() {
        let mut idx = RecallIndex::new();
        idx.index(
            "b3:a",
            FoldKind::Cleared,
            "shared term shared term unique_a",
        );
        idx.index(
            "b3:b",
            FoldKind::Cleared,
            "shared term shared term unique_b",
        );
        let a = idx.search("shared", 5);
        let b = idx.search("shared", 5);
        assert_eq!(a, b);
    }
}
