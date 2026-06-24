use async_trait::async_trait;
use ogham_core::{Compressed, CompressionContext, Compressor, Content, Result};
use std::collections::HashMap;
use tracing::debug;

use crate::ccr::{CcrStore, compute_key};

/// Simple semantic compressor: removes redundant whitespace and
/// deduplicates repeated sentences/paragraphs.
pub struct SemanticCompressor {
    ccr_store: Option<std::sync::Arc<dyn CcrStore>>,
}

impl SemanticCompressor {
    pub fn new() -> Self {
        Self { ccr_store: None }
    }

    pub fn with_ccr_store(ccr_store: std::sync::Arc<dyn CcrStore>) -> Self {
        Self {
            ccr_store: Some(ccr_store),
        }
    }

    pub fn compress_text(&self, text: &str) -> String {
        self.compress_text_focused(text, &[])
    }

    /// Compress text, additionally keeping paragraphs that match the `focus`
    /// terms at full length and never replacing them with a `[see paragraph N]`
    /// dedup reference. Unfocused paragraphs keep the default dedup/truncate
    /// behavior.
    pub fn compress_text_focused(&self, text: &str, focus: &[String]) -> String {
        // Split into paragraphs
        let paragraphs: Vec<&str> = text.split("\n\n").collect();
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut out: Vec<String> = Vec::new();
        let mut dup_count = 0usize;

        for para in paragraphs {
            let normalized = para.split_whitespace().collect::<Vec<_>>().join(" ");
            if normalized.is_empty() {
                continue;
            }
            let is_focus = crate::compressors::focus::matches(para, focus);
            // Focus paragraphs bypass dedup so they are always emitted in full.
            if !is_focus && let Some(&first_idx) = seen.get(&normalized) {
                dup_count += 1;
                if dup_count <= 3 {
                    out.push(format!("[see paragraph {}]", first_idx + 1));
                }
                continue;
            }
            // Remember the first position this paragraph appeared at.
            seen.entry(normalized.clone()).or_insert(out.len());
            // Trim non-focus paragraphs to a reasonable length (char-safe).
            let trimmed = if !is_focus && para.len() > 800 {
                format!(
                    "{}... [truncated]",
                    crate::compressors::truncate_on_boundary(para, 800)
                )
            } else {
                para.to_string()
            };
            out.push(trimmed);
        }

        // If very high duplication, add a summary line
        if dup_count > 5 && !out.is_empty() {
            out.push(format!("[{} duplicate paragraphs omitted]", dup_count));
        }

        out.join("\n\n")
    }
}

impl Default for SemanticCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Compressor for SemanticCompressor {
    fn name(&self) -> &'static str {
        "semantic"
    }

    async fn compress(&self, content: &Content, ctx: &CompressionContext) -> Result<Compressed> {
        debug!(
            "SemanticCompressor compressing {} bytes",
            content.data.len()
        );
        let text = String::from_utf8_lossy(&content.data);
        let focus = crate::compressors::focus::terms(ctx.question_hint.as_deref().unwrap_or(""));
        let compressed = self.compress_text_focused(&text, &focus);
        let compressed_tokens = compressed.len() / 4;
        let id = compute_key(content.data.as_ref());
        if ctx.reversible
            && let Some(store) = &self.ccr_store
        {
            store.save(&id, &text, None).await?;
        }
        Ok(Compressed {
            id,
            data: bytes::Bytes::from(compressed),
            original_tokens: text.len() / 4,
            compressed_tokens,
        })
    }

    async fn retrieve(&self, id: &str) -> Result<Option<String>> {
        if let Some(store) = &self.ccr_store {
            store.retrieve(id).await
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deduplicates_repeated_paragraphs() {
        let comp = SemanticCompressor::new();
        let input = "First paragraph.\n\nRepeated text.\n\nRepeated text.\n\nLast paragraph.";
        let out = comp.compress_text(input);
        assert!(out.contains("First paragraph"));
        assert!(out.contains("[see paragraph"));
        assert!(out.contains("Last paragraph"));
    }

    #[test]
    fn truncates_long_paragraphs() {
        let comp = SemanticCompressor::new();
        let input = "a".repeat(2000);
        let out = comp.compress_text(&input);
        assert!(out.contains("[truncated]"));
    }

    #[test]
    fn focus_keeps_paragraph_full_and_undeduped() {
        let comp = SemanticCompressor::new();
        let long = format!("auth token rotation {}", "x".repeat(1000));
        // Two identical focus-matching paragraphs.
        let input = format!("{long}\n\n{long}");
        let focus = crate::compressors::focus::terms("auth");
        let out = comp.compress_text_focused(&input, &focus);
        assert!(
            !out.contains("[truncated]"),
            "focus paragraph kept full length"
        );
        assert!(
            !out.contains("[see paragraph"),
            "focus paragraph not deduped"
        );
        assert_eq!(
            out.matches("auth token rotation").count(),
            2,
            "both focus paragraphs emitted in full"
        );
    }

    #[test]
    fn non_focus_paragraph_still_truncates_on_char_boundary() {
        let comp = SemanticCompressor::new();
        // Multibyte content longer than the 800-byte cap must not panic.
        let input = "é".repeat(1000);
        let out = comp.compress_text_focused(&input, &crate::compressors::focus::terms("auth"));
        assert!(out.contains("[truncated]"));
    }
}
