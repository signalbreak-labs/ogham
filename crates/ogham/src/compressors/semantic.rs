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
            if let Some(&first_idx) = seen.get(&normalized) {
                dup_count += 1;
                if dup_count <= 3 {
                    out.push(format!("[see paragraph {}]", first_idx + 1));
                }
                continue;
            }
            seen.insert(normalized.clone(), out.len());
            // Trim each paragraph to a reasonable length
            let trimmed = if para.len() > 800 {
                format!("{}... [truncated]", &para[..800])
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
        let compressed = self.compress_text(&text);
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
}
