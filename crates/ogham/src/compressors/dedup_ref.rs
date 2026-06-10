use async_trait::async_trait;
use ogham_core::{Compressed, CompressionContext, Compressor, Content, Result};
use std::collections::HashMap;
use tracing::debug;

use crate::ccr::{CcrStore, compute_key, marker_for};

/// Content-addressable deduplication compressor.
/// Replaces repeated content blocks with reference markers.
pub struct DedupRefCompressor {
    seen: std::sync::Mutex<HashMap<String, String>>,
    ccr_store: Option<std::sync::Arc<dyn CcrStore>>,
}

impl DedupRefCompressor {
    pub fn new() -> Self {
        Self {
            seen: std::sync::Mutex::new(HashMap::new()),
            ccr_store: None,
        }
    }

    pub fn with_ccr_store(ccr_store: std::sync::Arc<dyn CcrStore>) -> Self {
        Self {
            seen: std::sync::Mutex::new(HashMap::new()),
            ccr_store: Some(ccr_store),
        }
    }

    pub fn compress(&self, text: &str) -> String {
        let mut out = String::new();
        let mut total_saved = 0usize;
        for para in text.split("\n\n") {
            let trimmed = para.trim();
            if trimmed.is_empty() {
                continue;
            }
            let hash = compute_key(trimmed.as_bytes());
            let mut seen = self.seen.lock().unwrap();
            if let Some(first_seen) = seen.get(&hash) {
                if first_seen != trimmed {
                    // Hash collision — store the new one and emit inline
                    seen.insert(hash.clone(), trimmed.to_string());
                    drop(seen);
                    if !out.is_empty() {
                        out.push_str("\n\n");
                    }
                    out.push_str(trimmed);
                    continue;
                }
                drop(seen);
                total_saved += trimmed.len();
                if total_saved > 100 {
                    // Emit a reference marker
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(&marker_for(&hash));
                }
            } else {
                seen.insert(hash.clone(), trimmed.to_string());
                drop(seen);
                if !out.is_empty() {
                    out.push_str("\n\n");
                }
                out.push_str(trimmed);
            }
        }
        if total_saved > 0 && !out.contains("<<ccr:") {
            out.push_str(&format!(
                "\n[dedup: {} bytes of repeated content]",
                total_saved
            ));
        }
        out
    }
}

impl Default for DedupRefCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Compressor for DedupRefCompressor {
    fn name(&self) -> &'static str {
        "dedup_ref"
    }

    async fn compress(&self, content: &Content, ctx: &CompressionContext) -> Result<Compressed> {
        debug!(
            "DedupRefCompressor compressing {} bytes",
            content.data.len()
        );
        let text = String::from_utf8_lossy(&content.data);
        let compressed = self.compress(&text);
        let compressed_tokens = compressed.len() / 4;
        let id = compute_key(content.data.as_ref());
        if ctx.reversible {
            if let Some(store) = &self.ccr_store {
                let store_ref = store.clone();
                let id_clone = id.clone();
                let text_clone = text.to_string();
                tokio::spawn(async move {
                    let _ = store_ref.save(&id_clone, &text_clone, None).await;
                });
            }
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
    fn dedups_repeated_blocks() {
        let comp = DedupRefCompressor::new();
        let block = "This is a repeated paragraph that appears multiple times.";
        let input = format!("{}\n\n{}\n\nSome other text.\n\n{}", block, block, block);
        let out = comp.compress(&input);
        assert!(out.contains("Some other text"));
        assert!(out.contains("<<ccr:") || out.contains("dedup:"));
    }

    #[test]
    fn keeps_unique_blocks() {
        let comp = DedupRefCompressor::new();
        let input = "First unique block.\n\nSecond unique block.\n\nThird unique block.";
        let out = comp.compress(input);
        assert!(out.contains("First unique block"));
        assert!(out.contains("Second unique block"));
        assert!(out.contains("Third unique block"));
    }
}
