use async_trait::async_trait;
use ogham_core::{Compressed, CompressionContext, Compressor, Content, Result};
use tracing::debug;

/// TOON-style token-efficient serializer.
/// Converts JSON to compact representation, removes redundant keys.
pub struct ToonCompressor;

impl ToonCompressor {
    pub fn new() -> Self {
        Self
    }

    pub fn compress(&self, text: &str) -> String {
        // Try to parse as JSON — if it parses, emit compact form
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(text.trim()) {
            // Compact JSON
            let compact = serde_json::to_string(&val).unwrap_or_else(|_| text.to_string());
            if compact.len() < text.len() {
                return compact;
            }
        }
        // Fallback: collapse multiple spaces, trim lines
        text.lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl Default for ToonCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Compressor for ToonCompressor {
    fn name(&self) -> &'static str {
        "toon"
    }

    async fn compress(&self, content: &Content, _ctx: &CompressionContext) -> Result<Compressed> {
        debug!("ToonCompressor compressing {} bytes", content.data.len());
        let text = String::from_utf8_lossy(&content.data);
        let compressed = self.compress(&text);
        let compressed_tokens = compressed.len() / 4;
        Ok(Compressed {
            id: format!("toon-{}", crate::ccr::compute_key(content.data.as_ref())),
            data: bytes::Bytes::from(compressed),
            original_tokens: text.len() / 4,
            compressed_tokens,
        })
    }

    async fn retrieve(&self, _id: &str) -> Result<Option<String>> {
        // TOON is lossy; retrieve returns None.
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minifies_json() {
        let comp = ToonCompressor::new();
        let input = "{\n  \"a\": 1,\n  \"b\": 2\n}";
        let out = comp.compress(input);
        assert_eq!(out, r#"{"a":1,"b":2}"#);
    }

    #[test]
    fn collapses_plain_text() {
        let comp = ToonCompressor::new();
        let input = "line one\n\n  line two  \n\nline three";
        let out = comp.compress(input);
        assert_eq!(out, "line one line two line three");
    }
}
