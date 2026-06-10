pub mod error;
pub mod metrics;
pub mod token;

pub use error::{OghamError, Result};
pub use metrics::{
    CompressionEvent, Metrics, NoopMetrics, NoopObserver, Observer, PerCompressorStats,
    PipelineStats, TestMetrics, TestObserver,
};
pub use token::{OVERHEAD_PER_MESSAGE, TokenCounter};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// Optional annotations. Absent in serialized form when empty, so existing
    /// JSON wire formats are unchanged.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            metadata: HashMap::new(),
        }
    }
}

/// Well-known metadata keys. Values are plain strings.
pub mod meta_keys {
    /// One of the `AgentContentType` snake_case names (WP-4 writes this).
    pub const AGENT_CONTENT_TYPE: &str = "ogham.agent_content_type";
    /// "success" | "error" — status of a tool-result message.
    pub const TOOL_STATUS: &str = "ogham.tool_status";
    /// Name of the tool that produced a tool message.
    pub const TOOL_NAME: &str = "ogham.tool_name";
    /// CCR id of the original content, when this message was compressed/cleared.
    pub const CCR_ID: &str = "ogham.ccr_id";
    /// "ephemeral" — provider-cache breakpoint marker (WP-7 writes this).
    pub const CACHE_CONTROL: &str = "ogham.cache_control";
    /// "true" — message must never be compressed, cleared, or summarized.
    pub const PINNED: &str = "ogham.pinned";
}

/// Input content to be compressed.
#[derive(Debug, Clone)]
pub struct Content {
    pub data: Bytes,
    pub mime_or_lang: String,
    pub metadata: HashMap<String, String>,
}

/// Context passed to compressors so they can tailor output.
#[derive(Debug, Clone)]
pub struct CompressionContext {
    pub model: String,
    pub question_hint: Option<String>,
    pub max_tokens: Option<usize>,
    pub reversible: bool,
}

/// Result of compressing a single piece of content.
#[derive(Debug, Clone)]
pub struct Compressed {
    pub id: String,
    pub data: Bytes,
    pub original_tokens: usize,
    pub compressed_tokens: usize,
}

/// Statistics for a compression run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompressionStats {
    pub original_tokens: usize,
    pub compressed_tokens: usize,
    pub ratio: f64,
    pub compressor_used: String,
}

/// A batch of compressed messages plus stats.
#[derive(Debug, Clone)]
pub struct CompressedMessages {
    pub messages: Vec<Message>,
    pub stats: CompressionStats,
}

/// Trait implemented by every compression backend.
#[async_trait]
pub trait Compressor: Send + Sync {
    fn name(&self) -> &'static str;

    async fn compress(&self, content: &Content, ctx: &CompressionContext) -> Result<Compressed>;

    /// For reversible compressors: fetch the original by id.
    async fn retrieve(&self, id: &str) -> Result<Option<String>>;
}

/// A pipeline chains one or more compressors.
#[async_trait]
pub trait CompressionPipeline: Send + Sync {
    fn add_compressor(&mut self, compressor: Box<dyn Compressor>);

    async fn run(&self, messages: &[Message]) -> Result<CompressedMessages>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_metadata_roundtrip() {
        let msg = Message::new("user", "hi");
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = parsed.as_object().unwrap();
        assert!(obj.contains_key("role"));
        assert!(obj.contains_key("content"));
        assert!(
            !obj.contains_key("metadata"),
            "empty metadata must be omitted"
        );
        assert_eq!(obj.len(), 2);

        // Legacy JSON without metadata field must deserialize successfully.
        let legacy = r#"{"role":"user","content":"hi"}"#;
        let decoded: Message = serde_json::from_str(legacy).unwrap();
        assert_eq!(decoded.role, "user");
        assert_eq!(decoded.content, "hi");
        assert!(decoded.metadata.is_empty());
    }
}
