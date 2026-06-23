//! Host-neutral rich message content.
//!
//! The flat [`crate::Message`] (`content: String`) is enough for plain chat,
//! but agent transcripts carry structure: tool calls, tool results, thinking,
//! images, and references to files/memory. Forcing a host to flatten those into
//! a JSON string before compression creates an implicit lossy path. This module
//! adds a block-structured representation ([`RichMessage`] / [`ContentBlock`])
//! that round-trips losslessly through serde, plus an explicit, marked lossy
//! conversion down to a flat [`crate::Message`] for the text-only pipeline.
//!
//! The types are deliberately host-neutral: a host maps its own message format
//! into these blocks at the adapter boundary instead of stringifying it.

use crate::Message;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Metadata key set on a flat [`Message`] when [`RichMessage::to_flat_lossy`]
/// discarded block structure. Its presence means the text is a lossy rendering.
pub const META_FLATTENED: &str = "ogham.flattened";

/// Where image bytes live. Ogham keeps images opaque — it never decodes them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageSource {
    /// Base64-encoded image data with its media type (e.g. `image/png`).
    Base64 {
        /// The image media type.
        media_type: String,
        /// Base64-encoded image bytes.
        data: String,
    },
    /// A URL the host will resolve.
    Url {
        /// The image URL.
        url: String,
    },
}

/// A structured content block within a [`RichMessage`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text.
    Text {
        /// The text.
        text: String,
    },
    /// Model reasoning / thinking text.
    Thinking {
        /// The thinking text.
        text: String,
    },
    /// An image, kept opaque.
    Image {
        /// Where the image bytes live.
        source: ImageSource,
        /// Optional alt text.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        alt: Option<String>,
    },
    /// A tool call issued by the assistant.
    ToolUse {
        /// Provider tool-call id.
        id: String,
        /// Tool name.
        name: String,
        /// Tool input arguments.
        input: Value,
    },
    /// The result of a tool call.
    ToolResult {
        /// The `id` of the [`ContentBlock::ToolUse`] this answers.
        tool_use_id: String,
        /// Whether the tool reported an error.
        #[serde(default)]
        is_error: bool,
        /// Nested result content.
        content: Vec<ContentBlock>,
    },
    /// A reference to host-managed content (file, directory, memory, skill, …).
    Reference {
        /// Reference kind (e.g. `file`, `memory`).
        kind: String,
        /// Identifier or path of the referenced content.
        id_or_path: String,
        /// Host-defined metadata.
        #[serde(default, skip_serializing_if = "HashMap::is_empty")]
        metadata: HashMap<String, String>,
    },
}

/// Either flat text or a sequence of structured blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Flat text content.
    Text(String),
    /// Structured block content.
    Blocks(Vec<ContentBlock>),
}

/// A message whose content may be structured blocks rather than flat text.
///
/// Round-trips losslessly through serde. Convert from a flat [`Message`] with
/// [`From`], and down to one with [`RichMessage::to_flat_lossy`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RichMessage {
    /// Message role (e.g. `user`, `assistant`, `tool`).
    pub role: String,
    /// Structured or flat content.
    pub content: MessageContent,
    /// Optional annotations (same namespaced keys as [`crate::meta_keys`]).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

impl RichMessage {
    /// Construct a text [`RichMessage`].
    pub fn text(role: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: MessageContent::Text(text.into()),
            metadata: HashMap::new(),
        }
    }

    /// Construct a block [`RichMessage`].
    pub fn blocks(role: impl Into<String>, blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: role.into(),
            content: MessageContent::Blocks(blocks),
            metadata: HashMap::new(),
        }
    }

    /// True when flattening to a [`Message`] would discard structure.
    pub fn is_lossy_to_flatten(&self) -> bool {
        matches!(self.content, MessageContent::Blocks(_))
    }

    /// Flatten to a [`Message`].
    ///
    /// Text content is preserved exactly. Block content is rendered to text and
    /// the result is marked with [`META_FLATTENED`] so callers know the flat
    /// form is lossy and the structured original should be retrieved elsewhere.
    pub fn to_flat_lossy(&self) -> Message {
        match &self.content {
            MessageContent::Text(text) => Message {
                role: self.role.clone(),
                content: text.clone(),
                metadata: self.metadata.clone(),
            },
            MessageContent::Blocks(blocks) => {
                let mut metadata = self.metadata.clone();
                metadata.insert(META_FLATTENED.to_string(), "true".to_string());
                Message {
                    role: self.role.clone(),
                    content: render_blocks(blocks),
                    metadata,
                }
            }
        }
    }
}

impl From<Message> for RichMessage {
    /// Lift a flat message into the rich model as text content. Lossless.
    fn from(m: Message) -> Self {
        Self {
            role: m.role,
            content: MessageContent::Text(m.content),
            metadata: m.metadata,
        }
    }
}

/// Render blocks to a readable, deterministic text approximation.
fn render_blocks(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .map(render_block)
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_block(block: &ContentBlock) -> String {
    match block {
        ContentBlock::Text { text } => text.clone(),
        ContentBlock::Thinking { text } => format!("[thinking] {text}"),
        ContentBlock::Image { source, alt } => {
            let what = match source {
                ImageSource::Base64 { media_type, .. } => media_type.clone(),
                ImageSource::Url { url } => url.clone(),
            };
            match alt {
                Some(alt) => format!("[image: {what} — {alt}]"),
                None => format!("[image: {what}]"),
            }
        }
        ContentBlock::ToolUse { name, input, .. } => format!("[tool_use {name}: {input}]"),
        ContentBlock::ToolResult {
            is_error, content, ..
        } => {
            let tag = if *is_error {
                "tool_result error"
            } else {
                "tool_result"
            };
            format!("[{tag}] {}", render_blocks(content))
        }
        ContentBlock::Reference {
            kind, id_or_path, ..
        } => format!("[{kind}: {id_or_path}]"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn all_block_kinds() -> Vec<ContentBlock> {
        vec![
            ContentBlock::Text {
                text: "hello".into(),
            },
            ContentBlock::Thinking {
                text: "reasoning".into(),
            },
            ContentBlock::Image {
                source: ImageSource::Base64 {
                    media_type: "image/png".into(),
                    data: "QUJD".into(),
                },
                alt: Some("a chart".into()),
            },
            ContentBlock::ToolUse {
                id: "call_1".into(),
                name: "shell".into(),
                input: json!({ "cmd": "ls" }),
            },
            ContentBlock::ToolResult {
                tool_use_id: "call_1".into(),
                is_error: false,
                content: vec![ContentBlock::Text {
                    text: "file.txt".into(),
                }],
            },
            ContentBlock::Reference {
                kind: "file".into(),
                id_or_path: "/etc/hosts".into(),
                metadata: HashMap::new(),
            },
        ]
    }

    #[test]
    fn rich_message_serde_round_trips_losslessly() {
        let original = RichMessage::blocks("assistant", all_block_kinds());
        let json = serde_json::to_string(&original).expect("serialize");
        let back: RichMessage = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, back);
    }

    #[test]
    fn from_flat_message_is_lossless_text() {
        let mut m = Message::new("user", "just text");
        m.metadata.insert("k".into(), "v".into());
        let rich: RichMessage = m.clone().into();
        assert!(!rich.is_lossy_to_flatten());
        // text -> rich -> flat is a perfect round trip (no flatten marker).
        let flat = rich.to_flat_lossy();
        assert_eq!(flat.content, "just text");
        assert_eq!(flat.metadata.get("k").map(String::as_str), Some("v"));
        assert!(!flat.metadata.contains_key(META_FLATTENED));
    }

    #[test]
    fn blocks_flatten_lossily_and_are_marked() {
        let rich = RichMessage::blocks("assistant", all_block_kinds());
        assert!(rich.is_lossy_to_flatten());
        let flat = rich.to_flat_lossy();
        assert_eq!(
            flat.metadata.get(META_FLATTENED).map(String::as_str),
            Some("true")
        );
        // Rendering surfaces each block's salient text.
        assert!(flat.content.contains("hello"));
        assert!(flat.content.contains("[thinking] reasoning"));
        assert!(flat.content.contains("[tool_use shell:"));
        assert!(flat.content.contains("[tool_result]"));
        assert!(flat.content.contains("[file: /etc/hosts]"));
    }

    #[test]
    fn untagged_content_distinguishes_text_from_blocks() {
        let text: MessageContent = serde_json::from_str("\"hi\"").unwrap();
        assert_eq!(text, MessageContent::Text("hi".into()));
        let blocks: MessageContent =
            serde_json::from_str(r#"[{"type":"text","text":"hi"}]"#).unwrap();
        assert!(matches!(blocks, MessageContent::Blocks(_)));
    }
}
