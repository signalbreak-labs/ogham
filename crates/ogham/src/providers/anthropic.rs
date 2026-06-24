//! Anthropic server-side context editing (beta).
//!
//! Anthropic's API can clear old tool results server-side, mirroring
//! [`crate::agent::apply_agent_compression`] — but **irreversibly** (cleared
//! content is gone unless saved elsewhere) and only for Claude. Use this
//! adapter when a host targets the Claude API and wants the platform to do
//! first-line clearing, with Ogham handling everything the platform doesn't:
//! other providers, reversible CCR storage, budgets, and summaries.
//!
//! Requires the request beta header `anthropic-beta: context-management-2025-06-27`
//! (exposed as [`BETA_HEADER`]).

use crate::agent::AgentPolicy;
use ogham_core::{ContentBlock, ImageSource, Message, MessageContent, RichMessage, meta_keys};
use serde_json::{Value, json};

/// Value for the `anthropic-beta` request header that enables context editing.
pub const BETA_HEADER: &str = "context-management-2025-06-27";

/// Strategy identifier for server-side tool-result clearing.
pub const CLEAR_TOOL_USES: &str = "clear_tool_uses_20250919";

/// Configuration for Anthropic's `clear_tool_uses_20250919` strategy.
#[derive(Debug, Clone)]
pub struct AnthropicContextEditing {
    /// Clear when the prompt exceeds this many input tokens.
    pub trigger_input_tokens: usize,
    /// Number of most-recent tool uses to keep raw (maps from
    /// `AgentPolicy::keep_recent_tool_results`).
    pub keep_tool_uses: usize,
    /// Optional: clear at least this many tokens per edit, so the prompt-cache
    /// invalidation the edit causes is worthwhile.
    pub clear_at_least_tokens: Option<usize>,
    /// Tool names the server must never clear. Map your pinned/error-prone
    /// tools here — server-side clearing has no content-based error
    /// detection, unlike [`crate::agent::classify`].
    pub exclude_tools: Vec<String>,
    /// Also clear the tool-call inputs, not just results. Default false.
    pub clear_tool_inputs: bool,
}

impl AnthropicContextEditing {
    /// Derive a server-side config from an [`AgentPolicy`] and a clearing
    /// trigger. A sensible trigger is ~50–70% of your token budget so the
    /// platform clears proactively rather than at the context cliff.
    pub fn from_policy(policy: &AgentPolicy, trigger_input_tokens: usize) -> Self {
        Self {
            trigger_input_tokens,
            keep_tool_uses: policy.keep_recent_tool_results,
            clear_at_least_tokens: None,
            exclude_tools: Vec::new(),
            clear_tool_inputs: false,
        }
    }

    /// Render the `context_management` request-body fragment.
    ///
    /// Attach the returned value to the Messages API request as the
    /// `context_management` field, alongside the [`BETA_HEADER`].
    pub fn to_request_fragment(&self) -> Value {
        let mut edit = json!({
            "type": CLEAR_TOOL_USES,
            "trigger": { "type": "input_tokens", "value": self.trigger_input_tokens },
            "keep": { "type": "tool_uses", "value": self.keep_tool_uses },
        });
        if let Some(at_least) = self.clear_at_least_tokens {
            edit["clear_at_least"] = json!({ "type": "input_tokens", "value": at_least });
        }
        if !self.exclude_tools.is_empty() {
            edit["exclude_tools"] = json!(self.exclude_tools);
        }
        if self.clear_tool_inputs {
            edit["clear_tool_inputs"] = json!(true);
        }
        json!({ "edits": [edit] })
    }
}

/// Anthropic Messages API request parts rendered from a flat message list.
///
/// Splits system messages into the top-level `system` field and user/assistant
/// turns into `messages`, attaching `"cache_control": {"type": "ephemeral"}` to
/// the content block of any message annotated by
/// [`crate::cache_strategy::apply_cache_strategy`].
///
/// [`render_cache_control`] renders plain-text content only: a non-system role
/// other than `assistant` is normalized to `user` (valid Anthropic roles).
/// To preserve native tool-use / tool-result / image blocks, render
/// [`RichMessage`]s with [`render_cache_control_rich`] instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnthropicMessages {
    /// The `system` request field as an array of content blocks (possibly empty).
    pub system: Vec<Value>,
    /// The `messages` request field: user/assistant turns.
    pub messages: Vec<Value>,
}

/// Render `messages` into [`AnthropicMessages`], placing cache breakpoints on
/// blocks whose source message carries the `cache_control` annotation.
pub fn render_cache_control(messages: &[Message]) -> AnthropicMessages {
    let mut system = Vec::new();
    let mut turns = Vec::new();
    for m in messages {
        let mut block = json!({ "type": "text", "text": m.content });
        if m.metadata.contains_key(meta_keys::CACHE_CONTROL) {
            block["cache_control"] = json!({ "type": "ephemeral" });
        }
        if m.role == "system" {
            system.push(block);
        } else {
            let role = if m.role == "assistant" {
                "assistant"
            } else {
                "user"
            };
            turns.push(json!({ "role": role, "content": [block] }));
        }
    }
    AnthropicMessages {
        system,
        messages: turns,
    }
}

/// Render block-structured [`RichMessage`]s into [`AnthropicMessages`],
/// preserving native Anthropic content blocks instead of flattening to text.
///
/// Each [`ContentBlock`] maps to its Anthropic shape: text → `text`, tool calls
/// → `tool_use`, tool results → `tool_result` (with nested content rendered
/// recursively), images → `image` (base64 or url source). Two blocks have no
/// native Anthropic input shape and are rendered as `text` so their content
/// still reaches the model: `Thinking` (a replayable `thinking` block requires
/// the original provider signature, which the host-neutral model does not
/// carry) and `Reference` (rendered as `[kind: id_or_path]`).
///
/// Roles follow the Messages API: `system` messages become top-level `system`
/// blocks; `assistant` stays `assistant`; every other role (including `tool`)
/// becomes a `user` turn, which is where Anthropic expects `tool_result` blocks.
/// When a message carries the `cache_control` annotation, the breakpoint is
/// placed on its **last** block (so the whole message is within the cached
/// prefix). Messages that render to no blocks are skipped.
pub fn render_cache_control_rich(messages: &[RichMessage]) -> AnthropicMessages {
    let mut system = Vec::new();
    let mut turns = Vec::new();
    for m in messages {
        let mut blocks = render_message_content(&m.content);
        if blocks.is_empty() {
            continue;
        }
        if m.metadata.contains_key(meta_keys::CACHE_CONTROL)
            && let Some(last) = blocks.last_mut()
        {
            last["cache_control"] = json!({ "type": "ephemeral" });
        }
        if m.role == "system" {
            system.extend(blocks);
        } else {
            let role = if m.role == "assistant" {
                "assistant"
            } else {
                "user"
            };
            turns.push(json!({ "role": role, "content": blocks }));
        }
    }
    AnthropicMessages {
        system,
        messages: turns,
    }
}

fn render_message_content(content: &MessageContent) -> Vec<Value> {
    match content {
        MessageContent::Text(text) => vec![json!({ "type": "text", "text": text })],
        MessageContent::Blocks(blocks) => blocks.iter().map(render_block).collect(),
    }
}

fn render_block(block: &ContentBlock) -> Value {
    match block {
        ContentBlock::Text { text } => json!({ "type": "text", "text": text }),
        // No provider signature is available to replay a real `thinking` block,
        // so the reasoning text is preserved as a plain text block.
        ContentBlock::Thinking { text } => json!({ "type": "text", "text": text }),
        ContentBlock::Image { source, .. } => {
            json!({ "type": "image", "source": render_image_source(source) })
        }
        ContentBlock::ToolUse { id, name, input } => {
            json!({ "type": "tool_use", "id": id, "name": name, "input": input })
        }
        ContentBlock::ToolResult {
            tool_use_id,
            is_error,
            content,
        } => {
            let nested: Vec<Value> = content.iter().map(render_block).collect();
            json!({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "is_error": is_error,
                "content": nested,
            })
        }
        ContentBlock::Reference {
            kind, id_or_path, ..
        } => json!({ "type": "text", "text": format!("[{kind}: {id_or_path}]") }),
    }
}

fn render_image_source(source: &ImageSource) -> Value {
    match source {
        ImageSource::Base64 { media_type, data } => {
            json!({ "type": "base64", "media_type": media_type, "data": data })
        }
        ImageSource::Url { url } => json!({ "type": "url", "url": url }),
    }
}

/// Minimum input tokens before Anthropic prompt caching engages for `model`.
///
/// Anthropic's minimum cacheable prompt length is **versioned, not per-family** —
/// e.g. Haiku 4.5 needs 4096 while Sonnet 4.6 needs 1024, and Opus ranges from
/// 1024 (4.8) to 4096 (4.5/4.6). This table reflects the documented values as of
/// 2026-06 (Anthropic prompt-caching docs); the values change with new model
/// releases, so re-verify for a model not listed here. Matching is by substring
/// on a lowercased model id; unknown models fall back to 1024, the most common
/// current minimum.
pub fn min_cacheable_prefix_tokens(model: &str) -> usize {
    let m = model.to_ascii_lowercase();
    // Most specific / current first; each key is a distinct version token.
    const TABLE: &[(&str, usize)] = &[
        ("fable-5", 512),
        ("mythos-preview", 2048),
        ("mythos-5", 512),
        ("opus-4-8", 1024),
        ("opus-4-7", 2048),
        ("opus-4-6", 4096),
        ("opus-4-5", 4096),
        ("opus-4-1", 1024),
        ("sonnet-4-6", 1024),
        ("sonnet-4-5", 1024),
        ("haiku-4-5", 4096),
        ("haiku-3-5", 2048),
        ("3-5-haiku", 2048), // legacy id ordering (claude-3-5-haiku-…)
    ];
    for (key, min) in TABLE {
        if m.contains(key) {
            return *min;
        }
    }
    1024
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fragment_matches_documented_shape() {
        let cfg = AnthropicContextEditing {
            trigger_input_tokens: 30_000,
            keep_tool_uses: 3,
            clear_at_least_tokens: Some(5_000),
            exclude_tools: vec!["web_search".to_string()],
            clear_tool_inputs: false,
        };
        let frag = cfg.to_request_fragment();
        let edit = &frag["edits"][0];
        assert_eq!(edit["type"], CLEAR_TOOL_USES);
        assert_eq!(edit["trigger"]["type"], "input_tokens");
        assert_eq!(edit["trigger"]["value"], 30_000);
        assert_eq!(edit["keep"]["type"], "tool_uses");
        assert_eq!(edit["keep"]["value"], 3);
        assert_eq!(edit["clear_at_least"]["value"], 5_000);
        assert_eq!(edit["exclude_tools"][0], "web_search");
        assert!(edit.get("clear_tool_inputs").is_none());
    }

    #[test]
    fn optional_fields_omitted_by_default() {
        let cfg = AnthropicContextEditing::from_policy(&AgentPolicy::default(), 100_000);
        let frag = cfg.to_request_fragment();
        let edit = &frag["edits"][0];
        assert_eq!(edit["keep"]["value"], 3); // AgentPolicy default
        assert!(edit.get("clear_at_least").is_none());
        assert!(edit.get("exclude_tools").is_none());
    }

    #[test]
    fn deterministic_output() {
        let cfg = AnthropicContextEditing::from_policy(&AgentPolicy::default(), 50_000);
        assert_eq!(
            cfg.to_request_fragment().to_string(),
            cfg.to_request_fragment().to_string()
        );
    }

    #[test]
    fn render_places_cache_control_on_annotated_blocks() {
        let mut msgs = vec![
            Message::new("system", "rules"),
            Message::new("user", "hi"),
            Message::new("assistant", "hello"),
        ];
        msgs[0].metadata.insert(
            meta_keys::CACHE_CONTROL.to_string(),
            "ephemeral".to_string(),
        );

        let rendered = render_cache_control(&msgs);
        // System split out, with a cache breakpoint.
        assert_eq!(rendered.system.len(), 1);
        assert_eq!(rendered.system[0]["cache_control"]["type"], "ephemeral");
        // Two turns, neither annotated.
        assert_eq!(rendered.messages.len(), 2);
        assert_eq!(rendered.messages[0]["role"], "user");
        assert_eq!(rendered.messages[1]["role"], "assistant");
        assert!(
            rendered.messages[0]["content"][0]
                .get("cache_control")
                .is_none()
        );
    }

    #[test]
    fn render_normalizes_unknown_roles_to_user() {
        let msgs = vec![Message::new("tool", "result text")];
        let rendered = render_cache_control(&msgs);
        assert!(rendered.system.is_empty());
        assert_eq!(rendered.messages[0]["role"], "user");
        assert_eq!(rendered.messages[0]["content"][0]["text"], "result text");
    }

    #[test]
    fn render_without_annotations_has_no_cache_control() {
        let msgs = vec![Message::new("system", "rules"), Message::new("user", "hi")];
        let rendered = render_cache_control(&msgs);
        assert!(rendered.system[0].get("cache_control").is_none());
        assert!(
            rendered.messages[0]["content"][0]
                .get("cache_control")
                .is_none()
        );
    }

    // ─── Rich block rendering ────────────────────────────────────────────

    #[test]
    fn rich_renders_tool_use_and_tool_result_natively() {
        let assistant = RichMessage::blocks(
            "assistant",
            vec![
                ContentBlock::Text {
                    text: "let me check".into(),
                },
                ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "shell".into(),
                    input: json!({ "cmd": "ls" }),
                },
            ],
        );
        let tool = RichMessage::blocks(
            "tool",
            vec![ContentBlock::ToolResult {
                tool_use_id: "call_1".into(),
                is_error: false,
                content: vec![ContentBlock::Text {
                    text: "a.txt b.txt".into(),
                }],
            }],
        );

        let rendered = render_cache_control_rich(&[assistant, tool]);
        assert!(rendered.system.is_empty());
        assert_eq!(rendered.messages.len(), 2);

        // Assistant turn keeps text + tool_use.
        assert_eq!(rendered.messages[0]["role"], "assistant");
        assert_eq!(rendered.messages[0]["content"][0]["type"], "text");
        assert_eq!(rendered.messages[0]["content"][1]["type"], "tool_use");
        assert_eq!(rendered.messages[0]["content"][1]["id"], "call_1");
        assert_eq!(rendered.messages[0]["content"][1]["name"], "shell");
        assert_eq!(rendered.messages[0]["content"][1]["input"]["cmd"], "ls");

        // Tool result lands in a USER turn as a tool_result block.
        assert_eq!(rendered.messages[1]["role"], "user");
        let tr = &rendered.messages[1]["content"][0];
        assert_eq!(tr["type"], "tool_result");
        assert_eq!(tr["tool_use_id"], "call_1");
        assert_eq!(tr["is_error"], false);
        assert_eq!(tr["content"][0]["type"], "text");
        assert_eq!(tr["content"][0]["text"], "a.txt b.txt");
    }

    #[test]
    fn rich_renders_image_sources() {
        let base64 = RichMessage::blocks(
            "user",
            vec![ContentBlock::Image {
                source: ImageSource::Base64 {
                    media_type: "image/png".into(),
                    data: "AAAA".into(),
                },
                alt: Some("ignored by anthropic".into()),
            }],
        );
        let url = RichMessage::blocks(
            "user",
            vec![ContentBlock::Image {
                source: ImageSource::Url {
                    url: "https://example.com/x.png".into(),
                },
                alt: None,
            }],
        );
        let rendered = render_cache_control_rich(&[base64, url]);

        let img0 = &rendered.messages[0]["content"][0];
        assert_eq!(img0["type"], "image");
        assert_eq!(img0["source"]["type"], "base64");
        assert_eq!(img0["source"]["media_type"], "image/png");
        assert_eq!(img0["source"]["data"], "AAAA");

        let img1 = &rendered.messages[1]["content"][0];
        assert_eq!(img1["source"]["type"], "url");
        assert_eq!(img1["source"]["url"], "https://example.com/x.png");
    }

    #[test]
    fn rich_cache_control_lands_on_last_block() {
        let mut m = RichMessage::blocks(
            "user",
            vec![
                ContentBlock::Text { text: "a".into() },
                ContentBlock::Text { text: "b".into() },
            ],
        );
        m.metadata.insert(
            meta_keys::CACHE_CONTROL.to_string(),
            "ephemeral".to_string(),
        );
        let rendered = render_cache_control_rich(&[m]);
        let content = &rendered.messages[0]["content"];
        assert!(content[0].get("cache_control").is_none());
        assert_eq!(content[1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn rich_splits_system_and_renders_reference_and_thinking_as_text() {
        let system = RichMessage::text("system", "be helpful");
        let assistant = RichMessage::blocks(
            "assistant",
            vec![
                ContentBlock::Thinking { text: "hmm".into() },
                ContentBlock::Reference {
                    kind: "file".into(),
                    id_or_path: "src/main.rs".into(),
                    metadata: Default::default(),
                },
            ],
        );
        let rendered = render_cache_control_rich(&[system, assistant]);
        assert_eq!(rendered.system.len(), 1);
        assert_eq!(rendered.system[0]["text"], "be helpful");
        // Thinking + reference both become text blocks.
        assert_eq!(rendered.messages[0]["content"][0]["type"], "text");
        assert_eq!(rendered.messages[0]["content"][0]["text"], "hmm");
        assert_eq!(rendered.messages[0]["content"][1]["type"], "text");
        assert_eq!(
            rendered.messages[0]["content"][1]["text"],
            "[file: src/main.rs]"
        );
    }

    #[test]
    fn rich_render_is_deterministic() {
        let m = RichMessage::blocks(
            "assistant",
            vec![ContentBlock::ToolUse {
                id: "c".into(),
                name: "t".into(),
                input: json!({ "k": "v" }),
            }],
        );
        let a = render_cache_control_rich(std::slice::from_ref(&m));
        let b = render_cache_control_rich(&[m]);
        assert_eq!(a, b);
    }

    #[test]
    fn per_model_cache_thresholds() {
        // Documented Anthropic minimum cacheable prompt lengths (2026-06).
        assert_eq!(min_cacheable_prefix_tokens("claude-haiku-4-5"), 4096);
        assert_eq!(min_cacheable_prefix_tokens("claude-opus-4-8"), 1024);
        assert_eq!(min_cacheable_prefix_tokens("claude-opus-4-7"), 2048);
        assert_eq!(min_cacheable_prefix_tokens("claude-opus-4-5"), 4096);
        assert_eq!(min_cacheable_prefix_tokens("claude-sonnet-4-6"), 1024);
        assert_eq!(min_cacheable_prefix_tokens("claude-fable-5"), 512);
        // Unknown model falls back to the most common current minimum.
        assert_eq!(min_cacheable_prefix_tokens("some-future-model"), 1024);
    }
}
