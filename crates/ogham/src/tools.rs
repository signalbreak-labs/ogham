//! Agent-facing tool definitions.
//!
//! Compressed content carries `<<ccr:HASH>>` markers. For the model to
//! dereference them, the host must expose a retrieval tool. This module
//! ships the tool definition (JSON Schema) and a dispatcher so every host
//! wires it identically.
//!
//! ```no_run
//! # async fn demo(ccr: std::sync::Arc<dyn ogham::ccr::CcrStore>) {
//! // 1. advertise the tool to the model
//! let tools = vec![ogham::tools::retrieve_tool_definition()];
//! // 2. when the model calls it, dispatch:
//! let args = serde_json::json!({ "id": "86a33abc..." });
//! let result_text = ogham::tools::handle_retrieve_call(&args, ccr.as_ref()).await;
//! # }
//! ```
//!
//! The dispatcher is **fail-closed for the agent loop**: it always returns
//! a model-readable string (found content, a not-found notice, or an error
//! notice) and never an `Err`, so a storage hiccup can't crash a turn.

use crate::ccr::CcrStore;
use serde_json::{Value, json};

/// Name the model uses to call the retrieval tool.
pub const RETRIEVE_TOOL_NAME: &str = "ogham_retrieve";

/// Provider-agnostic definition of the retrieval tool.
///
/// The `input_schema` field is standard JSON Schema. Anthropic accepts this
/// shape directly; for OpenAI, move `input_schema` to `parameters` inside a
/// `function` object.
pub fn retrieve_tool_definition() -> Value {
    json!({
        "name": RETRIEVE_TOOL_NAME,
        "description": "Retrieve the full original content behind a <<ccr:HASH>> marker. \
            Compressed or cleared tool results in this conversation reference their \
            original content by hash; call this when the summary or stub is not enough.",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The hash from a <<ccr:HASH>> marker, without the marker syntax."
                }
            },
            "required": ["id"]
        }
    })
}

/// Execute a retrieval tool call. Always returns model-readable text.
pub async fn handle_retrieve_call(args: &Value, ccr: &dyn CcrStore) -> String {
    let Some(id) = args.get("id").and_then(Value::as_str) else {
        return "ogham_retrieve error: missing required string argument 'id' \
                (the hash from a <<ccr:HASH>> marker)."
            .to_string();
    };
    // Tolerate the model passing the full marker instead of the bare hash.
    let id = id
        .trim()
        .trim_start_matches("<<ccr:")
        .trim_end_matches(">>");

    match ccr.retrieve(id).await {
        Ok(Some(original)) => original,
        Ok(None) => format!(
            "ogham_retrieve: no content stored for id '{id}'. It may have \
             expired; proceed with the information already in context."
        ),
        Err(e) => format!(
            "ogham_retrieve: storage error while fetching '{id}' ({e}); \
             proceed with the information already in context."
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccr::in_memory::InMemoryCcrStore;

    #[test]
    fn definition_shape() {
        let def = retrieve_tool_definition();
        assert_eq!(def["name"], RETRIEVE_TOOL_NAME);
        assert_eq!(def["input_schema"]["type"], "object");
        assert_eq!(def["input_schema"]["required"][0], "id");
    }

    #[tokio::test]
    async fn retrieve_roundtrip() {
        let ccr = InMemoryCcrStore::new();
        let hash = crate::ccr::compute_key(b"the original");
        ccr.save(&hash, "the original", None).await.unwrap();
        let out = handle_retrieve_call(&serde_json::json!({ "id": hash }), &ccr).await;
        assert_eq!(out, "the original");
    }

    #[tokio::test]
    async fn retrieve_accepts_full_marker() {
        let ccr = InMemoryCcrStore::new();
        let hash = crate::ccr::compute_key(b"x");
        ccr.save(&hash, "x", None).await.unwrap();
        let marker = crate::ccr::marker_for(&hash);
        let out = handle_retrieve_call(&serde_json::json!({ "id": marker }), &ccr).await;
        assert_eq!(out, "x");
    }

    #[tokio::test]
    async fn retrieve_miss_is_model_readable() {
        let ccr = InMemoryCcrStore::new();
        let out = handle_retrieve_call(&serde_json::json!({ "id": "nope" }), &ccr).await;
        assert!(out.contains("no content stored"));
    }

    #[tokio::test]
    async fn retrieve_bad_args_is_model_readable() {
        let ccr = InMemoryCcrStore::new();
        let out = handle_retrieve_call(&serde_json::json!({}), &ccr).await;
        assert!(out.contains("missing required"));
    }
}
