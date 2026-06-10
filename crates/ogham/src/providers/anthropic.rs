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
}
