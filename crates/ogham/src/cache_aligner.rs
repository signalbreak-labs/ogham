//! CacheAligner: stabilise message prefixes so KV-cache slots match
//! across turns.
//!
//! LLM inference engines (vLLM, TGI, TensorRT-LLM) reuse KV-cache blocks
//! keyed by the *exact* prefix of the prompt.  If two prompts differ by
//! trailing whitespace, JSON key order, or stray blank lines the cache is
//! wasted.  CacheAligner normalises content so stable prefixes hit the
//! same cache slots.

use ogham_core::Message;

/// Normalise a piece of text for KV-cache alignment.
///
/// 1. Collapse runs of whitespace to a single space.  
/// 2. Trim leading/trailing whitespace.  
/// 3. Sort JSON object keys (if the text parses as JSON).  
///
/// This is intentionally *lossy* for whitespace only — it never mutates
/// semantic tokens.
pub fn align_text(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    // Try JSON normalisation first.
    if let Some(normalised) = align_json_keys(trimmed) {
        return normalised;
    }

    // Plain text: collapse whitespace.
    collapse_whitespace(trimmed)
}

/// Apply [`align_text`] to every message in the conversation.
pub fn align_messages(messages: &mut [Message]) {
    for msg in messages.iter_mut() {
        msg.content = align_text(&msg.content);
    }
}

fn collapse_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_ws = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !prev_ws {
                out.push(' ');
                prev_ws = true;
            }
        } else {
            out.push(ch);
            prev_ws = false;
        }
    }
    out.trim().to_string()
}

fn align_json_keys(text: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let aligned = sort_json_value(value);
    serde_json::to_string(&aligned).ok()
}

fn sort_json_value(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<_> = map.into_iter().collect();
            entries.sort_by(|(a, _), (b, _)| a.cmp(b));
            let sorted = entries
                .into_iter()
                .map(|(k, v)| (k, sort_json_value(v)))
                .collect();
            serde_json::Value::Object(sorted)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(sort_json_value).collect())
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collapse_whitespace() {
        assert_eq!(collapse_whitespace("hello   world"), "hello world");
        assert_eq!(collapse_whitespace("  a\n\n  b\t  c  "), "a b c");
    }

    #[test]
    fn test_align_json_sorts_keys() {
        let input = r#"{"b":1,"a":2}"#;
        assert_eq!(align_json_keys(input).unwrap(), r#"{"a":2,"b":1}"#);
    }

    #[test]
    fn test_align_json_recursive() {
        let input = r#"{"z":{"b":1,"a":2},"a":3}"#;
        assert_eq!(
            align_json_keys(input).unwrap(),
            r#"{"a":3,"z":{"a":2,"b":1}}"#
        );
    }

    #[test]
    fn test_align_messages() {
        let mut msgs = vec![
            Message::new("user", "  hello\n\nworld  "),
            Message::new("assistant", r#"{"b":1,"a":2}"#),
        ];
        align_messages(&mut msgs);
        assert_eq!(msgs[0].content, "hello world");
        assert_eq!(msgs[1].content, r#"{"a":2,"b":1}"#);
    }

    #[test]
    fn test_align_empty() {
        assert_eq!(align_text(""), "");
        assert_eq!(align_text("   "), "");
    }

    #[test]
    fn test_plain_text_unchanged_semantically() {
        let text = "The quick brown fox jumps over the lazy dog.";
        assert_eq!(align_text(text), text);
    }
}
