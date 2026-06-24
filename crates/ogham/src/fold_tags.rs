//! Structured, retrieval-friendly tags extracted from folded content.
//!
//! [`crate::recall::RecallIndex`] makes folded content searchable by free-text
//! relevance. Tags add the complementary *typed-field* axis: a host can ask
//! "which folds came from the `shell` tool?" or "which folds contain a panic?"
//! without a text query. Tags are extracted deterministically and offline
//! during compaction (zero runtime cost to hosts) from the fold's original
//! messages, and every field is sorted and deduplicated so the same input
//! always yields byte-identical tags.

use ogham_core::{ContentBlock, Message, MessageContent, RichMessage, meta_keys};

/// Maps each [`agent::ERROR_PATTERNS`](crate::agent::ERROR_PATTERNS) substring
/// to a normalized error-class name. Kept in lockstep with `agent::ERROR_PATTERNS`
/// (a unit test asserts parity) so tag extraction agrees with how the cascade
/// itself classifies errors.
const ERROR_CLASS_MAP: &[(&str, &str)] = &[
    ("Error:", "error"),
    ("error:", "error"),
    ("ERROR", "error"),
    ("Traceback (most recent call last)", "traceback"),
    ("panicked at", "panic"),
    ("Exception", "exception"),
    ("FAILED", "failed"),
    ("stderr:", "stderr"),
];

/// Error class recorded when a tool message is flagged `TOOL_STATUS == "error"`
/// but its content matches none of the textual patterns.
const TOOL_ERROR_CLASS: &str = "tool_error";

/// Bytes of each message scanned for file paths. Compaction runs on oversized
/// tool output, so path extraction is bounded to keep it from amplifying into a
/// memory/CPU cost on attacker-sized content.
const MAX_PATH_SCAN_BYTES: usize = 64 * 1024;
/// Hard cap on raw path candidates collected before dedup, so a pathological
/// input cannot grow the intermediate vector without bound.
const MAX_RAW_PATHS: usize = 4096;
/// Maximum tags kept per category after dedup. More than this in one fold is
/// low-signal for retrieval; the highest-sorted are kept deterministically.
const MAX_TAGS_PER_KIND: usize = 64;

/// File extensions treated as path-like even without a `/` separator, so a bare
/// `Cargo.toml` or `login.rs` is recognized while `e.g` or `1.2.3` is not.
const KNOWN_EXTENSIONS: &[&str] = &[
    "rs", "go", "ts", "tsx", "js", "jsx", "py", "rb", "java", "kt", "c", "h", "cc", "cpp", "hpp",
    "cs", "swift", "sh", "bash", "zsh", "sql", "html", "css", "scss", "toml", "yaml", "yml",
    "json", "md", "txt", "proto", "lock", "cfg", "ini", "env", "tf", "hcl",
];

/// A selectable category of structured fold tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldTagKind {
    /// Tool/command names (`metadata[TOOL_NAME]`).
    ToolName,
    /// Normalized error classes (e.g. `panic`, `traceback`, `error`).
    ErrorClass,
    /// File paths and path-like filenames mentioned in the content.
    FilePath,
}

/// Typed metadata extracted from a fold's original messages. Every field is
/// sorted and deduplicated for determinism.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FoldTags {
    /// Names of tools/commands whose output the fold contains.
    pub tool_names: Vec<String>,
    /// Normalized error classes detected in the fold's content.
    pub error_classes: Vec<String>,
    /// File paths and path-like filenames referenced in the fold's content.
    pub file_paths: Vec<String>,
}

impl FoldTags {
    /// Whether no tags were extracted.
    pub fn is_empty(&self) -> bool {
        self.tool_names.is_empty() && self.error_classes.is_empty() && self.file_paths.is_empty()
    }

    /// The values for one tag category.
    pub fn values(&self, kind: FoldTagKind) -> &[String] {
        match kind {
            FoldTagKind::ToolName => &self.tool_names,
            FoldTagKind::ErrorClass => &self.error_classes,
            FoldTagKind::FilePath => &self.file_paths,
        }
    }

    /// Whether this tag set contains `value` (case-insensitive) under `kind`.
    pub fn contains(&self, kind: FoldTagKind, value: &str) -> bool {
        self.values(kind)
            .iter()
            .any(|v| v.eq_ignore_ascii_case(value))
    }
}

/// Extract structured tags from a fold's original messages.
///
/// Deterministic: tool names come from `metadata[TOOL_NAME]`; error classes
/// from the same patterns the agent cascade uses to classify errors (over the
/// first 512 bytes of each message); file paths from a conservative path scan.
pub fn extract_fold_tags(originals: &[Message]) -> FoldTags {
    let mut tool_names = Vec::new();
    let mut error_classes = Vec::new();
    let mut file_paths = Vec::new();

    for msg in originals {
        if let Some(tool) = msg.metadata.get(meta_keys::TOOL_NAME)
            && !tool.is_empty()
        {
            tool_names.push(tool.clone());
        }
        collect_error_classes(msg, &mut error_classes);
        if file_paths.len() < MAX_RAW_PATHS {
            collect_file_paths(&msg.content, &mut file_paths);
        }
    }

    FoldTags {
        tool_names: sorted_unique_capped(tool_names),
        error_classes: sorted_unique_capped(error_classes),
        file_paths: sorted_unique_capped(file_paths),
    }
}

/// Extract structured tags from a [`RichMessage`], capturing rich-native signals
/// the lossy flat projection drops: tool-call names from `ToolUse` blocks and
/// the `is_error` flag on `ToolResult` blocks. Text-derived tags (error
/// patterns, file paths, `metadata[TOOL_NAME]`) still come from the flattened
/// text, so a block-compressed fold stays queryable by tool/error during
/// failure recovery.
pub fn extract_fold_tags_rich(message: &RichMessage) -> FoldTags {
    let mut tags = extract_fold_tags(std::slice::from_ref(&message.to_flat_lossy()));

    if let MessageContent::Blocks(blocks) = &message.content {
        let mut tools = Vec::new();
        let mut errors = Vec::new();
        collect_block_tags(blocks, &mut tools, &mut errors);
        if !tools.is_empty() {
            tools.append(&mut tags.tool_names);
            tags.tool_names = sorted_unique_capped(tools);
        }
        if !errors.is_empty() {
            errors.append(&mut tags.error_classes);
            tags.error_classes = sorted_unique_capped(errors);
        }
    }
    tags
}

/// Walk rich content blocks collecting tool names (`ToolUse.name`) and the
/// `tool_error` class for any errored `ToolResult`, recursing into nested
/// tool-result content.
fn collect_block_tags(blocks: &[ContentBlock], tools: &mut Vec<String>, errors: &mut Vec<String>) {
    for block in blocks {
        match block {
            ContentBlock::ToolUse { name, .. } if !name.is_empty() => tools.push(name.clone()),
            ContentBlock::ToolResult {
                is_error, content, ..
            } => {
                if *is_error {
                    errors.push(TOOL_ERROR_CLASS.to_string());
                }
                collect_block_tags(content, tools, errors);
            }
            _ => {}
        }
    }
}

fn collect_error_classes(msg: &Message, out: &mut Vec<String>) {
    if msg.metadata.get(meta_keys::TOOL_STATUS).map(String::as_str) == Some("error") {
        out.push(TOOL_ERROR_CLASS.to_string());
    }
    let window = prefix_to_char_boundary(&msg.content, 512);
    for (pattern, class) in ERROR_CLASS_MAP {
        if window.contains(pattern) {
            out.push((*class).to_string());
        }
    }
}

fn collect_file_paths(content: &str, out: &mut Vec<String>) {
    let window = prefix_to_char_boundary(content, MAX_PATH_SCAN_BYTES);
    for raw in window.split(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                '"' | '\''
                    | '`'
                    | '('
                    | ')'
                    | ','
                    | ';'
                    | ':'
                    | '['
                    | ']'
                    | '{'
                    | '}'
                    | '='
                    | '<'
                    | '>'
                    | '|'
                    | '*'
            )
    }) {
        if out.len() >= MAX_RAW_PATHS {
            break;
        }
        let token = raw
            .trim_matches(|c: char| !(c.is_alphanumeric() || matches!(c, '/' | '.' | '_' | '-')));
        if let Some(path) = looks_like_path(token) {
            out.push(path);
        }
    }
}

/// Recognize a path-like token: `stem.ext` where `ext` is short and alphabetic,
/// and either the token contains a `/` separator or `ext` is a known code
/// extension. Conservative on purpose — rejects `e.g`, `1.2.3`, `U.S.A`.
fn looks_like_path(token: &str) -> Option<String> {
    let (stem, ext) = token.rsplit_once('.')?;
    if stem.is_empty() || ext.is_empty() || ext.len() > 12 {
        return None;
    }
    if !ext.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    if !stem.chars().any(|c| c.is_alphanumeric()) {
        return None;
    }
    let ext_lower = ext.to_ascii_lowercase();
    if token.contains('/') || KNOWN_EXTENSIONS.contains(&ext_lower.as_str()) {
        Some(token.to_string())
    } else {
        None
    }
}

fn sorted_unique_capped(mut values: Vec<String>) -> Vec<String> {
    values.sort();
    values.dedup();
    values.truncate(MAX_TAGS_PER_KIND);
    values
}

/// Truncate to at most `max` bytes without splitting a UTF-8 char.
fn prefix_to_char_boundary(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::ERROR_PATTERNS;
    use ogham_core::ImageSource;

    #[test]
    fn rich_extractor_captures_native_tool_and_error_tags() {
        // A rich message whose text mentions nothing error-y, but whose blocks
        // carry a tool call and an errored tool result. The flat projection
        // would miss both; the rich extractor must not.
        let message = RichMessage::blocks(
            "assistant",
            vec![
                ContentBlock::Text {
                    text: "running the build".into(),
                },
                ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "shell".into(),
                    input: serde_json::json!({"cmd": "make"}),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "c1".into(),
                    is_error: true,
                    content: vec![ContentBlock::Image {
                        source: ImageSource::Url { url: "x".into() },
                        alt: None,
                    }],
                },
            ],
        );
        let tags = extract_fold_tags_rich(&message);
        assert!(
            tags.tool_names.contains(&"shell".to_string()),
            "ToolUse.name captured: {:?}",
            tags.tool_names
        );
        assert!(
            tags.error_classes.contains(&"tool_error".to_string()),
            "errored ToolResult captured: {:?}",
            tags.error_classes
        );
    }

    fn tool_msg(name: &str, content: &str) -> Message {
        let mut m = Message::new("tool", content);
        m.metadata
            .insert(meta_keys::TOOL_NAME.to_string(), name.to_string());
        m
    }

    #[test]
    fn error_class_map_matches_agent_patterns() {
        // Parity guard: every classifier pattern must have a tag mapping, so the
        // two never drift apart.
        let mapped: std::collections::BTreeSet<&str> =
            ERROR_CLASS_MAP.iter().map(|(p, _)| *p).collect();
        let patterns: std::collections::BTreeSet<&str> = ERROR_PATTERNS.iter().copied().collect();
        assert_eq!(mapped, patterns);
    }

    #[test]
    fn extracts_tool_names_deduped_and_sorted() {
        let tags = extract_fold_tags(&[
            tool_msg("shell", "ran ls"),
            tool_msg("editor", "opened a file"),
            tool_msg("shell", "ran cat"),
        ]);
        assert_eq!(tags.tool_names, vec!["editor", "shell"]);
    }

    #[test]
    fn extracts_error_classes_from_patterns() {
        let tags = extract_fold_tags(&[Message::new(
            "tool",
            "Traceback (most recent call last)\n  File x\npanicked at line 3",
        )]);
        assert!(tags.error_classes.contains(&"traceback".to_string()));
        assert!(tags.error_classes.contains(&"panic".to_string()));
        // Sorted + deduped.
        assert_eq!(
            tags.error_classes,
            sorted_unique_capped(tags.error_classes.clone())
        );
    }

    #[test]
    fn file_path_extraction_is_bounded() {
        // Pathological input: many distinct path-like tokens. Tags must stay
        // capped so they cannot amplify into fold records / recall.
        let mut content = String::with_capacity(2_000_000);
        for i in 0..200_000 {
            content.push_str(&format!("src/m{i}/file{i}.rs "));
        }
        let tags = extract_fold_tags(&[Message::new("tool", content)]);
        assert!(
            tags.file_paths.len() <= MAX_TAGS_PER_KIND,
            "file_paths capped at {MAX_TAGS_PER_KIND}, got {}",
            tags.file_paths.len()
        );
    }

    #[test]
    fn tool_status_error_tags_tool_error() {
        let mut m = Message::new("tool", "no textual pattern here, just data");
        m.metadata
            .insert(meta_keys::TOOL_STATUS.to_string(), "error".to_string());
        let tags = extract_fold_tags(&[m]);
        assert_eq!(tags.error_classes, vec!["tool_error"]);
    }

    #[test]
    fn extracts_file_paths_and_rejects_non_paths() {
        let tags = extract_fold_tags(&[Message::new(
            "tool",
            "edited src/auth/login.rs and Cargo.toml; version 1.2.3, e.g. the U.S.A",
        )]);
        assert!(tags.file_paths.contains(&"src/auth/login.rs".to_string()));
        assert!(tags.file_paths.contains(&"Cargo.toml".to_string()));
        // Non-paths must not leak in.
        assert!(!tags.file_paths.iter().any(|p| p.contains("1.2.3")));
        assert!(!tags.file_paths.iter().any(|p| p == "e.g"));
        assert!(!tags.file_paths.iter().any(|p| p.contains("U.S.A")));
    }

    #[test]
    fn values_and_contains_are_case_insensitive() {
        let tags = extract_fold_tags(&[tool_msg("Shell", "x")]);
        assert_eq!(tags.values(FoldTagKind::ToolName), &["Shell".to_string()]);
        assert!(tags.contains(FoldTagKind::ToolName, "shell"));
        assert!(!tags.contains(FoldTagKind::ToolName, "editor"));
    }

    #[test]
    fn empty_when_nothing_to_tag() {
        let tags = extract_fold_tags(&[Message::new("assistant", "just a normal reply")]);
        assert!(tags.is_empty());
    }
}
