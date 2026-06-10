use async_trait::async_trait;
use ogham_core::{Compressed, CompressionContext, Compressor, Content, Result};
use tracing::debug;

use crate::ccr::{CcrStore, compute_key};

/// Code compressor that removes comments, collapses blank lines, and
/// preserves structural lines (imports, signatures, braces).
pub struct AstCodeCompressor {
    ccr_store: Option<std::sync::Arc<dyn CcrStore>>,
}

impl AstCodeCompressor {
    pub fn new() -> Self {
        Self { ccr_store: None }
    }

    pub fn with_ccr_store(ccr_store: std::sync::Arc<dyn CcrStore>) -> Self {
        Self {
            ccr_store: Some(ccr_store),
        }
    }

    pub fn compress_code(&self, content: &str, _lang: &str) -> String {
        let lines: Vec<&str> = content.split('\n').collect();
        let mut out: Vec<String> = Vec::with_capacity(lines.len());
        let mut blank_run = 0usize;
        let mut in_block_comment = false;

        for line in &lines {
            let trimmed = line.trim();

            // Block comment handling (C-style /* */)
            if in_block_comment {
                if let Some(end) = trimmed.find("*/") {
                    in_block_comment = false;
                    let after = &trimmed[end + 2..];
                    if !after.trim().is_empty() {
                        out.push(after.trim_start().to_string());
                    }
                }
                continue;
            }
            if let Some(start) = trimmed.find("/*") {
                let before = &trimmed[..start];
                if !before.trim().is_empty() {
                    out.push(before.trim_end().to_string());
                }
                let after = &trimmed[start + 2..];
                if let Some(end) = after.find("*/") {
                    let between = &after[end + 2..];
                    if !between.trim().is_empty() {
                        out.push(between.trim_start().to_string());
                    }
                } else {
                    in_block_comment = true;
                }
                continue;
            }

            // Skip line comments
            let code_part = if let Some(idx) = line.find("//") {
                &line[..idx]
            } else if let Some(idx) = line.find('#') {
                // Python/Ruby/YAML style — only if # is first non-whitespace
                let leading_ws = line.len() - line.trim_start().len();
                if idx == leading_ws {
                    &line[..idx]
                } else {
                    *line
                }
            } else {
                *line
            };

            let code_trimmed = code_part.trim();
            if code_trimmed.is_empty() {
                blank_run += 1;
                if blank_run <= 1 {
                    out.push(String::new());
                }
                continue;
            }
            blank_run = 0;

            // Preserve structural lines at full length
            let is_structural = is_structural_line(code_trimmed);
            if is_structural {
                out.push(code_part.trim_end().to_string());
            } else {
                // Truncate long implementation lines
                let max_len = 120;
                if code_part.len() > max_len {
                    out.push(format!("{}...", &code_part[..max_len]));
                } else {
                    out.push(code_part.trim_end().to_string());
                }
            }
        }

        // Collapse trailing blanks
        while out.last().map(|s| s.is_empty()).unwrap_or(false) {
            out.pop();
        }

        out.join("\n")
    }
}

fn is_structural_line(line: &str) -> bool {
    let keywords = [
        "import ",
        "from ",
        "use ",
        "mod ",
        "pub ",
        "fn ",
        "def ",
        "class ",
        "struct ",
        "enum ",
        "impl ",
        "trait ",
        "interface ",
        "package ",
        "func ",
        "function ",
        "const ",
        "let ",
        "var ",
        "type ",
        "return",
        "yield",
        "async ",
        "await",
        "throw",
        "raise",
        "if ",
        "else",
        "for ",
        "while ",
        "match ",
        "switch ",
        "case ",
        "try",
        "catch",
        "except",
        "finally",
        "@",
    ];
    keywords.iter().any(|kw| line.starts_with(kw))
        || line.ends_with('{')
        || line.ends_with('}')
        || line.ends_with(':')
        || line.ends_with(';')
}

impl Default for AstCodeCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Compressor for AstCodeCompressor {
    fn name(&self) -> &'static str {
        "ast_code"
    }

    async fn compress(&self, content: &Content, ctx: &CompressionContext) -> Result<Compressed> {
        debug!("AstCodeCompressor compressing {} bytes", content.data.len());
        let text = String::from_utf8_lossy(&content.data);
        let lang = content.mime_or_lang.as_str();
        let compressed = self.compress_code(&text, lang);
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
    fn removes_comments() {
        let comp = AstCodeCompressor::new();
        let input = "fn main() {\n    // a comment\n    let x = 1;\n}\n";
        let out = comp.compress_code(input, "rust");
        assert!(!out.contains("comment"));
        assert!(out.contains("fn main()"));
        assert!(out.contains("let x = 1"));
    }

    #[test]
    fn collapses_blank_lines() {
        let comp = AstCodeCompressor::new();
        let input = "a\n\n\n\nb\n";
        let out = comp.compress_code(input, "rust");
        let lines: Vec<&str> = out.split('\n').collect();
        assert!(lines.iter().filter(|l| l.is_empty()).count() <= 1);
    }

    #[test]
    fn preserves_imports_and_signatures() {
        let comp = AstCodeCompressor::new();
        let input = "use std::sync::Arc;\n\nfn process() -> u32 { 42 }\n";
        let out = comp.compress_code(input, "rust");
        assert!(out.contains("use std::sync::Arc"));
        assert!(out.contains("fn process() -> u32"));
    }
}
