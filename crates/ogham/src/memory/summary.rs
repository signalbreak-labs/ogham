use ogham_core::Message;
use regex::Regex;
use std::sync::LazyLock;

/// Structured summary of a conversation span. All fields are plain data;
/// render_markdown() produces the prompt-ready form.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct StructuredSummary {
    /// One-line intent, best-effort (first user message's first sentence).
    pub session_intent: String,
    /// Unique file paths mentioned, sorted lexicographically.
    pub files_touched: Vec<String>,
    /// Lines that look like decisions, in order of appearance, deduplicated.
    pub key_decisions: Vec<String>,
    /// Error lines preserved verbatim (first line of each error block), in order.
    pub unresolved_errors: Vec<String>,
    /// Lines that look like TODO/next-step statements.
    pub next_steps: Vec<String>,
    /// Number of source turns this summary covers.
    pub turns_covered: usize,
}

impl StructuredSummary {
    /// Render with EXACTLY these section headers (omit empty sections):
    ///   "## Session intent", "## Files touched", "## Decisions",
    ///   "## Unresolved errors", "## Next steps"
    /// List items are "- " bullets. Ends without trailing newline.
    pub fn render_markdown(&self) -> String {
        let mut sections = Vec::new();

        if !self.session_intent.is_empty() {
            sections.push(format!("## Session intent\n- {}", self.session_intent));
        }
        if !self.files_touched.is_empty() {
            let mut s = "## Files touched".to_string();
            for f in &self.files_touched {
                s.push_str(&format!("\n- {}", f));
            }
            sections.push(s);
        }
        if !self.key_decisions.is_empty() {
            let mut s = "## Decisions".to_string();
            for d in &self.key_decisions {
                s.push_str(&format!("\n- {}", d));
            }
            sections.push(s);
        }
        if !self.unresolved_errors.is_empty() {
            let mut s = "## Unresolved errors".to_string();
            for e in &self.unresolved_errors {
                s.push_str(&format!("\n- {}", e));
            }
            sections.push(s);
        }
        if !self.next_steps.is_empty() {
            let mut s = "## Next steps".to_string();
            for n in &self.next_steps {
                s.push_str(&format!("\n- {}", n));
            }
            sections.push(s);
        }

        sections.join("\n\n")
    }

    /// Merge `newer` into `self`: union files (re-sort), append new decisions /
    /// errors / next_steps that are not exact duplicates, add turns_covered,
    /// keep self.session_intent unless empty.
    pub fn merge(&mut self, newer: &StructuredSummary) {
        let mut changed = false;
        if self.session_intent.is_empty() && !newer.session_intent.is_empty() {
            self.session_intent.clone_from(&newer.session_intent);
            changed = true;
        }
        for f in &newer.files_touched {
            if !self.files_touched.contains(f) {
                self.files_touched.push(f.clone());
                changed = true;
            }
        }
        if changed {
            self.files_touched.sort();
            self.files_touched.truncate(50);
        }
        for d in &newer.key_decisions {
            if !self.key_decisions.contains(d) {
                self.key_decisions.push(d.clone());
                changed = true;
            }
        }
        if changed {
            self.key_decisions.truncate(20);
        }
        for e in &newer.unresolved_errors {
            if !self.unresolved_errors.contains(e) {
                self.unresolved_errors.push(e.clone());
                changed = true;
            }
        }
        if changed {
            self.unresolved_errors.truncate(20);
        }
        for n in &newer.next_steps {
            if !self.next_steps.contains(n) {
                self.next_steps.push(n.clone());
                changed = true;
            }
        }
        if changed {
            self.next_steps.truncate(20);
        }
        if changed {
            self.turns_covered += newer.turns_covered;
        }
    }
}

static FILE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"[A-Za-z0-9_./\-]+\.(rs|py|ts|tsx|js|jsx|go|java|kt|rb|c|h|cpp|hpp|md|toml|yaml|yml|json|sql|sh)\b",
    )
    .unwrap()
});

fn truncate_to_char_boundary(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Deterministic, zero-LLM summarizer.
pub struct ExtractiveSummarizer;

impl ExtractiveSummarizer {
    pub fn summarize_sync(
        &self,
        turns: &[Message],
        existing: Option<&StructuredSummary>,
    ) -> StructuredSummary {
        let mut summary = existing.cloned().unwrap_or_default();

        // 1. session_intent
        if summary.session_intent.is_empty() {
            for turn in turns {
                if turn.role == "user" {
                    let sentence = turn
                        .content
                        .split(['.', '!', '?'])
                        .next()
                        .unwrap_or(&turn.content)
                        .trim();
                    summary.session_intent = truncate_to_char_boundary(sentence, 200).to_string();
                    break;
                }
            }
        }

        // 2. files_touched
        let mut files = std::collections::BTreeSet::new();
        for turn in turns {
            for m in FILE_RE.find_iter(&turn.content) {
                files.insert(m.as_str().to_string());
            }
        }
        for f in files {
            if !summary.files_touched.contains(&f) {
                summary.files_touched.push(f);
            }
        }
        summary.files_touched.sort();
        summary.files_touched.truncate(50);

        for turn in turns {
            for line in turn.content.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let lower = trimmed.to_lowercase();

                // 3. key_decisions
                if lower.starts_with("decision:")
                    || lower.starts_with("decided")
                    || lower.starts_with("we will ")
                    || lower.starts_with("we chose ")
                    || lower.starts_with("chose ")
                    || lower.starts_with("agreed ")
                {
                    if !summary.key_decisions.iter().any(|d| d.as_str() == trimmed) {
                        summary.key_decisions.push(trimmed.to_string());
                    }
                    continue;
                }

                // 4. unresolved_errors (same pattern list as agent::classify)
                if crate::agent::ERROR_PATTERNS
                    .iter()
                    .any(|p| trimmed.contains(p))
                {
                    let truncated = truncate_to_char_boundary(trimmed, 300).to_string();
                    if !summary.unresolved_errors.contains(&truncated) {
                        summary.unresolved_errors.push(truncated);
                    }
                    continue;
                }

                // 5. next_steps
                if (lower.starts_with("todo")
                    || lower.starts_with("next:")
                    || lower.starts_with("next step")
                    || trimmed.starts_with("- [ ]"))
                    && !summary.next_steps.iter().any(|n| n.as_str() == trimmed)
                {
                    summary.next_steps.push(trimmed.to_string());
                }
            }
        }

        summary.key_decisions.truncate(20);
        summary.unresolved_errors.truncate(20);
        summary.next_steps.truncate(20);
        summary.turns_covered += turns.len();

        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ogham_core::Message;

    #[test]
    fn extracts_files_sorted_deduped() {
        let turns = vec![Message::new(
            "user",
            "Looking at src/b.rs and src/a.rs and src/a.rs again.",
        )];
        let s = ExtractiveSummarizer.summarize_sync(&turns, None);
        assert_eq!(s.files_touched, vec!["src/a.rs", "src/b.rs"]);
    }

    #[test]
    fn extracts_errors_verbatim() {
        let turns = vec![Message::new("tool", "Error: connection refused")];
        let s = ExtractiveSummarizer.summarize_sync(&turns, None);
        assert!(
            s.unresolved_errors
                .contains(&"Error: connection refused".to_string())
        );
    }

    #[test]
    fn render_markdown_sections() {
        let s = StructuredSummary {
            session_intent: "fix bug".to_string(),
            files_touched: vec!["a.rs".to_string()],
            key_decisions: vec!["use sqlite".to_string()],
            unresolved_errors: vec!["Error: x".to_string()],
            next_steps: vec!["todo y".to_string()],
            turns_covered: 1,
        };
        let md = s.render_markdown();
        assert!(md.contains("## Session intent"));
        assert!(md.contains("## Files touched"));
        assert!(md.contains("## Decisions"));
        assert!(md.contains("## Unresolved errors"));
        assert!(md.contains("## Next steps"));
        assert!(!md.contains("\n\n\n"));
    }

    #[test]
    fn merge_is_idempotent() {
        let a = StructuredSummary {
            session_intent: "intent".to_string(),
            files_touched: vec!["a.rs".to_string()],
            key_decisions: vec!["d1".to_string()],
            unresolved_errors: vec!["e1".to_string()],
            next_steps: vec!["n1".to_string()],
            turns_covered: 2,
        };
        let mut once = a.clone();
        once.merge(&a);
        let mut twice = once.clone();
        twice.merge(&a);
        assert_eq!(once, twice);
    }

    #[test]
    fn summarizer_deterministic() {
        let turns = vec![
            Message::new("user", "We will use Rust."),
            Message::new("assistant", "src/main.rs looks good."),
        ];
        let s1 = ExtractiveSummarizer.summarize_sync(&turns, None);
        let s2 = ExtractiveSummarizer.summarize_sync(&turns, None);
        assert_eq!(s1, s2);
    }
}
