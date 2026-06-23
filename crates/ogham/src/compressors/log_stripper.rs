use async_trait::async_trait;
use ogham_core::{Compressed, CompressionContext, Compressor, Content, Result};
use std::collections::{BTreeMap, BTreeSet};
use tracing::debug;

use crate::adaptive_sizer::compute_optimal_k;
use crate::ccr::{CcrStore, compute_key};

// ─── Types ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogFormat {
    Pytest,
    Npm,
    Cargo,
    Jest,
    Make,
    Generic,
}

impl LogFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogFormat::Pytest => "pytest",
            LogFormat::Npm => "npm",
            LogFormat::Cargo => "cargo",
            LogFormat::Jest => "jest",
            LogFormat::Make => "make",
            LogFormat::Generic => "generic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogLevel {
    Error,
    Fail,
    Warn,
    Info,
    Debug,
    Trace,
    Unknown,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Error => "error",
            LogLevel::Fail => "fail",
            LogLevel::Warn => "warn",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
            LogLevel::Trace => "trace",
            LogLevel::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogLine {
    pub line_number: usize,
    pub content: String,
    pub level: LogLevel,
    pub is_stack_trace: bool,
    pub is_summary: bool,
    pub score: f32,
}

impl PartialEq for LogLine {
    fn eq(&self, other: &Self) -> bool {
        self.line_number == other.line_number
    }
}
impl Eq for LogLine {}
impl std::hash::Hash for LogLine {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.line_number.hash(state);
    }
}
impl PartialOrd for LogLine {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for LogLine {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.line_number.cmp(&other.line_number)
    }
}

impl LogLine {
    pub fn new(line_number: usize, content: impl Into<String>) -> Self {
        Self {
            line_number,
            content: content.into(),
            level: LogLevel::Unknown,
            is_stack_trace: false,
            is_summary: false,
            score: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogCompressorConfig {
    pub max_errors: usize,
    pub error_context_lines: usize,
    pub keep_first_error: bool,
    pub keep_last_error: bool,
    pub max_stack_traces: usize,
    pub stack_trace_max_lines: usize,
    pub max_warnings: usize,
    pub dedupe_warnings: bool,
    pub keep_summary_lines: bool,
    pub max_total_lines: usize,
    pub enable_ccr: bool,
    pub min_lines_for_ccr: usize,
    pub min_compression_ratio_for_ccr: f64,
}

impl Default for LogCompressorConfig {
    fn default() -> Self {
        Self {
            max_errors: 10,
            error_context_lines: 3,
            keep_first_error: true,
            keep_last_error: true,
            max_stack_traces: 3,
            stack_trace_max_lines: 20,
            max_warnings: 5,
            dedupe_warnings: true,
            keep_summary_lines: true,
            max_total_lines: 100,
            enable_ccr: true,
            min_lines_for_ccr: 50,
            min_compression_ratio_for_ccr: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogCompressionResult {
    pub compressed: String,
    pub original: String,
    pub original_line_count: usize,
    pub compressed_line_count: usize,
    pub format_detected: LogFormat,
    pub compression_ratio: f64,
    pub cache_key: Option<String>,
    pub stats: BTreeMap<String, u64>,
}

// ─── Format detector ─────────────────────────────────────────────────────

fn detect_format(lines: &[&str]) -> LogFormat {
    let sample: Vec<&str> = lines.iter().take(100).copied().collect();
    let mut best: Option<(LogFormat, usize)> = None;
    let patterns: &[(LogFormat, &[&str])] = &[
        (
            LogFormat::Pytest,
            &[
                "=== FAILURES",
                "=== ERRORS",
                "=== test session",
                "=== short test summary",
                "PASSED [",
                "FAILED [",
                "ERROR [",
                "SKIPPED [",
                "collected ",
            ],
        ),
        (
            LogFormat::Npm,
            &["npm ERR!", "npm WARN", "npm info", "npm http"],
        ),
        (
            LogFormat::Cargo,
            &[
                "Compiling ",
                "Finished ",
                "Running ",
                "warning: ",
                "error[E",
            ],
        ),
        (LogFormat::Jest, &["PASS ", "FAIL ", "Test Suites:"]),
        (
            LogFormat::Make,
            &["make[", "make:", "gcc ", "g++ ", "clang "],
        ),
    ];
    for (fmt, patts) in patterns {
        let score = sample
            .iter()
            .filter(|line| patts.iter().any(|p| line.contains(p)))
            .count();
        if score > 0 && best.map(|(_, s)| score > s).unwrap_or(true) {
            best = Some((*fmt, score));
        }
    }
    best.map(|(f, _)| f).unwrap_or(LogFormat::Generic)
}

// ─── Level classifier ────────────────────────────────────────────────────

fn classify_level(line: &str) -> LogLevel {
    let check = |pat: &str| -> bool {
        line.to_ascii_uppercase()
            .contains(&pat.to_ascii_uppercase())
    };
    if check("ERROR") || check("FATAL") || check("CRITICAL") {
        return LogLevel::Error;
    }
    if check("FAIL") || check("FAILED") {
        return LogLevel::Fail;
    }
    if check("WARN") || check("WARNING") {
        return LogLevel::Warn;
    }
    if check("INFO") {
        return LogLevel::Info;
    }
    if check("DEBUG") {
        return LogLevel::Debug;
    }
    if check("TRACE") {
        return LogLevel::Trace;
    }
    LogLevel::Unknown
}

// ─── Stack-trace detector ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TraceFlavor {
    PythonTraceback,
    Js,
    Java,
    RustError,
    Go,
}

fn flavor_for(line: &str) -> Option<TraceFlavor> {
    let trimmed = line.trim_start();
    if trimmed.starts_with("Traceback (most recent call last)") || is_python_file_frame(trimmed) {
        Some(TraceFlavor::PythonTraceback)
    } else if trimmed.starts_with("at ") && trimmed.contains('(') && trimmed.contains(')') {
        Some(TraceFlavor::Js)
    } else if trimmed.starts_with("at ") && trimmed.contains('.') && trimmed.contains('(') {
        Some(TraceFlavor::Java)
    } else if trimmed.starts_with("--> ") {
        Some(TraceFlavor::RustError)
    } else if is_go_frame(line) {
        Some(TraceFlavor::Go)
    } else {
        None
    }
}

fn is_python_file_frame(s: &str) -> bool {
    s.starts_with("File \"") && s.contains("\", line ")
}

fn is_go_frame(s: &str) -> bool {
    let trimmed = s.trim_start();
    let mut chars = trimmed.chars().peekable();
    let mut saw_digit = false;
    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            saw_digit = true;
            chars.next();
        } else {
            break;
        }
    }
    if !saw_digit || chars.next() != Some(':') {
        return false;
    }
    while chars.peek() == Some(&' ') {
        chars.next();
    }
    let rest: String = chars.collect();
    rest.starts_with("0x")
        && rest[2..]
            .chars()
            .take_while(|c| c.is_ascii_hexdigit())
            .count()
            > 0
}

fn terminates(flavor: TraceFlavor, line: &str) -> bool {
    let trimmed = line.trim_start();
    match flavor {
        TraceFlavor::PythonTraceback => {
            let is_indented_or_blank = line.starts_with([' ', '\t']) || line.is_empty();
            let is_continuation = trimmed.starts_with("Traceback")
                || trimmed.starts_with("File ")
                || trimmed.starts_with("During handling")
                || trimmed.starts_with("The above exception");
            if is_indented_or_blank || is_continuation {
                false
            } else {
                !trimmed.starts_with(char::is_uppercase)
            }
        }
        TraceFlavor::Js | TraceFlavor::Java => !trimmed.starts_with("at ") && !line.is_empty(),
        TraceFlavor::RustError => !trimmed.starts_with("--> ") && !line.is_empty(),
        TraceFlavor::Go => {
            !trimmed.chars().next().is_some_and(|c| c.is_ascii_digit()) && !line.is_empty()
        }
    }
}

// ─── Summary detector ────────────────────────────────────────────────────

fn is_summary_line(line: &str) -> bool {
    if line.starts_with("===") || line.starts_with("---") {
        return true;
    }
    let bytes = line.as_bytes();
    let leading_digits = bytes.iter().take_while(|b| b.is_ascii_digit()).count();
    if leading_digits > 0 && line[leading_digits..].starts_with(' ') {
        let rest = &line[leading_digits + 1..];
        for kw in &["passed", "failed", "skipped", "error", "warning"] {
            if rest.starts_with(kw) {
                return true;
            }
        }
    }
    for prefix in &[
        "Test ", "Tests ", "Tests:", "Test:", "Suite ", "Suites ", "Suites:", "Suite:",
    ] {
        if let Some(rest) = line.strip_prefix(prefix) {
            return rest
                .chars()
                .find(|c| !c.is_whitespace())
                .is_some_and(|c| c.is_ascii_digit());
        }
    }
    for prefix in &["TOTAL", "Total", "Summary"] {
        if line.starts_with(prefix) {
            return true;
        }
    }
    for prefix in &["Build", "Compile", "Test"] {
        if line.starts_with(prefix) {
            for outcome in &["succeeded", "failed", "complete"] {
                if line.contains(outcome) {
                    return true;
                }
            }
        }
    }
    false
}

// ─── Compressor ──────────────────────────────────────────────────────────

pub struct LogCompressor {
    config: LogCompressorConfig,
}

impl LogCompressor {
    pub fn new(config: LogCompressorConfig) -> Self {
        Self { config }
    }

    pub fn compress(&self, content: &str, bias: f64) -> LogCompressionResult {
        let lines: Vec<&str> = content.split('\n').collect();
        let original_line_count = lines.len();
        if original_line_count < self.config.min_lines_for_ccr {
            return LogCompressionResult {
                compressed: content.to_string(),
                original: content.to_string(),
                original_line_count,
                compressed_line_count: original_line_count,
                format_detected: LogFormat::Generic,
                compression_ratio: 1.0,
                cache_key: None,
                stats: BTreeMap::new(),
            };
        }
        let format = detect_format(&lines);
        let log_lines = self.parse_lines(&lines);
        let selected = self.select_lines(&log_lines, bias);
        let (compressed_body, output_stats) = self.format_output(&selected, &log_lines);
        let mut compressed = compressed_body;
        let ratio = compressed.len() as f64 / content.len().max(1) as f64;
        let mut cache_key = None;
        if self.config.enable_ccr && ratio < self.config.min_compression_ratio_for_ccr {
            let key = compute_key(content.as_bytes());
            let marker = format!(
                "\n[{} lines compressed to {}. Retrieve more: hash={}]",
                original_line_count,
                selected.len(),
                key
            );
            compressed.push_str(&marker);
            cache_key = Some(key);
        }
        LogCompressionResult {
            compressed,
            original: content.to_string(),
            original_line_count,
            compressed_line_count: selected.len(),
            format_detected: format,
            compression_ratio: ratio,
            cache_key,
            stats: output_stats,
        }
    }

    fn parse_lines(&self, lines: &[&str]) -> Vec<LogLine> {
        let mut out: Vec<LogLine> = Vec::with_capacity(lines.len());
        let mut active: Option<TraceFlavor> = None;
        let mut trace_lines = 0usize;
        for (i, line) in lines.iter().enumerate() {
            let mut entry = LogLine::new(i, *line);
            entry.level = classify_level(line);
            entry.is_summary = is_summary_line(line);
            if let Some(flavor) = active {
                if trace_lines >= self.config.stack_trace_max_lines || terminates(flavor, line) {
                    active = None;
                    trace_lines = 0;
                    if let Some(new_flavor) = flavor_for(line) {
                        active = Some(new_flavor);
                        trace_lines = 1;
                        entry.is_stack_trace = true;
                    }
                } else {
                    entry.is_stack_trace = true;
                    trace_lines += 1;
                }
            } else if let Some(flavor) = flavor_for(line) {
                active = Some(flavor);
                trace_lines = 1;
                entry.is_stack_trace = true;
            }
            entry.score = score_log_line(&entry);
            out.push(entry);
        }
        out
    }

    fn select_lines(&self, log_lines: &[LogLine], bias: f64) -> Vec<LogLine> {
        let all_strings: Vec<&str> = log_lines.iter().map(|l| l.content.as_str()).collect();
        let adaptive_max =
            compute_optimal_k(&all_strings, bias, 10, Some(self.config.max_total_lines));
        let mut errors: Vec<LogLine> = Vec::new();
        let mut fails: Vec<LogLine> = Vec::new();
        let mut warnings: Vec<LogLine> = Vec::new();
        let mut summaries: Vec<LogLine> = Vec::new();
        let mut stack_traces: Vec<Vec<LogLine>> = Vec::new();
        let mut current_stack: Vec<LogLine> = Vec::new();
        for line in log_lines {
            match line.level {
                LogLevel::Error => errors.push(line.clone()),
                LogLevel::Fail => fails.push(line.clone()),
                LogLevel::Warn => warnings.push(line.clone()),
                _ => {}
            }
            if line.is_stack_trace {
                current_stack.push(line.clone());
            } else if !current_stack.is_empty() {
                stack_traces.push(std::mem::take(&mut current_stack));
            }
            if line.is_summary {
                summaries.push(line.clone());
            }
        }
        if !current_stack.is_empty() {
            stack_traces.push(current_stack);
        }
        let mut selected: BTreeSet<LogLine> = BTreeSet::new();
        for line in self.select_with_first_last(&errors, self.config.max_errors) {
            selected.insert(line);
        }
        for line in self.select_with_first_last(&fails, self.config.max_errors) {
            selected.insert(line);
        }
        let warnings = if self.config.dedupe_warnings {
            self.dedupe_similar(warnings)
        } else {
            warnings
        };
        for line in warnings.into_iter().take(self.config.max_warnings) {
            selected.insert(line);
        }
        for stack in stack_traces.iter().take(self.config.max_stack_traces) {
            for line in stack.iter().take(self.config.stack_trace_max_lines) {
                selected.insert(line.clone());
            }
        }
        if self.config.keep_summary_lines {
            for line in summaries {
                selected.insert(line);
            }
        }
        let selected_indices: BTreeSet<usize> = selected.iter().map(|l| l.line_number).collect();
        let mut context_indices: BTreeSet<usize> = BTreeSet::new();
        for &idx in &selected_indices {
            let lo = idx.saturating_sub(self.config.error_context_lines);
            let hi = (idx + self.config.error_context_lines + 1).min(log_lines.len());
            for i in lo..hi {
                if i != idx {
                    context_indices.insert(i);
                }
            }
        }
        for idx in context_indices {
            if !selected_indices.contains(&idx) && idx < log_lines.len() {
                selected.insert(log_lines[idx].clone());
            }
        }
        let mut ordered: Vec<LogLine> = selected.into_iter().collect();
        if ordered.len() > adaptive_max {
            ordered.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.line_number.cmp(&b.line_number))
            });
            ordered.truncate(adaptive_max);
            ordered.sort_by_key(|l| l.line_number);
        }
        ordered
    }

    fn select_with_first_last(&self, lines: &[LogLine], max_count: usize) -> Vec<LogLine> {
        if lines.len() <= max_count {
            return lines.to_vec();
        }
        let mut out: Vec<LogLine> = Vec::with_capacity(max_count);
        let mut seen: BTreeSet<usize> = BTreeSet::new();
        let push = |line: LogLine, out: &mut Vec<LogLine>, seen: &mut BTreeSet<usize>| {
            if seen.insert(line.line_number) {
                out.push(line);
            }
        };
        if self.config.keep_first_error {
            push(lines[0].clone(), &mut out, &mut seen);
        }
        if self.config.keep_last_error {
            push(lines.last().unwrap().clone(), &mut out, &mut seen);
        }
        let remaining = max_count.saturating_sub(out.len());
        if remaining > 0 {
            let mut by_score = lines.to_vec();
            by_score.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.line_number.cmp(&b.line_number))
            });
            for line in by_score.into_iter() {
                if !seen.contains(&line.line_number) {
                    push(line, &mut out, &mut seen);
                    if out.len() >= max_count {
                        break;
                    }
                }
            }
        }
        out
    }

    fn dedupe_similar(&self, lines: Vec<LogLine>) -> Vec<LogLine> {
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut out: Vec<LogLine> = Vec::with_capacity(lines.len());
        for line in lines {
            let key = normalize_for_dedupe(&line.content);
            if seen.insert(key) {
                out.push(line);
            }
        }
        out
    }

    fn format_output(
        &self,
        selected: &[LogLine],
        all_lines: &[LogLine],
    ) -> (String, BTreeMap<String, u64>) {
        let mut stats: BTreeMap<String, u64> = BTreeMap::new();
        stats.insert("errors".into(), count_level(all_lines, LogLevel::Error));
        stats.insert("fails".into(), count_level(all_lines, LogLevel::Fail));
        stats.insert("warnings".into(), count_level(all_lines, LogLevel::Warn));
        stats.insert("info".into(), count_level(all_lines, LogLevel::Info));
        stats.insert("total".into(), all_lines.len() as u64);
        stats.insert("selected".into(), selected.len() as u64);
        let mut output: Vec<String> = selected.iter().map(|l| l.content.clone()).collect();
        let omitted = all_lines.len().saturating_sub(selected.len());
        if omitted > 0 {
            let mut summary_parts: Vec<String> = Vec::new();
            for (label, key) in [
                ("ERROR", "errors"),
                ("FAIL", "fails"),
                ("WARN", "warnings"),
                ("INFO", "info"),
            ] {
                let n = stats.get(key).copied().unwrap_or(0);
                if n > 0 {
                    summary_parts.push(format!("{} {}", n, label));
                }
            }
            if !summary_parts.is_empty() {
                output.push(format!(
                    "[{} lines omitted: {}]",
                    omitted,
                    summary_parts.join(", ")
                ));
            }
        }
        (output.join("\n"), stats)
    }
}

impl Default for LogCompressor {
    fn default() -> Self {
        Self::new(LogCompressorConfig::default())
    }
}

fn count_level(lines: &[LogLine], level: LogLevel) -> u64 {
    lines.iter().filter(|l| l.level == level).count() as u64
}

fn score_log_line(line: &LogLine) -> f32 {
    let level_score: f32 = match line.level {
        LogLevel::Error | LogLevel::Fail => 1.0,
        LogLevel::Warn => 0.5,
        LogLevel::Info | LogLevel::Unknown => 0.1,
        LogLevel::Debug => 0.05,
        LogLevel::Trace => 0.02,
    };
    let stack_boost: f32 = if line.is_stack_trace { 0.3 } else { 0.0 };
    let summary_boost: f32 = if line.is_summary { 0.4 } else { 0.0 };
    (level_score + stack_boost + summary_boost).min(1.0_f32)
}

fn normalize_for_dedupe(content: &str) -> String {
    let split_at = content.find([':', '=']).unwrap_or(content.len());
    let prefix = &content[..split_at];
    let suffix = &content[split_at..];
    let suffix_norm = suffix
        .replace(|c: char| c.is_ascii_digit(), "N")
        .replace(|c: char| c.is_ascii_hexdigit(), "H")
        .replace("/", "/P/");
    format!("{}{}", prefix, suffix_norm)
}

// ─── Compressor trait wrapper ────────────────────────────────────────────

pub struct LogStripper {
    compressor: LogCompressor,
    ccr_store: Option<std::sync::Arc<dyn CcrStore>>,
}

impl LogStripper {
    pub fn new() -> Self {
        Self {
            compressor: LogCompressor::new(LogCompressorConfig::default()),
            ccr_store: None,
        }
    }

    pub fn with_ccr_store(ccr_store: std::sync::Arc<dyn CcrStore>) -> Self {
        Self {
            compressor: LogCompressor::new(LogCompressorConfig::default()),
            ccr_store: Some(ccr_store),
        }
    }
}

impl Default for LogStripper {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Compressor for LogStripper {
    fn name(&self) -> &'static str {
        "log_stripper"
    }

    async fn compress(&self, content: &Content, ctx: &CompressionContext) -> Result<Compressed> {
        debug!("LogStripper compressing {} bytes", content.data.len());
        let text = String::from_utf8_lossy(&content.data);
        let bias = if ctx.max_tokens.map(|t| t < 1000).unwrap_or(false) {
            0.7
        } else {
            1.0
        };
        let result = self.compressor.compress(&text, bias);
        let compressed_tokens = result.compressed.len() / 4;
        let id = compute_key(content.data.as_ref());
        if ctx.reversible
            && let Some(store) = &self.ccr_store
        {
            store.save(&id, &text, None).await?;
        }
        Ok(Compressed {
            id,
            data: bytes::Bytes::from(result.compressed),
            original_tokens: result.original.len() / 4,
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
    fn detects_pytest_format() {
        let lines = [
            "============================= test session starts =============================",
            "collected 15 items",
            "tests/test_foo.py::test_basic PASSED [  6%]",
            "FAILED tests/test_foo.py::test_edge",
        ];
        assert_eq!(detect_format(&lines), LogFormat::Pytest);
    }

    #[test]
    fn compresses_log_output() {
        let cmp = LogCompressor::new(LogCompressorConfig {
            max_total_lines: 5,
            min_lines_for_ccr: 5,
            min_compression_ratio_for_ccr: 0.95,
            ..Default::default()
        });
        let mut content = String::new();
        for i in 0..50 {
            content.push_str(&format!("INFO line {}\n", i));
        }
        content.push_str("ERROR boom\n");
        let result = cmp.compress(&content, 1.0);
        assert!(result.compressed.contains("ERROR boom"));
        assert!(result.compressed_line_count <= 5);
    }

    #[test]
    fn short_log_passthrough() {
        let cmp = LogCompressor::new(LogCompressorConfig::default());
        let result = cmp.compress("a\nb\nc", 1.0);
        assert_eq!(result.compressed, "a\nb\nc");
        assert_eq!(result.compression_ratio, 1.0);
    }
}
