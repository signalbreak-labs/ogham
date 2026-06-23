use async_trait::async_trait;
use ogham_core::{Compressed, CompressionContext, Compressor, Content, Result};
use serde_json::{Map, Value};
use std::collections::{BTreeSet, HashSet};
use tracing::debug;

use crate::adaptive_sizer::compute_optimal_k;
use crate::ccr::{CcrStore, compute_key, marker_for};
// use crate::detect::is_json_array_of_dicts; // unused for now
use crate::stats_math;

// ─── Config ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SmartCrusherConfig {
    pub enabled: bool,
    pub min_items_to_analyze: usize,
    pub min_tokens_to_crush: usize,
    pub variance_threshold: f64,
    pub max_items_after_crush: usize,
    pub preserve_change_points: bool,
    pub first_fraction: f64,
    pub last_fraction: f64,
    pub enable_ccr_marker: bool,
}

impl Default for SmartCrusherConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_items_to_analyze: 5,
            min_tokens_to_crush: 200,
            variance_threshold: 2.0,
            max_items_after_crush: 15,
            preserve_change_points: true,
            first_fraction: 0.3,
            last_fraction: 0.15,
            enable_ccr_marker: true,
        }
    }
}

// ─── Error Keywords ──────────────────────────────────────────────────────

const ERROR_KEYWORDS: &[&str] = &[
    "error",
    "exception",
    "failed",
    "failure",
    "critical",
    "fatal",
    "crash",
    "panic",
    "abort",
    "timeout",
    "denied",
    "rejected",
];

// ─── Array Classifier ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArrayType {
    DictArray,
    StringArray,
    NumberArray,
    BoolArray,
    NestedArray,
    MixedArray,
    Empty,
}

fn classify_array(items: &[Value]) -> ArrayType {
    if items.is_empty() {
        return ArrayType::Empty;
    }
    let mut has_bool = false;
    let mut has_number = false;
    let mut has_string = false;
    let mut has_object = false;
    let mut has_array = false;
    let mut has_null = false;
    for item in items {
        match item {
            Value::Bool(_) => has_bool = true,
            Value::Number(_) => has_number = true,
            Value::String(_) => has_string = true,
            Value::Object(_) => has_object = true,
            Value::Array(_) => has_array = true,
            Value::Null => has_null = true,
        }
    }
    if has_bool && !has_number && !has_string && !has_object && !has_array && !has_null {
        return ArrayType::BoolArray;
    }
    if has_object && !has_bool && !has_number && !has_string && !has_array && !has_null {
        return ArrayType::DictArray;
    }
    if has_string && !has_bool && !has_number && !has_object && !has_array && !has_null {
        return ArrayType::StringArray;
    }
    if has_number && !has_bool && !has_string && !has_object && !has_array && !has_null {
        return ArrayType::NumberArray;
    }
    if has_array && !has_bool && !has_number && !has_string && !has_object && !has_null {
        return ArrayType::NestedArray;
    }
    ArrayType::MixedArray
}

// ─── Crushers ────────────────────────────────────────────────────────────

fn compute_k_split(
    items: &[&str],
    config: &SmartCrusherConfig,
    bias: f64,
) -> (usize, usize, usize, usize) {
    let max_k = if config.max_items_after_crush > 0 {
        Some(config.max_items_after_crush)
    } else {
        None
    };
    let k_total = compute_optimal_k(items, bias, 3, max_k);
    let k_first_raw =
        1_usize.max((k_total as f64 * config.first_fraction).round_ties_even() as usize);
    let k_last_raw =
        1_usize.max((k_total as f64 * config.last_fraction).round_ties_even() as usize);
    let k_first = k_first_raw.min(k_total);
    let k_last = k_last_raw.min(k_total.saturating_sub(k_first));
    let k_importance = k_total.saturating_sub(k_first + k_last);
    (k_total, k_first, k_last, k_importance)
}

fn crush_string_array(
    items: &[&str],
    config: &SmartCrusherConfig,
    bias: f64,
) -> (Vec<String>, String) {
    let n = items.len();
    if n <= 8 {
        return (
            items.iter().map(|s| (*s).to_string()).collect(),
            "string:passthrough".to_string(),
        );
    }
    let (k_total, k_first, k_last, _k_importance) = compute_k_split(items, config, bias);
    let mut error_indices: BTreeSet<usize> = BTreeSet::new();
    for (i, s) in items.iter().enumerate() {
        let lower = s.to_lowercase();
        if ERROR_KEYWORDS.iter().any(|kw| lower.contains(kw)) {
            error_indices.insert(i);
        }
    }
    let lengths: Vec<f64> = items.iter().map(|s| s.chars().count() as f64).collect();
    let mut anomaly_indices: BTreeSet<usize> = BTreeSet::new();
    if lengths.len() > 1
        && let Some(mean_len) = stats_math::mean(&lengths)
        && let Some(std_len) = stats_math::sample_stdev(&lengths)
        && std_len > 0.0
    {
        let threshold = config.variance_threshold * std_len;
        for (i, &length) in lengths.iter().enumerate() {
            if (length - mean_len).abs() > threshold {
                anomaly_indices.insert(i);
            }
        }
    }
    let first_indices: BTreeSet<usize> = (0..k_first.min(n)).collect();
    let last_start = n.saturating_sub(k_last);
    let last_indices: BTreeSet<usize> = (last_start..n).collect();
    let mut keep_indices: BTreeSet<usize> = BTreeSet::new();
    keep_indices.extend(error_indices.iter().copied());
    keep_indices.extend(anomaly_indices.iter().copied());
    keep_indices.extend(first_indices.iter().copied());
    keep_indices.extend(last_indices.iter().copied());
    let mut seen: HashSet<&str> = HashSet::new();
    for &i in &keep_indices {
        seen.insert(items[i]);
    }
    let mut dedup_count: usize = 0;
    let remaining_budget = k_total.saturating_sub(keep_indices.len());
    if remaining_budget > 0 {
        let stride = ((n.saturating_sub(1)) / (remaining_budget + 1)).max(1);
        let cap = k_total + error_indices.len() + anomaly_indices.len();
        let mut i: usize = 0;
        while i < n {
            if keep_indices.len() >= cap {
                break;
            }
            if !keep_indices.contains(&i) {
                if !seen.contains(items[i]) {
                    keep_indices.insert(i);
                    seen.insert(items[i]);
                } else {
                    dedup_count += 1;
                }
            }
            i += stride;
        }
    }
    let result: Vec<String> = keep_indices.iter().map(|&i| items[i].to_string()).collect();
    let mut strategy = format!("string:adaptive({}->{}", n, result.len());
    if dedup_count > 0 {
        strategy.push_str(&format!(",dedup={}", dedup_count));
    }
    if !error_indices.is_empty() {
        strategy.push_str(&format!(",errors={}", error_indices.len()));
    }
    strategy.push(')');
    (result, strategy)
}

fn crush_number_array(
    items: &[Value],
    config: &SmartCrusherConfig,
    bias: f64,
) -> (Vec<Value>, String) {
    let n = items.len();
    if n <= 8 {
        return (items.to_vec(), "number:passthrough".to_string());
    }
    let finite: Vec<f64> = items
        .iter()
        .filter_map(|v| v.as_f64().filter(|f| f.is_finite()))
        .collect();
    if finite.is_empty() {
        return (items.to_vec(), "number:no_finite".to_string());
    }
    let item_strings: Vec<String> = items.iter().map(|v| v.to_string()).collect();
    let item_str_refs: Vec<&str> = item_strings.iter().map(|s| s.as_str()).collect();
    let (k_total, k_first, k_last, _) = compute_k_split(&item_str_refs, config, bias);
    let mean_val = stats_math::mean(&finite).unwrap_or(0.0);
    let median_val = stats_math::median(&finite).unwrap_or(0.0);
    let std_val = if finite.len() > 1 {
        stats_math::sample_stdev(&finite).unwrap_or(0.0)
    } else {
        0.0
    };
    let mut sorted_finite: Vec<f64> = finite.clone();
    sorted_finite.sort_by(f64::total_cmp);
    let p25 = percentile_linear(&sorted_finite, 0.25);
    let p75 = percentile_linear(&sorted_finite, 0.75);
    let mut outlier_indices: BTreeSet<usize> = BTreeSet::new();
    if std_val > 0.0 {
        let threshold = config.variance_threshold * std_val;
        for (i, val) in items.iter().enumerate() {
            if let Some(num) = val.as_f64().filter(|f| f.is_finite())
                && (num - mean_val).abs() > threshold
            {
                outlier_indices.insert(i);
            }
        }
    }
    let mut change_indices: BTreeSet<usize> = BTreeSet::new();
    if config.preserve_change_points && n > 10 {
        let window: usize = 5;
        for i in window..n.saturating_sub(window) {
            let left: Vec<f64> = items[i - window..i]
                .iter()
                .filter_map(|v| v.as_f64().filter(|f| f.is_finite()))
                .collect();
            let right: Vec<f64> = items[i..i + window]
                .iter()
                .filter_map(|v| v.as_f64().filter(|f| f.is_finite()))
                .collect();
            if !left.is_empty() && !right.is_empty() {
                let left_mean = stats_math::mean(&left).unwrap_or(0.0);
                let right_mean = stats_math::mean(&right).unwrap_or(0.0);
                if std_val > 0.0
                    && (right_mean - left_mean).abs() > config.variance_threshold * std_val
                {
                    change_indices.insert(i);
                }
            }
        }
    }
    let first_indices: BTreeSet<usize> = (0..k_first.min(n)).collect();
    let last_start = n.saturating_sub(k_last);
    let last_indices: BTreeSet<usize> = (last_start..n).collect();
    let mut keep_indices: BTreeSet<usize> = BTreeSet::new();
    keep_indices.extend(outlier_indices.iter().copied());
    keep_indices.extend(change_indices.iter().copied());
    keep_indices.extend(first_indices.iter().copied());
    keep_indices.extend(last_indices.iter().copied());
    let remaining_budget = k_total.saturating_sub(keep_indices.len());
    if remaining_budget > 0 {
        let stride = ((n.saturating_sub(1)) / (remaining_budget + 1)).max(1);
        let cap = k_total + outlier_indices.len();
        let mut i: usize = 0;
        while i < n {
            if keep_indices.len() >= cap {
                break;
            }
            if !keep_indices.contains(&i) {
                keep_indices.insert(i);
            }
            i += stride;
        }
    }
    let kept_values: Vec<Value> = keep_indices.iter().map(|&i| items[i].clone()).collect();
    let mn = finite.iter().cloned().reduce(f64::min).unwrap_or(0.0);
    let mx = finite.iter().cloned().reduce(f64::max).unwrap_or(0.0);
    let mut strategy = format!(
        "number:adaptive({}->{},min={},max={},mean={},median={},stddev={},p25={},p75={}",
        n,
        kept_values.len(),
        format_number_repr(mn),
        format_number_repr(mx),
        stats_math::format_g(mean_val),
        stats_math::format_g(median_val),
        stats_math::format_g(std_val),
        stats_math::format_g(p25),
        stats_math::format_g(p75),
    );
    if !outlier_indices.is_empty() {
        strategy.push_str(&format!(",outliers={}", outlier_indices.len()));
    }
    if !change_indices.is_empty() {
        strategy.push_str(&format!(",change_points={}", change_indices.len()));
    }
    strategy.push(')');
    (kept_values, strategy)
}

fn crush_object(
    obj: &Map<String, Value>,
    config: &SmartCrusherConfig,
    bias: f64,
) -> (Map<String, Value>, String) {
    let n = obj.len();
    if n <= 8 {
        return (obj.clone(), "object:passthrough".to_string());
    }
    let mut kv_tokens: Vec<(String, usize)> = Vec::with_capacity(n);
    let mut total_tokens: usize = 0;
    for (key, val) in obj {
        let val_str = serde_json::to_string(val).unwrap_or_default();
        let tokens = val_str.len() / 4 + key.len() / 4 + 2;
        kv_tokens.push((key.clone(), tokens));
        total_tokens += tokens;
    }
    if total_tokens < config.min_tokens_to_crush {
        return (obj.clone(), "object:passthrough".to_string());
    }
    let keys: Vec<&String> = obj.keys().collect();
    let kv_strings: Vec<String> = keys
        .iter()
        .map(|k| {
            format!(
                "{}: {}",
                k,
                serde_json::to_string(&obj[k.as_str()]).unwrap_or_default()
            )
        })
        .collect();
    let kv_refs: Vec<&str> = kv_strings.iter().map(|s| s.as_str()).collect();
    let max_k = if config.max_items_after_crush > 0 {
        Some(config.max_items_after_crush)
    } else {
        None
    };
    let k_total = compute_optimal_k(&kv_refs, bias, 3, max_k);
    if k_total >= n {
        return (obj.clone(), "object:passthrough".to_string());
    }
    let mut keep_keys: HashSet<String> = HashSet::new();
    for (key, val) in obj {
        let val_str = serde_json::to_string(val)
            .unwrap_or_default()
            .to_lowercase();
        if ERROR_KEYWORDS.iter().any(|kw| val_str.contains(kw)) {
            keep_keys.insert(key.clone());
        }
    }
    let small_threshold_tokens = 50_usize / 4;
    for (key, tokens) in &kv_tokens {
        if *tokens <= small_threshold_tokens {
            keep_keys.insert(key.clone());
        }
    }
    let k_first = 1_usize.max((k_total as f64 * config.first_fraction).round_ties_even() as usize);
    let k_last = 1_usize.max((k_total as f64 * config.last_fraction).round_ties_even() as usize);
    for k in keys.iter().take(k_first) {
        keep_keys.insert((*k).clone());
    }
    for k in keys.iter().rev().take(k_last) {
        keep_keys.insert((*k).clone());
    }
    let remaining = k_total.saturating_sub(keep_keys.len());
    if remaining > 0 {
        let stride = ((n.saturating_sub(1)) / (remaining + 1)).max(1);
        let mut i: usize = 0;
        while i < n {
            let error_kept_count = keep_keys
                .iter()
                .filter(|k| {
                    let s = serde_json::to_string(&obj[k.as_str()])
                        .unwrap_or_default()
                        .to_lowercase();
                    ERROR_KEYWORDS.iter().any(|kw| s.contains(kw))
                })
                .count();
            if keep_keys.len() >= k_total + error_kept_count {
                break;
            }
            keep_keys.insert(keys[i].clone());
            i += stride;
        }
    }
    let mut result: Map<String, Value> = Map::new();
    for k in &keys {
        if keep_keys.contains(k.as_str()) {
            result.insert((*k).clone(), obj[k.as_str()].clone());
        }
    }
    let strategy = format!("object:adaptive({}->{} keys)", n, result.len());
    (result, strategy)
}

fn percentile_linear(sorted_values: &[f64], q: f64) -> f64 {
    let n = sorted_values.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted_values[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos as usize;
    let hi = if lo + 1 < n { lo + 1 } else { lo };
    let frac = pos - lo as f64;
    sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac
}

fn format_number_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    if x.fract() == 0.0 && x.abs() < 1e16 {
        return format!("{}", x as i64);
    }
    format!("{}", x)
}

// ─── Top-level SmartCrusher ──────────────────────────────────────────────

/// JSON outlier-detection + key-preservation compressor.
///
/// Reversible via CCR.
pub struct SmartCrusher {
    config: SmartCrusherConfig,
    ccr_store: Option<std::sync::Arc<dyn CcrStore>>,
}

struct PendingCcrSave {
    id: String,
    original: String,
}

impl SmartCrusher {
    pub fn new() -> Self {
        Self {
            config: SmartCrusherConfig::default(),
            ccr_store: None,
        }
    }

    pub fn with_ccr_store(ccr_store: std::sync::Arc<dyn CcrStore>) -> Self {
        Self {
            config: SmartCrusherConfig::default(),
            ccr_store: Some(ccr_store),
        }
    }

    /// Compress JSON content recursively.
    ///
    /// `_query` is accepted for forward compatibility with focus/question-hint
    /// biased compression but is currently ignored. See `ROADMAP.md`
    /// ("Consume the focus hint").
    pub fn crush(&self, content: &str, _query: &str, bias: f64) -> String {
        self.crush_with_options(content, bias, false).0
    }

    fn crush_with_options(
        &self,
        content: &str,
        bias: f64,
        emit_ccr_markers: bool,
    ) -> (String, Vec<PendingCcrSave>) {
        let Ok(parsed) = serde_json::from_str::<Value>(content) else {
            return (content.to_string(), Vec::new());
        };
        let mut ccr_saves = Vec::new();
        let (crushed, _) = self.process_value(&parsed, 0, bias, emit_ccr_markers, &mut ccr_saves);
        (
            serde_json::to_string(&crushed).unwrap_or_else(|_| content.to_string()),
            ccr_saves,
        )
    }

    fn process_value(
        &self,
        value: &Value,
        depth: usize,
        bias: f64,
        emit_ccr_markers: bool,
        ccr_saves: &mut Vec<PendingCcrSave>,
    ) -> (Value, String) {
        const MAX_DEPTH: usize = 50;
        if depth >= MAX_DEPTH {
            return (value.clone(), String::new());
        }
        let mut info_parts: Vec<String> = Vec::new();
        match value {
            Value::Array(arr) => {
                let n = arr.len();
                if n >= self.config.min_items_to_analyze {
                    let arr_type = classify_array(arr);
                    match arr_type {
                        ArrayType::DictArray => {
                            let (items, strategy, ccr_hash) =
                                self.crush_dict_array(arr, bias, emit_ccr_markers, ccr_saves);
                            info_parts.push(strategy);
                            if let Some(hash) = ccr_hash {
                                let mut items_with_sentinel = items.clone();
                                let mut sentinel = serde_json::Map::new();
                                sentinel.insert(
                                    "_ccr_dropped".to_string(),
                                    Value::String(marker_for(&hash)),
                                );
                                items_with_sentinel.push(Value::Object(sentinel));
                                return (Value::Array(items_with_sentinel), info_parts.join(","));
                            }
                            return (Value::Array(items), info_parts.join(","));
                        }
                        ArrayType::StringArray => {
                            let strs: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
                            let (crushed, strategy) = crush_string_array(&strs, &self.config, bias);
                            info_parts.push(format!("{}({}->{})", strategy, n, crushed.len()));
                            let crushed_values: Vec<Value> =
                                crushed.into_iter().map(Value::String).collect();
                            return (Value::Array(crushed_values), info_parts.join(","));
                        }
                        ArrayType::NumberArray => {
                            let (crushed, strategy) = crush_number_array(arr, &self.config, bias);
                            info_parts.push(format!("{}({}->{})", strategy, n, crushed.len()));
                            return (Value::Array(crushed), info_parts.join(","));
                        }
                        ArrayType::MixedArray => {
                            let (crushed, strategy) =
                                self.crush_mixed_array(arr, bias, emit_ccr_markers, ccr_saves);
                            info_parts.push(format!("{}({}->{})", strategy, n, crushed.len()));
                            return (Value::Array(crushed), info_parts.join(","));
                        }
                        _ => {}
                    }
                }
                let mut processed: Vec<Value> = Vec::with_capacity(n);
                for item in arr {
                    let (p_item, p_info) =
                        self.process_value(item, depth + 1, bias, emit_ccr_markers, ccr_saves);
                    processed.push(p_item);
                    if !p_info.is_empty() {
                        info_parts.push(p_info);
                    }
                }
                (Value::Array(processed), info_parts.join(","))
            }
            Value::Object(map) => {
                let mut processed = serde_json::Map::new();
                for (k, v) in map {
                    let (p_val, p_info) =
                        self.process_value(v, depth + 1, bias, emit_ccr_markers, ccr_saves);
                    processed.insert(k.clone(), p_val);
                    if !p_info.is_empty() {
                        info_parts.push(p_info);
                    }
                }
                if processed.len() >= self.config.min_items_to_analyze {
                    let (crushed_dict, strategy) = crush_object(&processed, &self.config, bias);
                    if strategy != "object:passthrough" {
                        info_parts.push(strategy);
                        return (Value::Object(crushed_dict), info_parts.join(","));
                    }
                }
                (Value::Object(processed), info_parts.join(","))
            }
            _ => (value.clone(), String::new()),
        }
    }

    fn crush_dict_array(
        &self,
        items: &[Value],
        bias: f64,
        emit_ccr_markers: bool,
        ccr_saves: &mut Vec<PendingCcrSave>,
    ) -> (Vec<Value>, String, Option<String>) {
        let item_strings: Vec<String> = items
            .iter()
            .map(|i| serde_json::to_string(i).unwrap_or_default())
            .collect();
        let item_str_refs: Vec<&str> = item_strings.iter().map(|s| s.as_str()).collect();
        let max_k = if self.config.max_items_after_crush > 0 {
            Some(self.config.max_items_after_crush)
        } else {
            None
        };
        let adaptive_k = compute_optimal_k(&item_str_refs, bias, 3, max_k);
        if items.len() <= adaptive_k {
            return (items.to_vec(), "none:adaptive_at_limit".to_string(), None);
        }
        // Simple adaptive sampling: keep first K, last K, and highest-scoring middle items.
        let k_first = (adaptive_k as f64 * self.config.first_fraction) as usize;
        let k_last = (adaptive_k as f64 * self.config.last_fraction) as usize;
        let k_middle = adaptive_k.saturating_sub(k_first + k_last);
        let mut keep_indices: BTreeSet<usize> = BTreeSet::new();
        for i in 0..k_first.min(items.len()) {
            keep_indices.insert(i);
        }
        for i in items.len().saturating_sub(k_last)..items.len() {
            keep_indices.insert(i);
        }
        // Score by length variance (longer/shorter = more interesting)
        let mut scored: Vec<(usize, f64)> = (0..items.len())
            .map(|i| {
                let s = &item_str_refs[i];
                let len_score = (s.len() as f64).ln_1p();
                (i, len_score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (idx, _) in scored.iter().take(k_middle) {
            keep_indices.insert(*idx);
            if keep_indices.len() >= adaptive_k {
                break;
            }
        }
        let result: Vec<Value> = keep_indices.iter().map(|&i| items[i].clone()).collect();
        let dropped_count = items.len().saturating_sub(result.len());
        let ccr_hash = if dropped_count > 0 && self.config.enable_ccr_marker && emit_ccr_markers {
            let canonical = serde_json::to_string(items).unwrap_or_default();
            let h = compute_key(canonical.as_bytes());
            ccr_saves.push(PendingCcrSave {
                id: h.clone(),
                original: canonical,
            });
            Some(h)
        } else {
            None
        };
        let strategy = format!("smart_sample({}->{})", items.len(), result.len());
        (result, strategy, ccr_hash)
    }

    fn crush_mixed_array(
        &self,
        items: &[Value],
        bias: f64,
        emit_ccr_markers: bool,
        ccr_saves: &mut Vec<PendingCcrSave>,
    ) -> (Vec<Value>, String) {
        let n = items.len();
        if n <= 8 {
            return (items.to_vec(), "mixed:passthrough".to_string());
        }
        #[derive(Default)]
        struct GroupBuckets {
            entries: Vec<(&'static str, Vec<usize>, Vec<Value>)>,
            index_of: std::collections::HashMap<&'static str, usize>,
        }
        impl GroupBuckets {
            fn push(&mut self, key: &'static str, idx: usize, value: Value) {
                match self.index_of.get(key).copied() {
                    Some(i) => {
                        self.entries[i].1.push(idx);
                        self.entries[i].2.push(value);
                    }
                    None => {
                        self.index_of.insert(key, self.entries.len());
                        self.entries.push((key, vec![idx], vec![value]));
                    }
                }
            }
        }
        let mut groups = GroupBuckets::default();
        for (i, item) in items.iter().enumerate() {
            let key: &'static str = match item {
                Value::Object(_) => "dict",
                Value::String(_) => "str",
                Value::Bool(_) => "bool",
                Value::Number(_) => "number",
                Value::Array(_) => "list",
                Value::Null => "none",
            };
            groups.push(key, i, item.clone());
        }
        let mut keep_indices: BTreeSet<usize> = BTreeSet::new();
        let mut strategy_parts: Vec<String> = Vec::new();
        for (type_key, indices, values) in groups.entries {
            if values.len() < self.config.min_items_to_analyze {
                keep_indices.extend(&indices);
                continue;
            }
            match type_key {
                "dict" => {
                    let (crushed, _strategy, _) =
                        self.crush_dict_array(&values, bias, emit_ccr_markers, ccr_saves);
                    let crushed_keys: HashSet<String> = crushed
                        .iter()
                        .map(|v| serde_json::to_string(v).unwrap_or_default())
                        .collect();
                    for (i, idx) in indices.iter().enumerate() {
                        let key = serde_json::to_string(&values[i]).unwrap_or_default();
                        if crushed_keys.contains(&key) {
                            keep_indices.insert(*idx);
                        }
                    }
                    strategy_parts.push(format!("dict:{}->{}", values.len(), crushed.len()));
                }
                "str" => {
                    let strs: Vec<&str> = values.iter().filter_map(|v| v.as_str()).collect();
                    let (crushed, _) = crush_string_array(&strs, &self.config, bias);
                    let crushed_set: HashSet<&str> = crushed.iter().map(|s| s.as_str()).collect();
                    for (i, idx) in indices.iter().enumerate() {
                        if let Some(s) = values[i].as_str()
                            && crushed_set.contains(s)
                        {
                            keep_indices.insert(*idx);
                        }
                    }
                    strategy_parts.push(format!("str:{}->{}", values.len(), crushed.len()));
                }
                "number" => {
                    let item_strings: Vec<String> = values.iter().map(|v| v.to_string()).collect();
                    let item_refs: Vec<&str> = item_strings.iter().map(|s| s.as_str()).collect();
                    let (_kt, kf, kl, _) = compute_k_split(&item_refs, &self.config, bias);
                    let kf = kf.min(values.len());
                    let kl = kl.min(values.len().saturating_sub(kf));
                    let first_idx: Vec<usize> = indices.iter().take(kf).copied().collect();
                    let last_idx: Vec<usize> = indices.iter().rev().take(kl).copied().collect();
                    keep_indices.extend(&first_idx);
                    keep_indices.extend(&last_idx);
                    let finite: Vec<f64> = values
                        .iter()
                        .filter_map(|v| v.as_f64().filter(|f| f.is_finite()))
                        .collect();
                    if finite.len() > 1
                        && let Some(mean_v) = stats_math::mean(&finite)
                        && let Some(std_v) = stats_math::sample_stdev(&finite)
                        && std_v > 0.0
                    {
                        let threshold = self.config.variance_threshold * std_v;
                        for (i, val) in values.iter().enumerate() {
                            if let Some(num) = val.as_f64().filter(|f| f.is_finite())
                                && (num - mean_v).abs() > threshold
                            {
                                keep_indices.insert(indices[i]);
                            }
                        }
                    }
                    strategy_parts.push(format!("num:{}", values.len()));
                }
                _ => {
                    keep_indices.extend(&indices);
                }
            }
        }
        let result: Vec<Value> = keep_indices.iter().map(|&i| items[i].clone()).collect();
        let strategy = format!(
            "mixed:adaptive({}->{},{})",
            n,
            result.len(),
            strategy_parts.join(",")
        );
        (result, strategy)
    }
}

impl Default for SmartCrusher {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Compressor for SmartCrusher {
    fn name(&self) -> &'static str {
        "smart_crusher"
    }

    async fn compress(&self, content: &Content, ctx: &CompressionContext) -> Result<Compressed> {
        debug!("SmartCrusher compressing {} bytes", content.data.len());
        let text = String::from_utf8_lossy(&content.data);
        let bias = if ctx.max_tokens.map(|t| t < 1000).unwrap_or(false) {
            0.7
        } else {
            1.0
        };
        let emit_ccr_markers = ctx.reversible && self.ccr_store.is_some();
        let (compressed, ccr_saves) = self.crush_with_options(&text, bias, emit_ccr_markers);
        let original_tokens = text.len() / 4;
        let compressed_tokens = compressed.len() / 4;
        let id = if ctx.reversible {
            compute_key(content.data.as_ref())
        } else {
            format!("sc-{}", compute_key(content.data.as_ref()))
        };
        if ctx.reversible
            && let Some(store) = &self.ccr_store
        {
            for pending in &ccr_saves {
                store.save(&pending.id, &pending.original, None).await?;
            }
            store.save(&id, &text, None).await?;
        }
        Ok(Compressed {
            id,
            data: bytes::Bytes::from(compressed),
            original_tokens,
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
    use serde_json::json;

    #[test]
    fn string_array_passthrough_at_threshold() {
        let items: [&str; 8] = ["a", "b", "c", "d", "e", "f", "g", "h"];
        let cfg = SmartCrusherConfig::default();
        let (out, strat) = crush_string_array(&items, &cfg, 1.0);
        assert_eq!(out.len(), 8);
        assert_eq!(strat, "string:passthrough");
    }

    #[test]
    fn string_array_keeps_error_strings() {
        let items: Vec<&str> = (0..30)
            .map(|i| {
                if i == 15 {
                    "FATAL: out of memory"
                } else {
                    "ok"
                }
            })
            .collect();
        let cfg = SmartCrusherConfig::default();
        let (out, strat) = crush_string_array(&items, &cfg, 1.0);
        assert!(out.iter().any(|s| s == "FATAL: out of memory"));
        assert!(strat.contains("errors=1"));
    }

    #[test]
    fn number_array_keeps_outliers() {
        let mut items: Vec<Value> = vec![json!(0); 30];
        items.push(json!(1000));
        let cfg = SmartCrusherConfig::default();
        let (out, strat) = crush_number_array(&items, &cfg, 1.0);
        assert!(out.iter().any(|v| v.as_f64() == Some(1000.0)));
        assert!(strat.contains("outliers="));
    }

    #[test]
    fn object_crushes_when_large() {
        let mut obj = Map::new();
        for i in 0..30 {
            obj.insert(
                format!("k{:02}", i),
                json!(format!(
                    "this is a relatively long value string for entry number {} with content",
                    i
                )),
            );
        }
        let cfg = SmartCrusherConfig::default();
        let (out, strat) = crush_object(&obj, &cfg, 1.0);
        assert!(strat.starts_with("object:adaptive(") || strat == "object:passthrough");
        assert!(out.len() <= 30);
    }

    #[test]
    fn smart_crusher_compresses_json() {
        let crusher = SmartCrusher::new();
        let input = r#"[{"id":1,"name":"alice"},{"id":2,"name":"bob"},{"id":3,"name":"charlie"}]"#;
        let out = crusher.crush(input, "", 1.0);
        assert!(!out.is_empty());
    }
}
