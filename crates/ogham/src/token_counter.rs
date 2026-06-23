#[cfg(feature = "tiktoken")]
use ogham_core::Result;
use ogham_core::{TokenCountKind, TokenCounter};
use std::sync::Arc;

/// Bytes-per-token heuristic, optionally calibrated per model family.
pub struct HeuristicCounter {
    /// Estimated bytes per token. Default 4.0 (English prose).
    pub bytes_per_token: f64,
}

impl HeuristicCounter {
    pub fn new() -> Self {
        Self {
            bytes_per_token: 4.0,
        }
    }

    /// Calibration table (exact strings; match by `starts_with` on the model id):
    ///   "gpt-"      -> 4.0
    ///   "claude-"   -> 3.5   (Claude tokenizes denser code/JSON; over-estimate tokens)
    ///   "gemini-"   -> 4.0
    ///   anything else -> 4.0
    pub fn for_model(model: &str) -> Self {
        let bpt = if model.starts_with("claude-") {
            3.5
        } else {
            // "gpt-", "gemini-", and everything else default to 4.0
            4.0
        };
        Self {
            bytes_per_token: bpt,
        }
    }
}

impl Default for HeuristicCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenCounter for HeuristicCounter {
    fn count(&self, text: &str) -> usize {
        ((text.len() as f64 / self.bytes_per_token).ceil() as usize).max(1)
    }
    fn is_exact(&self) -> bool {
        false
    }
    fn count_kind(&self) -> TokenCountKind {
        TokenCountKind::Estimated {
            method: format!("bytes/{:.1} heuristic", self.bytes_per_token),
            safety_margin: 0.05,
        }
    }
}

#[cfg(feature = "tiktoken")]
pub struct TiktokenCounter {
    bpe: tiktoken_rs::CoreBPE,
    exact: bool,
}

#[cfg(feature = "tiktoken")]
impl TiktokenCounter {
    /// Encoding selection (in this order):
    ///   model starts_with "gpt-4o" | "gpt-5" | "o1" | "o3" | "o4" -> o200k_base, exact=true
    ///   model starts_with "gpt-4" | "gpt-3.5"                     -> cl100k_base, exact=true
    ///   anything else (incl. "claude-*") -> o200k_base proxy, exact=false,
    ///       and count() multiplies the result by 1.1 rounded up (safety margin,
    ///       because we cannot tokenize Claude exactly).
    /// Returns Err only if the encoding tables fail to load.
    pub fn for_model(model: &str) -> Result<Self> {
        let (bpe, exact) = if model.starts_with("gpt-4o")
            || model.starts_with("gpt-5")
            || model.starts_with("o1")
            || model.starts_with("o3")
            || model.starts_with("o4")
        {
            (
                tiktoken_rs::o200k_base()
                    .map_err(|e| ogham_core::OghamError::CompressionFailed(e.to_string()))?,
                true,
            )
        } else if model.starts_with("gpt-4") || model.starts_with("gpt-3.5") {
            (
                tiktoken_rs::cl100k_base()
                    .map_err(|e| ogham_core::OghamError::CompressionFailed(e.to_string()))?,
                true,
            )
        } else {
            (
                tiktoken_rs::o200k_base()
                    .map_err(|e| ogham_core::OghamError::CompressionFailed(e.to_string()))?,
                false,
            )
        };
        Ok(Self { bpe, exact })
    }
}

#[cfg(feature = "tiktoken")]
impl TokenCounter for TiktokenCounter {
    fn count(&self, text: &str) -> usize {
        let tokens = self.bpe.encode_with_special_tokens(text).len();
        if self.exact {
            tokens.max(1)
        } else {
            ((tokens as f64 * 1.1).ceil() as usize).max(1)
        }
    }
    fn is_exact(&self) -> bool {
        self.exact
    }
    fn count_kind(&self) -> TokenCountKind {
        if self.exact {
            TokenCountKind::Exact
        } else {
            // Non-target families (e.g. Claude) use the o200k tokenizer scaled
            // by 1.1; report the residual margin so budgets stay honest.
            TokenCountKind::Estimated {
                method: "o200k proxy (x1.1)".to_string(),
                safety_margin: 0.05,
            }
        }
    }
}

/// Best available counter for `model`. Never fails:
/// with feature "tiktoken", tries TiktokenCounter::for_model and falls back to
/// HeuristicCounter::for_model on error; without the feature, returns the heuristic.
pub fn counter_for_model(model: &str) -> Arc<dyn TokenCounter> {
    #[cfg(feature = "tiktoken")]
    {
        if let Ok(counter) = TiktokenCounter::for_model(model) {
            return Arc::new(counter);
        }
    }
    Arc::new(HeuristicCounter::for_model(model))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ogham_core::OVERHEAD_PER_MESSAGE;

    #[test]
    fn heuristic_minimum_one() {
        assert_eq!(HeuristicCounter::new().count(""), 1);
    }

    #[test]
    fn heuristic_deterministic() {
        let c = HeuristicCounter::new();
        let s = "hello world";
        assert_eq!(c.count(s), c.count(s));
    }

    #[test]
    fn claude_overestimates() {
        let s = "a".repeat(1024);
        let claude = HeuristicCounter::for_model("claude-fable-5");
        let gpt = HeuristicCounter::for_model("gpt-4o");
        assert!(claude.count(&s) >= gpt.count(&s));
    }

    #[test]
    fn factory_never_panics() {
        let counter = counter_for_model("totally-unknown-model");
        assert!(counter.count("hi") >= 1);
    }

    #[test]
    fn heuristic_reports_estimated_kind_with_margin() {
        match HeuristicCounter::for_model("claude-fable-5").count_kind() {
            TokenCountKind::Estimated {
                method,
                safety_margin,
            } => {
                assert!(safety_margin > 0.0, "estimates must carry headroom");
                assert!(method.contains("3.5"), "method should name the calibration");
            }
            other => panic!("heuristic must report an estimate, got {other:?}"),
        }
        assert!(!HeuristicCounter::new().count_kind().is_exact());
    }

    #[test]
    fn overhead_constant_matches_spec() {
        assert_eq!(OVERHEAD_PER_MESSAGE, 4);
    }

    #[cfg(feature = "tiktoken")]
    mod tiktoken_tests {
        use super::*;

        #[test]
        fn tiktoken_exact_for_gpt4o() {
            let counter = TiktokenCounter::for_model("gpt-4o").unwrap();
            assert!(counter.is_exact());
            assert!(counter.count("hello world") > 0);
        }

        #[test]
        fn tiktoken_claude_is_estimate() {
            let counter = TiktokenCounter::for_model("claude-opus-4-8").unwrap();
            assert!(!counter.is_exact());
        }

        #[test]
        fn tiktoken_kind_matches_exactness() {
            assert_eq!(
                TiktokenCounter::for_model("gpt-4o").unwrap().count_kind(),
                TokenCountKind::Exact
            );
            assert!(
                !TiktokenCounter::for_model("claude-opus-4-8")
                    .unwrap()
                    .count_kind()
                    .is_exact()
            );
        }
    }
}
