use crate::Message;

/// Counts (or estimates) LLM tokens. Implementations must be deterministic
/// and must never panic.
pub trait TokenCounter: Send + Sync {
    /// Tokens for a raw string.
    fn count(&self, text: &str) -> usize;

    /// True if this counter is exact for its target model family,
    /// false if it is an estimate.
    fn is_exact(&self) -> bool;

    /// How this counter's counts are produced.
    ///
    /// The default derives from [`TokenCounter::is_exact`]; estimators should
    /// override it to report their method and a recommended safety margin, so a
    /// caller never presents an estimate as exact.
    fn count_kind(&self) -> TokenCountKind {
        if self.is_exact() {
            TokenCountKind::Exact
        } else {
            TokenCountKind::Estimated {
                method: "heuristic".to_string(),
                safety_margin: 0.05,
            }
        }
    }

    /// Tokens for a message list, including per-message wrapper overhead.
    /// Default: sum of content counts + OVERHEAD_PER_MESSAGE per message.
    fn count_messages(&self, messages: &[Message]) -> usize {
        messages
            .iter()
            .map(|m| self.count(&m.content))
            .sum::<usize>()
            + messages.len() * OVERHEAD_PER_MESSAGE
    }
}

/// How a token count was produced, so budgets and reports can distinguish exact
/// counts from estimates that need headroom.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenCountKind {
    /// Exact for the target tokenizer.
    Exact,
    /// Estimated. `method` names the heuristic; `safety_margin` is the
    /// recommended fractional headroom to add when budgeting (e.g. `0.05`).
    Estimated {
        /// Human-readable name of the estimation method.
        method: String,
        /// Recommended fractional safety margin for budgeting.
        safety_margin: f64,
    },
    /// Reported by the provider's own usage accounting.
    ProviderReported,
}

impl TokenCountKind {
    /// Recommended fractional safety margin for budgeting (`0.0` for exact and
    /// provider-reported counts).
    pub fn safety_margin(&self) -> f64 {
        match self {
            TokenCountKind::Exact | TokenCountKind::ProviderReported => 0.0,
            TokenCountKind::Estimated { safety_margin, .. } => *safety_margin,
        }
    }

    /// Whether the counts are exact (or provider-reported), i.e. need no margin.
    pub fn is_exact(&self) -> bool {
        matches!(
            self,
            TokenCountKind::Exact | TokenCountKind::ProviderReported
        )
    }
}

impl Default for TokenCountKind {
    /// Defaults to a conservative estimate, never a false claim of exactness.
    fn default() -> Self {
        TokenCountKind::Estimated {
            method: "unknown".to_string(),
            safety_margin: 0.05,
        }
    }
}

/// Approximate per-message wrapper cost (role tags, separators).
/// 4 matches OpenAI's documented ChatML overhead and is a safe
/// over-estimate for other providers.
pub const OVERHEAD_PER_MESSAGE: usize = 4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimated_kind_exposes_margin() {
        let kind = TokenCountKind::Estimated {
            method: "m".to_string(),
            safety_margin: 0.05,
        };
        assert!(!kind.is_exact());
        assert_eq!(kind.safety_margin(), 0.05);
    }

    #[test]
    fn exact_and_provider_have_no_margin() {
        assert!(TokenCountKind::Exact.is_exact());
        assert_eq!(TokenCountKind::Exact.safety_margin(), 0.0);
        assert!(TokenCountKind::ProviderReported.is_exact());
        assert_eq!(TokenCountKind::ProviderReported.safety_margin(), 0.0);
    }

    #[test]
    fn default_is_a_conservative_estimate() {
        assert!(!TokenCountKind::default().is_exact());
        assert!(TokenCountKind::default().safety_margin() > 0.0);
    }
}
