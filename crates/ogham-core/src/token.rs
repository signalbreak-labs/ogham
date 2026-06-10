use crate::Message;

/// Counts (or estimates) LLM tokens. Implementations must be deterministic
/// and must never panic.
pub trait TokenCounter: Send + Sync {
    /// Tokens for a raw string.
    fn count(&self, text: &str) -> usize;

    /// True if this counter is exact for its target model family,
    /// false if it is an estimate.
    fn is_exact(&self) -> bool;

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

/// Approximate per-message wrapper cost (role tags, separators).
/// 4 matches OpenAI's documented ChatML overhead and is a safe
/// over-estimate for other providers.
pub const OVERHEAD_PER_MESSAGE: usize = 4;
