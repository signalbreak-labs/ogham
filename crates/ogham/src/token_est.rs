//! Simple token estimation.
//!
//! Uses the standard heuristic: 1 token ≈ 4 bytes for English text.
//! This is not exact (GPT-4 cl100k_base uses BPE) but is deterministic
//! and good enough for compression ratio calculations.

/// Estimate the number of tokens in a string.
pub fn count_tokens(text: &str) -> usize {
    text.len().div_ceil(4).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_tokens() {
        assert_eq!(count_tokens("hello"), 2); // 5 bytes -> ceil(5/4)=2
        assert_eq!(count_tokens("hello world"), 3); // 11 bytes -> ceil(11/4)=3
        assert_eq!(count_tokens(""), 1); // min 1
    }
}
