pub mod fjall;
pub mod in_memory;
pub mod sqlite;

use async_trait::async_trait;
use ogham_core::Result;

/// Pluggable CCR storage backend.
#[async_trait]
pub trait CcrStore: Send + Sync {
    async fn save(&self, id: &str, original: &str, metadata: Option<&str>) -> Result<()>;
    async fn retrieve(&self, id: &str) -> Result<Option<String>>;
    async fn delete(&self, id: &str) -> Result<()>;
}

/// Compute a canonical CCR content address for a payload.
///
/// Returns a versioned, collision-resistant key of the form `b3:<32 hex>`
/// (a 128-bit BLAKE3 prefix). The `b3:` version tag lets the hash scheme
/// evolve without ambiguity; stores key on the literal string, so content
/// saved under an older scheme stays retrievable by its original id.
pub fn compute_key(payload: &[u8]) -> String {
    let hex = blake3::hash(payload).to_hex();
    format!("b3:{}", &hex[..32])
}

/// Standard CCR marker injected into compressed content.
pub fn marker_for(hash: &str) -> String {
    format!("<<ccr:{hash}>>")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_key_is_deterministic() {
        let a = compute_key(b"the same payload");
        let b = compute_key(b"the same payload");
        assert_eq!(a, b);
    }

    #[test]
    fn compute_key_is_versioned_and_distinct() {
        let key = compute_key(b"payload");
        assert!(
            key.starts_with("b3:"),
            "CCR ids must carry a hash-version prefix, got {key}"
        );
        assert_eq!(
            key.len(),
            "b3:".len() + 32,
            "fixed-width 128-bit content address"
        );
        assert_ne!(compute_key(b"a"), compute_key(b"b"));
    }

    #[test]
    fn marker_format() {
        assert_eq!(marker_for("abc123"), "<<ccr:abc123>>");
    }
}
