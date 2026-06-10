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

/// Compute a canonical CCR key for a payload.
pub fn compute_key(payload: &[u8]) -> String {
    let h = md5::compute(payload);
    format!("{:x}", h)
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
    fn marker_format() {
        assert_eq!(marker_for("abc123"), "<<ccr:abc123>>");
    }
}
