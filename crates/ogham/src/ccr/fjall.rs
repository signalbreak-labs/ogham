use super::{CcrPayload, CcrStore};
use async_trait::async_trait;
use ogham_core::Result;
use std::collections::HashMap;

/// Magic header marking a natively-framed [`CcrPayload`] value, distinguishing it
/// from a plain text `save` value (which is arbitrary UTF-8 and won't begin with
/// these bytes).
const PAYLOAD_MAGIC: &[u8] = b"OGHMccr\x01";

/// Frame a payload as `MAGIC | media_type_len:u32le | media_type | meta_len:u32le
/// | meta_json | raw bytes` — raw bytes, no hex envelope.
fn encode_native(payload: &CcrPayload) -> Vec<u8> {
    let media_type = payload.media_type.as_bytes();
    let metadata = serde_json::to_vec(&payload.metadata).unwrap_or_else(|_| b"{}".to_vec());
    let mut out = Vec::with_capacity(
        PAYLOAD_MAGIC.len() + 8 + media_type.len() + metadata.len() + payload.bytes.len(),
    );
    out.extend_from_slice(PAYLOAD_MAGIC);
    out.extend_from_slice(&(media_type.len() as u32).to_le_bytes());
    out.extend_from_slice(media_type);
    out.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
    out.extend_from_slice(&metadata);
    out.extend_from_slice(&payload.bytes);
    out
}

/// Read a little-endian `u32` length at `*pos`, advancing `*pos` past it.
fn read_u32(rest: &[u8], pos: &mut usize) -> Option<usize> {
    let end = pos.checked_add(4)?;
    let chunk = rest.get(*pos..end)?;
    *pos = end;
    Some(u32::from_le_bytes(chunk.try_into().ok()?) as usize)
}

/// Read a length-prefixed byte slice at `*pos`, advancing `*pos` past it.
fn read_chunk<'a>(rest: &'a [u8], pos: &mut usize) -> Option<&'a [u8]> {
    let len = read_u32(rest, pos)?;
    let end = pos.checked_add(len)?;
    let chunk = rest.get(*pos..end)?;
    *pos = end;
    Some(chunk)
}

/// Decode a native frame, or `None` if `bytes` is not framed (a plain `save`
/// value or a legacy text envelope).
fn decode_native(bytes: &[u8]) -> Option<CcrPayload> {
    let rest = bytes.strip_prefix(PAYLOAD_MAGIC)?;
    let mut pos = 0usize;
    let media_type = std::str::from_utf8(read_chunk(rest, &mut pos)?)
        .ok()?
        .to_string();
    let metadata: HashMap<String, String> =
        serde_json::from_slice(read_chunk(rest, &mut pos)?).ok()?;
    let payload_bytes = rest.get(pos..)?.to_vec();
    Some(CcrPayload {
        media_type,
        bytes: payload_bytes,
        metadata,
    })
}

/// CCR store backed by fjall LSM-tree.
pub struct FjallCcrStore {
    keyspace: fjall::Keyspace,
    partition: fjall::PartitionHandle,
}

impl FjallCcrStore {
    /// Open a new keyspace at `path`.
    pub fn new(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let keyspace = fjall::Config::new(path.as_ref())
            .open()
            .map_err(|e| ogham_core::OghamError::StoreError(e.to_string()))?;
        let partition = keyspace
            .open_partition("ccr", fjall::PartitionCreateOptions::default())
            .map_err(|e| ogham_core::OghamError::StoreError(e.to_string()))?;
        Ok(Self {
            keyspace,
            partition,
        })
    }

    /// Wrap an *existing* fjall keyspace so CCR shares the same DB file.
    /// This is critical for host integration where the application already
    /// owns a keyspace.
    pub fn from_keyspace(keyspace: &fjall::Keyspace, partition_name: &str) -> Result<Self> {
        let partition = keyspace
            .open_partition(partition_name, fjall::PartitionCreateOptions::default())
            .map_err(|e| ogham_core::OghamError::StoreError(e.to_string()))?;
        Ok(Self {
            keyspace: keyspace.clone(),
            partition,
        })
    }
}

#[async_trait]
impl CcrStore for FjallCcrStore {
    async fn save(&self, id: &str, original: &str, _metadata: Option<&str>) -> Result<()> {
        self.partition
            .insert(id, original)
            .map_err(|e| ogham_core::OghamError::StoreError(e.to_string()))?;
        self.keyspace
            .persist(fjall::PersistMode::SyncAll)
            .map_err(|e| ogham_core::OghamError::StoreError(e.to_string()))?;
        Ok(())
    }

    async fn retrieve(&self, id: &str) -> Result<Option<String>> {
        match self.partition.get(id) {
            Ok(Some(bytes)) => {
                // A natively-framed payload yields its content as text (lossless
                // for text payloads); a plain value is returned as-is.
                let text = match decode_native(&bytes) {
                    Some(payload) => String::from_utf8(payload.bytes).unwrap_or_default(),
                    None => String::from_utf8(bytes.to_vec()).unwrap_or_default(),
                };
                Ok(Some(text))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(ogham_core::OghamError::StoreError(e.to_string())),
        }
    }

    async fn delete(&self, id: &str) -> Result<()> {
        self.partition
            .remove(id)
            .map_err(|e| ogham_core::OghamError::StoreError(e.to_string()))?;
        Ok(())
    }

    /// Store a typed payload natively as a compact binary frame (raw bytes, no
    /// hex envelope), so binary payloads cost their real size.
    async fn save_payload(&self, id: &str, payload: &CcrPayload) -> Result<()> {
        self.partition
            .insert(id, encode_native(payload))
            .map_err(|e| ogham_core::OghamError::StoreError(e.to_string()))?;
        self.keyspace
            .persist(fjall::PersistMode::SyncAll)
            .map_err(|e| ogham_core::OghamError::StoreError(e.to_string()))?;
        Ok(())
    }

    async fn retrieve_payload(&self, id: &str) -> Result<Option<CcrPayload>> {
        match self.partition.get(id) {
            Ok(Some(bytes)) => Ok(Some(match decode_native(&bytes) {
                Some(payload) => payload,
                // A plain `save` value or a legacy text envelope.
                None => super::decode_payload(&String::from_utf8_lossy(&bytes)),
            })),
            Ok(None) => Ok(None),
            Err(e) => Err(ogham_core::OghamError::StoreError(e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new() -> Self {
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("ogham_ccr_fjall_{}_{n}", std::process::id()));
            Self(path)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn native_frame_round_trips_and_rejects_non_frames() {
        let payload = CcrPayload {
            media_type: "application/octet-stream".to_string(),
            bytes: vec![0xff, 0x00, 0x80, 0x01],
            metadata: HashMap::from([("k".to_string(), "v".to_string())]),
        };
        let framed = encode_native(&payload);
        assert_eq!(decode_native(&framed), Some(payload));
        // Plain text is not a frame.
        assert_eq!(decode_native(b"just text"), None);
        // A truncated frame must not panic.
        assert_eq!(decode_native(&framed[..PAYLOAD_MAGIC.len() + 2]), None);
    }

    #[tokio::test]
    async fn payload_round_trips_native_text_and_binary() {
        let dir = TempDir::new();
        let store = FjallCcrStore::new(&dir.0).unwrap();

        let text = CcrPayload {
            media_type: "application/json".to_string(),
            bytes: br#"{"a":1}"#.to_vec(),
            metadata: HashMap::from([("origin".to_string(), "tool".to_string())]),
        };
        store.save_payload("t", &text).await.unwrap();
        assert_eq!(
            store.retrieve_payload("t").await.unwrap().as_ref(),
            Some(&text)
        );
        // The text API reads the framed text payload's content back.
        assert_eq!(
            store.retrieve("t").await.unwrap().as_deref(),
            Some(r#"{"a":1}"#)
        );

        let binary = CcrPayload {
            media_type: "application/octet-stream".to_string(),
            bytes: vec![0xff, 0xfe, 0x00, 0x80],
            metadata: HashMap::new(),
        };
        store.save_payload("b", &binary).await.unwrap();
        assert_eq!(
            store.retrieve_payload("b").await.unwrap().as_ref(),
            Some(&binary)
        );
    }

    #[tokio::test]
    async fn retrieve_payload_on_plain_save_is_text() {
        let dir = TempDir::new();
        let store = FjallCcrStore::new(&dir.0).unwrap();
        store.save("p", "just text", None).await.unwrap();
        let payload = store.retrieve_payload("p").await.unwrap().unwrap();
        assert_eq!(payload.bytes, b"just text");
        assert!(payload.media_type.starts_with("text/plain"));
    }
}
