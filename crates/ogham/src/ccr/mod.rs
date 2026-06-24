#[cfg(feature = "ccr-fjall")]
pub mod fjall;
pub mod in_memory;
#[cfg(feature = "ccr-sqlite")]
pub mod sqlite;

use async_trait::async_trait;
use ogham_core::{Message, Result, meta_keys};
use std::collections::{BTreeSet, HashMap};

/// A typed CCR payload: raw bytes plus a media type and optional metadata.
///
/// Lets a host store an exact structured original (e.g. a serialized
/// [`ogham_core::RichMessage`]) for lossless undo, not just UTF-8 text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CcrPayload {
    /// Media type of `bytes`, e.g. `application/json` or `text/plain`.
    pub media_type: String,
    /// The original content bytes.
    pub bytes: Vec<u8>,
    /// Optional host metadata stored alongside the payload.
    pub metadata: HashMap<String, String>,
}

impl CcrPayload {
    /// Construct a UTF-8 text payload with the given media type.
    pub fn text(media_type: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            media_type: media_type.into(),
            bytes: text.into().into_bytes(),
            metadata: HashMap::new(),
        }
    }
}

/// Pluggable CCR storage backend.
#[async_trait]
pub trait CcrStore: Send + Sync {
    async fn save(&self, id: &str, original: &str, metadata: Option<&str>) -> Result<()>;
    async fn retrieve(&self, id: &str) -> Result<Option<String>>;
    async fn delete(&self, id: &str) -> Result<()>;

    /// Save a typed payload.
    ///
    /// The default implementation serializes the payload into the text store as
    /// a self-describing envelope, so every existing store gains payload support
    /// with no changes. Backends with native binary columns may override it.
    async fn save_payload(&self, id: &str, payload: &CcrPayload) -> Result<()> {
        self.save(id, &encode_payload(payload), None).await
    }

    /// Retrieve a typed payload.
    ///
    /// Returns a `text/plain` payload when the id holds a plain string saved via
    /// [`CcrStore::save`], so mixing the text and payload APIs degrades
    /// gracefully rather than erroring.
    async fn retrieve_payload(&self, id: &str) -> Result<Option<CcrPayload>> {
        Ok(self
            .retrieve(id)
            .await?
            .map(|stored| decode_payload(&stored)))
    }
}

/// Marker key identifying a [`CcrPayload`] envelope inside the text store.
const PAYLOAD_MARKER: &str = "ogham_ccr_payload";

/// Serialize a payload into a self-describing JSON envelope. UTF-8 bytes are
/// stored verbatim; binary bytes are hex-encoded so any payload round-trips.
fn encode_payload(payload: &CcrPayload) -> String {
    let (enc, data) = match std::str::from_utf8(&payload.bytes) {
        Ok(text) => ("utf8", text.to_string()),
        Err(_) => ("hex", to_hex(&payload.bytes)),
    };
    serde_json::json!({
        PAYLOAD_MARKER: 1,
        "media_type": payload.media_type,
        "enc": enc,
        "data": data,
        "metadata": payload.metadata,
    })
    .to_string()
}

/// Decode a stored string into a payload, falling back to `text/plain` for a
/// plain string that is not an envelope. Shared with the native backends so a
/// legacy text envelope (or a plain `save`) still decodes after they switch to
/// native storage.
pub(crate) fn decode_payload(stored: &str) -> CcrPayload {
    if let Some(payload) = try_decode_payload(stored) {
        return payload;
    }
    CcrPayload {
        media_type: "text/plain; charset=utf-8".to_string(),
        bytes: stored.as_bytes().to_vec(),
        metadata: HashMap::new(),
    }
}

fn try_decode_payload(stored: &str) -> Option<CcrPayload> {
    let value = serde_json::from_str::<serde_json::Value>(stored).ok()?;
    if value.get(PAYLOAD_MARKER)?.as_u64()? != 1 {
        return None;
    }
    let media_type = value.get("media_type")?.as_str()?.to_string();
    let data = value.get("data")?.as_str()?;
    let bytes = match value.get("enc")?.as_str()? {
        "utf8" => data.as_bytes().to_vec(),
        "hex" => from_hex(data)?,
        _ => return None,
    };
    let metadata = value
        .get("metadata")
        .and_then(|metadata| metadata.as_object())
        .map(|map| {
            map.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default();
    Some(CcrPayload {
        media_type,
        bytes,
        metadata,
    })
}

fn to_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

fn from_hex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    s.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let hi = (pair[0] as char).to_digit(16)?;
            let lo = (pair[1] as char).to_digit(16)?;
            Some(((hi << 4) | lo) as u8)
        })
        .collect()
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

/// Collect the CCR ids a message list still references — via `<<ccr:ID>>`
/// markers in message content and `meta_keys::CCR_ID` metadata.
///
/// This is the live set for garbage collection: an original whose id is not in
/// this set is no longer reachable from the working prompt and may be evicted
/// without breaking a live marker.
pub fn referenced_ccr_ids(messages: &[Message]) -> BTreeSet<String> {
    let mut ids = BTreeSet::new();
    for message in messages {
        if let Some(id) = message.metadata.get(meta_keys::CCR_ID) {
            ids.insert(id.clone());
        }
        collect_markers(&message.content, &mut ids);
    }
    ids
}

/// Extract every `<<ccr:ID>>` marker id embedded in `content`.
fn collect_markers(content: &str, ids: &mut BTreeSet<String>) {
    let mut rest = content;
    while let Some(start) = rest.find("<<ccr:") {
        let after = &rest[start + "<<ccr:".len()..];
        match after.find(">>") {
            Some(end) => {
                ids.insert(after[..end].to_string());
                rest = &after[end + 2..];
            }
            None => break,
        }
    }
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

    #[test]
    fn referenced_ids_collects_markers_and_metadata() {
        let mut m1 = Message::new("tool", "[tool:x] result cleared — via <<ccr:b3:aaa>>");
        m1.metadata
            .insert(meta_keys::CCR_ID.to_string(), "b3:aaa".to_string());
        // Multiple markers in one message.
        let m2 = Message::new(
            "tool",
            r#"[{"_ccr_dropped":"<<ccr:b3:bbb>>"},{"x":"<<ccr:b3:ccc>>"}]"#,
        );
        let m3 = Message::new("user", "no markers here");

        let ids = referenced_ccr_ids(&[m1, m2, m3]);
        assert_eq!(
            ids.into_iter().collect::<Vec<_>>(),
            vec![
                "b3:aaa".to_string(),
                "b3:bbb".to_string(),
                "b3:ccc".to_string()
            ]
        );
    }

    #[tokio::test]
    async fn payload_round_trips_text_and_binary() {
        let store = in_memory::InMemoryCcrStore::new();

        let mut metadata = HashMap::new();
        metadata.insert("origin".to_string(), "tool".to_string());
        let text = CcrPayload {
            media_type: "application/json".to_string(),
            bytes: br#"{"a":1}"#.to_vec(),
            metadata,
        };
        store.save_payload("t", &text).await.unwrap();
        assert_eq!(
            store.retrieve_payload("t").await.unwrap().as_ref(),
            Some(&text)
        );

        // Invalid UTF-8 must survive via hex encoding.
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
    async fn retrieve_payload_on_plain_string_is_text() {
        let store = in_memory::InMemoryCcrStore::new();
        store.save("p", "just text", None).await.unwrap();
        let payload = store.retrieve_payload("p").await.unwrap().unwrap();
        assert_eq!(payload.bytes, b"just text");
        assert!(payload.media_type.starts_with("text/plain"));
    }

    #[tokio::test]
    async fn retrieve_payload_on_plain_json_marker_collision_is_text() {
        let store = in_memory::InMemoryCcrStore::new();
        let plain = r#"{"ogham_ccr_payload":true,"data":"not an envelope"}"#;
        store.save("plain-json", plain, None).await.unwrap();

        let payload = store.retrieve_payload("plain-json").await.unwrap().unwrap();

        assert_eq!(payload.bytes, plain.as_bytes());
        assert!(payload.media_type.starts_with("text/plain"));
    }

    #[tokio::test]
    async fn malformed_payload_envelope_falls_back_to_plain_text() {
        let store = in_memory::InMemoryCcrStore::new();
        let malformed = r#"{"ogham_ccr_payload":1,"media_type":"x","enc":"hex","data":"abc"}"#;
        store.save("bad-envelope", malformed, None).await.unwrap();

        let payload = store
            .retrieve_payload("bad-envelope")
            .await
            .unwrap()
            .unwrap();

        assert_eq!(payload.bytes, malformed.as_bytes());
        assert!(payload.media_type.starts_with("text/plain"));
    }

    #[tokio::test]
    async fn rich_message_blocks_restore_exactly_via_ccr() {
        use ogham_core::{ContentBlock, RichMessage};

        let store = in_memory::InMemoryCcrStore::new();
        let original = RichMessage::blocks(
            "assistant",
            vec![
                ContentBlock::ToolUse {
                    id: "c1".to_string(),
                    name: "shell".to_string(),
                    input: serde_json::json!({ "cmd": "ls" }),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "c1".to_string(),
                    is_error: false,
                    content: vec![ContentBlock::Text {
                        text: "out".to_string(),
                    }],
                },
            ],
        );
        let json = serde_json::to_string(&original).unwrap();
        store
            .save_payload("m", &CcrPayload::text("application/json", json))
            .await
            .unwrap();

        let restored_bytes = store.retrieve_payload("m").await.unwrap().unwrap().bytes;
        let restored: RichMessage =
            serde_json::from_str(&String::from_utf8(restored_bytes).unwrap()).unwrap();
        assert_eq!(
            restored, original,
            "tool ids and structure must survive CCR"
        );
    }
}
