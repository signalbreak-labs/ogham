use super::{CcrPayload, CcrStore};
use async_trait::async_trait;
use ogham_core::{OghamError, Result};
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Run an `ALTER TABLE ... ADD COLUMN` migration, treating only the benign
/// "column already exists" case (a re-open of an already-migrated DB) as success
/// and surfacing any other failure so a genuinely failed migration is not hidden.
fn add_column_if_missing(conn: &Connection, alter_sql: &str) -> rusqlite::Result<()> {
    match conn.execute(alter_sql, []) {
        Ok(_) => Ok(()),
        Err(e) if e.to_string().contains("duplicate column name") => Ok(()),
        Err(e) => Err(e),
    }
}

/// SQLite-backed CCR store.
pub struct SqliteCcrStore {
    conn: Mutex<Connection>,
    default_ttl_seconds: u64,
    _path: PathBuf,
}

impl SqliteCcrStore {
    pub fn open(path: impl AsRef<Path>, default_ttl_seconds: u64) -> rusqlite::Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let conn = Connection::open(&path_buf)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ccr_entries (
                 hash         TEXT PRIMARY KEY,
                 original     BLOB NOT NULL,
                 created_at   INTEGER NOT NULL,
                 ttl_seconds  INTEGER NOT NULL
             )",
            [],
        )?;
        // Native typed-payload columns. Added by migration so a database created
        // before payload support gains them; the `original` BLOB then holds raw
        // payload bytes (no hex envelope). NULL `media_type` marks a plain text
        // `save` (or a legacy envelope), which retrieves via the text decoder. A
        // real migration failure surfaces; only "column exists" on re-open is ok.
        add_column_if_missing(&conn, "ALTER TABLE ccr_entries ADD COLUMN media_type TEXT")?;
        add_column_if_missing(&conn, "ALTER TABLE ccr_entries ADD COLUMN metadata TEXT")?;
        Ok(Self {
            conn: Mutex::new(conn),
            default_ttl_seconds,
            _path: path_buf,
        })
    }

    fn now_unix_seconds() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

#[async_trait]
impl CcrStore for SqliteCcrStore {
    async fn save(&self, id: &str, original: &str, _metadata: Option<&str>) -> Result<()> {
        let now = Self::now_unix_seconds();
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO ccr_entries (hash, original, media_type, metadata, created_at, ttl_seconds)
             VALUES (?1, ?2, NULL, NULL, ?3, ?4)
             ON CONFLICT(hash) DO UPDATE SET
                 original    = excluded.original,
                 media_type  = NULL,
                 metadata    = NULL,
                 created_at  = excluded.created_at,
                 ttl_seconds = excluded.ttl_seconds",
            params![
                id,
                original.as_bytes(),
                now as i64,
                self.default_ttl_seconds as i64
            ],
        )
        .map_err(|err| {
            // Fail closed: a marker must never be emitted for an unstored original.
            tracing::warn!(hash = %id, error = %err, "ccr_sqlite_save_failed");
            OghamError::StoreError(err.to_string())
        })?;
        Ok(())
    }

    async fn retrieve(&self, id: &str) -> Result<Option<String>> {
        let now = Self::now_unix_seconds();
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute(
            "DELETE FROM ccr_entries WHERE created_at + ttl_seconds <= ?1",
            params![now as i64],
        );
        let row: Option<Vec<u8>> = conn
            .query_row(
                "SELECT original FROM ccr_entries WHERE hash = ?1 AND created_at + ttl_seconds > ?2",
                params![id, now as i64],
                |r| r.get::<_, Vec<u8>>(0),
            )
            .optional()
            .map_err(|err| {
                tracing::warn!(hash = %id, error = %err, "ccr_sqlite_get_failed");
                OghamError::StoreError(err.to_string())
            })?;
        match row {
            None => Ok(None),
            Some(bytes) => String::from_utf8(bytes).map(Some).map_err(|_| {
                // Plain `save` values are always valid UTF-8, so non-UTF-8 bytes
                // are a native binary payload fetched via the text API — fail
                // closed rather than report it empty/missing.
                OghamError::StoreError(format!(
                    "CCR entry {id} is a binary payload; use retrieve_payload"
                ))
            }),
        }
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute("DELETE FROM ccr_entries WHERE hash = ?1", params![id]);
        Ok(())
    }

    /// Store a typed payload. A UTF-8 payload keeps the self-describing text
    /// envelope (no hex penalty, and readable by older binaries that predate the
    /// native columns); only a *binary* payload is stored natively (raw bytes in
    /// the `original` BLOB plus the media-type/metadata columns) so it costs its
    /// real size instead of a 2x hex envelope.
    async fn save_payload(&self, id: &str, payload: &CcrPayload) -> Result<()> {
        if std::str::from_utf8(&payload.bytes).is_ok() {
            return self.save(id, &super::encode_payload(payload), None).await;
        }
        let now = Self::now_unix_seconds();
        let metadata_json =
            serde_json::to_string(&payload.metadata).unwrap_or_else(|_| "{}".to_string());
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO ccr_entries (hash, original, media_type, metadata, created_at, ttl_seconds)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(hash) DO UPDATE SET
                 original    = excluded.original,
                 media_type  = excluded.media_type,
                 metadata    = excluded.metadata,
                 created_at  = excluded.created_at,
                 ttl_seconds = excluded.ttl_seconds",
            params![
                id,
                payload.bytes,
                payload.media_type,
                metadata_json,
                now as i64,
                self.default_ttl_seconds as i64
            ],
        )
        .map_err(|err| {
            // Fail closed: the rich path keeps the original uncompressed if the
            // payload save errors, so it must not be swallowed as Ok.
            tracing::warn!(hash = %id, error = %err, "ccr_sqlite_save_payload_failed");
            OghamError::StoreError(err.to_string())
        })?;
        Ok(())
    }

    async fn retrieve_payload(&self, id: &str) -> Result<Option<CcrPayload>> {
        let now = Self::now_unix_seconds();
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute(
            "DELETE FROM ccr_entries WHERE created_at + ttl_seconds <= ?1",
            params![now as i64],
        );
        // A read error is not "missing" — surface it so an exact restore fails
        // closed rather than looking like an absent original.
        let row: Option<(Vec<u8>, Option<String>, Option<String>)> = conn
            .query_row(
                "SELECT original, media_type, metadata FROM ccr_entries
                 WHERE hash = ?1 AND created_at + ttl_seconds > ?2",
                params![id, now as i64],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()
            .map_err(|err| {
                tracing::warn!(hash = %id, error = %err, "ccr_sqlite_get_payload_failed");
                OghamError::StoreError(err.to_string())
            })?;
        let Some((bytes, media_type, metadata)) = row else {
            return Ok(None);
        };
        match media_type {
            // A native payload: reconstruct from the raw bytes + columns. Malformed
            // metadata on a native row is corruption — fail closed, don't drop it.
            Some(media_type) => {
                let metadata = match metadata {
                    Some(json) => serde_json::from_str(&json).map_err(|e| {
                        OghamError::StoreError(format!(
                            "corrupt CCR payload metadata for {id}: {e}"
                        ))
                    })?,
                    None => HashMap::new(),
                };
                Ok(Some(CcrPayload {
                    media_type,
                    bytes,
                    metadata,
                }))
            }
            // No media type: a plain `save` or a legacy text envelope — degrade
            // through the shared text decoder.
            None => Ok(Some(super::decode_payload(&String::from_utf8_lossy(
                &bytes,
            )))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A unique temp path; the returned guard removes the db (and WAL/SHM
    /// sidecars) on drop.
    struct TempDb(PathBuf);
    impl TempDb {
        fn new() -> Self {
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("ogham_ccr_sqlite_{}_{n}.db", std::process::id()));
            Self(path)
        }
    }
    impl Drop for TempDb {
        fn drop(&mut self) {
            for suffix in ["", "-wal", "-shm"] {
                let _ = std::fs::remove_file(format!("{}{suffix}", self.0.display()));
            }
        }
    }

    #[tokio::test]
    async fn payload_round_trips_text_envelope_and_native_binary() {
        let db = TempDb::new();
        let store = SqliteCcrStore::open(&db.0, 300).unwrap();

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
        // Rollback-readable: a UTF-8 payload is kept as the text envelope, so an
        // older binary's text decoder can still reconstruct it.
        let stored = store.retrieve("t").await.unwrap().unwrap();
        assert_eq!(crate::ccr::decode_payload(&stored), text);

        // Binary survives natively (no hex envelope).
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
        let db = TempDb::new();
        let store = SqliteCcrStore::open(&db.0, 300).unwrap();
        store.save("p", "just text", None).await.unwrap();
        let payload = store.retrieve_payload("p").await.unwrap().unwrap();
        assert_eq!(payload.bytes, b"just text");
        assert!(payload.media_type.starts_with("text/plain"));
    }

    #[test]
    fn migration_helper_classifies_errors() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute("CREATE TABLE t (a INTEGER)", []).unwrap();
        // First add succeeds; re-adding the same column is the benign duplicate.
        add_column_if_missing(&conn, "ALTER TABLE t ADD COLUMN b TEXT").unwrap();
        add_column_if_missing(&conn, "ALTER TABLE t ADD COLUMN b TEXT").unwrap();
        // A genuinely failing migration (no such table) must surface, not be hidden.
        assert!(add_column_if_missing(&conn, "ALTER TABLE missing ADD COLUMN c TEXT").is_err());
    }

    #[tokio::test]
    async fn retrieve_text_api_on_binary_payload_fails_closed() {
        let db = TempDb::new();
        let store = SqliteCcrStore::open(&db.0, 300).unwrap();
        let binary = CcrPayload {
            media_type: "application/octet-stream".to_string(),
            bytes: vec![0xff, 0xfe, 0x00],
            metadata: HashMap::new(),
        };
        store.save_payload("b", &binary).await.unwrap();
        // The typed API restores it exactly; the text API must not report empty.
        assert_eq!(
            store.retrieve_payload("b").await.unwrap().as_ref(),
            Some(&binary)
        );
        assert!(store.retrieve("b").await.is_err());
    }

    #[tokio::test]
    async fn corrupt_native_metadata_fails_closed() {
        let db = TempDb::new();
        let store = SqliteCcrStore::open(&db.0, 300).unwrap();
        // Binary bytes so the payload is stored in the native columns.
        let payload = CcrPayload {
            media_type: "application/octet-stream".to_string(),
            bytes: vec![0xff, 0x00],
            metadata: HashMap::new(),
        };
        store.save_payload("m", &payload).await.unwrap();
        // Corrupt the metadata column of the native row.
        store
            .conn
            .lock()
            .unwrap()
            .execute(
                "UPDATE ccr_entries SET metadata = ?1 WHERE hash = ?2",
                params!["not json", "m"],
            )
            .unwrap();
        // A native row with unparseable metadata must error, not lose metadata.
        assert!(store.retrieve_payload("m").await.is_err());
    }

    #[tokio::test]
    async fn read_error_is_propagated_not_hidden() {
        let db = TempDb::new();
        let store = SqliteCcrStore::open(&db.0, 300).unwrap();
        // Drop the table so the next read genuinely errors.
        store
            .conn
            .lock()
            .unwrap()
            .execute("DROP TABLE ccr_entries", [])
            .unwrap();
        // A read error must surface as Err, not be hidden as Ok(None).
        assert!(store.retrieve_payload("any").await.is_err());
    }

    #[tokio::test]
    async fn reopen_migrates_and_preserves_payload() {
        let db = TempDb::new();
        let payload = CcrPayload {
            media_type: "image/png".to_string(),
            bytes: vec![0x89, 0x50, 0x4e, 0x47, 0x00, 0xff],
            metadata: HashMap::new(),
        };
        {
            let store = SqliteCcrStore::open(&db.0, 300).unwrap();
            store.save_payload("m", &payload).await.unwrap();
        }
        // Reopening runs the migration again (idempotent) and the data persists.
        let store = SqliteCcrStore::open(&db.0, 300).unwrap();
        assert_eq!(
            store.retrieve_payload("m").await.unwrap().as_ref(),
            Some(&payload)
        );
    }
}
