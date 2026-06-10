use super::CcrStore;
use async_trait::async_trait;
use ogham_core::Result;
use rusqlite::{Connection, OptionalExtension, params};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

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
        let res = conn.execute(
            "INSERT INTO ccr_entries (hash, original, created_at, ttl_seconds)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(hash) DO UPDATE SET
                 original    = excluded.original,
                 created_at  = excluded.created_at,
                 ttl_seconds = excluded.ttl_seconds",
            params![
                id,
                original.as_bytes(),
                now as i64,
                self.default_ttl_seconds as i64
            ],
        );
        if let Err(err) = res {
            tracing::warn!(hash = %id, error = %err, "ccr_sqlite_save_failed");
        }
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
            .unwrap_or_else(|err| {
                tracing::warn!(hash = %id, error = %err, "ccr_sqlite_get_failed");
                None
            });
        Ok(row.and_then(|bytes| String::from_utf8(bytes).ok()))
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute("DELETE FROM ccr_entries WHERE hash = ?1", params![id]);
        Ok(())
    }
}
