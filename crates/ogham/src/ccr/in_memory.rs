use super::CcrStore;
use async_trait::async_trait;
use ogham_core::Result;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// In-memory CCR store backed by a mutex-protected HashMap.
///
/// By default it bounds memory with a capacity (LRU-style eviction) and a TTL.
/// For durable reversibility — where an original must never disappear while a
/// live `<<ccr:...>>` marker still references it — construct it with
/// [`InMemoryCcrStore::unbounded`] and manage lifecycle explicitly (e.g. via
/// `ContextSession` retention or `delete`).
pub struct InMemoryCcrStore {
    map: Mutex<HashMap<String, Entry>>,
    /// Entry lifetime; `None` never expires.
    ttl: Option<Duration>,
    /// Maximum entries before oldest-eviction; `None` is unbounded.
    capacity: Option<usize>,
}

#[derive(Clone)]
struct Entry {
    payload: String,
    inserted: Instant,
}

impl InMemoryCcrStore {
    pub fn new() -> Self {
        Self::with_capacity_and_ttl(1000, Duration::from_secs(300))
    }

    pub fn with_capacity_and_ttl(capacity: usize, ttl: Duration) -> Self {
        Self {
            map: Mutex::new(HashMap::with_capacity(capacity.min(1024))),
            ttl: Some(ttl),
            capacity: Some(capacity),
        }
    }

    /// A non-evicting store: no capacity cap and no TTL, so a stored original is
    /// never silently dropped. Use this for durable sessions and delete entries
    /// explicitly (or let `ContextSession` retention garbage-collect them).
    pub fn unbounded() -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
            ttl: None,
            capacity: None,
        }
    }
}

impl Default for InMemoryCcrStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryCcrStore {
    /// Return a snapshot of all live stored entries.
    pub fn get_all(&self) -> Vec<(String, String)> {
        let map = self.map.lock().unwrap();
        map.iter()
            .filter(|(_, e)| !self.is_expired(e))
            .map(|(k, e)| (k.clone(), e.payload.clone()))
            .collect()
    }

    fn is_expired(&self, entry: &Entry) -> bool {
        self.ttl.is_some_and(|ttl| entry.inserted.elapsed() > ttl)
    }
}

#[async_trait]
impl CcrStore for InMemoryCcrStore {
    async fn save(&self, id: &str, original: &str, _metadata: Option<&str>) -> Result<()> {
        let mut map = self.map.lock().unwrap();
        // Evict the oldest entry if a capacity cap is set and reached.
        if let Some(capacity) = self.capacity
            && map.len() >= capacity
            && !map.contains_key(id)
        {
            let oldest = map
                .iter()
                .min_by_key(|(_, e)| e.inserted)
                .map(|(k, _)| k.clone());
            if let Some(k) = oldest {
                map.remove(&k);
            }
        }
        map.insert(
            id.to_string(),
            Entry {
                payload: original.to_string(),
                inserted: Instant::now(),
            },
        );
        Ok(())
    }

    async fn retrieve(&self, id: &str) -> Result<Option<String>> {
        let mut map = self.map.lock().unwrap();
        if let Some(entry) = map.get(id) {
            if !self.is_expired(entry) {
                return Ok(Some(entry.payload.clone()));
            }
            map.remove(id);
        }
        Ok(None)
    }

    async fn delete(&self, id: &str) -> Result<()> {
        self.map.lock().unwrap().remove(id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_then_get() {
        let store = InMemoryCcrStore::new();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            store.save("abc", "payload", None).await.unwrap();
            assert_eq!(
                store.retrieve("abc").await.unwrap(),
                Some("payload".to_string())
            );
        });
    }

    #[test]
    fn unbounded_never_evicts() {
        let store = InMemoryCcrStore::unbounded();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            for i in 0..2_000 {
                store
                    .save(&format!("id{i}"), &format!("payload{i}"), None)
                    .await
                    .unwrap();
            }
            // The very first entry survives 2000 inserts (no capacity cap, no TTL).
            assert_eq!(
                store.retrieve("id0").await.unwrap(),
                Some("payload0".to_string())
            );
            assert_eq!(store.get_all().len(), 2_000);
        });
    }

    #[test]
    fn expired_entries_dropped() {
        let store = InMemoryCcrStore::with_capacity_and_ttl(10, Duration::from_millis(10));
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            store.save("a", "1", None).await.unwrap();
            std::thread::sleep(Duration::from_millis(25));
            assert_eq!(store.retrieve("a").await.unwrap(), None);
        });
    }
}
