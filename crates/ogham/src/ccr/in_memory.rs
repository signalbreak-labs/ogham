use super::CcrStore;
use async_trait::async_trait;
use ogham_core::Result;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// In-memory CCR store backed by a mutex-protected HashMap.
pub struct InMemoryCcrStore {
    map: Mutex<HashMap<String, Entry>>,
    ttl: Duration,
    capacity: usize,
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
            map: Mutex::new(HashMap::with_capacity(capacity)),
            ttl,
            capacity,
        }
    }
}

impl Default for InMemoryCcrStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryCcrStore {
    /// Return a snapshot of all stored entries.
    pub fn get_all(&self) -> Vec<(String, String)> {
        let map = self.map.lock().unwrap();
        map.iter()
            .filter(|(_, e)| e.inserted.elapsed() <= self.ttl)
            .map(|(k, e)| (k.clone(), e.payload.clone()))
            .collect()
    }
}

#[async_trait]
impl CcrStore for InMemoryCcrStore {
    async fn save(&self, id: &str, original: &str, _metadata: Option<&str>) -> Result<()> {
        let mut map = self.map.lock().unwrap();
        // Evict oldest if at capacity.
        if map.len() >= self.capacity && !map.contains_key(id) {
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
            if entry.inserted.elapsed() <= self.ttl {
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
