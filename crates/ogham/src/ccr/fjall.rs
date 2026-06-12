use super::CcrStore;
use async_trait::async_trait;
use ogham_core::Result;

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
            Ok(Some(bytes)) => Ok(Some(
                String::from_utf8(bytes.to_vec()).unwrap_or_else(|_| String::new()),
            )),
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
}
