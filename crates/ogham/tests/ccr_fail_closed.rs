use async_trait::async_trait;
use ogham::ccr::CcrStore;
use ogham::pipeline::DefaultCompressionPipeline;
use ogham::{CompressionPipeline, Message, OghamError};
use std::sync::Arc;

struct FailingStore;

#[async_trait]
impl CcrStore for FailingStore {
    async fn save(&self, _id: &str, _original: &str, _metadata: Option<&str>) -> ogham::Result<()> {
        Err(OghamError::StoreError("forced save failure".to_string()))
    }

    async fn retrieve(&self, _id: &str) -> ogham::Result<Option<String>> {
        Ok(None)
    }

    async fn delete(&self, _id: &str) -> ogham::Result<()> {
        Ok(())
    }
}

fn json_array() -> String {
    let values: Vec<_> = (0..40)
        .map(|i| serde_json::json!({ "id": i, "name": format!("item_{i}"), "score": i }))
        .collect();
    serde_json::to_string(&values).unwrap()
}

#[tokio::test]
async fn ccr_save_failure_returns_original() {
    let input = json_array();
    let pipeline = DefaultCompressionPipeline::with_builtin_compressors(
        Some(Arc::new(FailingStore)),
        &["smart_crusher"],
    )
    .unwrap();

    let result = pipeline
        .run(&[Message::new("tool", input.clone())])
        .await
        .unwrap();

    assert_eq!(result.messages[0].content, input);
    assert_eq!(result.stats.compressor_used, "none");
}
