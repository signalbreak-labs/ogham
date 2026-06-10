pub mod adaptive_sizer;
pub mod agent;
pub mod budget;
pub mod cache_aligner;
pub mod cache_strategy;
pub mod ccr;
pub mod compressors;
pub mod conversation;
pub mod detect;
pub mod memory;
pub mod pipeline;
pub mod stats_math;
pub mod token_counter;
pub mod token_est;

pub use ogham_core::*;
pub use token_counter::{HeuristicCounter, counter_for_model};

use crate::ccr::CcrStore;
use crate::pipeline::DefaultCompressionPipeline;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Library configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressConfig {
    pub reversible: bool,
    pub use_cache_aligner: bool,
    pub compressors: Vec<String>,
    pub ccr_store_path: Option<std::path::PathBuf>,
}

impl Default for CompressConfig {
    fn default() -> Self {
        Self {
            reversible: true,
            use_cache_aligner: false,
            compressors: vec![
                "smart_crusher".into(),
                "ast_code".into(),
                "log_stripper".into(),
            ],
            ccr_store_path: None,
        }
    }
}

/// Build a pipeline with the built-in compressors enabled by `config`.
pub fn default_pipeline() -> DefaultCompressionPipeline {
    DefaultCompressionPipeline::new(None, None)
}

/// Build a pipeline with a shared CCR store.
pub fn pipeline_with_ccr(ccr_store: Arc<dyn CcrStore>) -> DefaultCompressionPipeline {
    DefaultCompressionPipeline::with_ccr_store(ccr_store)
}

/// Convenience: compress a full message session in one call.
pub async fn compress_messages(
    messages: Vec<Message>,
    config: CompressConfig,
) -> Result<CompressedMessages> {
    let pipeline = if let Some(path) = &config.ccr_store_path {
        let ccr_store = Arc::new(
            crate::ccr::sqlite::SqliteCcrStore::open(path, 300)
                .map_err(|e| OghamError::StoreError(e.to_string()))?,
        );
        DefaultCompressionPipeline::with_ccr_store(ccr_store)
    } else {
        let ccr_store = Arc::new(crate::ccr::in_memory::InMemoryCcrStore::new());
        DefaultCompressionPipeline::with_ccr_store(ccr_store)
    };
    pipeline.run(&messages).await
}

/// Detect the content type of a string.
pub fn detect(content: &str) -> crate::detect::DetectionResult {
    crate::detect::detect_content_type(content)
}

/// Check if content is a JSON array of dictionaries.
pub fn is_json_dict_array(content: &str) -> bool {
    crate::detect::is_json_array_of_dicts(content)
}
