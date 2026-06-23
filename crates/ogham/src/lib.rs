//! # Ogham — LLM context engineering SDK
//!
//! Ogham compresses, prunes, and budgets LLM conversation context —
//! entirely in-process. No subprocesses, no network calls, no
//! background tasks.
//!
//! ## The two API levels
//!
//! **Content level** — compress individual payloads (tool outputs,
//! logs, code, JSON) with content-type detection and reversible CCR
//! markers:
//!
//! ```no_run
//! # async fn demo() -> ogham::Result<()> {
//! use ogham::{compress_messages, CompressConfig, Message};
//!
//! let out = compress_messages(
//!     vec![Message::new("tool", r#"[{"id":1},{"id":2}]"#)],
//!     CompressConfig::default(),
//! ).await?;
//! println!("{} -> {} tokens", out.stats.original_tokens, out.stats.compressed_tokens);
//! # Ok(()) }
//! ```
//!
//! **Conversation level** — agent-aware rules and token budgets over a
//! whole message history. The canonical order is:
//!
//! 1. [`agent::apply_agent_compression`] — clear stale successful tool
//!    results to retrievable CCR markers; never touch errors, system
//!    prompts, or the latest user query.
//! 2. [`budget::enforce_budget`] — escalate through compression,
//!    summarization, and dropping until the history fits a token
//!    budget, or fail closed with [`OghamError::BudgetExceeded`].
//! 3. [`cache_aligner::align_messages`] +
//!    [`cache_strategy::apply_cache_strategy`] — stabilize bytes and
//!    annotate provider cache breakpoints.
//!
//! ## Guarantees
//!
//! - **Fail-closed:** any internal error leaves the original content
//!   unchanged; oversized prompts return an error instead of being sent.
//! - **Deterministic:** same input + config ⇒ byte-identical output.
//! - **Reversible by default:** originals are stored in a [`ccr`] store
//!   (in-memory, SQLite, or fjall) and retrievable via
//!   `<<ccr:HASH>>` markers.
//! - **Honest token counting:** exact for OpenAI encodings with the
//!   `tiktoken` feature; calibrated estimates with an explicit safety
//!   margin otherwise (Claude tokenizers are not public).
//!
//! ## Feature flags
//!
//! - `ccr-sqlite` *(default)* — persistent SQLite CCR store
//!   (`ccr::sqlite::SqliteCcrStore`, pulls `rusqlite`). Required for
//!   `CompressConfig::ccr_store_path`.
//! - `ccr-fjall` *(default)* — embedded-KV CCR store
//!   (`ccr::fjall::FjallCcrStore`, pulls `fjall`).
//! - `tiktoken` — exact OpenAI token counts via
//!   `token_counter::TiktokenCounter` (adds the `tiktoken-rs` dependency).
//!
//! Build with `--no-default-features` for a lean dependency set: in-memory CCR
//! only, no `rusqlite`/`fjall`. Re-enable persistence with `ccr-sqlite` and/or
//! `ccr-fjall`.

pub mod adaptive_sizer;
pub mod agent;
pub mod budget;
pub mod cache_aligner;
pub mod cache_strategy;
pub mod ccr;
pub mod compact;
pub mod compressors;
pub mod conversation;
pub mod detect;
pub mod memory;
pub mod pipeline;
pub mod providers;
pub mod rich;
pub mod stats_math;
pub mod token_counter;
pub mod token_est;
pub mod tools;

pub use ogham_core::*;
pub use token_counter::{HeuristicCounter, counter_for_model};

use crate::ccr::CcrStore;
use crate::pipeline::{DEFAULT_COMPRESSORS, DefaultCompressionPipeline};
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
                "semantic".into(),
            ],
            ccr_store_path: None,
        }
    }
}

/// Build a pipeline with the default built-in compressors and an in-memory CCR store.
pub fn default_pipeline() -> DefaultCompressionPipeline {
    let ccr_store = Arc::new(crate::ccr::in_memory::InMemoryCcrStore::new());
    DefaultCompressionPipeline::with_builtin_compressors(Some(ccr_store), DEFAULT_COMPRESSORS)
        .expect("static default compressor names must be valid")
}

/// Build an empty pipeline for tests or custom compressor registration.
pub fn empty_pipeline() -> DefaultCompressionPipeline {
    DefaultCompressionPipeline::new(None, None)
}

/// Build a pipeline with a shared CCR store.
pub fn pipeline_with_ccr(ccr_store: Arc<dyn CcrStore>) -> DefaultCompressionPipeline {
    DefaultCompressionPipeline::with_ccr_store(ccr_store)
}

/// Open the persistent CCR store backing `ccr_store_path`.
///
/// Requires the `ccr-sqlite` feature; without it, a configured path is a
/// typed error rather than a silent fallback.
#[cfg(feature = "ccr-sqlite")]
fn open_path_ccr_store(path: &std::path::Path) -> Result<Arc<dyn CcrStore>> {
    Ok(Arc::new(
        crate::ccr::sqlite::SqliteCcrStore::open(path, 300)
            .map_err(|e| OghamError::StoreError(e.to_string()))?,
    ) as Arc<dyn CcrStore>)
}

#[cfg(not(feature = "ccr-sqlite"))]
fn open_path_ccr_store(path: &std::path::Path) -> Result<Arc<dyn CcrStore>> {
    Err(OghamError::StoreError(format!(
        "ccr_store_path ({}) requires the `ccr-sqlite` feature",
        path.display()
    )))
}

/// Convenience: compress a full message session in one call.
pub async fn compress_messages(
    messages: Vec<Message>,
    config: CompressConfig,
) -> Result<CompressedMessages> {
    let ccr_store = if config.reversible {
        match &config.ccr_store_path {
            Some(path) => Some(open_path_ccr_store(path)?),
            None => {
                Some(Arc::new(crate::ccr::in_memory::InMemoryCcrStore::new()) as Arc<dyn CcrStore>)
            }
        }
    } else {
        None
    };
    let mut pipeline =
        DefaultCompressionPipeline::with_builtin_compressors(ccr_store, &config.compressors)?
            .with_reversible(config.reversible);
    if config.use_cache_aligner {
        pipeline = pipeline.with_align_cache();
    }
    pipeline.run(&messages).await
}

pub use compact::{
    CachePlan, CachePolicy, CcrPolicy, CompactConfig, CompactResult, CompressionPolicy, FoldKind,
    FoldRecord, ProtectedReport, compact_conversation,
};
pub use rich::{RichCompressionPolicy, compress_rich_messages, restore_rich_message};

/// Detect the content type of a string.
pub fn detect(content: &str) -> crate::detect::DetectionResult {
    crate::detect::detect_content_type(content)
}

/// Check if content is a JSON array of dictionaries.
pub fn is_json_dict_array(content: &str) -> bool {
    crate::detect::is_json_array_of_dicts(content)
}
