pub mod summary;

pub use summary::{ExtractiveSummarizer, StructuredSummary};

use async_trait::async_trait;
use ogham_core::{Message, Result};

/// Summarizes a span of messages. The library ships ExtractiveSummarizer;
/// hosts may implement this with an LLM (e.g. Brehon).
#[async_trait]
pub trait Summarizer: Send + Sync {
    async fn summarize(
        &self,
        turns: &[Message],
        existing: Option<&StructuredSummary>,
    ) -> Result<StructuredSummary>;
}

#[async_trait]
impl Summarizer for ExtractiveSummarizer {
    async fn summarize(
        &self,
        turns: &[Message],
        existing: Option<&StructuredSummary>,
    ) -> Result<StructuredSummary> {
        Ok(self.summarize_sync(turns, existing))
    }
}
