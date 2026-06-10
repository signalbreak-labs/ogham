use thiserror::Error;

#[derive(Error, Debug)]
pub enum OghamError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("compression failed: {0}")]
    CompressionFailed(String),

    #[error("retrieve failed: {0}")]
    RetrieveFailed(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("unsupported content type: {0}")]
    UnsupportedContentType(String),

    #[error("store error: {0}")]
    StoreError(String),

    #[error("token budget exceeded: need {needed} tokens but limit is {limit}")]
    BudgetExceeded { needed: usize, limit: usize },
}

pub type Result<T> = std::result::Result<T, OghamError>;
