//! Embeddable HTTP server for the [Ogham](https://github.com/signalbreak-labs/ogham)
//! context engineering SDK.
//!
//! Run standalone (`ogham-server` binary, configured via `OGHAM_*`
//! environment variables) or mount the router into an existing Axum
//! application:
//!
//! ```no_run
//! use ogham_server::{app_with_state, AppState};
//!
//! let router = axum::Router::new().nest("/ogham", app_with_state(AppState::new()));
//! ```
//!
//! Endpoints: `POST /compress`, `POST /retrieve`, `POST /detect`,
//! `GET /health`, `GET /stats`. Request/response shapes are documented
//! on [`app`]. The binary binds `127.0.0.1:3000` by default and never
//! listens on all interfaces unless explicitly configured.

use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use ogham::{
    CompressionPipeline, Message,
    ccr::{CcrStore, in_memory::InMemoryCcrStore},
    detect,
    pipeline::DefaultCompressionPipeline,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

/// Backend selection for the server's CCR store.
#[derive(Debug, Clone)]
pub enum CcrBackendConfig {
    InMemory,
    Sqlite {
        path: std::path::PathBuf,
        ttl_seconds: u64,
    },
    Fjall {
        path: std::path::PathBuf,
    },
}

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Default: 127.0.0.1:3000 (was 0.0.0.0 — do not listen on all
    /// interfaces by default).
    pub bind: std::net::SocketAddr,
    pub ccr: CcrBackendConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: std::net::SocketAddr::from(([127, 0, 0, 1], 3000)),
            ccr: CcrBackendConfig::InMemory,
        }
    }
}

/// Application state shared across handlers.
#[derive(Clone)]
pub struct AppState {
    pub start_time: Instant,
    pub request_count: Arc<Mutex<u64>>,
    pub pipeline: Arc<DefaultCompressionPipeline>,
    pub ccr_store: Arc<dyn CcrStore>,
}

impl AppState {
    pub fn new() -> Self {
        let ccr_store = Arc::new(InMemoryCcrStore::new());
        let pipeline = Arc::new(
            DefaultCompressionPipeline::builder()
                .ccr_store(ccr_store.clone())
                .align_cache()
                .build(),
        );
        Self {
            start_time: Instant::now(),
            request_count: Arc::new(Mutex::new(0)),
            pipeline,
            ccr_store,
        }
    }

    /// Build with any store.
    pub fn with_store(store: Arc<dyn CcrStore>) -> Self {
        let pipeline = Arc::new(
            DefaultCompressionPipeline::builder()
                .ccr_store(store.clone())
                .align_cache()
                .build(),
        );
        Self {
            start_time: Instant::now(),
            request_count: Arc::new(Mutex::new(0)),
            pipeline,
            ccr_store: store,
        }
    }

    pub async fn bump_requests(&self) {
        let mut guard = self.request_count.lock().await;
        *guard += 1;
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the MCP/REST router with shared state.
///
/// Request/response contracts:
///
/// ```jsonc
/// // POST /compress   request:  {"messages":[{"role":"user","content":"..."}]}
/// //                  response: {"messages":[...], "stats":{"original_tokens":N,
/// //                             "compressed_tokens":N,"ratio":F,"compressor_used":"..."}}
/// // POST /retrieve   request:  {"id":"<md5hex>"}
/// //                  response: {"found":true,"original":"..."} or {"found":false}
/// // POST /detect     request:  {"content":"..."}
/// //                  response: {"content_type":"json_array","confidence":0.9}
/// // GET  /health     {"status":"ok"}
/// // GET  /stats      {"uptime_seconds":N,"requests":N}
/// ```
pub fn app() -> Router {
    app_with_state(AppState::new())
}

/// Mountable router over caller-provided state — this is what host applications embed.
pub fn app_with_state(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/compress", post(compress_handler))
        .route("/retrieve", post(retrieve_handler))
        .route("/detect", post(detect_handler))
        .route("/stats", get(stats_handler))
        .with_state(state)
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

#[derive(Deserialize)]
struct CompressRequest {
    messages: Vec<Message>,
}

#[derive(Serialize)]
struct CompressResponse {
    messages: Vec<Message>,
    stats: Value,
}

async fn compress_handler(
    State(state): State<AppState>,
    Json(req): Json<CompressRequest>,
) -> Json<CompressResponse> {
    state.bump_requests().await;

    match state.pipeline.run(&req.messages).await {
        Ok(result) => Json(CompressResponse {
            messages: result.messages,
            stats: serde_json::to_value(&result.stats).unwrap_or_else(|_| serde_json::json!({})),
        }),
        Err(e) => Json(CompressResponse {
            // Fail-closed: return the originals unchanged rather than nothing.
            messages: req.messages,
            stats: serde_json::json!({"error": e.to_string()}),
        }),
    }
}

#[derive(Deserialize)]
struct RetrieveRequest {
    id: String,
}

#[derive(Serialize)]
struct RetrieveResponse {
    found: bool,
    original: Option<String>,
}

async fn retrieve_handler(
    State(state): State<AppState>,
    Json(req): Json<RetrieveRequest>,
) -> Json<RetrieveResponse> {
    state.bump_requests().await;
    let original = state.ccr_store.retrieve(&req.id).await.ok().flatten();
    Json(RetrieveResponse {
        found: original.is_some(),
        original,
    })
}

#[derive(Deserialize)]
struct DetectRequest {
    content: String,
}

#[derive(Serialize)]
struct DetectResponse {
    content_type: String,
    confidence: f64,
    metadata: Value,
}

async fn detect_handler(
    State(state): State<AppState>,
    Json(req): Json<DetectRequest>,
) -> Json<DetectResponse> {
    state.bump_requests().await;
    let result = detect(&req.content);
    Json(DetectResponse {
        content_type: result.content_type.as_str().to_string(),
        confidence: result.confidence,
        metadata: serde_json::to_value(&result.metadata).unwrap_or_else(|_| serde_json::json!({})),
    })
}

#[derive(Serialize)]
struct StatsResponse {
    uptime_seconds: u64,
    requests_total: u64,
}

async fn stats_handler(State(state): State<AppState>) -> Json<StatsResponse> {
    let requests = *state.request_count.lock().await;
    Json(StatsResponse {
        uptime_seconds: state.start_time.elapsed().as_secs(),
        requests_total: requests,
    })
}
