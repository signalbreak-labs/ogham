//! Metrics and observability traits for ogham.
//!
//! Downstream systems (agent runtimes, hosts) implement [`Metrics`] and [`Observer`]
//! to forward compression events into their own telemetry (tracing spans,
//! audit logs, Prometheus, etc.).

use std::sync::Arc;

/// Per-compressor breakdown within a pipeline run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PerCompressorStats {
    pub name: String,
    pub content_type: String,
    pub original_tokens: usize,
    pub compressed_tokens: usize,
    pub ratio: f64,
    pub latency_ms: u64,
}

/// Extended stats returned by the pipeline.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PipelineStats {
    pub original_tokens: usize,
    pub compressed_tokens: usize,
    pub ratio: f64,
    pub compressor_used: String,
    pub content_type: String,
    pub latency_ms: u64,
    pub message_count: usize,
    pub per_compressor: Vec<PerCompressorStats>,
    pub ccr_retrievals: usize,
    pub ccr_hits: usize,
}

/// Structured compression events for observability.
#[derive(Debug, Clone)]
pub enum CompressionEvent {
    /// Pipeline started processing a batch.
    PipelineStarted { message_count: usize },
    /// Content type was detected for a message.
    ContentDetected {
        content_type: String,
        confidence: f64,
    },
    /// A compressor was selected for a message.
    CompressorSelected { name: String, content_type: String },
    /// A single message was compressed.
    MessageCompressed {
        compressor: String,
        original_tokens: usize,
        compressed_tokens: usize,
        latency_ms: u64,
    },
    /// The full pipeline run completed.
    PipelineCompleted { stats: PipelineStats },
    /// A CCR retrieve was attempted.
    RetrieveAttempt { id: String },
    /// CCR retrieve succeeded.
    RetrieveHit { id: String, latency_ms: u64 },
    /// CCR retrieve failed or missed.
    RetrieveMiss { id: String },
    /// Original content was saved to CCR store.
    CcrSaved { id: String, bytes: usize },
    /// CCR entry was deleted or expired.
    CcrExpired { id: String },
    /// An error occurred during compression.
    Error { stage: String, message: String },
}

/// Metrics sink for aggregate counters / histograms.
pub trait Metrics: Send + Sync {
    /// Record a single compression operation.
    fn record_compress(
        &self,
        compressor: &str,
        content_type: &str,
        original_tokens: usize,
        compressed_tokens: usize,
        latency_ms: u64,
    );

    /// Record a CCR retrieve attempt.
    fn record_retrieve(&self, hit: bool, latency_ms: u64);

    /// Record current CCR store size (called periodically).
    fn record_ccr_store_size(&self, entries: usize);

    /// Record which compressor was selected for a content type.
    fn record_routing_decision(&self, content_type: &str, compressor: &str);
}

/// Event observer for streaming compression diagnostics.
pub trait Observer: Send + Sync {
    fn on_event(&self, event: &CompressionEvent);
}

// Blanket impl: Arc<dyn Metrics> is Metrics
impl<M: Metrics + ?Sized> Metrics for Arc<M> {
    fn record_compress(
        &self,
        compressor: &str,
        content_type: &str,
        original_tokens: usize,
        compressed_tokens: usize,
        latency_ms: u64,
    ) {
        (**self).record_compress(
            compressor,
            content_type,
            original_tokens,
            compressed_tokens,
            latency_ms,
        )
    }
    fn record_retrieve(&self, hit: bool, latency_ms: u64) {
        (**self).record_retrieve(hit, latency_ms)
    }
    fn record_ccr_store_size(&self, entries: usize) {
        (**self).record_ccr_store_size(entries)
    }
    fn record_routing_decision(&self, content_type: &str, compressor: &str) {
        (**self).record_routing_decision(content_type, compressor)
    }
}

// Blanket impl: Arc<dyn Observer> is Observer
impl<O: Observer + ?Sized> Observer for Arc<O> {
    fn on_event(&self, event: &CompressionEvent) {
        (**self).on_event(event)
    }
}

/// No-op metrics sink — used when downstream does not care.
#[derive(Debug, Clone, Copy)]
pub struct NoopMetrics;

impl Metrics for NoopMetrics {
    fn record_compress(
        &self,
        _compressor: &str,
        _content_type: &str,
        _original_tokens: usize,
        _compressed_tokens: usize,
        _latency_ms: u64,
    ) {
    }
    fn record_retrieve(&self, _hit: bool, _latency_ms: u64) {}
    fn record_ccr_store_size(&self, _entries: usize) {}
    fn record_routing_decision(&self, _content_type: &str, _compressor: &str) {}
}

/// No-op observer — used when downstream does not care.
#[derive(Debug, Clone, Copy)]
pub struct NoopObserver;

impl Observer for NoopObserver {
    fn on_event(&self, _event: &CompressionEvent) {}
}

/// In-memory metrics collector for testing.
#[derive(Debug, Clone, Default)]
pub struct TestMetrics {
    pub compress_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub retrieve_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub routing_calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Metrics for TestMetrics {
    fn record_compress(
        &self,
        _compressor: &str,
        _content_type: &str,
        _original_tokens: usize,
        _compressed_tokens: usize,
        _latency_ms: u64,
    ) {
        self.compress_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    fn record_retrieve(&self, _hit: bool, _latency_ms: u64) {
        self.retrieve_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    fn record_ccr_store_size(&self, _entries: usize) {}
    fn record_routing_decision(&self, _content_type: &str, _compressor: &str) {
        self.routing_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// In-memory observer collector for testing.
#[derive(Debug, Clone, Default)]
pub struct TestObserver {
    pub events: std::sync::Arc<std::sync::Mutex<Vec<CompressionEvent>>>,
}

impl Observer for TestObserver {
    fn on_event(&self, event: &CompressionEvent) {
        self.events.lock().unwrap().push(event.clone());
    }
}
