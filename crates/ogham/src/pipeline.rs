use async_trait::async_trait;
use ogham_core::{
    CompressedMessages, CompressionContext, CompressionEvent, CompressionPipeline,
    CompressionStats, Compressor, Message, Metrics, NoopMetrics, NoopObserver, Observer,
    PerCompressorStats, PipelineStats, Result,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{info, warn};

use crate::cache_aligner::align_messages;
use crate::ccr::CcrStore;
use crate::compressors::{
    ast_code::AstCodeCompressor, dedup_ref::DedupRefCompressor, log_stripper::LogStripper,
    semantic::SemanticCompressor, smart_crusher::SmartCrusher, toon::ToonCompressor,
};
use crate::detect::{ContentType, detect_content_type};

/// The default compression pipeline.
///
/// Detects content type per-message then routes to the best-matching
/// compressor.  Supports reversible CCR, metrics, observer hooks, and
/// KV-cache alignment.
///
/// # Fail-closed design
/// If any compressor returns an error the pipeline keeps the **original**
/// message content unchanged.  This guarantees we never corrupt LLM calls.
pub struct DefaultCompressionPipeline {
    pub(crate) compressors: Arc<std::sync::RwLock<Vec<Arc<dyn Compressor>>>>,
    ccr_store: Option<Arc<dyn CcrStore>>,
    metrics: Arc<dyn Metrics>,
    observer: Arc<dyn Observer>,
    align_cache: bool,
}

impl DefaultCompressionPipeline {
    pub fn new(metrics: Option<Arc<dyn Metrics>>, observer: Option<Arc<dyn Observer>>) -> Self {
        Self {
            compressors: Arc::new(std::sync::RwLock::new(Vec::new())),
            ccr_store: None,
            metrics: metrics.unwrap_or_else(|| Arc::new(NoopMetrics)),
            observer: observer.unwrap_or_else(|| Arc::new(NoopObserver)),
            align_cache: false,
        }
    }

    pub fn with_ccr_store(ccr_store: Arc<dyn CcrStore>) -> Self {
        let mut pipeline = Self::new(None, None);
        pipeline.ccr_store = Some(ccr_store.clone());
        let compressors: Vec<Arc<dyn Compressor>> = vec![
            Arc::new(SmartCrusher::with_ccr_store(ccr_store.clone())),
            Arc::new(LogStripper::with_ccr_store(ccr_store.clone())),
            Arc::new(AstCodeCompressor::with_ccr_store(ccr_store.clone())),
            Arc::new(SemanticCompressor::with_ccr_store(ccr_store.clone())),
            Arc::new(DedupRefCompressor::with_ccr_store(ccr_store.clone())),
            Arc::new(ToonCompressor::new()),
        ];
        pipeline.compressors = Arc::new(std::sync::RwLock::new(compressors));
        pipeline
    }

    /// Build a pipeline with all optional components.
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::default()
    }

    /// Enable KV-cache alignment (sort JSON keys, normalise whitespace).
    pub fn with_align_cache(mut self) -> Self {
        self.align_cache = true;
        self
    }

    /// Register a compressor at setup time.
    pub async fn register(&self, compressor: Box<dyn Compressor>) {
        let mut guard = self.compressors.write().unwrap_or_else(|e| e.into_inner());
        info!("registering compressor: {}", compressor.name());
        guard.push(Arc::from(compressor));
    }

    /// Look up a registered compressor by name.
    pub fn get_compressor(&self, name: &str) -> Option<Arc<dyn Compressor>> {
        let guard = self.compressors.read().unwrap_or_else(|e| e.into_inner());
        guard.iter().find(|c| c.name() == name).cloned()
    }

    /// Compress a single message, returning the (content, per-msg stats).
    /// Fail-closed: on error the original text is returned unchanged.
    /// Events are emitted through `observer` (the per-run counting wrapper).
    async fn compress_one(
        &self,
        msg: &Message,
        observer: &dyn Observer,
    ) -> (String, Option<PerCompressorStats>, Option<String>) {
        let content_str = &msg.content;
        let original_tokens = crate::token_est::count_tokens(content_str);

        let detected = detect_content_type(content_str);
        observer.on_event(&CompressionEvent::ContentDetected {
            content_type: detected.content_type.as_str().to_string(),
            confidence: detected.confidence,
        });

        let comp_name: Option<&'static str> = match detected.content_type {
            ContentType::JsonArray => Some("smart_crusher"),
            ContentType::BuildOutput => Some("log_stripper"),
            ContentType::SourceCode | ContentType::GitDiff => Some("ast_code"),
            ContentType::SearchResults | ContentType::Html | ContentType::PlainText => {
                Some("semantic")
            }
        };

        let start = std::time::Instant::now();

        // Clone the Arc out so no lock guard is held across an await.
        let comp = comp_name.and_then(|n| {
            let guard = self.compressors.read().unwrap_or_else(|e| e.into_inner());
            guard.iter().find(|c| c.name() == n).cloned()
        });

        {
            if let Some(comp) = comp {
                observer.on_event(&CompressionEvent::CompressorSelected {
                    name: comp.name().to_string(),
                    content_type: detected.content_type.as_str().to_string(),
                });
                self.metrics
                    .record_routing_decision(detected.content_type.as_str(), comp.name());

                let ctx = CompressionContext {
                    model: "default".to_string(),
                    question_hint: None,
                    max_tokens: None,
                    reversible: self.ccr_store.is_some(),
                };
                let content = ogham_core::Content {
                    data: bytes::Bytes::from(content_str.clone().into_bytes()),
                    mime_or_lang: detected.content_type.as_str().to_string(),
                    metadata: std::collections::HashMap::new(),
                };

                match comp.compress(&content, &ctx).await {
                    Ok(compressed) => {
                        let latency_ms = start.elapsed().as_millis() as u64;
                        let compressed_tokens = compressed.compressed_tokens;
                        let out = String::from_utf8_lossy(&compressed.data).to_string();

                        self.metrics.record_compress(
                            comp.name(),
                            detected.content_type.as_str(),
                            original_tokens,
                            compressed_tokens,
                            latency_ms,
                        );
                        observer.on_event(&CompressionEvent::MessageCompressed {
                            compressor: comp.name().to_string(),
                            original_tokens,
                            compressed_tokens,
                            latency_ms,
                        });

                        let stats = PerCompressorStats {
                            name: comp.name().to_string(),
                            content_type: detected.content_type.as_str().to_string(),
                            original_tokens,
                            compressed_tokens,
                            ratio: if original_tokens > 0 {
                                compressed_tokens as f64 / original_tokens as f64
                            } else {
                                1.0
                            },
                            latency_ms,
                        };
                        (out, Some(stats), Some(comp.name().to_string()))
                    }
                    Err(e) => {
                        warn!(
                            "compression failed ({}): {}, keeping original",
                            comp.name(),
                            e
                        );
                        observer.on_event(&CompressionEvent::Error {
                            stage: comp.name().to_string(),
                            message: e.to_string(),
                        });
                        (content_str.clone(), None, None)
                    }
                }
            } else {
                (content_str.clone(), None, None)
            }
        }
    }
}

impl Default for DefaultCompressionPipeline {
    fn default() -> Self {
        Self::new(None, None)
    }
}

/// Internal observer that counts CCR-related events for honest stats reporting.
struct CountingObserver {
    inner: Arc<dyn Observer>,
    retrieve_attempts: AtomicUsize,
    retrieve_hits: AtomicUsize,
    ccr_saves: AtomicUsize,
}

impl Observer for CountingObserver {
    fn on_event(&self, event: &CompressionEvent) {
        match event {
            CompressionEvent::RetrieveAttempt { .. } => {
                self.retrieve_attempts.fetch_add(1, Ordering::Relaxed);
            }
            CompressionEvent::RetrieveHit { .. } => {
                self.retrieve_hits.fetch_add(1, Ordering::Relaxed);
            }
            CompressionEvent::CcrSaved { .. } => {
                self.ccr_saves.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        self.inner.on_event(event);
    }
}

#[async_trait]
impl CompressionPipeline for DefaultCompressionPipeline {
    fn add_compressor(&mut self, compressor: Box<dyn Compressor>) {
        let mut guard = self.compressors.write().unwrap_or_else(|e| e.into_inner());
        info!("registering compressor: {}", compressor.name());
        guard.push(Arc::from(compressor));
    }

    async fn run(&self, messages: &[Message]) -> Result<CompressedMessages> {
        let pipeline_start = std::time::Instant::now();
        info!("running default pipeline on {} messages", messages.len());

        let counting_observer = Arc::new(CountingObserver {
            inner: self.observer.clone(),
            retrieve_attempts: AtomicUsize::new(0),
            retrieve_hits: AtomicUsize::new(0),
            ccr_saves: AtomicUsize::new(0),
        });

        // All events for this run flow through the counting wrapper (which
        // forwards to the configured observer), so the CCR counts reported in
        // PipelineStats reflect events that actually fired.
        counting_observer.on_event(&CompressionEvent::PipelineStarted {
            message_count: messages.len(),
        });

        let mut working = messages.to_vec();
        if self.align_cache {
            align_messages(&mut working);
        }

        let mut compressed_messages: Vec<Message> = Vec::with_capacity(working.len());
        let mut total_original = 0usize;
        let mut total_compressed = 0usize;
        let mut last_compressor = "none".to_string();
        let mut per_compressor_stats: Vec<PerCompressorStats> = Vec::new();

        for msg in &working {
            let (compressed_content, stats, _comp_name) =
                self.compress_one(msg, counting_observer.as_ref()).await;

            total_original += crate::token_est::count_tokens(&msg.content);
            if let Some(s) = stats {
                total_compressed += s.compressed_tokens;
                last_compressor = s.name.clone();
                per_compressor_stats.push(s);
            } else {
                total_compressed += crate::token_est::count_tokens(&msg.content);
            }

            compressed_messages.push(Message {
                role: msg.role.clone(),
                content: compressed_content,
                metadata: msg.metadata.clone(),
            });
        }

        let ccr_retrievals = counting_observer.retrieve_attempts.load(Ordering::Relaxed);
        let ccr_hits = counting_observer.retrieve_hits.load(Ordering::Relaxed);

        let ratio = if total_original > 0 {
            total_compressed as f64 / total_original as f64
        } else {
            1.0
        };
        let latency_ms = pipeline_start.elapsed().as_millis() as u64;

        let pipeline_stats = PipelineStats {
            original_tokens: total_original,
            compressed_tokens: total_compressed,
            ratio,
            compressor_used: last_compressor.clone(),
            content_type: "mixed".to_string(),
            latency_ms,
            message_count: messages.len(),
            per_compressor: per_compressor_stats.clone(),
            ccr_retrievals,
            ccr_hits,
        };

        counting_observer.on_event(&CompressionEvent::PipelineCompleted {
            stats: pipeline_stats.clone(),
        });

        Ok(CompressedMessages {
            messages: compressed_messages,
            stats: CompressionStats {
                original_tokens: total_original,
                compressed_tokens: total_compressed,
                ratio,
                compressor_used: last_compressor,
            },
        })
    }
}

/// Builder for ergonomic pipeline construction.
#[derive(Default)]
pub struct PipelineBuilder {
    ccr_store: Option<Arc<dyn CcrStore>>,
    metrics: Option<Arc<dyn Metrics>>,
    observer: Option<Arc<dyn Observer>>,
    align_cache: bool,
}

impl PipelineBuilder {
    pub fn ccr_store(mut self, store: Arc<dyn CcrStore>) -> Self {
        self.ccr_store = Some(store);
        self
    }
    pub fn metrics(mut self, metrics: Arc<dyn Metrics>) -> Self {
        self.metrics = Some(metrics);
        self
    }
    pub fn observer(mut self, observer: Arc<dyn Observer>) -> Self {
        self.observer = Some(observer);
        self
    }
    pub fn align_cache(mut self) -> Self {
        self.align_cache = true;
        self
    }
    pub fn build(self) -> DefaultCompressionPipeline {
        let mut pipeline = DefaultCompressionPipeline::new(self.metrics, self.observer);
        if let Some(store) = self.ccr_store {
            pipeline = pipeline.with_ccr_store_reuse(store);
        }
        if self.align_cache {
            pipeline.align_cache = true;
        }
        pipeline
    }
}

// Helper to reuse CCR store in builder
impl DefaultCompressionPipeline {
    fn with_ccr_store_reuse(mut self, ccr_store: Arc<dyn CcrStore>) -> Self {
        self.ccr_store = Some(ccr_store.clone());
        let compressors: Vec<Arc<dyn Compressor>> = vec![
            Arc::new(SmartCrusher::with_ccr_store(ccr_store.clone())),
            Arc::new(LogStripper::with_ccr_store(ccr_store.clone())),
            Arc::new(AstCodeCompressor::with_ccr_store(ccr_store.clone())),
            Arc::new(SemanticCompressor::with_ccr_store(ccr_store.clone())),
            Arc::new(DedupRefCompressor::with_ccr_store(ccr_store.clone())),
            Arc::new(ToonCompressor::new()),
        ];
        self.compressors = Arc::new(std::sync::RwLock::new(compressors));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ogham_core::{Compressed, CompressionEvent, Observer};
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    struct NoopCompressor;

    #[async_trait]
    impl Compressor for NoopCompressor {
        fn name(&self) -> &'static str {
            "noop"
        }
        async fn compress(
            &self,
            _content: &ogham_core::Content,
            _ctx: &CompressionContext,
        ) -> Result<Compressed> {
            Ok(Compressed {
                id: "id".into(),
                data: bytes::Bytes::new(),
                original_tokens: 0,
                compressed_tokens: 0,
            })
        }
        async fn retrieve(&self, _id: &str) -> Result<Option<String>> {
            Ok(None)
        }
    }

    #[test]
    fn add_compressor_never_drops() {
        let mut pipeline = DefaultCompressionPipeline::new(None, None);
        pipeline.add_compressor(Box::new(NoopCompressor));
        assert!(pipeline.get_compressor("noop").is_some());
    }

    #[test]
    fn get_compressor_unknown_returns_none() {
        let pipeline = DefaultCompressionPipeline::new(None, None);
        assert!(pipeline.get_compressor("nope").is_none());
    }

    #[derive(Default)]
    struct StatsObserver {
        ccr_hits: AtomicUsize,
    }

    impl Observer for StatsObserver {
        fn on_event(&self, event: &CompressionEvent) {
            if let CompressionEvent::PipelineCompleted { stats } = event {
                self.ccr_hits.store(stats.ccr_hits, Ordering::Relaxed);
            }
        }
    }

    #[tokio::test]
    async fn ccr_stats_not_fabricated() {
        let obs = Arc::new(StatsObserver::default());
        let pipeline = DefaultCompressionPipeline::builder()
            .observer(obs.clone())
            .build();
        let msgs = vec![Message::new("user", "hello world")];
        let out = pipeline.run(&msgs).await.unwrap();
        assert_eq!(out.stats.compressed_tokens, out.stats.original_tokens);
        // No CCR store configured → hits must be 0.
        assert_eq!(obs.ccr_hits.load(Ordering::Relaxed), 0);
    }
}
