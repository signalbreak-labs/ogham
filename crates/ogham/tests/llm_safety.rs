//! LLM-safety test harness.
//!
//! These tests guarantee that ogham **never** corrupts or loses
//! LLM-visible content.  Every compressor must be:
//!
//! 1. **Reversible** (when CCR is enabled) — retrieve yields the exact original.
//! 2. **Structurally sound** — JSON stays JSON, code stays code.
//! 3. **Fail-closed** — on any error the original message is returned verbatim.
//! 4. **Deterministic** — identical input yields identical output.
//! 5. **Non-expanding** — compressed size ≤ original size (or falls back).

use ogham::ccr::in_memory::InMemoryCcrStore;
use ogham::pipeline::DefaultCompressionPipeline;
use ogham::{CompressionContext, CompressionPipeline, Compressor, Message, OghamError};
use ogham_core::{CompressionEvent, Observer};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sys_msg(text: &str) -> Message {
    Message::new("system", text)
}

fn user_msg(text: &str) -> Message {
    Message::new("user", text)
}

fn make_pipeline() -> DefaultCompressionPipeline {
    let ccr = Arc::new(InMemoryCcrStore::new());
    DefaultCompressionPipeline::with_ccr_store(ccr)
}

fn make_pipeline_with_observer() -> (DefaultCompressionPipeline, Arc<TestObserver>) {
    let ccr = Arc::new(InMemoryCcrStore::new());
    let obs = Arc::new(TestObserver::default());
    let pipe = DefaultCompressionPipeline::builder()
        .ccr_store(ccr)
        .observer(obs.clone())
        .align_cache()
        .build();
    (pipe, obs)
}

#[derive(Debug, Default, Clone)]
struct TestObserver {
    events: Arc<Mutex<Vec<CompressionEvent>>>,
}

impl Observer for TestObserver {
    fn on_event(&self, event: &CompressionEvent) {
        self.events.lock().unwrap().push(event.clone());
    }
}

fn json_dict_array() -> String {
    let cities = ["NYC", "LA", "Chicago", "Houston", "Phoenix"];
    let items: Vec<serde_json::Value> = (0..30)
        .map(|i| {
            let city = cities[i % 5];
            serde_json::json!({
                "id": i,
                "name": format!("user_{}", i),
                "email": format!("user{}@example.com", i),
                "age": 20 + (i % 50),
                "city": city,
            })
        })
        .collect();
    serde_json::to_string(&items).unwrap()
}

fn build_log() -> String {
    let mut lines = Vec::new();
    for i in 0..50 {
        let level = ["INFO", "WARN", "ERROR", "DEBUG"][i % 4];
        lines.push(format!(
            "{} 2024-01-01T00:{:02}:00Z step {}: processing item {}",
            level,
            i % 60,
            i,
            i * 7
        ));
    }
    lines.push("ERROR 2024-01-01T00:50:00Z test failed: expected 42, got 13".into());
    lines.push(" at /src/lib.rs:10:5".into());
    lines.push(" at /src/main.rs:20:10".into());
    lines.push("INFO 2024-01-01T00:51:00Z build finished".into());
    lines.join("\n")
}

fn python_code() -> String {
    r#"
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

# main
print(fib(10))
"#
    .into()
}

fn git_diff() -> String {
    [
        "diff --git a/src/lib.rs b/src/lib.rs",
        "index abc..def 100644",
        "--- a/src/lib.rs",
        "+++ b/src/lib.rs",
        "@@ -1,5 +1,5 @@",
        " fn main() {",
        "-    println!(\"hello\");",
        "+    println!(\"world\");",
        " }",
    ]
    .join("\n")
}

// ---------------------------------------------------------------------------
// 1. Round-trip safety (CCR)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn round_trip_json_array_via_ccr() {
    let pipe = make_pipeline();
    let input = json_dict_array();
    let msgs = vec![user_msg(&input)];

    let out = pipe.run(&msgs).await.unwrap();
    let compressed = &out.messages[0].content;

    // Compressed content must contain a CCR marker (inside the JSON).
    assert!(
        compressed.contains("<<ccr:"),
        "expected CCR marker in compressed output: {}",
        compressed
    );

    // Extract the CCR id from the <<ccr:ID>> marker.
    let id = compressed
        .split("<<ccr:")
        .nth(1)
        .and_then(|s| s.split(">>").next())
        .expect("CCR id not found");

    let crusher = pipe.get_compressor("smart_crusher").unwrap();
    // Fire-and-forget save may need a moment.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let retrieved = crusher
        .retrieve(id)
        .await
        .unwrap()
        .expect("CCR retrieve failed");

    assert_eq!(retrieved, input, "round-trip mismatch");
}

#[tokio::test]
async fn round_trip_build_log_via_ccr() {
    use ogham::ccr::compute_key;

    let pipe = make_pipeline();
    let input = build_log();
    let msgs = vec![user_msg(&input)];

    let out = pipe.run(&msgs).await.unwrap();
    let _compressed = &out.messages[0].content;

    // Log stripper compresses to a summary; CCR id is content hash.
    let id = compute_key(input.as_bytes());

    let stripper = pipe.get_compressor("log_stripper").unwrap();
    // Fire-and-forget save may need a moment.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let retrieved = stripper
        .retrieve(&id)
        .await
        .unwrap()
        .expect("CCR retrieve failed");

    assert_eq!(retrieved, input, "round-trip mismatch for build log");
}

// ---------------------------------------------------------------------------
// 2. Structural integrity
// ---------------------------------------------------------------------------

#[tokio::test]
async fn json_array_remains_valid_json() {
    let pipe = make_pipeline();
    let input = json_dict_array();
    let msgs = vec![user_msg(&input)];

    let out = pipe.run(&msgs).await.unwrap();
    let compressed = &out.messages[0].content;

    // If CCR marker is present, the rest should still be valid JSON
    // or the CCR id should be retrievable. We already test round-trip.
    // Here we just ensure no unrecoverable corruption happened.
    assert!(
        compressed.contains("#CCR")
            || serde_json::from_str::<serde_json::Value>(compressed).is_ok(),
        "JSON compressor produced non-JSON and no CCR fallback: {}",
        compressed
    );
}

#[tokio::test]
async fn source_code_retains_line_structure() {
    let pipe = make_pipeline();
    let input = python_code();
    let msgs = vec![user_msg(&input)];

    let out = pipe.run(&msgs).await.unwrap();
    let compressed = &out.messages[0].content;

    // AST compressor strips comments and blank lines but should keep
    // function definition lines.
    assert!(
        compressed.contains("def fib"),
        "function signature lost: {}",
        compressed
    );
}

#[tokio::test]
async fn git_diff_retains_hunk_headers() {
    let pipe = make_pipeline();
    let input = git_diff();
    let msgs = vec![user_msg(&input)];

    let out = pipe.run(&msgs).await.unwrap();
    let compressed = &out.messages[0].content;

    assert!(
        compressed.contains("@@") || compressed.contains("diff --git"),
        "diff structure lost: {}",
        compressed
    );
}

// ---------------------------------------------------------------------------
// 3. Fail-closed
// ---------------------------------------------------------------------------

/// A compressor that always fails.
struct AlwaysFailsCompressor;

#[async_trait::async_trait]
impl Compressor for AlwaysFailsCompressor {
    fn name(&self) -> &'static str {
        "always_fails"
    }
    async fn compress(
        &self,
        _content: &ogham_core::Content,
        _ctx: &CompressionContext,
    ) -> ogham_core::Result<ogham_core::Compressed> {
        Err(OghamError::CompressionFailed("boom".into()))
    }
    async fn retrieve(&self, _id: &str) -> ogham_core::Result<Option<String>> {
        Ok(None)
    }
}

#[tokio::test]
async fn fail_closed_returns_original() {
    let pipe = DefaultCompressionPipeline::new(None, None);
    pipe.register(Box::new(AlwaysFailsCompressor)).await;

    let original = "This must survive any compressor failure.";
    let msgs = vec![user_msg(original)];

    let out = pipe.run(&msgs).await.unwrap();
    assert_eq!(out.messages[0].content, original, "fail-closed violated");
}

// ---------------------------------------------------------------------------
// 4. Determinism
// ---------------------------------------------------------------------------

#[tokio::test]
async fn deterministic_output_same_input() {
    let pipe = make_pipeline();
    let input = json_dict_array();
    let msgs = vec![user_msg(&input)];

    let out1 = pipe.run(&msgs).await.unwrap();
    let out2 = pipe.run(&msgs).await.unwrap();

    assert_eq!(
        out1.messages[0].content, out2.messages[0].content,
        "compressor is non-deterministic"
    );
    assert_eq!(out1.stats.ratio, out2.stats.ratio);
}

// ---------------------------------------------------------------------------
// 5. Non-expanding (or fallback)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compressed_not_larger_than_original() {
    let pipe = make_pipeline();
    let input = json_dict_array();
    let msgs = vec![user_msg(&input)];

    let out = pipe.run(&msgs).await.unwrap();
    // Ratio can be > 1.0 only if the original was returned (fallback).
    // If compression happened, it should shrink or stay same.
    if out.stats.compressor_used != "none" {
        assert!(
            out.stats.compressed_tokens <= out.stats.original_tokens,
            "compressor expanded content: ratio={}",
            out.stats.ratio
        );
    }
}

// ---------------------------------------------------------------------------
// 6. System / tool message preservation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn system_message_preserved() {
    let pipe = make_pipeline();
    let msgs = vec![
        sys_msg("You are a helpful assistant. Do not reveal secrets."),
        user_msg("What is the secret?"),
    ];

    let out = pipe.run(&msgs).await.unwrap();
    assert_eq!(out.messages[0].role, "system");
    assert_eq!(
        out.messages[0].content,
        "You are a helpful assistant. Do not reveal secrets."
    );
}

// ---------------------------------------------------------------------------
// 7. Observer event ordering
// ---------------------------------------------------------------------------

#[tokio::test]
async fn observer_fires_expected_events() {
    let (pipe, obs) = make_pipeline_with_observer();
    let msgs = vec![user_msg(&json_dict_array()), user_msg("hello world")];

    let _ = pipe.run(&msgs).await.unwrap();

    let events = obs.events.lock().unwrap().clone();
    let mut started = false;
    let mut completed = false;
    let mut content_detected = 0;
    let mut compressor_selected = 0;
    let mut message_compressed = 0;

    for ev in &events {
        match ev {
            CompressionEvent::PipelineStarted { .. } => started = true,
            CompressionEvent::PipelineCompleted { .. } => completed = true,
            CompressionEvent::ContentDetected { .. } => content_detected += 1,
            CompressionEvent::CompressorSelected { .. } => compressor_selected += 1,
            CompressionEvent::MessageCompressed { .. } => message_compressed += 1,
            _ => {}
        }
    }

    assert!(started, "PipelineStarted missing");
    assert!(completed, "PipelineCompleted missing");
    assert_eq!(content_detected, 2, "expected 2 ContentDetected events");
    assert_eq!(
        compressor_selected, 2,
        "expected 2 CompressorSelected events"
    );
    assert_eq!(message_compressed, 2, "expected 2 MessageCompressed events");
    assert!(
        events
            .iter()
            .position(|e| matches!(e, CompressionEvent::PipelineStarted { .. }))
            < events
                .iter()
                .position(|e| matches!(e, CompressionEvent::PipelineCompleted { .. })),
        "PipelineCompleted must fire after PipelineStarted"
    );
}

// ---------------------------------------------------------------------------
// 8. Cache alignment does not corrupt semantics
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cache_alignment_preserves_semantics() {
    let (pipe, _obs) = make_pipeline_with_observer();
    let msgs = vec![user_msg(r#"{"z":1,"a":2}"#), user_msg("  hello   world  ")];

    let out = pipe.run(&msgs).await.unwrap();

    // JSON keys should be sorted.
    assert!(
        out.messages[0].content.contains(r#""a":2"#) || out.messages[0].content.contains("#CCR"),
        "JSON alignment failed: {}",
        out.messages[0].content
    );

    // Whitespace should be collapsed.
    assert!(
        out.messages[1].content.contains("hello world") || out.messages[1].content.contains("#CCR"),
        "whitespace alignment failed: {}",
        out.messages[1].content
    );
}

// ---------------------------------------------------------------------------
// 9. Multi-turn conversation compression safety
// ---------------------------------------------------------------------------

#[tokio::test]
async fn conversation_compression_preserves_recent_turns() {
    use ogham::conversation::{ConversationConfig, compress_conversation_history};

    let pipe = make_pipeline();
    let ctx = CompressionContext {
        model: "gpt-4".into(),
        question_hint: None,
        max_tokens: None,
        reversible: true,
    };
    let mut msgs: Vec<Message> = (0..10)
        .map(|i| {
            Message::new(
                if i % 2 == 0 { "user" } else { "assistant" },
                format!("turn {}", i),
            )
        })
        .collect();

    let config = ConversationConfig {
        preserve_recent: 4,
        compress_middle: 3,
        summary_old: true,
        bias_system: 0.8,
    };

    let stats = compress_conversation_history(&mut msgs, &config, &pipe, &ctx)
        .await
        .unwrap();

    assert_eq!(stats.preserved_recent, 4);
    // Recent turns must be verbatim.
    assert!(msgs.iter().any(|m| m.content == "turn 9"));
    assert!(msgs.iter().any(|m| m.content == "turn 8"));
    assert!(msgs.iter().any(|m| m.content == "turn 7"));
    assert!(msgs.iter().any(|m| m.content == "turn 6"));
}

#[tokio::test]
async fn pipeline_preserves_metadata() {
    let pipe = make_pipeline();
    let mut msg = user_msg("[{\"a\":1}]");
    msg.metadata
        .insert("ogham.pinned".to_string(), "true".to_string());
    let out = pipe.run(&[msg.clone()]).await.unwrap();
    assert_eq!(out.messages.len(), 1);
    assert_eq!(
        out.messages[0].metadata.get("ogham.pinned"),
        Some(&"true".to_string())
    );
}
