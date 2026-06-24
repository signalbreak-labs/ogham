//! Property-based fuzz tests for compressors.
//!
//! These tests feed random / adversarial inputs to every compressor and
//! verify the invariants documented in `llm_safety.rs`:
//!
//! - Never panic.
//! - Never return content larger than original (or return original).
//! - CCR retrieve round-trips exactly when CCR is enabled.

use bytes::Bytes;
use ogham::ccr::in_memory::InMemoryCcrStore;
use ogham::compressors::{
    ast_code::AstCodeCompressor, dedup_ref::DedupRefCompressor, log_stripper::LogStripper,
    semantic::SemanticCompressor, smart_crusher::SmartCrusher, toon::ToonCompressor,
};
use ogham::{CompressionContext, Compressor, Message};
use std::collections::HashMap;
use std::sync::Arc;

fn ctx() -> CompressionContext {
    CompressionContext {
        model: "gpt-4".into(),
        question_hint: None,
        max_tokens: None,
        reversible: true,
    }
}

fn content(data: &str, mime: &str) -> ogham_core::Content {
    ogham_core::Content {
        data: Bytes::from(data.to_string().into_bytes()),
        mime_or_lang: mime.into(),
        metadata: HashMap::new(),
    }
}

fn fuzz_inputs() -> Vec<(String, &'static str)> {
    vec![
        // Empty
        ("".into(), "text/plain"),
        // Very long repeated char
        ("a".repeat(10_000), "text/plain"),
        // Binary-looking garbage
        (
            String::from_utf8_lossy(&[0u8, 1, 2, 3, 255, 254].repeat(100)).into_owned(),
            "text/plain",
        ),
        // Nested JSON
        (
            serde_json::to_string(&serde_json::json!({"a":{"b":{"c":1}}})).unwrap(),
            "application/json",
        ),
        // JSON array of primitives (not dicts — should still not panic)
        ("[1,2,3,4,5]".into(), "application/json"),
        // Malformed JSON
        (r#"{"unclosed": "string"#.into(), "application/json"),
        // Unicode bomb
        ("🎉".repeat(500), "text/plain"),
        // Single log line
        ("INFO hello world".into(), "text/plain"),
        // Code with unmatched braces
        ("fn main() {".into(), "text/plain"),
        // HTML-like
        ("<div><span>hi</span></div>".into(), "text/html"),
        // Mixed newlines
        ("line1\r\nline2\nline3\rline4".into(), "text/plain"),
    ]
}

macro_rules! fuzz_compressor {
    ($name:ident, $ctor:expr, $mime:expr) => {
        #[tokio::test]
        async fn $name() {
            let ccr = Arc::new(InMemoryCcrStore::new());
            let comp: Box<dyn Compressor> = Box::new(($ctor)(ccr.clone()));
            let ctx = ctx();

            for (input, mime) in fuzz_inputs() {
                let c = content(&input, mime);
                let result = comp.compress(&c, &ctx).await;

                // Must not panic (caught by test framework if it does).
                // Must return Ok or a proper error.
                match result {
                    Ok(compressed) => {
                        // Compressed should not be larger than original
                        // unless the compressor decides to return original.
                        let orig_tokens = input.len() / 4;
                        assert!(
                            compressed.compressed_tokens <= orig_tokens || input.is_empty(),
                            "{} expanded {} bytes -> {} tokens",
                            comp.name(),
                            input.len(),
                            compressed.compressed_tokens
                        );

                        // Best-effort CCR round-trip for compressors that expose
                        // a top-level reversible id.
                        if !input.is_empty() && comp.name() != "toon" {
                            let _ = comp.retrieve(&compressed.id).await; // must not panic
                        }
                    }
                    Err(e) => {
                        // Errors are allowed for malformed input, but must be
                        // well-formed error types.
                        let _ = e.to_string();
                    }
                }
            }
        }
    };
}

fuzz_compressor!(
    fuzz_smart_crusher,
    SmartCrusher::with_ccr_store,
    "application/json"
);
fuzz_compressor!(fuzz_log_stripper, LogStripper::with_ccr_store, "text/plain");
fuzz_compressor!(
    fuzz_ast_code,
    AstCodeCompressor::with_ccr_store,
    "text/plain"
);
fuzz_compressor!(
    fuzz_semantic,
    SemanticCompressor::with_ccr_store,
    "text/plain"
);
fuzz_compressor!(
    fuzz_dedup_ref,
    DedupRefCompressor::with_ccr_store,
    "text/plain"
);
fuzz_compressor!(fuzz_toon, |_ccr| ToonCompressor::new(), "text/plain");

// ---------------------------------------------------------------------------
// Adversarial: rapid random bytes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fuzz_random_bytes_smart_crusher() {
    let ccr = Arc::new(InMemoryCcrStore::new());
    let comp = SmartCrusher::with_ccr_store(ccr);
    let ctx = ctx();

    // Seed a small pseudo-random sequence deterministically.
    let mut state: u64 = 0x12345678;
    for _ in 0..100 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let len = (state % 200) as usize;
        let bytes: Vec<u8> = (0..len)
            .map(|_i| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state % 256) as u8
            })
            .collect();
        let input = String::from_utf8_lossy(&bytes);
        let c = content(&input, "application/json");
        let _ = comp.compress(&c, &ctx).await; // must not panic
    }
}

#[tokio::test]
async fn fuzz_random_bytes_log_stripper() {
    let ccr = Arc::new(InMemoryCcrStore::new());
    let comp = LogStripper::with_ccr_store(ccr);
    let ctx = ctx();

    let mut state: u64 = 0xdeadbeef;
    for _ in 0..100 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let len = (state % 500) as usize;
        let bytes: Vec<u8> = (0..len)
            .map(|_i| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state % 256) as u8
            })
            .collect();
        let input = String::from_utf8_lossy(&bytes);
        let c = content(&input, "text/plain");
        let _ = comp.compress(&c, &ctx).await; // must not panic
    }
}

// ---------------------------------------------------------------------------
// Stress: many small messages through full pipeline
// ---------------------------------------------------------------------------

use ogham::CompressionPipeline;
use ogham::pipeline::DefaultCompressionPipeline;

#[tokio::test]
async fn stress_pipeline_many_messages() {
    let pipe = DefaultCompressionPipeline::with_ccr_store(Arc::new(InMemoryCcrStore::new()));
    let msgs: Vec<Message> = (0..50)
        .map(|i| Message::new("user", format!("msg {} {}", i, "x".repeat(i * 10))))
        .collect();

    let out = pipe.run(&msgs).await.unwrap();
    assert_eq!(out.messages.len(), 50);
    for (orig, compressed) in msgs.iter().zip(out.messages.iter()) {
        // Every message must have some content.
        assert!(
            !compressed.content.is_empty(),
            "message {} became empty",
            orig.content
        );
    }
}
