use ogham::{
    CompressConfig, Message, ccr::compute_key, compress_messages, default_pipeline, empty_pipeline,
    meta_keys,
};

fn json_array() -> String {
    let values: Vec<_> = (0..40)
        .map(|i| serde_json::json!({ "id": i, "name": format!("item_{i}"), "score": i }))
        .collect();
    serde_json::to_string(&values).unwrap()
}

#[tokio::test]
async fn default_pipeline_registers_builtins() {
    let pipeline = default_pipeline();
    assert!(pipeline.get_compressor("smart_crusher").is_some());
    assert!(pipeline.get_compressor("ast_code").is_some());
    assert!(pipeline.get_compressor("log_stripper").is_some());
    assert!(pipeline.get_compressor("semantic").is_some());
}

#[test]
fn empty_pipeline_has_no_builtins() {
    let pipeline = empty_pipeline();
    assert!(pipeline.get_compressor("smart_crusher").is_none());
    assert!(pipeline.get_compressor("semantic").is_none());
}

#[tokio::test]
async fn compress_config_respects_compressor_allowlist() {
    let input = json_array();
    let disabled = compress_messages(
        vec![Message::new("tool", input.clone())],
        CompressConfig {
            reversible: false,
            compressors: vec!["log_stripper".to_string()],
            ..CompressConfig::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(disabled.messages[0].content, input);
    assert_eq!(disabled.stats.compressor_used, "none");

    let enabled = compress_messages(
        vec![Message::new("tool", input.clone())],
        CompressConfig {
            reversible: false,
            compressors: vec!["smart_crusher".to_string()],
            ..CompressConfig::default()
        },
    )
    .await
    .unwrap();
    assert_ne!(enabled.messages[0].content, input);
    assert_eq!(enabled.stats.compressor_used, "smart_crusher");
}

#[tokio::test]
async fn compress_config_reversible_false_writes_no_ccr_marker() {
    let result = compress_messages(
        vec![Message::new("tool", json_array())],
        CompressConfig {
            reversible: false,
            compressors: vec!["smart_crusher".to_string()],
            ..CompressConfig::default()
        },
    )
    .await
    .unwrap();

    assert!(!result.messages[0].content.contains("<<ccr:"));
    assert!(!result.messages[0].metadata.contains_key(meta_keys::CCR_ID));
}

#[tokio::test]
async fn compress_config_reversible_true_records_ccr_id_metadata() {
    let input =
        "first paragraph\n\nsecond paragraph\n\nfirst paragraph\n\nfirst paragraph\n\n".repeat(16);
    let expected_id = compute_key(input.as_bytes());
    let result = compress_messages(
        vec![Message::new("user", input)],
        CompressConfig {
            reversible: true,
            compressors: vec!["semantic".to_string()],
            ..CompressConfig::default()
        },
    )
    .await
    .unwrap();

    assert_eq!(result.stats.compressor_used, "semantic");
    assert_eq!(
        result.messages[0].metadata.get(meta_keys::CCR_ID),
        Some(&expected_id)
    );
}

#[tokio::test]
async fn compress_config_use_cache_aligner_normalizes_messages() {
    let result = compress_messages(
        vec![Message::new("user", r#"  {"z":1,"a":2}  "#)],
        CompressConfig {
            reversible: false,
            use_cache_aligner: true,
            compressors: Vec::new(),
            ..CompressConfig::default()
        },
    )
    .await
    .unwrap();

    assert_eq!(result.messages[0].content, r#"{"a":2,"z":1}"#);
}
