use ogham::{CompressConfig, compress_messages};
use ogham_core::Message;
use std::path::PathBuf;

fn run_golden(name: &str) {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let input_path = manifest
        .join("tests/golden/inputs")
        .join(format!("{}.txt", name));
    let expected_path = manifest
        .join("tests/golden/expected")
        .join(format!("{}.txt", name));

    let input = std::fs::read_to_string(&input_path)
        .unwrap_or_else(|e| panic!("failed to read input {}: {}", input_path.display(), e));

    let rt = tokio::runtime::Runtime::new().unwrap();
    let output = rt.block_on(async {
        let result =
            compress_messages(vec![Message::new("tool", input)], CompressConfig::default())
                .await
                .unwrap_or_else(|e| panic!("compression failed for {}: {}", name, e));
        result.messages.into_iter().next().unwrap().content
    });

    if std::env::var("UPDATE_GOLDEN").is_ok() {
        std::fs::write(&expected_path, &output).unwrap_or_else(|e| {
            panic!(
                "failed to write expected {}: {}",
                expected_path.display(),
                e
            )
        });
        return;
    }

    let expected = std::fs::read_to_string(&expected_path)
        .unwrap_or_else(|e| {
            panic!(
                "golden mismatch for {}; if intentional, rerun with UPDATE_GOLDEN=1; also failed to read expected file: {}",
                name, e
            )
        });

    assert_eq!(
        output, expected,
        "golden mismatch for {}; if intentional, rerun with UPDATE_GOLDEN=1",
        name
    );
}

#[test]
fn golden_json_array() {
    run_golden("json_array");
}

#[test]
fn golden_build_log() {
    run_golden("build_log");
}

#[test]
fn golden_source_code() {
    run_golden("source_code");
}

#[test]
fn golden_git_diff() {
    run_golden("git_diff");
}

#[test]
fn golden_prose() {
    run_golden("prose");
}
