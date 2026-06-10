use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ogham::compress_messages;
use ogham::compressors::log_stripper::{LogCompressor, LogCompressorConfig};
use ogham::compressors::smart_crusher::SmartCrusher;
use ogham::{CompressConfig, detect};
use ogham_core::Message;

fn build_input(size_kb: usize) -> String {
    let chunk = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ".repeat(20);
    let repeats = (size_kb * 1024).saturating_div(chunk.len()).max(1);
    chunk.repeat(repeats)
}

/// JSON array of dicts — the shape SmartCrusher actually routes on.
fn build_json_array(size_kb: usize) -> String {
    let mut items = Vec::new();
    let mut total = 2usize;
    let mut i = 0usize;
    while total < size_kb * 1024 {
        let item = format!(
            r#"{{"id":{i},"name":"item_{i}","status":"{}","latency_ms":{},"region":"us-east-1"}}"#,
            if i % 97 == 0 { "failed" } else { "ok" },
            (i * 37) % 500,
        );
        total += item.len() + 1;
        items.push(item);
        i += 1;
    }
    format!("[{}]", items.join(","))
}

/// Timestamped build-log lines — the shape LogCompressor actually handles.
fn build_log(size_kb: usize) -> String {
    let mut lines = Vec::new();
    let mut total = 0usize;
    let mut i = 0usize;
    while total < size_kb * 1024 {
        let line = format!(
            "2026-06-10T12:{:02}:{:02}Z {} module::worker: processed batch {} in {}ms",
            (i / 60) % 60,
            i % 60,
            match i % 50 {
                0 => "ERROR",
                1..=3 => "WARN",
                _ => "INFO",
            },
            i,
            (i * 13) % 900,
        );
        total += line.len() + 1;
        lines.push(line);
        i += 1;
    }
    lines.join("\n")
}

fn bench_detect(c: &mut Criterion) {
    let mut group = c.benchmark_group("detect");
    for size in &[1, 32, 256] {
        let input = build_input(*size);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}kb", size)),
            &input,
            |b, i| {
                b.iter(|| detect(black_box(i)));
            },
        );
    }
    group.finish();
}

fn bench_smart_crusher_json(c: &mut Criterion) {
    let mut group = c.benchmark_group("smart_crusher_json");
    let crusher = SmartCrusher::new();
    for size in &[1, 32, 256] {
        let input = build_json_array(*size);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}kb", size)),
            &input,
            |b, i| {
                b.iter(|| crusher.crush(black_box(i), "", 1.0));
            },
        );
    }
    group.finish();
}

fn bench_log_stripper(c: &mut Criterion) {
    let mut group = c.benchmark_group("log_stripper");
    let stripper = LogCompressor::new(LogCompressorConfig::default());
    for size in &[1, 32, 256] {
        let input = build_log(*size);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}kb", size)),
            &input,
            |b, i| {
                b.iter(|| stripper.compress(black_box(i), 1.0));
            },
        );
    }
    group.finish();
}

fn bench_pipeline_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_end_to_end");
    let rt = tokio::runtime::Runtime::new().unwrap();
    for size in &[1, 32, 256] {
        let input = build_input(*size);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}kb", size)),
            &input,
            |b, i| {
                b.iter(|| {
                    let msgs = vec![Message::new("tool", i.clone())];
                    rt.block_on(async {
                        compress_messages(msgs, CompressConfig::default())
                            .await
                            .unwrap()
                    });
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_detect,
    bench_smart_crusher_json,
    bench_log_stripper,
    bench_pipeline_end_to_end,
);
criterion_main!(benches);
