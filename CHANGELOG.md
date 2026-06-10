# Changelog

All notable changes to Ogham are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/) (pre-1.0: minor
versions may contain breaking changes).

## [Unreleased]

## [0.1.0] - 2026-06-10

Initial release.

### Added

- **`ogham-core`** — trait/type vocabulary: `Message` (with namespaced
  `ogham.*` metadata), `Compressor`, `CompressionPipeline`, `TokenCounter`,
  `Metrics`, `Observer`, `OghamError`.
- **Content-level compression** with automatic content-type detection
  (JSON arrays, build logs, source code, git diffs, HTML, search results,
  prose) routed to six compressors: `smart_crusher`, `log_stripper`,
  `ast_code`, `semantic`, `dedup_ref`, `toon`.
- **Reversible CCR storage** (`<<ccr:HASH>>` markers) with in-memory,
  SQLite (WAL), and fjall LSM-tree backends.
- **Agent-aware rules**: message classification, tool-result clearing to
  CCR stubs, hard protection for tool errors, system instructions, the
  latest user query, and pinned messages.
- **Token budget enforcement** with a graceful degradation cascade
  (`agent_rules → compress_middle → summarize_old → drop_old`) that fails
  closed with `BudgetExceeded` rather than overflowing.
- **Token counting**: calibrated heuristic counter everywhere; exact OpenAI
  counts behind the `tiktoken` feature; estimates carry an automatic safety
  margin in budget enforcement.
- **Structured summaries**: deterministic `ExtractiveSummarizer` (files,
  decisions, errors, next steps) and a `Summarizer` trait for LLM-backed
  implementations; anchored merging for incremental updates.
- **Prompt-cache support**: byte-stabilizing cache aligner and provider
  breakpoint annotation (`apply_cache_strategy`).
- **`ogham-server`** — embeddable Axum router (`/compress`, `/retrieve`,
  `/detect`, `/health`, `/stats`), configurable via `OGHAM_*` environment
  variables, binding `127.0.0.1:3000` by default.
- **Test suite**: 118 tests including fuzz (no-panic), LLM-safety
  invariants, golden-file regression, and needle-in-haystack probes;
  criterion benchmarks.

### Attribution

Design substantially derived from
[Headroom](https://github.com/chopratejas/headroom) (Apache-2.0) — see
[NOTICE](NOTICE).

[Unreleased]: https://github.com/signalbreak-labs/ogham/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/signalbreak-labs/ogham/releases/tag/v0.1.0
