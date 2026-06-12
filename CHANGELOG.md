# Changelog

All notable changes to Ogham are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/) (pre-1.0: minor
versions may contain breaking changes).

## [Unreleased]

## [0.2.2] - 2026-06-11

### Changed

- Replaced a machine-specific path in a golden-test fixture and
  genericized references to a downstream project in docs and doc
  comments. No behavior changes.
- First version available on crates.io: the 0.2.1 packages were
  deleted from the registry before this release and were never
  generally consumed.

## [0.2.1] - 2026-06-11

### Added

- **crates.io publishing**: releases now publish `ogham-core`, `ogham`,
  and `ogham-server` to crates.io (in dependency order) in addition to
  the GitHub release with binaries. Depend on the SDK with
  `ogham = "0.2"` instead of a git dependency.

## [0.2.0] - 2026-06-10

Informed by a June 2026 state-of-the-art review (Anthropic context editing,
ACON, Hermes/Claude Code production patterns).

### Added

- **Pair-safe dropping**: the budget cascade's `drop_old` step now removes
  an assistant tool call and its consecutive tool results as one atomic
  group — provider APIs reject orphaned tool results. A protected tool
  result (error or pinned) now also protects the assistant message that
  invoked it. The conversation summarize-band boundary is aligned the same
  way so draining old turns can never split a pair.
- **Anthropic server-side context editing adapter**
  (`ogham::providers::anthropic`): translates an `AgentPolicy` into the
  `context_management` / `clear_tool_uses_20250919` request fragment
  (beta header `context-management-2025-06-27`), so Claude-targeting hosts
  can delegate first-line clearing to the platform while Ogham covers
  other providers, reversibility, budgets, and summaries.
- **Agent-facing retrieval tool** (`ogham::tools`):
  `retrieve_tool_definition()` (provider-agnostic JSON Schema) and
  `handle_retrieve_call()` — a dispatcher that always returns
  model-readable text (content, not-found, or error notice) and never
  errors the agent loop. Tolerates being passed a full `<<ccr:HASH>>`
  marker instead of the bare hash.
- `meta_keys::TOOL_CALL_ID` (`ogham.tool_call_id`) for linking tool calls
  to their results in audits and tests.

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

[Unreleased]: https://github.com/signalbreak-labs/ogham/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/signalbreak-labs/ogham/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/signalbreak-labs/ogham/releases/tag/v0.1.0
