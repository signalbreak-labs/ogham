# Changelog

All notable changes to Ogham are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/) (pre-1.0: minor
versions may contain breaking changes).

## [Unreleased]

### Added

- Focus-hint steering across all built-in compressors. Previously only
  `SmartCrusher` consumed `CompactConfig.focus`; now `LogStripper`,
  `AstCodeCompressor`, and `SemanticCompressor` do too, via the shared
  `compressors::focus` module. `LogStripper` retains log lines matching the
  hint (scored below errors/fails, so focus never evicts a diagnostic),
  `AstCodeCompressor` keeps matching code lines at full length instead of
  truncating them, and `SemanticCompressor` keeps matching paragraphs full and
  un-deduplicated. An empty/noise-only hint is byte-identical to the no-hint
  path. (Also hardened: the code/semantic truncation now slices on a UTF-8 char
  boundary, fixing a latent multibyte panic.)
- Structured fold tags (`ogham::fold_tags`): every `FoldRecord` now carries a
  deterministic `FoldTags { tool_names, error_classes, file_paths }`, extracted
  offline during compaction from the fold's original messages (tool names from
  `metadata[TOOL_NAME]`, error classes from the same patterns the agent cascade
  uses — a parity test guards drift — and a conservative file-path scan). This
  adds the typed-field axis to recall: `RecallHit` carries the tags and
  `RecallIndex::find_by_tag(FoldTagKind, value)` returns folds by tool/error/path
  (case-insensitive) without a free-text query, so a host can ask "all folds from
  the `shell` tool" or "all folds with a `panic`". `ContextSession` indexes tags
  automatically. Re-exports `FoldTags`, `FoldTagKind`, `extract_fold_tags`.
- Searchable fold recall (`ogham::recall`): a deterministic BM25 keyword index
  over folded content, addressable by CCR id. Reversible CCR is exact-id-only —
  you can retrieve an original only if you still hold its `<<ccr:HASH>>` marker —
  so `RecallIndex` lets a host (or agent) `search()` folded content by relevance
  to recover the ids to `retrieve()`. It is pure and deterministic (no
  embeddings, no network); `extract_terms` splits path-like and identifier-like
  tokens (`src/auth/login.rs` also matches `auth`/`login`/`rs`; `parseToolResult`
  also matches `parse`/`tool`/`result`). `ContextSession` maintains one
  automatically — each compaction indexes the original text of newly folded
  content under its CCR id and drops entries whose originals are
  garbage-collected — exposed via `ContextSession::recall()`. Re-exports
  `RecallHit`, `RecallIndex`, and `extract_terms`.
- Durable CCR retention and stub-eviction: `InMemoryCcrStore::unbounded()` is a
  non-evicting store (so a referenced original is never silently dropped by
  capacity/TTL); `ccr::referenced_ccr_ids()` collects the CCR ids a message list
  still references (markers + metadata); and `ContextSession` gains a
  `RetentionPolicy` (`KeepAll` default, or `EvictFinalized { max_finalized,
  evict_originals }`) that bounds finalized-stub growth and garbage-collects the
  CCR originals of evicted stubs — never one a live marker still references.
  `SessionStep` gains an `evicted` field listing the GC'd ids.
- `ContextSession` (`ogham::session`): stateful, incremental conversation
  compaction. `push()` appends turns; `compact()` folds only the active tail and
  freezes already-folded messages (marked finalized + pinned), so each turn does
  work proportional to new content instead of recompacting the whole history.
  The fold ledger is append-only (stable undo/UI references) and the finalized
  region is byte-stable (preserves provider prompt-cache reuse). Adds
  `SessionConfig` and `SessionStep`.
- Honest token-count provenance: `TokenCountKind`
  (`Exact` / `Estimated { method, safety_margin }` / `ProviderReported`) and a
  `TokenCounter::count_kind()` default method, so a counter declares how its
  counts are produced. `HeuristicCounter` and the tiktoken counter report their
  method and recommended safety margin.
- `compact_rich()` plus `CompactRichConfig`/`CompactRichResult`: the block-aware
  analogue of `compact_conversation()`. It runs block-aware text compression and
  then the agent/budget cascade over `Vec<RichMessage>`, returning
  structure-preserving output — messages the cascade keeps retain their tool ids
  and non-text blocks; messages it folds become reversible flat text — plus fold
  records, protected-tail evidence, optional budget/agent reports, and a cache
  plan. Kept block-compressed messages restore to exact blocks via
  `restore_rich_message`.
- `ogham::rich`: block-aware compression. `compress_rich_messages()` compresses
  the bulky text *inside* `RichMessage` blocks (routing each text payload through
  the content-type compressors) while preserving tool-call ids, non-text blocks
  (images/references), error tool results, and roles — so a host never flattens
  structured content to a JSON string. Reversibility is message-level: each
  rewritten message's original blocks are stored as a `CcrPayload` and tagged
  with `meta_keys::CCR_ID`, and `restore_rich_message()` returns the exact
  original. Saves are awaited and fail closed.
- Block-aware CCR payloads: `ccr::CcrPayload` (media type + bytes + metadata)
  with `CcrStore::save_payload` / `retrieve_payload` default methods, so every
  store can persist and restore exact structured originals (e.g. serialized
  `RichMessage` blocks) for lossless undo. The default impl wraps the payload in
  a self-describing envelope over the text store (UTF-8 verbatim, binary
  hex-encoded); `retrieve_payload` degrades a plain string, including JSON that
  merely collides with the envelope marker key, to a `text/plain` payload.
- `ogham_core::content`: a host-neutral rich message model — `RichMessage`,
  `MessageContent` (text or blocks), and `ContentBlock`
  (text/thinking/image/tool-use/tool-result/reference). It round-trips
  losslessly through serde, so a host can map its structured messages in without
  flattening to a JSON string. `RichMessage::to_flat_lossy()` renders blocks to
  text for the text pipeline and marks the result with `META_FLATTENED` so the
  lossy path is explicit. Re-exported from `ogham`.
- Provider cache planning across `providers`:
  - `providers::openai` — `stable_prefix_report()` reports the cacheable
    stable-prefix boundary, whether it clears OpenAI's ~1024-token auto-cache
    threshold, and a deterministic `prompt_cache_key()` (no invented request
    fields; OpenAI caching is automatic).
  - `providers::gemini` — `cache_candidate()` reports the explicit-cache
    candidate prefix span, token estimate, and a deterministic `content_id` to
    detect when a Gemini `CachedContent` must be refreshed.
  - `providers::anthropic::render_cache_control()` renders messages into the
    Anthropic `system`/`messages` request parts, attaching
    `cache_control: ephemeral` to annotated blocks.
  - `providers::content_key()` — shared deterministic content identity for a
    message span.
  All are advisory pure data-structure builders; Ogham never calls a provider.
- `compact_conversation()` plus `CompactConfig`/`CompactResult` and the
  `FoldRecord`, `FoldKind`, `ProtectedReport`, `CachePlan`, `CompressionPolicy`,
  `CcrPolicy`, and `CachePolicy` types: a high-level conversation compaction API
  that returns auditable fold records (cleared/compressed/summarized/dropped),
  a protected-tail report, optional budget/agent reports, provider cache
  annotations, and warnings — so hosts no longer scrape `<<ccr:...>>` markers.
- `ogham::empty_pipeline()` and
  `DefaultCompressionPipeline::with_builtin_compressors()` for explicit,
  allowlist-driven pipeline construction.

### Changed

- Conversation compression now fully honors the `PINNED` contract: a message
  marked `meta_keys::PINNED` is never rewritten — not by middle-band compression
  and not by the old-band summary/drop (previously only clearing and budget-drop
  respected it, so a pinned message could be compressed or summarized, and in a
  session a finalized summary stub could be recursively re-summarized).
- A tool result that is cleared and then dropped within the same budget pass now
  keeps its CCR id in the `Dropped` `FoldRecord` (the cascade threads the dropped
  stub through to fold-record building; `BudgetReport` gains a `dropped` field),
  so the audit pointer to the stored original is no longer orphaned.
- Dropped fold records keep CCR provenance without counting removed stubs as
  emitted replacement tokens, and `ContextSession` disambiguates repeated
  content-addressed fold ids so its append-only ledger has stable unique event
  references.
- `compact_rich` now runs the agent/budget cascade on the verbatim conversation
  FIRST and block-compresses only the kept, non-protected messages afterward.
  This fixes two correctness bugs found by audit: (1) the cascade no longer
  clears/recompresses already-compressed content, so a cleared/folded message's
  CCR id and fold record resolve to the exact verbatim original (previously they
  resolved to a lossy projection and `restore_rich_message` could orphan the
  real payload); (2) `compact_rich` now recounts the true size of the emitted
  rich output (counting opaque image/tool-input block bytes the flat projection
  hides) and fails closed with `BudgetExceeded`, instead of returning an
  over-budget payload while reporting that it fit.
- `BudgetReport` now carries `count_kind` and the `safety_margin` actually
  applied, so a report never presents an estimate as exact. The budget cascade
  derives its margin from the counter's `count_kind` (behavior-preserving: exact
  counts use `0.0`, estimates use the counter's declared margin, e.g. `0.05`).
- `CompactResult.cache_plan` (`CachePlan`) now reports stable-prefix accounting:
  `stable_prefix_messages`, `stable_prefix_tokens`, `cacheable`, a content-keyed
  `content_key` (set for OpenAI/Gemini/Generic, `None` for Anthropic), and
  `notes`. `CachePolicy::None` reports no stable prefix and clears stale cache
  annotations, while Anthropic planning replaces old breakpoint annotations
  instead of accumulating them. (Additive fields on a returned struct.)
- The heavy embedded CCR stores are now feature-gated: `ccr-sqlite` (`rusqlite`)
  and `ccr-fjall` (`fjall`). Both remain in the default feature set, so existing
  builds are unchanged; `cargo build --no-default-features` now yields a lean,
  in-memory-only dependency set that excludes `rusqlite` and `fjall`. Calling
  `compress_messages` with `ccr_store_path` set without `ccr-sqlite` returns a
  typed `StoreError` instead of failing to compile. The unused `aho-corasick`
  dependency was dropped.
- `AgentPolicy::keep_recent_assistant` is now enforced: under budget pressure the
  compression cascade preserves at least the span covering the most-recent N
  assistant replies, keeping them raw instead of compressing them in the middle
  band. Previously the field was accepted but ignored.
- CCR content addresses (`ccr::compute_key`) are now versioned, collision-resistant
  BLAKE3 keys of the form `b3:<32 hex>` instead of bare MD5. The `b3:` tag lets the
  hash scheme evolve unambiguously; stores key on the literal id, so content saved
  under an older scheme stays retrievable. Emitted `<<ccr:...>>` markers change
  accordingly.
- The focus / question hint (`CompactConfig.focus`,
  `DefaultCompressionPipeline::with_question_hint()`) is now consumed:
  `SmartCrusher` boosts records whose serialized form matches the hint so they
  survive sampling of large JSON arrays, end to end through `compact_conversation`
  and the budget cascade. An empty hint leaves output unchanged, and protected
  content (system prompts, errors, latest user query, protected tail) is never
  overridden. Other built-in compressors still ignore the hint.
- `ogham::default_pipeline()` now registers the default built-in compressors
  with an in-memory CCR store instead of returning an empty pipeline. Use
  `ogham::empty_pipeline()` (or `DefaultCompressionPipeline::default()`) for the
  previous empty behavior.
- `compress_messages()` now honors every `CompressConfig` field: `reversible`
  (suppresses CCR construction and `<<ccr:...>>` markers when false),
  `use_cache_aligner`, the `compressors` allowlist, and `ccr_store_path`
  (opened only when `reversible` is true).
- Compressor CCR saves are now awaited and fail-closed: a save error keeps the
  original message content instead of being silently dropped by a detached
  `tokio::spawn`.
- Reversible pipeline compression now annotates rewritten messages with
  `metadata["ogham.ccr_id"]`, giving `FoldRecord::ccr_id` a durable top-level
  restore key even when compressed text has no embedded CCR marker.

## [0.3.0] - 2026-06-12

### Added

- `AgentPolicy::protected_tail_tokens`, an optional token-budgeted suffix
  guard for coding-agent transcripts. When set, Ogham keeps every message
  overlapping the estimated trailing token window byte-for-byte across agent
  clearing and budget enforcement; `None` preserves 0.2.x behavior.

### Changed

- Relaxed the `rusqlite` dependency requirement to `>=0.38, <0.41`; the
  workspace test suite passes at both range edges.

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

[Unreleased]: https://github.com/signalbreak-labs/ogham/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/signalbreak-labs/ogham/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/signalbreak-labs/ogham/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/signalbreak-labs/ogham/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/signalbreak-labs/ogham/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/signalbreak-labs/ogham/releases/tag/v0.1.0
