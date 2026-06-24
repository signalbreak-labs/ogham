# Ogham Roadmap

Ogham is a pure-Rust, in-process SDK for LLM context engineering: deterministic
compression, reversible clearing (CCR), agent-aware pruning, token budgeting, and
cache-aware prompt shaping. It runs entirely in the host process — no sidecar, no
network calls, no background tasks.

This document describes where Ogham is, where it is going, and the principles
that constrain how it gets there. It is a living plan, not a contract; see
`CHANGELOG.md` for what has actually shipped.

## What Ogham is — and is not

Ogham aims to be:

1. A pure-Rust dependency a host application can vendor or path-depend on without
   starting a daemon.
2. A deterministic compression and compaction engine with fail-closed semantics.
3. A fold/CCR engine that returns durable, auditable records — not just opaque
   marker strings.
4. A provider-aware cache planner that protects stable prefixes and emits
   provider-specific annotations.
5. A bridge between generic chat messages and richer, block-structured agent
   messages.
6. A token-budget gate that returns `BudgetExceeded` rather than emitting an
   oversized prompt.
7. A library with a small default dependency footprint and feature gates for the
   heavier backends and compressors.

Ogham is deliberately **not**:

1. A replacement for provider-side prompt caching or context editing.
2. A default LLM-summarization service (model-assisted compression is opt-in).
3. A hidden background-task system that marks content retrievable before storage
   succeeds.
4. A lossy serializer of a host's rich message format.
5. A universal tokenizer crate or provider HTTP client.

## Current status

The safety and honesty foundation is in place:

- **Honest public API.** `default_pipeline()` registers the built-in compressors
  with an in-memory CCR store; `empty_pipeline()` is the explicit empty
  constructor. `CompressConfig` honors every field (`reversible`,
  `use_cache_aligner`, `compressors` allowlist, `ccr_store_path`).
- **Fail-closed CCR.** Reversible saves are awaited before a marker is emitted; a
  save failure keeps the original content rather than producing an unretrievable
  marker. There are no detached background saves.
- **Agent-aware rules.** Stale successful tool results are cleared to retrievable
  CCR stubs; errors, system prompts, the latest user query, pinned messages, and
  a token-budgeted protected tail are never touched.
- **Budget cascade.** Compression → summarization → pair-safe dropping, failing
  closed with `BudgetExceeded` instead of overflowing. Assistant/tool-call pairs
  are dropped atomically so no orphaned tool results reach the provider.
- **First-class compaction results.** `compact_conversation()` returns
  `FoldRecord`s (cleared / compressed / summarized / dropped), a protected-tail
  report, optional budget/agent reports, and provider cache annotations — so
  hosts never have to scrape marker strings to infer what happened.
- **Stateful incremental compaction.** `ContextSession` compacts append-only:
  push turns, compact only the active tail, freeze already-folded messages. Each
  turn's work is proportional to new content, the fold ledger is append-only, and
  the finalized region stays byte-stable for prompt-cache reuse.
- **Durable CCR lifecycle.** A non-evicting store (`InMemoryCcrStore::unbounded`)
  keeps referenced originals durable; `ccr::referenced_ccr_ids` computes the live
  set; and `ContextSession`'s `RetentionPolicy` bounds stub growth by evicting
  the oldest finalized stubs and garbage-collecting only unreferenced originals.
- **Honest token counting.** Heuristic estimates carry an explicit safety margin;
  exact OpenAI counts are available behind the `tiktoken` feature. Estimated
  counts are never presented as exact.
- **Searchable fold recall.** `ogham::recall::RecallIndex` is a deterministic
  BM25 keyword index over folded content, addressable by CCR id, so a host can
  search folds by relevance to recover the ids to retrieve — closing the
  exact-id-only gap in reversible CCR. `ContextSession` maintains one
  automatically (index on fold, drop on GC). Pure and deterministic: no
  embeddings, no network.
- **Structured fold tags.** Every `FoldRecord` carries a deterministic
  `FoldTags { tool_names, error_classes, file_paths }` extracted offline during
  compaction, and `RecallIndex::find_by_tag` filters folds by typed field — so a
  host can fetch "all folds from the `shell` tool" or "all folds with a `panic`"
  without a free-text query.
- **Block-aware rich content.** `ogham_core::content` defines a host-neutral
  block-structured `RichMessage`; `ogham::compact_rich` runs the agent/budget
  cascade over rich messages with structure preserved for kept messages and
  reversible flat text for folded ones — so a host never flattens tool calls,
  images, or references to a JSON string.
- **Provider cache planning.** `providers::{openai, gemini, anthropic}` emit
  provider-shaped cache plans (stable-prefix reports, content-keyed candidates,
  native Anthropic block rendering with per-model thresholds); Ogham emits the
  plan, the host owns the HTTP call.
- **Lean dependency footprint.** The default build is in-memory CCR only; the
  persistent SQLite/fjall backends are opt-in features and store binary CCR
  payloads natively (no hex envelope), a CI guard keeps the default tree lean.

## Roadmap

Status legend: **Done** items have shipped (see `CHANGELOG.md`); **Planned**
items are not yet implemented. Everything in the near- and mid-term tiers below
is done; the longer-term tier is the remaining (optional) work.

### ✅ Near-term: correctness and dependency hygiene — done

- **Slim the default feature set.** Done (0.4, breaking). `ccr-sqlite`
  (`rusqlite`) and `ccr-fjall` (`fjall`) are no longer in the default set, so the
  default `ogham` build is lean (in-memory CCR only); consumers opt into a
  persistent store with `features = ["ccr-sqlite"]` / `["ccr-fjall"]`.
  `ogham-server` enables both, and a CI guard keeps the default dependency tree
  free of the heavy backends. The lighter deps (`regex` in content detection,
  `flate2` in adaptive sizing) sit in core paths and stay non-optional; the TOON
  encoder is pure Rust and pulls no backend.
- **Broaden focus-hint steering.** Done. Every built-in compressor now biases
  retention on the `CompactConfig.focus` hint via the shared `compressors::focus`
  module: `SmartCrusher` (JSON-array records), `LogStripper` (log lines, scored
  below errors so focus never evicts a diagnostic), `AstCodeCompressor` (code
  lines kept full-length), and `SemanticCompressor` (paragraphs kept full and
  un-deduplicated). Protected content is still never overridden, and an
  empty/noise-only hint is byte-identical to the no-hint path.

### ✅ Mid-term: richer host content and provider planning — done

- **Host-neutral rich content model.** Done. `ogham_core::content` defines a
  block-structured `RichMessage` (text / thinking / image / tool-use /
  tool-result / reference) that round-trips losslessly through serde, with an
  explicit marked lossy flattening to the text `Message`; and
  `ogham::rich::compress_rich_messages` compresses the text *inside* blocks while
  preserving tool ids and non-text blocks, with message-level reversible undo, so
  hosts no longer flatten to a JSON string at all. `ogham::compact_rich` is the
  high-level entry point: it folds block-aware compaction into the agent/budget
  cascade and returns the same audit records (fold records, protected report,
  cache plan) as `compact_conversation`, with structure preserved for kept
  messages and reversible flat text for folded ones.
- **Block-aware CCR payloads.** Done. `ccr::CcrPayload` plus
  `CcrStore::save_payload` / `retrieve_payload` let a host store and restore
  exact structured originals; the in-memory store uses a self-describing text
  envelope, while the SQLite backend stores payloads in native BLOB +
  media-type/metadata columns and the fjall backend in a compact
  length-prefixed binary frame — no hex envelope, so large binary payloads cost
  their real size. Both native backends fall back to the shared text decoder for
  plain `save`s and legacy envelopes, so existing stores keep working
  (a re-open runs an idempotent column migration).
- **Provider cache planning.** Done. OpenAI stable-prefix reports
  (`providers::openai`), Gemini cache candidates (`providers::gemini`), an
  Anthropic `cache_control` request renderer (`providers::anthropic`), and
  stable-prefix accounting folded into `CompactResult`'s `CachePlan`.
  `render_cache_control_rich` renders native Anthropic tool-use / tool-result /
  image blocks from the rich content model (no flattening), and
  `min_cacheable_prefix_tokens(model)` gives per-model Anthropic cache
  thresholds. `CachePolicy::Gemini` is now first-class in the integrated
  `CachePlan`: content-keyed (no inline breakpoints — matching Gemini's
  `CachedContent` model), thresholded by `gemini::MIN_CACHEABLE_PREFIX_TOKENS`,
  and annotated with refresh guidance, rather than emitting a generic plan with
  a warning. Ogham emits plans — hosts own the HTTP calls and auth.
- **Token-counter reporting.** Done. `TokenCountKind` distinguishes `Exact`,
  `Estimated { method, safety_margin }`, and `ProviderReported`; `count_kind()`
  reports it per counter; `BudgetReport` surfaces the count kind and applied
  margin. Model-family selection lives in `counter_for_model` (calibrated by
  prefix). Possible follow-up: a `CountedTokens { tokens, kind }` return type for
  per-count provenance.

### ⬜ Longer-term: optional power features — planned (not yet implemented)

- **Planned — selective structured encodings.** Wire the existing (registered
  but not default-routed) TOON encoder into content-type-gated selective use:
  apply only to uniform arrays after validation, always preserving the CCR
  original and benchmarking against tokenizer counts rather than byte length.
- **Planned — model-assisted compression boundary.** Define a
  `ModelAssistedCompressor` trait / feature boundary for aggressive (e.g.
  LLMLingua-style) compression so the default path stays deterministic and
  zero-network. Require evaluation gates before enabling semantic token dropping
  on tool, error, or system content.
- **Retrieval-friendly metadata.** Largely done. A deterministic BM25 recall
  index (`ogham::recall`) and structured fold tags (`ogham::fold_tags`:
  `tool_names` / `error_classes` / `file_paths`, queryable via
  `RecallIndex::find_by_tag`) have shipped. *Planned remainder:* broaden typed
  tagging to summaries and add more categories (e.g. symbol identifiers) as
  evaluation shows they help. Ogham integrates with retrieval; it does not
  become a vector database.

## Context landscape

Local compression is one layer of a larger stack: provider prompt caching,
provider context editing, retrieval for large static corpora, structured
summaries, and cache-prefix stability all matter. Two consequences shape the
roadmap:

- **Protect stable prefixes, not just token count.** A pass that lowers tokens
  but breaks a cacheable prefix can increase real cost and latency. Cache savings
  and token savings are reported separately.
- **Optimize for recall per token, not ratio alone.** For coding and agent
  workloads the valuable working set is the current task, recent tool results,
  active diffs, constraints, and unresolved errors. Old successful tool outputs
  are the ideal CCR/fold candidates; instructions and errors stay raw.

## Design principles

| Principle | Rationale |
|---|---|
| Pure Rust, in-process. | An optimal dependency should not require an HTTP sidecar. |
| Deterministic default compression. | Agent context carries instructions and errors where probabilistic compression is risky. |
| Provider plans, not provider clients. | Provider APIs change quickly; hosts own HTTP calls and auth. |
| Fold records are first-class. | UIs, ledgers, and undo must not depend on scraping marker strings. |
| Preserve rich blocks. | Agent semantics live below the message text. |
| Feature-gate heavy components. | Dependency weight matters across many consumers. |
| Versioned, collision-resistant IDs. | Durable content addresses should not rely on a broken hash. |

## Risk register

| Risk | Severity | Why it matters | Mitigation |
|---|---:|---|---|
| Unretrievable CCR markers | Critical | Breaks undo and the fail-closed promise. | Await saves; emit no marker on save failure. |
| Rich-content loss | Critical | Tool calls, images, and references can collapse to text. | Block-aware content model and payload CCR. |
| Config/docs mismatch | High | Users depend on behavior that is not implemented. | Honest defaults, named APIs, config tests. |
| Cache damage despite token savings | High | Provider cost/latency can worsen if stable prefixes change. | `CachePlan` and stable-prefix protection. |
| Over-aggressive semantic compression | High | Loses instructions/errors agents need. | Deterministic default; model-assisted compressors are opt-in. |
| Weak content-address hash | Medium | Poor durable/public content-address story. | Versioned, collision-resistant IDs with legacy retrieval. |
| Dependency bloat | Medium | Makes Ogham less optimal as a dependency. | Feature-gate stores, compressors, and server. |
| Estimated counts presented as exact | Medium | Budget errors and overconfident stats. | Count-kind reporting and provider-specific counters. |
