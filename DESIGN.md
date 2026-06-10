# Context-Compress: Implementation Design & Build Plan

> **RENAME (2026-06-10):** the project is now **Ogham**. Crate mapping:
> `context-compress-core` → `ogham-core`, `context-compress` → `ogham`,
> `context-compress-server` → `ogham-server`; error type
> `ContextCompressError` → `OghamError`; metadata key prefix `cc.` → `ogham.`;
> server env vars `CC_*` → `OGHAM_*`. This document predates the rename;
> apply the mapping when reading.

**Version:** 2.0
**Date:** 2026-06-10
**Status:** Authoritative build plan (all work packages completed; kept as the engineering record)
**Audience:** An implementing engineer or code-generation model. Every work package below is
self-contained: exact file paths, exact signatures, exact behavior, exact tests, exact
done-criteria. Do not invent APIs that are not specified here.

---

## How to Use This Document

1. Read §1 (Ground Truth) — it describes code that **already exists and passes tests**.
   Do not re-implement anything in §1.
2. Read §2 (Implementer Rules) — these are hard constraints on every change.
3. Execute work packages §WP-0 through §WP-9 **in order**. Each WP ends with a
   Definition of Done. Do not start the next WP until the current one's DoD passes.
4. After every WP run, in the workspace root:
   ```bash
   cargo fmt --all
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   ```
   All three must succeed before a WP is considered done.

---

## Table of Contents

1. [Ground Truth: What Already Exists](#1-ground-truth-what-already-exists)
2. [Implementer Rules (Hard Constraints)](#2-implementer-rules-hard-constraints)
3. [Corrections vs. Design v1.0](#3-corrections-vs-design-v10)
4. [Target Architecture](#4-target-architecture)
5. [Work Packages](#5-work-packages)
   - [WP-0: Repository Hygiene](#wp-0-repository-hygiene)
   - [WP-1: Bug Fixes in Existing Code](#wp-1-bug-fixes-in-existing-code)
   - [WP-2: Token Counting](#wp-2-token-counting)
   - [WP-3: Message Metadata](#wp-3-message-metadata)
   - [WP-4: Agent-Aware Compression Rules](#wp-4-agent-aware-compression-rules)
   - [WP-5: Token Budget Enforcement](#wp-5-token-budget-enforcement)
   - [WP-6: Tiered Memory & Structured Summaries](#wp-6-tiered-memory--structured-summaries)
   - [WP-7: Prompt-Cache Annotation](#wp-7-prompt-cache-annotation)
   - [WP-8: Server v2](#wp-8-server-v2)
   - [WP-9: Evaluation, Golden Files & Benchmarks](#wp-9-evaluation-golden-files--benchmarks)
6. [Execution Order & Dependency Graph](#6-execution-order--dependency-graph)
7. [Invariants (Must Always Hold)](#7-invariants-must-always-hold)
8. [Brehon Integration (External — Informational)](#8-brehon-integration-external--informational)
9. [Decision Records](#9-decision-records)
10. [References](#10-references)

---

## 1. Ground Truth: What Already Exists

The workspace **builds clean and all 70 tests pass** (`cargo test --workspace`:
49 unit tests + 9 fuzz tests + 12 llm_safety tests). The following is the as-built
inventory, verified against source on 2026-06-10. Treat this as the API baseline.

### 1.1 Workspace Layout

```
Cargo.toml                  workspace: resolver=3, edition=2024, rust-version=1.85, Apache-2.0
crates/
  context-compress-core/    traits + types, no heavy deps          (~300 LOC, DONE)
  context-compress/         compressors, CCR, pipeline, detection  (~3,100 LOC, DONE)
  context-compress-server/  Axum HTTP server                       (~230 LOC, DONE, needs WP-8)
  brehon-compress/          ABANDONED STUB — delete in WP-0
  brehon-compress-core/     ABANDONED STUB — delete in WP-0
  brehon-compress-server/   ABANDONED STUB — delete in WP-0
DESIGN.md                   stale pointer file — fix in WP-0
DESIGN.md                   this document (the build plan and engineering record)
```

Workspace `members` lists only the three `context-compress*` crates. The `brehon-*`
crates are dead code from an earlier naming scheme: incompatible trait signatures,
all bodies `todo!()`, zero tests, not compiled by `cargo build`.

### 1.2 `context-compress-core` (as built)

Files: `src/lib.rs`, `src/error.rs`, `src/metrics.rs`.
Deps: `async-trait`, `bytes`, `serde`, `serde_json`, `thiserror`. No tokio, no storage.

Existing public API (verbatim — do not change except where a WP says so):

```rust
// types
pub struct Message { pub role: String, pub content: String }            // WP-3 extends this
pub struct Content { pub data: Bytes, pub mime_or_lang: String,
                     pub metadata: HashMap<String, String> }
pub struct CompressionContext { pub model: String, pub question_hint: Option<String>,
                                pub max_tokens: Option<usize>, pub reversible: bool }
pub struct Compressed { pub id: String, pub data: Bytes,
                        pub original_tokens: usize, pub compressed_tokens: usize }
pub struct CompressedMessages { pub messages: Vec<Message>, pub stats: CompressionStats }
pub struct CompressionStats { pub original_tokens: usize, pub compressed_tokens: usize,
                              pub ratio: f64, pub compressor_used: String }
pub struct PerCompressorStats { pub name: String, pub content_type: String,
                                pub original_tokens: usize, pub compressed_tokens: usize,
                                pub ratio: f64, pub latency_ms: u64 }
pub struct PipelineStats { /* totals + per_compressor + ccr_retrievals + ccr_hits */ }

// traits
#[async_trait]
pub trait Compressor: Send + Sync {
    fn name(&self) -> &'static str;
    async fn compress(&self, content: &Content, ctx: &CompressionContext) -> Result<Compressed>;
    async fn retrieve(&self, id: &str) -> Result<Option<String>>;
}
#[async_trait]
pub trait CompressionPipeline: Send + Sync {
    fn add_compressor(&mut self, compressor: Box<dyn Compressor>);
    async fn run(&self, messages: &[Message]) -> Result<CompressedMessages>;
}
pub trait Metrics: Send + Sync { /* record_compress, record_retrieve,
                                    record_ccr_store_size, record_routing_decision */ }
pub trait Observer: Send + Sync { fn on_event(&self, event: &CompressionEvent); }
pub enum CompressionEvent { PipelineStarted{..}, ContentDetected{..}, CompressorSelected{..},
    MessageCompressed{..}, PipelineCompleted{..}, RetrieveAttempt{..}, RetrieveHit{..},
    RetrieveMiss{..}, CcrSaved{..}, CcrExpired{..}, Error{..} }

// error
pub enum ContextCompressError { Io(..), Serde(..), CompressionFailed(String),
    RetrieveFailed(String), NotFound(String), UnsupportedContentType(String),
    StoreError(String) }                                                 // WP-5 adds a variant
pub type Result<T> = std::result::Result<T, ContextCompressError>;

// sinks
pub struct NoopMetrics; pub struct NoopObserver;
pub struct TestMetrics { /* atomic counters */ } pub struct TestObserver { /* event vec */ }
```

### 1.3 `context-compress` (as built)

| File | Status | Notes |
|---|---|---|
| `src/lib.rs` | DONE | `CompressConfig`, `default_pipeline()`, `pipeline_with_ccr()`, `compress_messages()`, `detect()`, `is_json_dict_array()` |
| `src/pipeline.rs` | DONE | `DefaultCompressionPipeline`, `PipelineBuilder`; fail-closed per-message; routing table below. Has known defects fixed in WP-1. |
| `src/detect.rs` | DONE | `ContentType { JsonArray, SourceCode, SearchResults, BuildOutput, GitDiff, Html, PlainText }`, `DetectionResult { content_type, confidence, metadata }`, `detect_content_type()`, `is_json_array_of_dicts()`. Regex-based, 12 tests. |
| `src/compressors/smart_crusher.rs` | DONE | JSON arrays: outlier keep + adaptive sampling (knee-point) + CCR markers. 8 tests. |
| `src/compressors/log_stripper.rs` | DONE | Build logs: format detect, level scoring, dedup, CCR. 3 tests. |
| `src/compressors/ast_code.rs` | DONE | Code/diffs: strip comments/blanks, keep signatures. 3 tests. |
| `src/compressors/semantic.rs` | DONE | Prose: paragraph dedup + truncation. 2 tests. |
| `src/compressors/dedup_ref.rs` | DONE | Content-addressed block dedup via CCR. 2 tests. |
| `src/compressors/toon.rs` | DONE | JSON minify / whitespace collapse, lossy, zero-dep. 2 tests. |
| `src/ccr/mod.rs` | DONE | `trait CcrStore { save, retrieve, delete }`; `compute_key(payload) -> md5 hex`; `marker_for(hash) -> "<<ccr:HASH>>"` |
| `src/ccr/in_memory.rs` | DONE | TTL + capacity eviction. |
| `src/ccr/sqlite.rs` | DONE | `SqliteCcrStore::open(path, ttl_seconds)`, WAL mode. |
| `src/ccr/fjall.rs` | DONE | `FjallCcrStore` over fjall keyspace/partition. |
| `src/conversation.rs` | DONE | `ConversationConfig { preserve_recent=4, compress_middle=8, summary_old=true, bias_system=0.8 }`, `compress_conversation_history(&mut Vec<Message>, ..) -> ConversationStats`. Bands: recent verbatim → middle via pipeline → old extractive summary or drop. **Has UTF-8 panic bug — WP-1.** |
| `src/cache_aligner.rs` | DONE | `align_text()`, `align_messages()`: whitespace collapse + recursive JSON key sort. 6 tests. |
| `src/adaptive_sizer.rs` | DONE | `compute_optimal_k()`, `find_knee()` — simhash + knee detection. 4 tests. |
| `src/token_est.rs` | DONE | `count_tokens(text) = ceil(len/4).max(1)` heuristic. Superseded-but-kept by WP-2. |
| `src/stats_math.rs` | DONE | mean/variance/stdev/median/format_g. 5 tests. |
| `tests/fuzz.rs` | DONE | 9 adversarial-input tests (no panics). |
| `tests/llm_safety.rs` | DONE | 12 tests: round-trip CCR, fail-closed, determinism, structural integrity, observer events. |

Pipeline routing table (as built, in `pipeline.rs::compress_one`):

| ContentType | Compressor |
|---|---|
| JsonArray | `smart_crusher` |
| BuildOutput | `log_stripper` |
| SourceCode | `ast_code` |
| GitDiff | `ast_code` |
| SearchResults | `semantic` |
| Html | `semantic` |
| PlainText | `semantic` |

### 1.4 `context-compress-server` (as built)

`src/lib.rs`: `AppState { start_time, request_count, pipeline, ccr_store: Arc<InMemoryCcrStore> }`,
`app() -> Router` with routes `GET /health`, `POST /compress`, `POST /retrieve`,
`POST /detect`, `GET /stats`. `src/main.rs`: binds `0.0.0.0:3000` hardcoded.
WP-8 makes state/store/bind configurable and fixes the bind address.

### 1.5 Known Defects in Existing Code (verified by reading source)

These are fixed in WP-1. Listed here so they are not treated as intended behavior:

1. **UTF-8 panic:** `conversation.rs::summarise_turns` does `&joined[..800]` — panics if
   byte 800 is not a char boundary (any multibyte text). `tests/fuzz.rs` does not cover it.
2. **Inline token math:** `pipeline.rs` uses `content.len() / 4` directly (3 sites) instead
   of `token_est::count_tokens`, which uses `div_ceil` — the two disagree by ±1.
3. **Silent lock failure:** `CompressionPipeline::add_compressor` for
   `DefaultCompressionPipeline` uses `try_write()` and **silently drops the compressor**
   if the lock is contended.
4. **Panic in library code:** `DefaultCompressionPipeline::get_compressor` calls
   `panic!("unknown compressor name in test: {name}")` for unknown names.
5. **Fabricated stats:** `pipeline.rs::run` sets `ccr_hits = per_compressor_stats.len()`
   and `ccr_retrievals = messages.len()` — these are guesses, not measurements.

---

## 2. Implementer Rules (Hard Constraints)

1. **Fail-closed, always.** Any error during compression returns the original content
   unchanged. Never propagate a compression error into message content. Never send/return
   corrupted text.
2. **No panics in library code paths.** No `unwrap()`, `expect()`, `panic!()`, or
   unchecked slice indexing (`&s[..n]`) outside `#[cfg(test)]` code. Use `?`,
   `unwrap_or_default()`, or explicit error returns.
3. **No subprocesses, no network calls** in `context-compress-core` and
   `context-compress`. The library never spawns processes, opens sockets, or calls LLMs.
4. **No background tasks.** The library never spawns detached tokio tasks. All compaction
   is caller-driven (the host decides when to run it, sync or in its own task).
5. **Determinism.** Same input + same config ⇒ byte-identical output. No
   `SystemTime`-dependent output, no randomness, no HashMap-iteration-order-dependent
   output (sort keys before emitting anything derived from a map).
6. **Dependency policy.** New dependencies require a feature gate unless listed in a WP.
   `context-compress-core` may only depend on: `async-trait`, `bytes`, `serde`,
   `serde_json`, `thiserror`.
7. **Don't break the baseline.** Existing public APIs in §1 stay source-compatible except
   where a WP explicitly amends them (WP-3 amends `Message`; the WP tells you how to fix
   call sites). All 70 existing tests must keep passing (modulo tests a WP explicitly
   updates).
8. **Edition 2024, rustfmt-formatted, clippy-clean** (`-D warnings`).
9. **Doc comments** (`///`) on every new public item, stating behavior and failure modes.
10. **Honest claims only.** Token counts for Claude-family models are estimates (Anthropic
    does not publish its tokenizer). Never name a function/field "exact" for a model we
    cannot tokenize exactly.

---

## 3. Corrections vs. Design v1.0

v1.0 of this document was written as a greenfield spec. Reality diverged. The following
v1.0 claims are corrected; the rest of v1.0's intent is preserved in the WPs below.

| v1.0 said | Reality / v2.0 position |
|---|---|
| Phases 0–2 to be built over 4 weeks | Already built and tested (§1). Remaining work is WP-1…WP-9. |
| §15.2: `msg.metadata.insert("cache_control", ...)` | `Message` has no `metadata` field. WP-3 adds it; WP-7 uses it. |
| "Exact token counting" via `tiktoken-rs` + `HuggingFaceCounter` | tiktoken is exact only for OpenAI encodings. Anthropic's tokenizer is private — counts for Claude are estimates with a safety margin. HF `tokenizers` dep dropped (heavy, no current consumer). WP-2 specifies the honest design. |
| LTM uses "heavy abstractive summarization" | The library cannot call an LLM (Rule 3). WP-6 ships a deterministic extractive summarizer and a `Summarizer` trait the host (Brehon) can implement with its own LLM. |
| ADR-4: "proactive async" background compaction | Violates Rule 4. Compaction is caller-driven; the host may call it from its own background task. ADR-4 rewritten (§9). |
| Server crate "future" | Exists. WP-8 hardens it (configurable bind/store, mountable router). |
| `headroom-core` facts (pinned rev, proxy subprocess, 2 code paths) | Still accurate per the Brehon analysis doc; integration itself happens in the Brehon repo (§8), not here. |
| 14-week roadmap with week counts | Replaced by ordered WPs with done-criteria. No calendar estimates. |
| `ToolCall`/`WorkingSet` structs with rich fields (`Vec<ToolCall>`, `Vec<ErrorTrace>`) | Over-modeled. v2.0 keeps everything on `Message` + metadata tags (WP-3/WP-4) so the library stays provider-agnostic and Brehon doesn't need lossy conversions. |

---

## 4. Target Architecture

```
                         Host application (Brehon, CLI, tests)
                                        │
        ┌───────────────────────────────┼─────────────────────────────────┐
        │ content-level                 │ conversation-level              │
        │  compress_messages()          │  compress_conversation_history()│
        │  detect()                     │  apply_agent_compression() WP-4 │
        │                               │  enforce_budget()          WP-5 │
        │                               │  TieredMemory              WP-6 │
        │                               │  apply_cache_strategy()    WP-7 │
        └───────────────┬───────────────┴───────────────┬─────────────────┘
                        ▼                               ▼
              context-compress (lib)          context-compress-server (optional HTTP)
   detect → DefaultCompressionPipeline → compressors (6) → CCR markers
                        │                         │
              TokenCounter (WP-2)        CcrStore: in-memory / sqlite / fjall
                        ▼
              context-compress-core (traits + types only)
```

Layering rule: `core` ← `context-compress` ← `server`. Nothing in `core` knows about
storage, tokio, or HTTP.

The conversation-level features compose in this order when a host uses all of them:

```
1. classify + agent rules (WP-4)   — tag & clear stale tool results, protect errors
2. budget enforcement (WP-5)       — cascade until under ContextBudget
3. cache alignment (existing)      — stabilize bytes
4. cache annotation (WP-7)         — mark breakpoints for the provider adapter
```

---

## 5. Work Packages

---

### WP-0: Repository Hygiene

**Goal:** Remove dead code and stale pointers so the repo contains exactly one source of truth.

**Steps (exact):**

1. Delete directories `crates/brehon-compress`, `crates/brehon-compress-core`,
   `crates/brehon-compress-server` (they are not workspace members; nothing imports them).
2. Overwrite root `DESIGN.md` with exactly:
   ```markdown
   # Context-Compress — Design

   The authoritative design and build plan is [.planning/DESIGN.md](.planning/DESIGN.md).
   All other documents under `.planning/` are superseded drafts kept for history.
   ```
3. Append to `.gitignore` (if not already present): `.agora/` and `target/`
   (check the existing file first; do not duplicate lines).
4. Do not modify `NOTICE` (it carries required attribution).

**Definition of Done:**
- `cargo build --workspace && cargo test --workspace` passes (nothing referenced the deleted crates).
- `grep -r "brehon-compress" crates/ Cargo.toml` returns nothing.

---

### WP-1: Bug Fixes in Existing Code

**Goal:** Fix the five defects in §1.5 without changing any public signatures.

#### 1a. UTF-8-safe truncation (`crates/context-compress/src/conversation.rs`)

Add a private helper and use it in `summarise_turns`:

```rust
/// Truncate to at most `max` bytes without splitting a UTF-8 char.
fn truncate_to_char_boundary(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}
```

Replace the existing block

```rust
let cap = 800;
if joined.len() > cap { format!("{}…", &joined[..cap]) } else { joined }
```

with

```rust
let cap = 800;
if joined.len() > cap {
    format!("{}…", truncate_to_char_boundary(&joined, cap))
} else {
    joined
}
```

Test to add in the same file's `mod tests`:

```rust
#[tokio::test]
async fn test_summary_multibyte_no_panic() {
    // 12 turns of multibyte text long enough to force summary truncation at byte 800.
    let mut msgs: Vec<Message> = (0..12).map(|i| Message {
        role: if i % 2 == 0 { "user" } else { "assistant" }.into(),
        content: "héllo wörld 日本語のテキストです。".repeat(20),
    }).collect();
    let pipeline = DefaultCompressionPipeline::new(None, None);
    let ctx = CompressionContext { model: "m".into(), question_hint: None,
                                   max_tokens: None, reversible: true };
    let cfg = ConversationConfig { preserve_recent: 2, compress_middle: 2,
                                   summary_old: true, bias_system: 0.8 };
    let stats = compress_conversation_history(&mut msgs, &cfg, &pipeline, &ctx)
        .await.unwrap();
    assert!(stats.old_turns_summarised > 0);
    assert!(msgs[0].content.is_char_boundary(msgs[0].content.len()));
}
```

(Note: WP-3 changes `Message` construction; if WP-3 has already run, use `Message::new(...)`.)

#### 1b. Unify token math (`crates/context-compress/src/pipeline.rs`)

Replace every occurrence of `content_str.len() / 4` and `msg.content.len() / 4` with
`crate::token_est::count_tokens(content_str)` / `crate::token_est::count_tokens(&msg.content)`.
There are 3 occurrences (one in `compress_one`, two in `run`). Existing tests that assert
token totals may shift by ±1 per message due to `div_ceil`; update those assertions if
they fail — the new values are correct.

#### 1c. `add_compressor` must not silently drop

In `impl CompressionPipeline for DefaultCompressionPipeline`, replace the `try_write`
body with a guaranteed write. Since the trait method is sync and the lock is
`tokio::sync::RwLock`, switch the field to a synchronous lock:

1. Change the field to `pub(crate) compressors: Arc<std::sync::RwLock<Vec<Box<dyn Compressor>>>>`.
2. Replace every `self.compressors.read().await` with
   `self.compressors.read().unwrap_or_else(|e| e.into_inner())`
   and every `.write().await` similarly with `.write().unwrap_or_else(|e| e.into_inner())`
   (poison recovery — never panic, per Rule 2).
3. Remove the now-unneeded `use tokio::sync::RwLock;` import; keep critical sections
   short (clone/iterate, don't hold across `.await`). **Important:** in `compress_one`,
   the read guard is currently held across `comp.compress(...).await`. Restructure so the
   guard is dropped first: find the compressor name + index under the read lock, drop the
   guard, then re-acquire briefly OR (simpler and correct) keep the guard usage but wrap
   compressor selection and invocation so no `std` guard lives across an await:

   ```rust
   // inside compress_one, replacing the current selection block:
   let comp_name: Option<&'static str> = match detected.content_type {
       ContentType::JsonArray => Some("smart_crusher"),
       ContentType::BuildOutput => Some("log_stripper"),
       ContentType::SourceCode | ContentType::GitDiff => Some("ast_code"),
       ContentType::SearchResults | ContentType::Html | ContentType::PlainText =>
           Some("semantic"),
   };
   // Find the index without holding the guard across await:
   let comp_idx = {
       let guard = self.compressors.read().unwrap_or_else(|e| e.into_inner());
       comp_name.and_then(|n| guard.iter().position(|c| c.name() == n))
   };
   ```

   Then, because `Box<dyn Compressor>` can't be cloned out, invoke through a re-acquired
   guard **with the future boxed and awaited after the guard drops is not possible** —
   so instead make the invocation safe by changing the storage to
   `Vec<Arc<dyn Compressor>>` (Arc instead of Box). This is the actual fix:

   - Change the field type to `Arc<std::sync::RwLock<Vec<Arc<dyn Compressor>>>>`.
   - `add_compressor(&mut self, compressor: Box<dyn Compressor>)` converts via
     `Arc::from(compressor)` (trait signature unchanged).
   - In `compress_one`: clone the `Arc<dyn Compressor>` out under the lock, drop the
     guard, then `comp.compress(...).await`.
   - `with_ccr_store` / `with_ccr_store_reuse` / `register` construct `Arc<dyn Compressor>`
     values directly (`Arc::new(SmartCrusher::with_ccr_store(...))` etc.).

#### 1d. `get_compressor` must not panic

Change the unknown-name arm from `panic!(...)` to returning `None`. Easiest correct
form given 1c: with `Vec<Arc<dyn Compressor>>` storage, the whole hand-rolled
re-instantiation `match` is deleted and the method becomes:

```rust
/// Look up a registered compressor by name.
pub fn get_compressor(&self, name: &str) -> Option<Arc<dyn Compressor>> {
    let guard = self.compressors.read().unwrap_or_else(|e| e.into_inner());
    guard.iter().find(|c| c.name() == name).cloned()
}
```

(Signature change from `Option<Box<dyn Compressor>>` + async → sync `Option<Arc<..>>` is
permitted; update its callers in tests.)

#### 1e. Honest CCR stats

Delete the heuristic block in `run()` (`ccr_hits = per_compressor_stats.len()` etc.).
Instead, wrap the observer with an internal counting observer:

```rust
// new private type in pipeline.rs
struct CountingObserver {
    inner: Arc<dyn Observer>,
    retrieve_attempts: AtomicUsize,
    retrieve_hits: AtomicUsize,
    ccr_saves: AtomicUsize,
}
impl Observer for CountingObserver {
    fn on_event(&self, event: &CompressionEvent) {
        match event {
            CompressionEvent::RetrieveAttempt { .. } =>
                { self.retrieve_attempts.fetch_add(1, Ordering::Relaxed); }
            CompressionEvent::RetrieveHit { .. } =>
                { self.retrieve_hits.fetch_add(1, Ordering::Relaxed); }
            CompressionEvent::CcrSaved { .. } =>
                { self.ccr_saves.fetch_add(1, Ordering::Relaxed); }
            _ => {}
        }
        self.inner.on_event(event);
    }
}
```

`run()` reads the counters into `PipelineStats { ccr_retrievals, ccr_hits, .. }` at the
end. If compressors do not currently emit `CcrSaved` when they store originals, add that
emission is **out of scope** — report whatever is actually observed (zeros are honest;
guesses are not).

**Tests to add** (in `pipeline.rs` `mod tests` or `tests/llm_safety.rs`):
- `add_compressor_never_drops`: register a custom no-op compressor via the trait method,
  then assert `get_compressor("noop").is_some()`.
- `get_compressor_unknown_returns_none`: assert `get_compressor("nope").is_none()`.
- `ccr_stats_not_fabricated`: run pipeline without a CCR store; assert
  `stats` pipeline event reports `ccr_hits == 0`.

**Definition of Done:** fmt/clippy/test clean; the three new tests plus
`test_summary_multibyte_no_panic` pass; no `panic!`/`unwrap()` added in non-test code.

---

### WP-2: Token Counting

**Goal:** Replace "len/4 everywhere" with a pluggable `TokenCounter`, exact for OpenAI
encodings (feature-gated), calibrated-heuristic otherwise, honest for Claude.

#### Files

- `crates/context-compress-core/src/token.rs` (new) — trait only.
- `crates/context-compress/src/token_counter.rs` (new) — implementations + factory.
- `crates/context-compress/src/token_est.rs` — keep as-is (the zero-dep fallback).

#### Core trait (`context-compress-core/src/token.rs`, re-export from `lib.rs`)

```rust
use crate::Message;

/// Counts (or estimates) LLM tokens. Implementations must be deterministic
/// and must never panic.
pub trait TokenCounter: Send + Sync {
    /// Tokens for a raw string.
    fn count(&self, text: &str) -> usize;

    /// True if this counter is exact for its target model family,
    /// false if it is an estimate.
    fn is_exact(&self) -> bool;

    /// Tokens for a message list, including per-message wrapper overhead.
    /// Default: sum of content counts + OVERHEAD_PER_MESSAGE per message.
    fn count_messages(&self, messages: &[Message]) -> usize {
        messages.iter().map(|m| self.count(&m.content)).sum::<usize>()
            + messages.len() * OVERHEAD_PER_MESSAGE
    }
}

/// Approximate per-message wrapper cost (role tags, separators).
/// 4 matches OpenAI's documented ChatML overhead and is a safe
/// over-estimate for other providers.
pub const OVERHEAD_PER_MESSAGE: usize = 4;
```

#### Implementations (`context-compress/src/token_counter.rs`)

```rust
/// Bytes-per-token heuristic, optionally calibrated per model family.
pub struct HeuristicCounter {
    /// Estimated bytes per token. Default 4.0 (English prose).
    pub bytes_per_token: f64,
}

impl HeuristicCounter {
    pub fn new() -> Self { Self { bytes_per_token: 4.0 } }
    /// Calibration table (exact strings; match by `starts_with` on the model id):
    ///   "gpt-"      -> 4.0
    ///   "claude-"   -> 3.5   (Claude tokenizes denser code/JSON; over-estimate tokens)
    ///   "gemini-"   -> 4.0
    ///   anything else -> 4.0
    pub fn for_model(model: &str) -> Self;
}

impl TokenCounter for HeuristicCounter {
    fn count(&self, text: &str) -> usize {
        ((text.len() as f64 / self.bytes_per_token).ceil() as usize).max(1)
    }
    fn is_exact(&self) -> bool { false }
}
```

Feature-gated exact counter:

```rust
#[cfg(feature = "tiktoken")]
pub struct TiktokenCounter { bpe: tiktoken_rs::CoreBPE, exact: bool }

#[cfg(feature = "tiktoken")]
impl TiktokenCounter {
    /// Encoding selection (in this order):
    ///   model starts_with "gpt-4o" | "gpt-5" | "o1" | "o3" | "o4" -> o200k_base, exact=true
    ///   model starts_with "gpt-4" | "gpt-3.5"                     -> cl100k_base, exact=true
    ///   anything else (incl. "claude-*") -> o200k_base proxy, exact=false,
    ///       and count() multiplies the result by 1.1 rounded up (safety margin,
    ///       because we cannot tokenize Claude exactly).
    /// Returns Err only if the encoding tables fail to load.
    pub fn for_model(model: &str) -> context_compress_core::Result<Self>;
}
```

Factory (always available):

```rust
/// Best available counter for `model`. Never fails:
/// with feature "tiktoken", tries TiktokenCounter::for_model and falls back to
/// HeuristicCounter::for_model on error; without the feature, returns the heuristic.
pub fn counter_for_model(model: &str) -> std::sync::Arc<dyn TokenCounter>;
```

#### Cargo changes (`crates/context-compress/Cargo.toml`)

```toml
[features]
default = []
tiktoken = ["dep:tiktoken-rs"]

[dependencies]
tiktoken-rs = { version = "0.7", optional = true }
```

(If `0.7` does not resolve, use the latest published version; check with
`cargo add tiktoken-rs --dry-run`.)

#### Tests (`token_counter.rs` `mod tests`)

- `heuristic_minimum_one`: `HeuristicCounter::new().count("") == 1`.
- `heuristic_deterministic`: two calls on same input equal.
- `claude_overestimates`: `HeuristicCounter::for_model("claude-fable-5").count(s)
   >= HeuristicCounter::for_model("gpt-4o").count(s)` for a 1 KB ASCII string.
- `factory_never_panics`: `counter_for_model("totally-unknown-model").count("hi") >= 1`.
- Behind `#[cfg(feature = "tiktoken")]`:
  - `tiktoken_exact_for_gpt4o`: `is_exact() == true` and `count("hello world") > 0`.
  - `tiktoken_claude_is_estimate`: `is_exact() == false` for `"claude-opus-4-8"`.

Also add a CI-visible check: `cargo test -p context-compress --features tiktoken` must pass.

**Non-goals:** do NOT wire the counter into the pipeline's per-message stats (the
heuristic there is fine for ratio reporting); the counter's consumer is WP-5 budget
enforcement and host applications.

**Definition of Done:** fmt/clippy/test clean with and without `--features tiktoken`;
`pub use` the trait from `context-compress-core/src/lib.rs` and the impls + factory from
`context-compress/src/lib.rs`.

---

### WP-3: Message Metadata

**Goal:** Give `Message` an optional string-keyed metadata map so later WPs can tag
messages (agent content type, tool status, cache breakpoints, CCR ids) without changing
the type again.

#### Change (`crates/context-compress-core/src/lib.rs`)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// Optional annotations. Absent in serialized form when empty, so existing
    /// JSON wire formats are unchanged.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into(), metadata: HashMap::new() }
    }
}
```

Add a module of well-known keys (same file or `core/src/meta.rs`, re-exported):

```rust
/// Well-known metadata keys. Values are plain strings.
pub mod meta_keys {
    /// One of the `AgentContentType` snake_case names (WP-4 writes this).
    pub const AGENT_CONTENT_TYPE: &str = "cc.agent_content_type";
    /// "success" | "error" — status of a tool-result message.
    pub const TOOL_STATUS: &str = "cc.tool_status";
    /// Name of the tool that produced a tool message.
    pub const TOOL_NAME: &str = "cc.tool_name";
    /// CCR id of the original content, when this message was compressed/cleared.
    pub const CCR_ID: &str = "cc.ccr_id";
    /// "ephemeral" — provider-cache breakpoint marker (WP-7 writes this).
    pub const CACHE_CONTROL: &str = "cc.cache_control";
    /// "true" — message must never be compressed, cleared, or summarized.
    pub const PINNED: &str = "cc.pinned";
}
```

#### Mechanical follow-up

Adding a field breaks every `Message { role, content }` struct literal. Fix strategy:
run `cargo build --workspace 2>&1`, and for each error either switch the literal to
`Message::new(role, content)` (preferred) or add `metadata: Default::default()`.
Known literal sites: `pipeline.rs::run` (1), `conversation.rs` (2 + test helper),
`tests/llm_safety.rs`, `tests/fuzz.rs`, `context-compress-server/src/lib.rs` handlers.
Preserve behavior exactly — this is a mechanical change.

Pipeline `run()` must **propagate metadata** when rebuilding messages:

```rust
compressed_messages.push(Message {
    role: msg.role.clone(),
    content: compressed_content,
    metadata: msg.metadata.clone(),
});
```

#### Tests

- `core`: `message_metadata_roundtrip` — serialize `Message::new("user","hi")` to JSON,
  assert the JSON object has exactly keys `role` and `content` (no `metadata`);
  deserialize legacy JSON `{"role":"user","content":"hi"}` successfully.
- `context-compress` (`tests/llm_safety.rs`): `pipeline_preserves_metadata` — run the
  pipeline on a message with `metadata["cc.pinned"]="true"` and assert the output message
  still carries it.

**Definition of Done:** workspace builds; all tests pass; serialized form of
metadata-less messages is byte-identical to before (asserted by the roundtrip test).

---

### WP-4: Agent-Aware Compression Rules

**Goal:** Implement the highest-ROI agent technique — classify messages by agent role,
clear stale successful tool results to CCR markers, and never touch errors, system
prompts, user queries, or recent context.

#### File: `crates/context-compress/src/agent.rs` (new, `pub mod agent;` in `lib.rs`)

```rust
use context_compress_core::{meta_keys, Message, Result};
use crate::ccr::CcrStore;
use std::sync::Arc;

/// Agent-semantic classification of a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentContentType {
    SystemInstruction,
    UserQuery,
    AssistantReply,
    ToolResultSuccess,
    ToolResultError,
    Unknown,
}

impl AgentContentType {
    pub fn as_str(&self) -> &'static str { /* snake_case names, e.g. "tool_result_error" */ }
    pub fn from_str_loose(s: &str) -> Self { /* inverse of as_str; unknown -> Unknown */ }
}

/// Classify one message. Precedence (first match wins):
/// 1. metadata[AGENT_CONTENT_TYPE] if present and parseable
/// 2. role == "system"                          -> SystemInstruction
/// 3. role == "tool" or role == "function":
///       metadata[TOOL_STATUS] == "error"       -> ToolResultError
///       else if content matches ERROR_PATTERN  -> ToolResultError
///       else                                   -> ToolResultSuccess
/// 4. role == "user"                            -> UserQuery
/// 5. role == "assistant"                       -> AssistantReply
/// 6. anything else                             -> Unknown
///
/// ERROR_PATTERN (case-sensitive, any-of, checked against the first 512 bytes only):
///   "Error:", "error:", "ERROR", "Traceback (most recent call last)",
///   "panicked at", "Exception", "FAILED", "stderr:"
pub fn classify(msg: &Message) -> AgentContentType;

/// Policy knobs. All counts are in *messages of that kind*, newest-first.
#[derive(Debug, Clone)]
pub struct AgentPolicy {
    /// Keep this many most-recent ToolResultSuccess messages raw. Default 3.
    pub keep_recent_tool_results: usize,
    /// Tool results older than the kept window are CLEARED (replaced by a stub +
    /// CCR marker) if true, otherwise compressed via the pipeline. Default true.
    pub clear_old_tool_results: bool,
    /// Keep this many most-recent AssistantReply messages raw. Default 2.
    pub keep_recent_assistant: usize,
}
impl Default for AgentPolicy { /* values above */ }

/// What was done, for observability.
#[derive(Debug, Clone, Default)]
pub struct AgentCompressionStats {
    pub tool_results_cleared: usize,
    pub tool_results_kept_raw: usize,
    pub errors_preserved: usize,
    pub tokens_before: usize,
    pub tokens_after: usize,
}

/// Apply agent-aware rules in place. Fail-closed: any per-message failure
/// (e.g. CCR save error) leaves that message unchanged.
pub async fn apply_agent_compression(
    messages: &mut Vec<Message>,
    policy: &AgentPolicy,
    ccr: Option<Arc<dyn CcrStore>>,
) -> Result<AgentCompressionStats>;
```

#### Behavior of `apply_agent_compression` (exact algorithm)

1. Compute `tokens_before = HeuristicCounter::new().count_messages(messages)`
   (use the WP-2 heuristic; do not require a model id here).
2. Walk messages **from newest to oldest**, maintaining `seen_tool_success = 0`.
3. For each message, `let kind = classify(msg)`. Apply this rule table:

| AgentContentType | Rule |
|---|---|
| SystemInstruction | Never modified. |
| UserQuery | Never modified. |
| ToolResultError | Never modified. Increment `errors_preserved`. |
| AssistantReply | Never modified (compression of old replies is WP-6's summarizer's job, not this pass). |
| Unknown | Never modified. |
| ToolResultSuccess | If `metadata[PINNED]=="true"`: never modified. Else if `seen_tool_success < policy.keep_recent_tool_results`: keep raw, increment both `seen_tool_success` and `tool_results_kept_raw`. Else: **clear** (below) when `policy.clear_old_tool_results`, otherwise leave unchanged. |

4. **Clearing a tool result** (only when a CCR store was provided; if `ccr` is `None`,
   clearing is skipped and the message is left unchanged — fail-closed):
   1. `let hash = crate::ccr::compute_key(msg.content.as_bytes());`
   2. `ccr.save(&hash, &msg.content, None).await` — on `Err`, leave the message
      unchanged and continue (do not abort the pass).
   3. Replace content with exactly:
      ```text
      [tool:{tool_name}] result cleared ({n} tokens) — original retrievable via <<ccr:{hash}>>
      ```
      where `{tool_name}` is `metadata[TOOL_NAME]` or `"unknown"`, and `{n}` is the
      heuristic token count of the original content.
   4. Set `metadata[CCR_ID] = hash` and `metadata[AGENT_CONTENT_TYPE] = "tool_result_success"`.
   5. Increment `tool_results_cleared`.
5. `tokens_after` recomputed the same way as step 1. Return stats.

#### Tests (`agent.rs` `mod tests`, all `#[tokio::test]` where async)

- `classify_precedence_metadata_wins`: role "user" + metadata
  `AGENT_CONTENT_TYPE="tool_result_error"` classifies as `ToolResultError`.
- `classify_tool_error_by_content`: role "tool", content
  `"Traceback (most recent call last):\n..."` → `ToolResultError`.
- `errors_never_cleared`: 10 tool messages, 5 of them errors, policy keeps 0 recent;
  after the pass all 5 error messages have unchanged content.
- `recent_tool_results_kept_raw`: 6 success tool messages, default policy → exactly the
  3 newest unchanged, the 3 oldest cleared; stub content starts with `"[tool:"` and
  contains `"<<ccr:"`.
- `cleared_content_roundtrip`: clear one tool result with an `InMemoryCcrStore`, extract
  the hash from `metadata[CCR_ID]`, `ccr.retrieve(hash)` returns the original string.
- `no_ccr_store_means_no_clearing`: same input with `ccr=None` → zero modifications.
- `pinned_never_cleared`: oldest tool result has `PINNED="true"` → unchanged.
- `system_and_user_untouched`: mixed conversation; system/user messages byte-identical
  after the pass.

**Definition of Done:** fmt/clippy/test clean; all 8 tests pass; `pub use agent::*` is
NOT added to lib root (namespace it: callers use `context_compress::agent::...`).

---

### WP-5: Token Budget Enforcement

**Goal:** A single entry point that guarantees a message list fits a token budget, by
escalating through cheap → lossy steps, and errors out rather than overflowing.

#### Error variant (`crates/context-compress-core/src/error.rs`)

Add:

```rust
#[error("token budget exceeded: need {needed} tokens but limit is {limit}")]
BudgetExceeded { needed: usize, limit: usize },
```

#### File: `crates/context-compress/src/budget.rs` (new, `pub mod budget;`)

```rust
use context_compress_core::{Message, Result, TokenCounter};
use crate::agent::AgentPolicy;
use crate::conversation::ConversationConfig;
use crate::pipeline::DefaultCompressionPipeline;
use crate::ccr::CcrStore;
use std::sync::Arc;

/// Token budget for one LLM call.
#[derive(Debug, Clone)]
pub struct ContextBudget {
    /// Hard cap for the whole message list (prompt side), e.g. 180_000.
    pub total_limit: usize,
    /// Fraction of total_limit held back as safety margin for tokenizer
    /// inexactness. Default 0.05 when the counter is not exact, 0.0 when it is.
    /// effective_limit = total_limit - ceil(total_limit * margin).
    pub safety_margin: Option<f64>,
}

/// What the cascade did.
#[derive(Debug, Clone, Default)]
pub struct BudgetReport {
    pub tokens_initial: usize,
    pub tokens_final: usize,
    pub effective_limit: usize,
    /// Names of the steps that ran, in order: subset of
    /// ["agent_rules", "compress_middle", "summarize_old", "drop_old"].
    pub steps_applied: Vec<String>,
}

/// Make `messages` fit `budget`. Mutates in place. Fail-closed per step;
/// fails CLOSED overall: if after all steps the list still exceeds the
/// effective limit, returns Err(BudgetExceeded) and `messages` is left in
/// its (partially compacted, still valid) state — the caller must not send it.
pub async fn enforce_budget(
    messages: &mut Vec<Message>,
    budget: &ContextBudget,
    counter: &dyn TokenCounter,
    pipeline: &DefaultCompressionPipeline,
    agent_policy: &AgentPolicy,
    ccr: Option<Arc<dyn CcrStore>>,
) -> Result<BudgetReport>;
```

#### Cascade (exact; re-count with `counter.count_messages` after every step and stop as soon as `tokens <= effective_limit`)

```
effective_limit = total_limit - ceil(total_limit * margin)
   where margin = budget.safety_margin.unwrap_or(if counter.is_exact() {0.0} else {0.05})

Step 0: count. Already under limit -> return report with steps_applied = [].
Step 1: "agent_rules"      -> agent::apply_agent_compression(messages, agent_policy, ccr)
Step 2: "compress_middle"  -> conversation::compress_conversation_history with
                              ConversationConfig { preserve_recent: 4, compress_middle:
                              messages.len().saturating_sub(4), summary_old: false,
                              bias_system: 0.8 }
Step 3: "summarize_old"    -> compress_conversation_history with
                              ConversationConfig { preserve_recent: 4, compress_middle: 4,
                              summary_old: true, bias_system: 0.8 }
Step 4: "drop_old"         -> while over limit AND more than 4 messages remain:
                              remove the OLDEST message whose classify() is not
                              SystemInstruction / UserQuery(latest) / ToolResultError and
                              whose metadata[PINNED] != "true". "UserQuery(latest)" means
                              only the most recent user message is protected; older user
                              messages are droppable. If a full scan finds nothing
                              droppable, break.
Final:  if still over limit -> Err(ContextCompressError::BudgetExceeded {
            needed: current_count, limit: effective_limit })
```

Notes:
- Steps 2/3 reuse existing code; do not duplicate their logic.
- Protected messages in step 4 are found via `crate::agent::classify`.
- Never drop the last remaining system message.

#### Tests (`budget.rs` `mod tests`)

- `under_budget_is_noop`: small list, huge limit → `steps_applied` empty, messages
  byte-identical.
- `agent_rules_sufficient`: list whose only bloat is old tool results; limit chosen so
  step 1 alone fits → `steps_applied == ["agent_rules"]`.
- `cascade_reaches_drop`: 50 long plain-text messages, tiny limit → report contains
  `"drop_old"`, final count ≤ effective limit.
- `budget_exceeded_errors`: 1 system + 1 user message, both pinned, limit of 1 token →
  returns `Err(BudgetExceeded { .. })`.
- `errors_survive_cascade`: include a `ToolResultError` older than everything; after a
  full cascade that drops other messages, the error message is still present verbatim.
- `safety_margin_applied`: counter with `is_exact() == false`, `total_limit = 100`,
  `safety_margin = None` → `report.effective_limit == 95`.

**Definition of Done:** fmt/clippy/test clean; six tests pass; the function is
re-exported as `context_compress::budget::enforce_budget`.

---

### WP-6: Tiered Memory & Structured Summaries

**Goal:** A deterministic structured-summary builder plus a `Summarizer` extension point,
so hosts get Claude-Code-style LTM without the library calling an LLM.

Scope note: full 4-tier orchestration (working set / STM / LTM / archive) is the host's
loop. This WP ships the **building blocks**: the summary type, the deterministic
summarizer, the trait for LLM-backed summarizers, and one helper that upgrades
`conversation.rs`'s old-turn summary to use them.

#### File: `crates/context-compress/src/memory/mod.rs` + `memory/summary.rs` (new)

```rust
// memory/summary.rs
use context_compress_core::{Message, Result};
use async_trait::async_trait;

/// Structured summary of a conversation span. All fields are plain data;
/// render_markdown() produces the prompt-ready form.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct StructuredSummary {
    /// One-line intent, best-effort (first user message's first sentence).
    pub session_intent: String,
    /// Unique file paths mentioned, sorted lexicographically.
    pub files_touched: Vec<String>,
    /// Lines that look like decisions, in order of appearance, deduplicated.
    pub key_decisions: Vec<String>,
    /// Error lines preserved verbatim (first line of each error block), in order.
    pub unresolved_errors: Vec<String>,
    /// Lines that look like TODO/next-step statements.
    pub next_steps: Vec<String>,
    /// Number of source turns this summary covers.
    pub turns_covered: usize,
}

impl StructuredSummary {
    /// Render with EXACTLY these section headers (omit empty sections):
    ///   "## Session intent", "## Files touched", "## Decisions",
    ///   "## Unresolved errors", "## Next steps"
    /// List items are "- " bullets. Ends without trailing newline.
    pub fn render_markdown(&self) -> String;

    /// Merge `newer` into `self`: union files (re-sort), append new decisions /
    /// errors / next_steps that are not exact duplicates, add turns_covered,
    /// keep self.session_intent unless empty.
    pub fn merge(&mut self, newer: &StructuredSummary);
}

/// Summarizes a span of messages. The library ships ExtractiveSummarizer;
/// hosts may implement this with an LLM (e.g. Brehon).
#[async_trait]
pub trait Summarizer: Send + Sync {
    async fn summarize(
        &self,
        turns: &[Message],
        existing: Option<&StructuredSummary>,
    ) -> Result<StructuredSummary>;
}

/// Deterministic, zero-LLM summarizer.
pub struct ExtractiveSummarizer;
```

#### `ExtractiveSummarizer` extraction rules (exact, applied per message line)

Operate on `turns` in order; split content into lines; trim each line.

1. `session_intent`: first sentence (split on `.`, `!`, `?`) of the first message with
   `role == "user"`, truncated UTF-8-safely to 200 bytes. Empty string if no user message.
2. `files_touched`: every regex match of
   `[A-Za-z0-9_./\\-]+\.(rs|py|ts|tsx|js|jsx|go|java|kt|rb|c|h|cpp|hpp|md|toml|yaml|yml|json|sql|sh)\b`
   across all content. Deduplicate, sort, cap at 50 entries (keep first 50 after sort).
3. `key_decisions`: lines starting (case-insensitive) with one of
   `"decision:"`, `"decided"`, `"we will "`, `"we chose "`, `"chose "`, `"agreed "`.
   Cap 20.
4. `unresolved_errors`: lines containing any WP-4 `ERROR_PATTERN` string. Keep the line
   verbatim, UTF-8-safe-truncated to 300 bytes. Cap 20.
5. `next_steps`: lines starting (case-insensitive) with `"todo"`, `"next:"`,
   `"next step"`, `"- [ ]"`. Cap 20.
6. `turns_covered = turns.len()`.
7. If `existing` is `Some`, start from a clone of it and `merge` the freshly extracted
   summary into it (anchored summarization — never regenerate from scratch).

Use the `regex` crate (already a dependency) with `std::sync::LazyLock` for compiled
patterns. Deterministic ordering everywhere (Rule 5).

#### Wire into `conversation.rs`

Add a sibling function (do not change the existing one's signature):

```rust
/// Like compress_conversation_history, but old turns are summarized with the
/// given Summarizer into a StructuredSummary rendered as markdown, inserted as
/// one system message prefixed "[Earlier conversation summary]\n".
pub async fn compress_conversation_history_with_summarizer(
    messages: &mut Vec<Message>,
    config: &ConversationConfig,
    pipeline: &DefaultCompressionPipeline,
    summarizer: &dyn crate::memory::Summarizer,
) -> Result<ConversationStats>
```

Behavior identical to the existing function except the old-turn band calls
`summarizer.summarize(&old_turns, None)` and inserts
`Message::new("system", format!("[Earlier conversation summary]\n{}", s.render_markdown()))`.
On summarizer `Err`, fall back to the existing `summarise_turns` output (fail-closed).

#### Tests (`memory/summary.rs` `mod tests`)

- `extracts_files_sorted_deduped`: content mentioning `src/b.rs`, `src/a.rs`, `src/a.rs`
  → `["src/a.rs", "src/b.rs"]`.
- `extracts_errors_verbatim`: a line `Error: connection refused` appears in
  `unresolved_errors`.
- `render_markdown_sections`: non-empty summary renders headers in the exact specified
  order; empty sections absent.
- `merge_is_idempotent`: `merge(x)` twice with the same `x` equals once.
- `summarizer_deterministic`: two runs over the same turns produce equal structs.
- In `conversation.rs`: `with_summarizer_fail_closed` — a Summarizer impl that always
  errors; result must equal the legacy extractive path (no error propagated, summary
  message present).

**Definition of Done:** fmt/clippy/test clean; `serde` round-trip of `StructuredSummary`
works (it derives Serialize/Deserialize); six tests pass.

---

### WP-7: Prompt-Cache Annotation

**Goal:** Mark cache breakpoints on messages via metadata so host provider-adapters can
translate them into real API fields (e.g. Anthropic `cache_control`). The library only
annotates — it never talks to providers.

#### File: `crates/context-compress/src/cache_strategy.rs` (new, `pub mod cache_strategy;`)

```rust
use context_compress_core::{meta_keys, Message};

/// Provider cache behavior the host is targeting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStrategy {
    /// Anthropic explicit breakpoints (max 4 per request).
    Anthropic,
    /// OpenAI: automatic prefix caching — alignment only, no annotations.
    OpenAi,
    /// Unknown provider: alignment only.
    Generic,
}

/// Annotate breakpoints. Returns how many breakpoints were set.
///
/// Anthropic: set metadata[CACHE_CONTROL] = "ephemeral" on (in priority order,
/// skipping duplicates, max 4):
///   1. the LAST message with role == "system"
///   2. the message immediately BEFORE the last `stable_suffix` messages
///      (i.e. index len - stable_suffix - 1), if that index is > breakpoint 1's index
/// Other strategies: remove any existing CACHE_CONTROL keys and return 0.
///
/// `stable_suffix` is the number of trailing messages expected to change every
/// turn (typically preserve_recent from ConversationConfig). Saturate at 0.
pub fn apply_cache_strategy(
    messages: &mut [Message],
    strategy: CacheStrategy,
    stable_suffix: usize,
) -> usize;
```

Rationale recorded for the implementer: provider caches key on exact prefixes, so the
breakpoint must sit at the end of the longest stable prefix; combined with
`align_messages` (existing) this maximizes hit rate. The host adapter maps
`metadata["cc.cache_control"] == "ephemeral"` to the provider's wire format.

#### Tests

- `anthropic_sets_system_breakpoint`: system+user+assistant; system message gets the key.
- `anthropic_max_two_distinct`: 10-message conversation, `stable_suffix=4` → returns ≤ 2,
  annotated indices are unique and the count ≤ 4 invariant holds.
- `openai_strips_annotations`: pre-annotated messages + `OpenAi` → 0 returned, no message
  has the key.
- `empty_messages_ok`: empty slice → 0, no panic.

**Definition of Done:** fmt/clippy/test clean; four tests pass.

---

### WP-8: Server v2

**Goal:** Make the server embeddable and configurable; fix the insecure default bind.

#### Changes (`crates/context-compress-server/src/lib.rs`, `src/main.rs`)

```rust
/// Backend selection for the server's CCR store.
pub enum CcrBackendConfig {
    InMemory,
    Sqlite { path: std::path::PathBuf, ttl_seconds: u64 },
    Fjall { path: std::path::PathBuf },
}

pub struct ServerConfig {
    /// Default: 127.0.0.1:3000  (was 0.0.0.0 — do not listen on all
    /// interfaces by default).
    pub bind: std::net::SocketAddr,
    pub ccr: CcrBackendConfig,
}
impl Default for ServerConfig { /* values above, InMemory */ }

impl AppState {
    /// Existing behavior preserved: pub fn new() -> Self (in-memory).
    /// New: build with any store.
    pub fn with_store(store: Arc<dyn CcrStore>) -> Self;
}
// AppState.ccr_store field type changes: Arc<InMemoryCcrStore> -> Arc<dyn CcrStore>.

/// Existing: pub fn app() -> Router  (keep, delegating to app_with_state(AppState::new()))
/// New: mountable router over caller-provided state — this is what Brehon embeds.
pub fn app_with_state(state: AppState) -> Router;
```

`main.rs` reads env vars (no CLI parsing dependency):

| Env var | Meaning | Default |
|---|---|---|
| `CC_BIND` | socket addr to bind | `127.0.0.1:3000` |
| `CC_CCR` | `memory` \| `sqlite` \| `fjall` | `memory` |
| `CC_CCR_PATH` | db path for sqlite/fjall | required when not memory; exit(1) with a clear message if missing |
| `CC_CCR_TTL_SECONDS` | sqlite TTL | `86400` |

Invalid values → print one-line error to stderr and exit code 1 (this is binary code,
not library code; `std::process::exit` is acceptable in `main.rs` only).

#### Request/response contracts (document in `lib.rs` doc comments; shapes already exist — make them explicit and add the missing fields)

```jsonc
// POST /compress   request:  {"messages":[{"role":"user","content":"..."}]}
//                  response: {"messages":[...], "stats":{"original_tokens":N,
//                             "compressed_tokens":N,"ratio":F,"compressor_used":"..."}}
// POST /retrieve   request:  {"id":"<md5hex>"}
//                  response: {"found":true,"original":"..."} or {"found":false}
// POST /detect     request:  {"content":"..."}
//                  response: {"content_type":"json_array","confidence":0.9}
// GET  /health     {"status":"ok"}
// GET  /stats      {"uptime_seconds":N,"requests":N}
```

#### Tests (`crates/context-compress-server/tests/http.rs`, new — use `tower::ServiceExt::oneshot`, add `dev-dependencies`: `tower = { workspace = true, features = ["util"] }`, `http-body-util = "0.1"`)

- `health_ok`: GET /health → 200, body contains `"ok"`.
- `compress_roundtrip`: POST /compress with one JSON-array message → 200, response
  parses, `stats.original_tokens >= stats.compressed_tokens` is NOT asserted (compression
  may be a no-op) — assert only that `messages.len() == 1` and stats fields exist.
- `retrieve_missing_is_found_false`: POST /retrieve with random id → 200 `{found:false}`.
- `detect_json`: POST /detect with `[{"a":1},{"a":2}]` → `content_type == "json_array"`.

**Definition of Done:** fmt/clippy/test clean; `app()` behavior unchanged for existing
callers; binary refuses bad env config with exit 1; four HTTP tests pass.

---

### WP-9: Evaluation, Golden Files & Benchmarks

**Goal:** Lock current behavior (golden regression), measure quality (probes), and
measure speed (criterion), so future changes are caught.

#### 9a. Golden files (`crates/context-compress/tests/golden.rs` + `tests/golden/` dir)

Layout:

```
tests/golden/
  inputs/
    json_array.txt        (≥ 50-element JSON array of objects with 1 outlier)
    build_log.txt         (≥ 200 lines: timestamps, INFO spam, 3 errors)
    source_code.txt       (≥ 100 lines of Rust with comments + docstrings)
    git_diff.txt          (a real-looking multi-file diff)
    prose.txt             (≥ 5 paragraphs with 2 duplicated paragraphs)
  expected/
    json_array.txt        (generated)
    build_log.txt
    source_code.txt
    git_diff.txt
    prose.txt
```

Harness behavior (single test fn per fixture, plus a shared helper):

1. Read input, run `compress_messages(vec![Message::new("tool", input)],
   CompressConfig::default())` (in-memory CCR → markers will contain content hashes,
   which are deterministic, so outputs are stable).
2. If env var `UPDATE_GOLDEN=1`: write the output to `expected/<name>.txt` and pass.
3. Else: read `expected/<name>.txt` and `assert_eq!` (on mismatch the failure message
   says: `golden mismatch for <name>; if intentional, rerun with UPDATE_GOLDEN=1`).

Author the five input fixtures by hand (synthetic but realistic; no copyrighted text),
then generate expected files once with `UPDATE_GOLDEN=1 cargo test -p context-compress
--test golden` and commit both.

#### 9b. Probe harness (`crates/context-compress/tests/probes.rs`)

Deterministic, no-LLM probes: after compression, key facts must still be findable by
substring search.

```rust
struct Probe { name: &'static str, needle: &'static str }
```

Build one synthetic 30-message agent conversation containing: an error trace
(`"Error: ECONNREFUSED 127.0.0.1:5432"`), a file path (`"src/billing/invoice.rs"`),
a decision line (`"Decision: use sqlite for the cache"`), and a next step
(`"TODO: add retry with backoff"`). Run, in order: `apply_agent_compression` (default
policy, in-memory CCR) then `enforce_budget` with a limit ~50% of the initial count.
Assert each probe needle is still present in the joined final message contents
**or** (for tool-result content) retrievable via a `<<ccr:...>>` marker present in the
final text. The error trace and decision must be present **directly** (not just via CCR).

#### 9c. Benchmarks (`crates/context-compress/benches/compress.rs`)

`Cargo.toml` additions:

```toml
[dev-dependencies]
criterion = { version = "0.5", default-features = false }

[[bench]]
name = "compress"
harness = false
```

Benches (input sizes 1 KB / 32 KB / 256 KB, generated deterministically in code):
`detect`, `smart_crusher_json`, `log_stripper`, `pipeline_end_to_end`. Use
`tokio::runtime::Runtime::new().unwrap().block_on(...)` inside the bench closures
(unwrap allowed: bench code is not library code). No performance assertions — benches
are for measurement, the target (<10 ms median on typical agent payloads ≤ 32 KB) is
tracked manually.

**Definition of Done:** `cargo test --workspace` passes including golden + probes;
`cargo bench -p context-compress --no-run` compiles; golden expected files committed.

---

## 6. Execution Order & Dependency Graph

```
WP-0 (hygiene)
  └─ WP-1 (bug fixes)
       └─ WP-2 (token counting)      ── independent of WP-3
       └─ WP-3 (message metadata)
            └─ WP-4 (agent rules)     needs WP-2 (heuristic counter) + WP-3 (tags)
                 └─ WP-5 (budget)     needs WP-2 + WP-4
                      └─ WP-6 (memory/summaries)   (uses conversation.rs; after WP-5 to
                                                    avoid merge conflicts, not a hard dep)
            └─ WP-7 (cache annotation) needs only WP-3; may run any time after it
  └─ WP-8 (server v2)                 needs WP-3 (Message serde) — schedule after WP-3
       └─ WP-9 (eval/golden/bench)    LAST — locks final behavior
```

Strict serial order `WP-0 → WP-1 → WP-2 → WP-3 → WP-4 → WP-5 → WP-6 → WP-7 → WP-8 → WP-9`
is always valid and is the recommended path for a single implementer.

---

## 7. Invariants (Must Always Hold)

Each invariant is enforced by at least one named test; if you change behavior, the test
must still encode the invariant.

| # | Invariant | Enforced by |
|---|---|---|
| I1 | Compressor error ⇒ original content returned unchanged | `tests/llm_safety.rs` fail-closed tests |
| I2 | Same input + config ⇒ identical output | llm_safety determinism tests; WP-6 `summarizer_deterministic` |
| I3 | No panics on arbitrary input | `tests/fuzz.rs`; WP-1 `test_summary_multibyte_no_panic` |
| I4 | `ToolResultError` messages are never modified or dropped | WP-4 `errors_never_cleared`; WP-5 `errors_survive_cascade` |
| I5 | System instructions and the latest user query are never modified or dropped | WP-4 `system_and_user_untouched`; WP-5 cascade rules |
| I6 | CCR round-trip: marker hash retrieves the exact original | llm_safety round-trip; WP-4 `cleared_content_roundtrip` |
| I7 | Over-budget prompts are never returned as "ok" — `BudgetExceeded` instead | WP-5 `budget_exceeded_errors` |
| I8 | Reported stats are measured, never fabricated | WP-1e `ccr_stats_not_fabricated` |
| I9 | Empty-metadata `Message` serializes identically to the pre-WP-3 format | WP-3 `message_metadata_roundtrip` |
| I10 | Library spawns no processes, threads(detached), or network connections | code review; Rule 3/4 (no test can prove a negative — keep deps minimal) |

---

## 8. Brehon Integration (External — Informational)

This work happens **in the Brehon repository**, not here. It is listed so the library's
API surface is designed against a real consumer. Verified facts about Brehon's current
state (verified directly against the Brehon codebase during planning): `headroom-core` pinned to git
rev `ec7d0065`; proxy mode spawns a `headroom` subprocess with port allocation and
45-second `/livez` health-check polling (`brehon-cli/src/commands/run/headroom_proxy.rs`,
~1,035 lines); two divergent code paths in `brehon-mcp/src/tools/context_efficiency.rs`;
`chars/4` token heuristic; no conversation-level budget.

Integration sequence (each step maps to library APIs that exist after the WPs):

1. **Dependency swap:** `headroom-core = { git = ... }` → `context-compress = { path/git }`.
   In-process calls move to `compress_messages` / `DefaultCompressionPipeline`.
2. **Kill the proxy subprocess:** replace `HeadroomProxyManager` with
   `context_compress_server::app_with_state(...)` mounted in Brehon's existing Axum
   router (WP-8), or drop HTTP entirely and call the library.
3. **Audit logging:** implement `Observer` writing Brehon's `audit.jsonl` event format.
4. **Budget:** call `budget::enforce_budget` in Brehon's message-assembly path with
   `counter_for_model(model)`.
5. **Agent rules:** Brehon tags messages with `meta_keys::TOOL_NAME` / `TOOL_STATUS` at
   the point it records tool results; calls `agent::apply_agent_compression`.
6. **LTM:** Brehon implements `Summarizer` with its own LLM client; persists
   `StructuredSummary` (it's `serde`-serializable) in its fjall event store; passes
   `FjallCcrStore::from_keyspace`-style sharing if desired (constructor exists).
7. **Config mapping** (Brehon yaml):
   ```yaml
   context:
     compression:
       enabled: true
       mode: context_compress     # was: headroom (+command/proxy keys, now removed)
       compact_memories: true
       compact_rules: true
       never_compress: []         # maps to metadata cc.pinned = "true"
   ```

---

## 9. Decision Records

### ADR-1: Rule-based core, ML optional (unchanged)
Deterministic rule-based compression for JSON/logs/code/tool-clearing. ML (perplexity
pruning) remains out of scope until the rule-based system is integrated and measured.
Determinism is a safety requirement (Rule 5).

### ADR-2: Reversible STM, lossy LTM, archival source of truth (unchanged)
CCR makes message-level compression reversible. Structured summaries are intentionally
lossy. The CCR store retains originals subject to TTL; hosts choose durable backends.

### ADR-3: Optimize total-tokens-to-task-completion, not ratio (unchanged)
A 70% reduction that preserves file paths and errors beats a 99% reduction that loses
them (Factory.ai finding). This is why WP-6 extraction rules prioritize paths, errors,
decisions — and why probes (WP-9b) test recall, not ratio.

### ADR-4 (REVISED): Caller-driven compaction
v1.0 specified background async compaction. Rejected: a library spawning detached tasks
is a footgun (runtime coupling, shutdown races, surprise CPU). All compaction APIs are
plain `async fn`s the host calls; the host may schedule them in its own background task.

### ADR-5: No subprocesses (unchanged)
The library never spawns processes. The server is an optional embeddable router, not a
sidecar requirement.

### ADR-6 (NEW): Honest token counting
Exact counting exists only where the tokenizer is public (OpenAI encodings via
`tiktoken-rs`, feature-gated). Claude counts are estimates; budget enforcement
compensates with an explicit safety margin (default 5% for inexact counters). No API is
named or documented as "exact" for models we cannot tokenize.

### ADR-7 (NEW): Deterministic summarizer by default, LLM via trait
The library cannot and does not call LLMs. `ExtractiveSummarizer` is deterministic and
testable; hosts that want abstractive quality implement `Summarizer` with their own LLM
client. This keeps Rule 3/5 intact and makes summaries reproducible in CI.

### ADR-8 (NEW): Metadata over typed message hierarchies
Agent semantics (tool name/status, pinning, cache breakpoints, CCR ids) ride on
`Message.metadata: HashMap<String,String>` with namespaced `cc.*` keys, instead of new
message types. Rationale: provider-agnostic, serde-backward-compatible (skip-if-empty),
and hosts already have their own message types — string tags are the cheapest stable
interop surface.

### ADR-9 (NEW): MD5 for CCR keys is fine
CCR keys are content addresses for dedup/lookup, not security boundaries. MD5 is
deterministic, fast, and already shipped (`ccr::compute_key`). Do not "upgrade" it —
changing the hash breaks every stored marker.

---

## 10. References

- Brehon codebase: `/Users/recursive/workspace/brehon`
- Headroom: https://github.com/chopratejas/headroom (attribution in `NOTICE`)
- Anthropic, "Effective Context Engineering for AI Agents":
  https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Factory.ai, "Evaluating Context Compression": https://factory.ai/news/evaluating-compression
- Microsoft LLMLingua-2: https://llmlingua.com/llmlingua2.html
- Acon (Agent Context Optimization): https://arxiv.org/abs/2510.00615
