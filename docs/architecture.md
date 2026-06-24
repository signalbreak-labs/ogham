# Architecture

Ogham is three crates with a strict one-way dependency flow:

```
┌─────────────────────────────────────────────────────────────┐
│                    Host application                          │
│        (an agent runtime, a CLI, your Axum service)          │
└──────────────┬──────────────────────────┬───────────────────┘
               │ library calls            │ HTTP (optional)
┌──────────────▼──────────────┐  ┌────────▼───────────────────┐
│           ogham             │◄─┤       ogham-server         │
│  detection · compressors    │  │  embeddable Axum router    │
│  pipeline · CCR stores      │  │  /compress /retrieve       │
│  agent rules · budgets      │  │  /detect /health /stats    │
│  compaction · sessions      │  └────────────────────────────┘
│  rich content · fold recall │
│  provider cache planning    │
└──────────────┬──────────────┘
┌──────────────▼──────────────┐
│         ogham-core          │
│  Message · Compressor       │
│  TokenCounter · Observer    │
│  Metrics · OghamError       │
└─────────────────────────────┘
```

- **`ogham-core`** holds only traits, types, and the error enum. It has five
  small dependencies and no async runtime, storage, or algorithms. Implement
  `Compressor`, `Observer`, or `TokenCounter` against this crate alone.
- **`ogham`** is the SDK: everything below.
- **`ogham-server`** is optional. Nothing in the SDK requires it.

## Data flow: content level

```
text ──► detect() ──► route by ContentType ──► compressor ──► compressed text
                                                   │
                                                   ▼ (reversible mode)
                                             CcrStore.save(original)
                                             output contains <<ccr:HASH>>
```

| ContentType | Compressor | Strategy |
|---|---|---|
| `JsonArray` | `smart_crusher` | keep outliers, sample representatives (knee-point detection), CCR markers for the rest |
| `BuildOutput` | `log_stripper` | format detection, line importance scoring (errors > warnings > info), dedup |
| `SourceCode`, `GitDiff` | `ast_code` | strip comments/blank lines, keep signatures and imports |
| `SearchResults`, `Html`, `PlainText` | `semantic` | paragraph dedup, long-paragraph truncation |
| (registered, not routed by default) | `dedup_ref` | content-addressed block dedup |
| (registered, not routed by default) | `toon` | JSON minify / whitespace collapse, lossy, zero-dep |

Routing lives in `DefaultCompressionPipeline::compress_one`
(`crates/ogham/src/pipeline.rs`).

## Data flow: conversation level

The conversation-level passes compose in a fixed order (each is optional):

```
1. agent::apply_agent_compression   tag & clear stale tool results → CCR
2. budget::enforce_budget           cascade until under ContextBudget
3. cache_aligner::align_messages    stabilize bytes for KV-cache reuse
4. cache_strategy::apply_cache_strategy   annotate provider breakpoints
```

The budget cascade escalates cheap → lossy, re-counting after every step and
stopping as soon as the history fits:

```
agent_rules → compress_middle → summarize_old → drop_old → Err(BudgetExceeded)
```

`drop_old` never removes system instructions, tool errors, the latest user
query, or anything tagged `ogham.pinned = "true"`. If nothing more can be
dropped and the history still exceeds the limit, `enforce_budget` returns
`OghamError::BudgetExceeded` — an oversized prompt is never reported as ok.

## High-level entry points

Most hosts use one of three unified entry points rather than composing the
passes by hand; each runs the cascade and returns auditable records:

| Entry point | Module | For |
|---|---|---|
| `compact_conversation()` | `compact` | one-shot compaction of a flat `Vec<Message>` → `CompactResult` (folds, protected report, budget/agent reports, cache plan, warnings) |
| `compact_rich()` | `rich` | the block-aware analogue over `Vec<RichMessage>` — keeps tool ids, images, and references structured; folds become reversible flat text |
| `ContextSession` | `session` | stateful, incremental compaction: `push()` turns, `compact()` only the active tail, freeze already-folded messages (work proportional to new content; append-only fold ledger; byte-stable finalized region for prompt-cache reuse) |

Two subsystems make the fold ledger queryable:

- **`recall::RecallIndex`** — a deterministic BM25 keyword index over folded
  content, addressable by CCR id, so a host can search folds by relevance to
  recover the ids to retrieve. `ContextSession` maintains one automatically.
- **`fold_tags`** — each `FoldRecord` carries typed `FoldTags`
  (`tool_names` / `error_classes` / `file_paths`), filterable via
  `RecallIndex::find_by_tag`.

Provider cache planning lives in `providers::{openai, gemini, anthropic}`: Ogham
emits a provider-shaped `CachePlan` (stable-prefix accounting, content keys,
breakpoint annotations, native Anthropic block rendering); the host owns the HTTP
call and auth.

## CCR (Compression-Compaction-Reference)

Compression is reversible by default. Originals are stored under a versioned,
collision-resistant content address — a 128-bit BLAKE3 prefix of the form
`b3:<32 hex>` (`ccr::compute_key`) — and referenced inline as `<<ccr:HASH>>`.
The hash is a content address, not a security boundary; the `b3:` version tag
lets the scheme evolve while older ids stay retrievable.

| Store | Feature | Use case |
|---|---|---|
| `InMemoryCcrStore` | (always) | tests, ephemeral sessions (TTL + capacity eviction); `::unbounded()` never evicts, for durable sessions |
| `SqliteCcrStore` | `ccr-sqlite` | single-node persistence (WAL mode, TTL) |
| `FjallCcrStore` | `ccr-fjall` | LSM-tree production store; can share an existing fjall keyspace |

The persistent backends are **opt-in features** (not in the default build).
Beyond plain text, a `CcrStore` can persist a typed `CcrPayload`
(`save_payload` / `retrieve_payload`) for lossless structured originals (e.g.
serialized `RichMessage` blocks). UTF-8 payloads use a self-describing text
envelope (rollback-safe); binary payloads are stored natively — SQLite in BLOB +
media-type/metadata columns, fjall in a length-prefixed frame — so binary costs
its real size with no hex overhead.

## Design invariants

These hold everywhere and are enforced by tests (`tests/llm_safety.rs`,
`tests/fuzz.rs`, `tests/probes.rs`):

1. **Fail-closed.** Any error returns the original content unchanged.
2. **Deterministic.** Same input + config ⇒ byte-identical output. No clocks,
   no randomness, no map-iteration-order dependence.
3. **No side effects.** The SDK spawns no processes, opens no sockets, calls
   no LLMs, and starts no background tasks. Compaction is caller-driven.
4. **Protected content.** Tool errors, system instructions, and the latest
   user query survive every pass verbatim.
5. **Honest accounting.** Reported stats are measured, never fabricated;
   token counts are labelled exact or estimated (`TokenCounter::is_exact`).

## Why these choices (decision records)

The full ADRs live in [DESIGN.md](../DESIGN.md) §9. The short version:

- **Rule-based core, ML optional** — determinism is a safety requirement.
- **Caller-driven compaction** — a library that spawns background tasks is a
  footgun; hosts schedule compaction themselves.
- **Optimize tokens-to-task-completion, not ratio** — a 70% reduction that
  preserves file paths and errors beats a 99% reduction that loses them.
- **Metadata over typed hierarchies** — agent semantics ride on
  `Message.metadata` string tags (`ogham.*`), keeping the SDK
  provider-agnostic.
