# Ogham

[![CI](https://github.com/signalbreak-labs/ogham/actions/workflows/ci.yml/badge.svg)](https://github.com/signalbreak-labs/ogham/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/ogham.svg)](https://crates.io/crates/ogham)
[![docs.rs](https://img.shields.io/docsrs/ogham)](https://docs.rs/ogham)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

> Ogham — the early-Irish script that carved language into compact notches.
> This SDK does the same to your LLM context.

Ogham is a pure-Rust SDK for **LLM context engineering**: it compresses,
prunes, and budgets conversation context before it reaches the model —
entirely in-process. Agentic workloads routinely burn 60–90% of their tokens
on stale tool output; Ogham reclaims them without losing the things that
matter (errors, decisions, file paths), and everything it removes stays
retrievable.

**No subprocesses. No network calls. No background tasks.** Every operation
is a deterministic library call.

## What it does

| Capability | In one line |
|---|---|
| **Content compression** | detects JSON / logs / code / diffs / prose and routes each to a purpose-built compressor |
| **Reversible by default** | originals live in a CCR store (in-memory, SQLite, fjall); compressed text carries `<<ccr:HASH>>` markers the model can dereference |
| **Agent-aware rules** | clears stale successful tool results to one-line stubs; **never** touches errors, system prompts, the latest user query, or pinned messages |
| **Token budgets** | a degradation cascade (compress → summarize → drop) that fits any history into a limit — or refuses with `BudgetExceeded` rather than overflow |
| **Honest token counting** | exact for OpenAI encodings (`tiktoken` feature); calibrated estimates with an explicit safety margin elsewhere — Claude tokenizers aren't public and Ogham doesn't pretend otherwise |
| **Structured summaries** | deterministic extraction of files / decisions / errors / next-steps, plus a `Summarizer` trait for LLM-backed memory |
| **Prompt-cache support** | byte-stable prefixes + provider breakpoint annotation for KV-cache reuse |
| **Observable** | every decision streams through `Observer`/`Metrics` traits — auditable, testable |

## Install

```toml
[dependencies]
ogham = "0.1"
# exact OpenAI token counting:
# ogham = { version = "0.1", features = ["tiktoken"] }
```

## Sixty seconds

**Compress a tool result** (content level):

```rust
use ogham::{compress_messages, CompressConfig, Message};

let out = compress_messages(
    vec![Message::new("tool", huge_json_or_log_output)],
    CompressConfig::default(),
).await?;
println!("{} -> {} tokens ({:.0}% saved)",
    out.stats.original_tokens, out.stats.compressed_tokens,
    (1.0 - out.stats.ratio) * 100.0);
```

**Fit a whole conversation into a budget** (conversation level):

```rust
use std::sync::Arc;
use ogham::agent::{apply_agent_compression, AgentPolicy};
use ogham::budget::{enforce_budget, ContextBudget};
use ogham::ccr::in_memory::InMemoryCcrStore;
use ogham::pipeline::DefaultCompressionPipeline;
use ogham::counter_for_model;

let ccr = Arc::new(InMemoryCcrStore::new());
let pipeline = DefaultCompressionPipeline::with_ccr_store(ccr.clone());
let policy = AgentPolicy::default();

// 1. clear stale tool results to retrievable CCR stubs
apply_agent_compression(&mut messages, &policy, Some(ccr.clone())).await?;

// 2. guarantee the prompt fits — or get Err(BudgetExceeded), never an oversized send
enforce_budget(
    &mut messages,
    &ContextBudget { total_limit: 180_000, safety_margin: None },
    counter_for_model("claude-fable-5").as_ref(),
    &pipeline, &policy, Some(ccr),
).await?;
```

A cleared tool result looks like this — and the original is one
`ccr.retrieve(hash)` away:

```text
[tool:file_read] result cleared (512 tokens) — original retrievable via <<ccr:86a33abc...>>
```

## Guarantees

1. **Fail-closed.** Any internal error returns the original content
   unchanged. A budget that can't be met is an error, not a truncated prompt.
2. **Deterministic.** Same input + same config ⇒ byte-identical output.
   No clocks, no randomness.
3. **Protected content.** Tool errors, system instructions, and the latest
   user query survive every pass verbatim — enforced by tests, not
   convention.
4. **Measured, not fabricated.** Stats reflect events that actually fired;
   estimated token counts are labelled as estimates.

## Documentation

| Doc | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | crate layering, data flow, invariants, design decisions |
| [docs/compression.md](docs/compression.md) | detection, the six compressors, CCR stores, cache alignment |
| [docs/agent-context.md](docs/agent-context.md) | classification, tool clearing, budgets, summaries, cache strategy |
| [docs/server.md](docs/server.md) | the optional HTTP server: env vars, endpoints, embedding |
| [docs/integration.md](docs/integration.md) | wiring Ogham into an agent runtime, end to end |
| [DESIGN.md](DESIGN.md) | the full engineering record: build plan, work packages, ADRs |
| [CONTRIBUTING.md](CONTRIBUTING.md) · [RELEASING.md](RELEASING.md) · [CHANGELOG.md](CHANGELOG.md) | process |

API reference: [docs.rs/ogham](https://docs.rs/ogham) ·
[docs.rs/ogham-core](https://docs.rs/ogham-core) ·
[docs.rs/ogham-server](https://docs.rs/ogham-server)

## Crates

| Crate | What | When to depend on it |
|---|---|---|
| [`ogham`](crates/ogham) | the SDK: compressors, CCR, pipeline, agent rules, budgets, summaries | almost always |
| [`ogham-core`](crates/ogham-core) | traits + types only (5 small deps, no runtime) | implementing custom compressors/observers |
| [`ogham-server`](crates/ogham-server) | embeddable Axum HTTP server + standalone binary | non-Rust clients, standalone deployment |

## Status

`0.1.0` — APIs may change before 1.0 (semver-minor may break). The test
suite covers fail-closed behavior, determinism, fuzzing, golden-file
regression, and needle-in-haystack survival probes; see
[CONTRIBUTING.md](CONTRIBUTING.md) for the gate every change must pass.

## License & attribution

Apache-2.0 — see [LICENSE](LICENSE).

Ogham's architecture is substantially derived from
[Headroom](https://github.com/chopratejas/headroom) (Apache-2.0): the CCR
reversible-storage concept, content-type routing, and the SmartCrusher /
LogStripper compressor designs originate there. The agent-aware rules,
token budgets, tiered summaries, and cache integration are informed by
published research from Anthropic, Factory.ai, and Microsoft (LLMLingua).
Full attribution in [NOTICE](NOTICE).
