# Ogham

> Ogham — the early-Irish script that carved language into compact notches.
> This SDK does the same to your LLM context.

Ogham is a pure-Rust SDK for LLM context engineering: reversible compression of
tool outputs, logs, code, and JSON; agent-aware rules (never touch errors, clear
stale tool results to retrievable markers); conversation-level token budgets
with a graceful degradation cascade; structured summaries; and prompt-cache
breakpoint annotation. No subprocesses, no network calls, no background tasks —
everything is a library call.

## Crates

| Crate | Purpose |
|---|---|
| `ogham-core` | Traits and types only (`Message`, `Compressor`, `TokenCounter`, `Observer`, …) |
| `ogham` | Compressors, CCR stores (in-memory / SQLite / fjall), pipeline, agent rules, budgets, summaries |
| `ogham-server` | Optional embeddable Axum HTTP server (`/compress`, `/retrieve`, `/detect`, `/stats`) |

## Quick taste

```rust
use ogham::budget::{ContextBudget, enforce_budget};
use ogham::agent::{AgentPolicy, apply_agent_compression};
use ogham::counter_for_model;

// Clear stale tool results to CCR markers, then fit the conversation
// into a token budget — errors and the latest user query always survive.
apply_agent_compression(&mut messages, &AgentPolicy::default(), Some(ccr)).await?;
enforce_budget(&mut messages, &ContextBudget { total_limit: 180_000, safety_margin: None },
               counter_for_model("claude-fable-5").as_ref(), &pipeline,
               &AgentPolicy::default(), Some(ccr)).await?;
```

Design guarantees: fail-closed (errors return originals unchanged),
deterministic (same input + config ⇒ identical output), and honest token
counting (exact for OpenAI encodings via the `tiktoken` feature; calibrated
estimates with a safety margin elsewhere).

The full design and build plan lives in [DESIGN.md](DESIGN.md).

## License & attribution

Apache-2.0. Ogham's architecture is substantially derived from
[Headroom](https://github.com/chopratejas/headroom) (Apache-2.0) — see
[NOTICE](NOTICE) for full attribution.
