# Agent-Aware Context Management

This guide covers the conversation-level API: classification, tool-result
clearing, token budgets, structured summaries, and prompt-cache annotation.

## Metadata keys

Agent semantics ride on `Message.metadata` (a `HashMap<String, String>`),
using namespaced keys defined in `ogham_core::meta_keys`:

| Constant | Key | Meaning |
|---|---|---|
| `AGENT_CONTENT_TYPE` | `ogham.agent_content_type` | explicit classification override (snake_case `AgentContentType` name) |
| `TOOL_STATUS` | `ogham.tool_status` | `"success"` or `"error"` for tool-result messages |
| `TOOL_NAME` | `ogham.tool_name` | name of the tool that produced a result |
| `CCR_ID` | `ogham.ccr_id` | content hash of the original, set when a message is cleared |
| `CACHE_CONTROL` | `ogham.cache_control` | `"ephemeral"` — provider cache breakpoint |
| `PINNED` | `ogham.pinned` | `"true"` — never compress, clear, summarize, or drop |

Empty metadata is omitted from serialized JSON, so plain
`{"role": ..., "content": ...}` wire formats are unchanged.

**Tag at the source.** When your runtime records a tool result, set
`TOOL_NAME` and `TOOL_STATUS` immediately — classification is much more
reliable from tags than from content sniffing.

## Classification

```rust
use ogham::agent::{classify, AgentContentType};

match classify(&msg) {
    AgentContentType::ToolResultError => { /* untouchable */ }
    AgentContentType::ToolResultSuccess => { /* clearable when stale */ }
    _ => {}
}
```

Precedence (first match wins):

1. `metadata[AGENT_CONTENT_TYPE]`, if present and valid
2. role `system` → `SystemInstruction`
3. role `tool`/`function` → `ToolResultError` if `TOOL_STATUS == "error"` or
   the first 512 bytes contain an error pattern (`Error:`, `Traceback`,
   `panicked at`, `Exception`, `FAILED`, …; see `agent::ERROR_PATTERNS`);
   otherwise `ToolResultSuccess`
4. role `user` → `UserQuery`; role `assistant` → `AssistantReply`
5. anything else → `Unknown`

## Tool-result clearing

The highest-ROI technique for agent workloads. Old successful tool results
become one-line stubs whose full content stays retrievable:

```rust
use ogham::agent::{apply_agent_compression, AgentPolicy};

let stats = apply_agent_compression(
    &mut messages,
    &AgentPolicy::default(),   // keep newest 3 tool results raw, clear older
    Some(ccr_store.clone()),
).await?;
// stats.tool_results_cleared, stats.tokens_before, stats.tokens_after
```

A cleared message looks like:

```text
[tool:file_read] result cleared (512 tokens) — original retrievable via <<ccr:86a33abc...>>
```

Guarantees, in priority order:

- **Errors are never touched.** `ToolResultError` survives verbatim, always.
- **System instructions and user queries are never touched.**
- **Pinned messages (`ogham.pinned = "true"`) are never touched.**
- The newest `keep_recent_tool_results` successes stay raw (default 3).
- With no CCR store, nothing is cleared — content is never discarded
  irretrievably (fail-closed).

## Token budgets

```rust
use ogham::budget::{enforce_budget, ContextBudget};
use ogham::counter_for_model;

let report = enforce_budget(
    &mut messages,
    &ContextBudget { total_limit: 180_000, safety_margin: None },
    counter_for_model(model).as_ref(),
    &pipeline,
    &AgentPolicy::default(),
    Some(ccr_store),
).await?;
// report.steps_applied: e.g. ["agent_rules", "compress_middle"]
```

The cascade runs until the history fits, escalating cheap → lossy:

| Step | What it does |
|---|---|
| `agent_rules` | tool-result clearing (above) |
| `compress_middle` | compress all but the 4 newest turns through the pipeline |
| `summarize_old` | fold turns older than the middle band into one summary message |
| `drop_old` | remove oldest droppable messages one at a time |

`drop_old` never removes system instructions, tool errors, the most recent
user message, or pinned messages. If the history still doesn't fit,
`enforce_budget` returns `OghamError::BudgetExceeded { needed, limit }` —
treat that as "do not send this prompt."

`safety_margin: None` means: 0% when the counter is exact, 5% otherwise
(Claude counts are estimates — see [compression.md](compression.md#token-counting)).

## Structured summaries

Freeform summaries silently lose file paths and decisions over time. Ogham's
`StructuredSummary` keeps them in explicit sections:

```rust
use ogham::memory::{ExtractiveSummarizer, Summarizer, StructuredSummary};

let summary: StructuredSummary = ExtractiveSummarizer
    .summarize(&old_turns, previous_summary.as_ref())   // anchored: merges, never regenerates
    .await?;
println!("{}", summary.render_markdown());
```

Rendered output (empty sections omitted):

```markdown
## Session intent
- migrate the billing service to sqlite

## Files touched
- src/billing/invoice.rs

## Decisions
- Decision: use sqlite for the cache

## Unresolved errors
- Error: ECONNREFUSED 127.0.0.1:5432

## Next steps
- TODO: add retry with backoff
```

`ExtractiveSummarizer` is deterministic and LLM-free (regex extraction of
file paths, decision lines, error lines, TODOs). For abstractive quality,
implement the `Summarizer` trait with your own LLM client — the SDK never
makes network calls itself. `StructuredSummary` is `serde`-serializable for
cross-session persistence.

To use a summarizer for the old band of a conversation:

```rust
use ogham::conversation::compress_conversation_history_with_summarizer;

compress_conversation_history_with_summarizer(
    &mut messages, &config, &pipeline, &my_summarizer,
).await?;
// on summarizer error this falls back to the built-in extractive path
```

## Prompt-cache annotation

Ogham marks where provider cache breakpoints should go; your provider
adapter translates the marks into the wire format (e.g. Anthropic
`cache_control`):

```rust
use ogham::cache_strategy::{apply_cache_strategy, CacheStrategy};

// stable_suffix = how many trailing messages change every turn
let n = apply_cache_strategy(&mut messages, CacheStrategy::Anthropic, 4);
// messages with metadata["ogham.cache_control"] == "ephemeral" are breakpoints
```

Anthropic strategy sets at most two breakpoints (well under the 4-per-request
limit): the last system message, and the end of the stable prefix.
`CacheStrategy::OpenAi` / `Generic` remove any annotations (those providers
cache prefixes automatically) — pair with `align_messages` for byte-stable
prefixes either way.

## Putting it together

```rust
// 1. agent rules — clear stale tool results
apply_agent_compression(&mut msgs, &policy, Some(ccr.clone())).await?;

// 2. budget — guarantee the prompt fits, or refuse
enforce_budget(&mut msgs, &budget, counter.as_ref(), &pipeline, &policy,
               Some(ccr.clone())).await?;

// 3. byte stability + cache breakpoints
align_messages(&mut msgs);
apply_cache_strategy(&mut msgs, CacheStrategy::Anthropic, 4);

// 4. translate metadata to your provider's wire format and send
```
