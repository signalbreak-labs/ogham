# Integrating Ogham into an Agent Runtime

This is the host-integration guide: what to wire where when embedding Ogham
into an agent orchestrator, LLM gateway, or chat application.

## The five integration points

### 1. Tag messages at the source

Classification works best from explicit tags. When your runtime records a
tool invocation, tag the result message immediately:

```rust
use ogham_core::{meta_keys, Message};

let mut msg = Message::new("tool", tool_output);
msg.metadata.insert(meta_keys::TOOL_NAME.into(), "file_read".into());
msg.metadata.insert(
    meta_keys::TOOL_STATUS.into(),
    if ok { "success" } else { "error" }.into(),
);
```

Anything the model must always see gets pinned:

```rust
msg.metadata.insert(meta_keys::PINNED.into(), "true".into());
```

### 2. One CCR store per deployment, shared everywhere

Use a single store for the pipeline, agent rules, and your retrieve tool so
every `<<ccr:HASH>>` marker resolves:

```rust
use std::sync::Arc;
use ogham::ccr::{CcrStore, fjall::FjallCcrStore};

let ccr: Arc<dyn CcrStore> = Arc::new(FjallCcrStore::new(data_dir.join("ccr"))?);
```

Already running fjall? Share the keyspace instead of opening a second
database — see `FjallCcrStore` constructors. Parallel agents sharing one
store deduplicate automatically (keys are content hashes).

**Expose retrieval to the model.** Use the built-in definition and
dispatcher — `ogham::tools::retrieve_tool_definition()` and
`ogham::tools::handle_retrieve_call(args, ccr)` — so every host wires it
identically. The model will use it when a cleared result turns out to
matter. Budget for retrievals: clearing saves tokens *net of* the
occasional re-fetch.

**Tag tool-call pairs.** Set `meta_keys::TOOL_CALL_ID` on both the
assistant message that makes a tool call and its result message(s). Ogham
keeps pairs intact positionally either way, but the tag makes audits and
tests exact.

**On Claude, optionally delegate first-line clearing to the platform** via
`ogham::providers::anthropic::AnthropicContextEditing` (see
[agent-context.md](agent-context.md#delegating-to-anthropics-server-side-clearing)).

### 3. Compress at the message-assembly boundary

Run the conversation-level passes once per LLM call, right before provider
serialization (full snippet in
[agent-context.md](agent-context.md#putting-it-together)):

```rust
apply_agent_compression(&mut msgs, &policy, Some(ccr.clone())).await?;
enforce_budget(&mut msgs, &budget, counter.as_ref(), &pipeline, &policy,
               Some(ccr.clone())).await?;   // Err(BudgetExceeded) => do NOT send
align_messages(&mut msgs);
apply_cache_strategy(&mut msgs, CacheStrategy::Anthropic, policy_recent);
```

Your provider adapter then maps `metadata["ogham.cache_control"] ==
"ephemeral"` onto the provider's cache-control wire format, and strips
`ogham.*` keys from what is actually sent.

### 4. Audit via `Observer`, measure via `Metrics`

Every compression decision is observable. Write them to your audit log:

```rust
use ogham_core::{CompressionEvent, Observer};

struct AuditObserver { /* your sink */ }

impl Observer for AuditObserver {
    fn on_event(&self, event: &CompressionEvent) {
        match event {
            CompressionEvent::MessageCompressed { compressor, original_tokens,
                                                  compressed_tokens, .. } => { /* log */ }
            CompressionEvent::Error { stage, message } => { /* log */ }
            _ => {}
        }
    }
}

let pipeline = DefaultCompressionPipeline::builder()
    .ccr_store(ccr.clone())
    .observer(Arc::new(AuditObserver { /* ... */ }))
    .build();
```

`Metrics` is the aggregate counterpart (counters/histograms for Prometheus
and friends). Both default to no-ops.

### 5. LLM-backed summaries (optional)

The built-in summarizer is extractive and deterministic. If you want
abstractive long-term memory, implement `Summarizer` with your own LLM
client — Ogham deliberately makes no network calls:

```rust
use ogham::memory::{StructuredSummary, Summarizer};

struct LlmSummarizer { client: MyLlmClient }

#[async_trait::async_trait]
impl Summarizer for LlmSummarizer {
    async fn summarize(&self, turns: &[Message], existing: Option<&StructuredSummary>)
        -> ogham::Result<StructuredSummary>
    {
        // call your model, parse into StructuredSummary sections,
        // merge with `existing` (anchored summarization)
        // on any failure: return Err — ogham falls back to extractive
    }
}
```

Persist `StructuredSummary` (it's `Serialize`/`Deserialize`) to carry memory
across sessions.

## Choosing budgets

`ContextBudget.total_limit` should be your model's context window minus the
response headroom you need, e.g. for a 200k window reserving 20k for output:

```rust
ContextBudget { total_limit: 180_000, safety_margin: None }
```

Leave `safety_margin: None` unless you've measured: it auto-selects 0% for
exact counters and 5% for estimates. Enable the `tiktoken` feature for exact
OpenAI counts:

```toml
ogham = { git = "https://github.com/signalbreak-labs/ogham", tag = "v0.2.0", features = ["tiktoken"] }
```

## Configuration mapping example

A typical host config block and what it maps to:

```yaml
context:
  compression:
    enabled: true                 # gate all passes
    keep_recent_tool_results: 3   # AgentPolicy.keep_recent_tool_results
    budget_tokens: 180000         # ContextBudget.total_limit
    ccr: fjall                    # which CcrStore to construct
    never_compress:               # tag matching messages ogham.pinned = "true"
      - system_prompt
```

## Failure-mode cheat sheet

| Situation | What Ogham does | What you should do |
|---|---|---|
| Compressor errors on a message | passes the original through, emits `CompressionEvent::Error` | nothing; optionally alert on frequency |
| CCR save fails during clearing | leaves that message uncleaned | nothing |
| `retrieve` misses (TTL expiry) | returns `Ok(None)` | tell the model the content expired |
| Budget can't be met | `Err(BudgetExceeded { needed, limit })` | do **not** send; surface to the user or raise the limit |
| Tokenizer unavailable | heuristic counter, `is_exact() == false`, 5% margin | nothing |
