# Contributing to Ogham

## The gate

Every change must pass, from the workspace root:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test -p ogham --features tiktoken
```

CI runs the same gate on Linux, macOS, and Windows, plus an MSRV (1.85)
check and `cargo doc` with warnings denied.

## Hard rules

These are design invariants, not style preferences. PRs that violate them
will be declined regardless of functionality:

1. **Fail-closed.** Any error during compression returns the original
   content unchanged. Never propagate an error into message content.
2. **No panics in library code.** No `unwrap()`, `expect()`, `panic!()`, or
   unchecked slicing (`&s[..n]` — use a char-boundary-safe helper) outside
   `#[cfg(test)]`, test files, benches, and the server binary's `main.rs`.
3. **No side effects.** `ogham-core` and `ogham` must not spawn processes,
   open sockets, call LLMs, or start background tasks.
4. **Determinism.** Same input + config ⇒ byte-identical output. No clocks
   or randomness in output paths; sort before emitting anything derived
   from a map.
5. **Honest reporting.** Stats are measured, never fabricated. Don't call an
   estimate "exact".
6. **`ogham-core` stays tiny.** Allowed deps: `async-trait`, `bytes`,
   `serde`, `serde_json`, `thiserror`. New `ogham` deps need a feature gate
   unless discussed first.
7. **Never weaken a test to make the gate pass.**

Public items need `///` docs covering behavior and failure modes.

## Test layout

| Location | What |
|---|---|
| `src/**` `mod tests` | unit tests beside the code |
| `tests/llm_safety.rs` | invariant tests: fail-closed, determinism, round-trips |
| `tests/fuzz.rs` | adversarial inputs must not panic |
| `tests/golden.rs` + `tests/golden/` | output-stability regression tests |
| `tests/probes.rs` | needle-in-haystack survival across the full agent pipeline |
| `benches/compress.rs` | criterion benchmarks (`cargo bench -p ogham`) |

### Golden files

If you intentionally change compression output:

```bash
UPDATE_GOLDEN=1 cargo test -p ogham --test golden
```

then commit the updated `tests/golden/expected/*` files **and explain the
output change in your PR description** — golden diffs are the review surface
for behavior changes.

## Adding a compressor

1. Create `crates/ogham/src/compressors/your_name.rs` implementing
   `ogham_core::Compressor`. Name it in `snake_case` via `name()`.
2. Fail-closed inside `compress`: on any internal error return `Err(..)` —
   the pipeline keeps the original automatically.
3. If reversible, accept an `Arc<dyn CcrStore>` (see `with_ccr_store` on the
   existing compressors) and embed `ogham::ccr::marker_for(hash)` markers.
4. Register it in `DefaultCompressionPipeline::with_ccr_store` /
   `with_ccr_store_reuse`, and route a `ContentType` to it in
   `compress_one` if it should be selected automatically.
5. Add unit tests + a fuzz case + (if output-bearing) a golden fixture.

## Commit style

Imperative subject ≤ 72 chars, body explains *why* when it isn't obvious.
One logical change per commit; the gate must pass at every commit.

## Releases

See [RELEASING.md](RELEASING.md). The design history and decision records
live in [DESIGN.md](DESIGN.md).
