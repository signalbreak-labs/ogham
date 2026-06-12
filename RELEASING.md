# Releasing Ogham

Releases are tag-driven and automated by
[`.github/workflows/release.yml`](.github/workflows/release.yml).
Each release publishes the crates to **crates.io** and creates a
**GitHub release** with `ogham-server` binaries.

## Cutting a release

1. Update the version once, in the workspace root `Cargo.toml`:

   ```toml
   [workspace.package]
   version = "0.2.0"
   ```

   and the two pinned path-dep versions right below it:

   ```toml
   ogham-core = { path = "crates/ogham-core", version = "0.2.0" }
   ogham = { path = "crates/ogham", version = "0.2.0" }
   ```

2. Add a `## [0.2.0] - YYYY-MM-DD` section to `CHANGELOG.md`
   (keep-a-changelog format — the GitHub release notes are extracted
   from it).

3. Verify locally:

   ```bash
   cargo fmt --all --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   cargo test -p ogham --features tiktoken
   ```

4. Commit, tag, push:

   ```bash
   git commit -am "release: v0.2.0"
   git tag -a v0.2.0 -m "Ogham v0.2.0"
   git push origin main v0.2.0
   ```

## What the automation does

```
tag v0.2.0
   │
   ├─ verify ............ tag == workspace version, full gate
   ├─ publish-crates .... cargo publish: ogham-core → ogham → ogham-server
   └─ create-release .... GitHub release, notes from CHANGELOG.md
        └─ binaries ..... ogham-server for 4 targets + sha256 checksums
                          (linux x86_64/aarch64, macOS universal,
                           windows x86_64)
```

The verify job hard-fails if the tag doesn't match the workspace version, so
a mistagged release publishes nothing. Publishing uses the
`CARGO_REGISTRY_TOKEN` repository secret (a crates.io API token with publish
scope) and runs in the `crates-io` GitHub environment — add required
reviewers to that environment for a manual approval gate.

## Consuming releases

Rust users depend on the crates via crates.io:

```toml
[dependencies]
ogham = "0.2"
```

Server binaries with sha256 checksums are attached to each
[GitHub release](https://github.com/signalbreak-labs/ogham/releases).

## Versioning policy

Single version for all three crates (workspace-inherited). Pre-1.0, minor
bumps may break APIs; breaking changes are listed under a **Changed**/
**Removed** heading in the changelog.

## crates.io publishing notes

The `publish-crates` job publishes in dependency order (`ogham-core` →
`ogham` → `ogham-server`); `cargo publish` waits for each crate to land in
the index before the next one starts.

If a publish fails partway, published versions are immutable: publish the
remainder by hand with `cargo publish -p <crate>` (re-running the whole job
fails on the crates that already went out), or `cargo yank` and bump the
patch version if a published crate is broken.

To rotate the token: create a new crates.io API token with publish scope
(https://crates.io/settings/tokens) and update the `CARGO_REGISTRY_TOKEN`
repository secret.
