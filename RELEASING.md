# Releasing Ogham

Releases are tag-driven and automated by
[`.github/workflows/release.yml`](.github/workflows/release.yml).
Releases are currently **GitHub-only** (no crates.io publishing — see the
last section to enable it later).

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
   └─ create-release .... GitHub release, notes from CHANGELOG.md
        └─ binaries ..... ogham-server for 4 targets + sha256 checksums
                          (linux x86_64/aarch64, macOS universal,
                           windows x86_64)
```

The verify job hard-fails if the tag doesn't match the workspace version, so
a mistagged release publishes nothing.

## Consuming releases

Rust users depend on the crates via git:

```toml
[dependencies]
ogham = { git = "https://github.com/signalbreak-labs/ogham", tag = "v0.1.0" }
```

Server binaries with sha256 checksums are attached to each
[GitHub release](https://github.com/signalbreak-labs/ogham/releases).

## Versioning policy

Single version for all three crates (workspace-inherited). Pre-1.0, minor
bumps may break APIs; breaking changes are listed under a **Changed**/
**Removed** heading in the changelog.

## Publishing to crates.io (currently disabled)

When ready:

1. Create a crates.io account and an API token with publish scope
   (https://crates.io/settings/tokens).
2. Add it as the repository secret `CARGO_REGISTRY_TOKEN`.
3. (Recommended) Create a GitHub environment named `crates-io` with
   required reviewers for a manual approval gate.
4. Restore this job in `release.yml`:

   ```yaml
   publish-crates:
     name: Publish to crates.io
     needs: verify
     runs-on: ubuntu-latest
     environment: crates-io
     steps:
       - uses: actions/checkout@v6
       - uses: dtolnay/rust-toolchain@stable
       - uses: Swatinem/rust-cache@v2
       - name: Publish (dependency order)
         env:
           CARGO_REGISTRY_TOKEN: ${{ secrets.CARGO_REGISTRY_TOKEN }}
         run: |
           cargo publish -p ogham-core
           cargo publish -p ogham
           cargo publish -p ogham-server
   ```

   Crate-name availability was verified 2026-06-10 (`ogham`, `ogham-core`,
   `ogham-server` all free); re-check before first publish. If a publish
   fails partway, published versions are immutable: re-run the job (cargo
   errors on already-published crates — publish the remainder by hand), or
   `cargo yank` and bump the patch version if the published crate is broken.
