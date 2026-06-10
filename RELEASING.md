# Releasing Ogham

Releases are tag-driven and fully automated by
[`.github/workflows/release.yml`](.github/workflows/release.yml).

## One-time setup

1. Create a crates.io API token with publish scope
   (https://crates.io/settings/tokens).
2. Add it as the repository secret `CARGO_REGISTRY_TOKEN`.
3. (Recommended) Create a GitHub environment named `crates-io` and require
   reviewers on it — the publish job runs in that environment, giving you a
   manual approval step before anything hits crates.io.

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
   (keep-a-changelog format — the release notes are extracted from it).

3. Verify locally:

   ```bash
   cargo fmt --all --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   cargo test -p ogham --features tiktoken
   cargo publish -p ogham-core --dry-run
   ```

4. Commit, tag, push:

   ```bash
   git commit -am "release: v0.2.0"
   git tag v0.2.0
   git push origin main v0.2.0
   ```

## What the automation does

```
tag v0.2.0
   │
   ├─ verify ............ tag == workspace version, full gate
   ├─ create-release .... GitHub release, notes from CHANGELOG.md
   │    └─ binaries ..... ogham-server for 5 targets + sha256 checksums
   │                      (linux x86_64/aarch64, macOS x86_64/aarch64,
   │                       windows x86_64)
   └─ publish-crates .... crates.io, in dependency order:
                          ogham-core → ogham → ogham-server
                          (gated by the `crates-io` environment)
```

The verify job hard-fails if the tag doesn't match the workspace version, so
a mistagged release publishes nothing.

## If publishing fails partway

`cargo publish` is per-crate, and published versions are immutable. If, say,
`ogham-core` published but `ogham` failed:

- Fix the issue **without changing the version** if the fix doesn't touch
  published code, re-run the `publish-crates` job (it skips already-published
  versions with an "already exists" error — re-run publishes the remainder;
  if cargo aborts on the first crate, comment it out locally and publish the
  rest by hand: `cargo publish -p ogham && cargo publish -p ogham-server`).
- If the published crate itself is broken, `cargo yank` it, bump the patch
  version, and start over.

## Versioning policy

Single version for all three crates (workspace-inherited). Pre-1.0, minor
bumps may break APIs; breaking changes are listed under a **Changed**/
**Removed** heading in the changelog.
