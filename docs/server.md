# HTTP Server (`ogham-server`)

The server is optional — the SDK is library-first. The server exists for
non-Rust clients, standalone debugging, and embedding a compression endpoint
into an existing Axum app.

## Running standalone

```bash
cargo install --git https://github.com/signalbreak-labs/ogham ogham-server
# or grab a prebuilt binary from the GitHub releases page
ogham-server
# listening on 127.0.0.1:3000
```

### Configuration (environment variables)

| Variable | Values | Default | Notes |
|---|---|---|---|
| `OGHAM_BIND` | socket address | `127.0.0.1:3000` | never binds all interfaces unless you ask |
| `OGHAM_CCR` | `memory` \| `sqlite` \| `fjall` | `memory` | CCR backend |
| `OGHAM_CCR_PATH` | path | — | **required** for `sqlite`/`fjall`; exits 1 if missing |
| `OGHAM_CCR_TTL_SECONDS` | integer | `86400` | sqlite entry TTL |
| `RUST_LOG` | tracing filter | `ogham_server=debug` | standard `tracing` env filter |

Invalid configuration prints a one-line error to stderr and exits with
code 1.

```bash
OGHAM_BIND=0.0.0.0:8080 OGHAM_CCR=sqlite OGHAM_CCR_PATH=/var/lib/ogham/ccr.db ogham-server
```

There is no authentication layer — front it with your own auth/proxy before
exposing it beyond localhost.

## Embedding in your Axum app

`ogham-server` enables both persistent CCR backends (`ccr-sqlite`, `ccr-fjall`),
so the store types below are available when you depend on the server crate. If
you reference `ogham` directly instead, add the matching feature.

```rust
use ogham_server::{app_with_state, AppState};
use ogham::ccr::fjall::FjallCcrStore;
use std::sync::Arc;

let store = Arc::new(FjallCcrStore::new("/var/lib/myapp/ccr")?);
let router = axum::Router::new()
    .nest("/ogham", app_with_state(AppState::with_store(store)));
```

## API

All endpoints return `200` with JSON bodies. The compress endpoint is
fail-closed: on internal error it returns the input messages unchanged with
an `error` field in `stats`, never an empty or corrupted list.

### `POST /compress`

```jsonc
// request
{ "messages": [ { "role": "tool", "content": "[{\"id\":1},{\"id\":2}]" } ] }

// response — message metadata (ogham.* keys) is preserved round-trip
{
  "messages": [ { "role": "tool", "content": "..." } ],
  "stats": {
    "original_tokens": 120,
    "compressed_tokens": 36,
    "ratio": 0.3,
    "compressor_used": "smart_crusher"
  }
}
```

### `POST /retrieve`

Fetch an original by the hash from a `<<ccr:HASH>>` marker.

```jsonc
// request
{ "id": "86a33abcdadc3e60d9e6fe9896046d97" }

// response
{ "found": true, "original": "full original text" }
// or
{ "found": false, "original": null }
```

### `POST /detect`

```jsonc
// request
{ "content": "[{\"a\":1},{\"a\":2}]" }

// response
{ "content_type": "json_array", "confidence": 0.95, "metadata": {} }
```

### `GET /health`

```json
{ "status": "ok" }
```

### `GET /stats`

```json
{ "uptime_seconds": 42, "requests_total": 7 }
```
