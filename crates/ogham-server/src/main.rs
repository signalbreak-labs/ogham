use ogham_server::{AppState, CcrBackendConfig, ServerConfig};
use std::sync::Arc;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ogham_server=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = read_config_from_env();

    let state = match config.ccr {
        CcrBackendConfig::InMemory => AppState::new(),
        CcrBackendConfig::Sqlite { path, ttl_seconds } => {
            let store = Arc::new(
                ogham::ccr::sqlite::SqliteCcrStore::open(&path, ttl_seconds)
                    .expect("failed to open sqlite ccr store"),
            );
            AppState::with_store(store)
        }
        CcrBackendConfig::Fjall { path } => {
            let store = Arc::new(
                ogham::ccr::fjall::FjallCcrStore::new(&path)
                    .expect("failed to open fjall ccr store"),
            );
            AppState::with_store(store)
        }
    };

    let app = ogham_server::app_with_state(state);

    let listener = tokio::net::TcpListener::bind(config.bind)
        .await
        .expect("failed to bind");

    tracing::info!(
        "ogham-server listening on {}",
        listener.local_addr().unwrap()
    );

    axum::serve(listener, app).await.expect("server failed");
}

fn read_config_from_env() -> ServerConfig {
    let mut config = ServerConfig::default();

    if let Ok(bind_str) = std::env::var("OGHAM_BIND") {
        match bind_str.parse() {
            Ok(addr) => config.bind = addr,
            Err(_) => {
                eprintln!("invalid OGHAM_BIND: {}", bind_str);
                std::process::exit(1);
            }
        }
    }

    let ccr = std::env::var("OGHAM_CCR").unwrap_or_else(|_| "memory".to_string());
    config.ccr = match ccr.as_str() {
        "memory" => CcrBackendConfig::InMemory,
        "sqlite" => {
            let path = std::env::var("OGHAM_CCR_PATH").unwrap_or_else(|_| {
                eprintln!("OGHAM_CCR_PATH required when OGHAM_CCR=sqlite");
                std::process::exit(1);
            });
            let ttl = std::env::var("OGHAM_CCR_TTL_SECONDS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(86400);
            CcrBackendConfig::Sqlite {
                path: path.into(),
                ttl_seconds: ttl,
            }
        }
        "fjall" => {
            let path = std::env::var("OGHAM_CCR_PATH").unwrap_or_else(|_| {
                eprintln!("OGHAM_CCR_PATH required when OGHAM_CCR=fjall");
                std::process::exit(1);
            });
            CcrBackendConfig::Fjall { path: path.into() }
        }
        _ => {
            eprintln!("invalid OGHAM_CCR: {} (expected memory|sqlite|fjall)", ccr);
            std::process::exit(1);
        }
    };

    config
}
