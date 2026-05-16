//! Anamnesis MCP server binary — stdio mode.
//!
//! Reads one JSON-RPC message per line from stdin, dispatches it to
//! `AnamnesisServer::handle`, writes the response to stdout. Designed to
//! be launched as a subprocess by an MCP client (Claude Desktop,
//! Claude Code, ghast, …).

#![forbid(unsafe_code)]

use std::path::PathBuf;

use anamnesis_mcp_server::protocol::{JsonRpcRequest, JsonRpcResponse};
use anamnesis_mcp_server::AnamnesisServer;
use anamnesis_store::Store;
use anyhow::{Context, Result};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

fn resolve_data_dir() -> Result<PathBuf> {
    if let Some(d) = std::env::var_os("ANAMNESIS_DATA_DIR") {
        return Ok(PathBuf::from(d));
    }
    let base = if let Some(xdg) = std::env::var_os("XDG_DATA_HOME") {
        PathBuf::from(xdg)
    } else if cfg!(target_os = "macos") {
        dirs_home()?.join("Library/Application Support")
    } else if cfg!(windows) {
        std::env::var_os("APPDATA")
            .map(PathBuf::from)
            .context("APPDATA not set")?
    } else {
        dirs_home()?.join(".local/share")
    };
    Ok(base.join("anamnesis"))
}

fn dirs_home() -> Result<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .context("HOME not set")
}

#[cfg(feature = "local-fastembed")]
fn try_open_provider(
    data_dir: &std::path::Path,
    active_model: Option<&str>,
) -> Option<Box<dyn anamnesis_core::EmbeddingProvider>> {
    let key = active_model?.split(':').nth(1)?;
    match anamnesis_embedder::LocalFastembedProvider::new(key, data_dir.join("models")) {
        Ok(p) => Some(Box::new(p)),
        Err(e) => {
            tracing::warn!(
                model = key,
                error = %e,
                "failed to open active embedding model; search will degrade to FTS-only"
            );
            None
        }
    }
}

#[cfg(not(feature = "local-fastembed"))]
fn try_open_provider(
    _data_dir: &std::path::Path,
    _active_model: Option<&str>,
) -> Option<Box<dyn anamnesis_core::EmbeddingProvider>> {
    None
}

#[tokio::main]
async fn main() -> Result<()> {
    // Log to stderr; stdout is reserved for JSON-RPC frames.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let data_dir = resolve_data_dir()?;
    let db_path = data_dir.join("anamnesis.sqlite");
    let store = Store::open(&db_path).with_context(|| format!("open {}", db_path.display()))?;
    let active_model = store.active_model().ok().flatten();
    let provider = try_open_provider(&data_dir, active_model.as_deref());

    let server = AnamnesisServer::new(store, provider, data_dir.clone());
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        data_dir = %data_dir.display(),
        active_model = active_model.as_deref().unwrap_or("<unset>"),
        "anamnesis-mcp stdio server ready",
    );

    let mut stdout = tokio::io::stdout();
    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<JsonRpcRequest>(&line) {
            Ok(req) => {
                let is_note = req.is_notification();
                let resp = server.handle(req).await;
                if is_note {
                    // Notifications expect no reply.
                    continue;
                }
                resp
            }
            Err(e) => {
                JsonRpcResponse::err(serde_json::Value::Null, -32700, format!("parse error: {e}"))
            }
        };
        let line_out = serde_json::to_string(&response)? + "\n";
        stdout.write_all(line_out.as_bytes()).await?;
        stdout.flush().await?;
    }
    Ok(())
}
