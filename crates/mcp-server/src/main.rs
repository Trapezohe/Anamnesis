//! Anamnesis MCP server binary — stdio + (optional) HTTP transports.

#![forbid(unsafe_code)]

use std::path::PathBuf;

use anamnesis_mcp_server::AnamnesisServer;
use anamnesis_store::Store;
use anyhow::{Context, Result};
use clap::Parser;

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
    if let Some(h) = std::env::var_os("HOME") {
        return Ok(PathBuf::from(h));
    }
    // Windows: HOME isn't set by default, but USERPROFILE is.
    if let Some(h) = std::env::var_os("USERPROFILE") {
        return Ok(PathBuf::from(h));
    }
    Err(anyhow::anyhow!("neither HOME nor USERPROFILE is set"))
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

#[derive(Parser, Debug)]
#[command(
    name = "anamnesis-mcp",
    version,
    about = "Anamnesis MCP server (stdio default, --sse for HTTP)"
)]
struct Cli {
    /// Bind an HTTP transport on the given TCP port (loopback only).
    /// Requires the `sse` cargo feature (on by default).
    #[arg(long)]
    sse: Option<u16>,
    /// Pre-shared bearer token for HTTP mode. If omitted a fresh 64-byte
    /// token is generated and printed to stderr on startup.
    #[arg(long, env = "ANAMNESIS_MCP_TOKEN")]
    token: Option<String>,
    /// Expose admin tools (currently `import_source`) over MCP.
    ///
    /// Off by default — an MCP server should be a read-only memory
    /// provider. Only enable this when every connected client is trusted
    /// (the bearer-token-gated HTTP transport on loopback is the typical
    /// case). See BLUEPRINT §17.5 PR-A.
    #[arg(long, env = "ANAMNESIS_MCP_ALLOW_ADMIN")]
    allow_admin_tools: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let data_dir = resolve_data_dir()?;
    let db_path = data_dir.join("anamnesis.sqlite");
    let store = Store::open(&db_path).with_context(|| format!("open {}", db_path.display()))?;
    let active_model = store.active_model().ok().flatten();
    let provider = try_open_provider(&data_dir, active_model.as_deref());
    let server = AnamnesisServer::new(store, provider, data_dir.clone())
        .with_admin_tools(cli.allow_admin_tools);

    if cli.allow_admin_tools {
        tracing::warn!(
            "admin tools enabled — `import_source` is callable over MCP. \
             Only use this with trusted clients."
        );
    }

    if let Some(port) = cli.sse {
        run_http(server, port, cli.token).await
    } else {
        tracing::info!(
            version = env!("CARGO_PKG_VERSION"),
            data_dir = %data_dir.display(),
            active_model = active_model.as_deref().unwrap_or("<unset>"),
            admin_tools = cli.allow_admin_tools,
            "anamnesis-mcp stdio server ready",
        );
        anamnesis_mcp_server::stdio::run(server).await
    }
}

#[cfg(feature = "sse")]
async fn run_http(server: AnamnesisServer, port: u16, token: Option<String>) -> Result<()> {
    let config = anamnesis_mcp_server::sse::HttpServerConfig { port, token };
    anamnesis_mcp_server::sse::run(server, config).await
}

#[cfg(not(feature = "sse"))]
async fn run_http(_server: AnamnesisServer, _port: u16, _token: Option<String>) -> Result<()> {
    Err(anyhow::anyhow!(
        "this anamnesis-mcp build lacks the `sse` feature; rebuild with `--features sse`"
    ))
}
