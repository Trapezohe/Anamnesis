//! `anamnesis` CLI entry point.

#![forbid(unsafe_code)]
// Aligned table headers are clearer than building format strings with
// inlined idents — clippy's literal-format-arg lint isn't load-bearing here.
#![allow(clippy::print_literal)]

use std::path::PathBuf;

use anamnesis_adapter_claude_code::{ClaudeCodeAdapter, ClaudeCodeConfig, ClaudeCodeDetector};
use anamnesis_adapter_codex::{codex_adapter, CodexDetector};
use anamnesis_adapter_ghast::{ghast_adapter, GhastDetector};
use anamnesis_adapter_hermes::{hermes_adapter, HermesDetector};
use anamnesis_adapter_letta::{letta_adapter, LettaSqliteDetector};
use anamnesis_adapter_mem0::{sqlite_adapter as mem0_sqlite_adapter, Mem0SqliteDetector};
use anamnesis_adapter_openclaw::{openclaw_adapter, OpenClawDetector};
use anamnesis_core::discovery::{DetectOpts, Discovery};
use anamnesis_embedder::registry;
use anamnesis_importer::{ImportOptions, ImportService};
use anamnesis_search::{pack, ContextBudget, HybridOpts, HybridSearcher, SearchMode};
use anamnesis_store::Store;
use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "anamnesis",
    version,
    about = "Cross-agent memory layer (搜魂术)",
    long_about = None,
)]
struct Cli {
    /// Override data directory (defaults to XDG_DATA_HOME/anamnesis).
    #[arg(long, global = true, env = "ANAMNESIS_DATA_DIR")]
    data_dir: Option<PathBuf>,

    /// Override config file path (defaults to
    /// XDG_CONFIG_HOME/anamnesis/config.toml).
    #[arg(long, global = true, env = "ANAMNESIS_CONFIG")]
    config: Option<PathBuf>,

    /// Log level (trace, debug, info, warn, error).
    #[arg(long, global = true, env = "ANAMNESIS_LOG", default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Initialize a new data directory and database.
    Init {
        /// Override the default embedding model (curated key).
        #[arg(long)]
        model: Option<String>,
    },

    /// Show database stats and active model.
    Status {
        /// Emit JSON instead of the human-friendly table.
        #[arg(long)]
        json: bool,
    },

    /// Scan default paths for known memory sources (read-only).
    Discover,

    /// Manage configured memory sources.
    #[command(subcommand)]
    Source(SourceCmd),

    /// Run an import job for one source.
    Import {
        /// Adapter name, optionally `adapter:instance`.
        target: String,
        /// Full re-scan: ignore any incremental window (`--since` or
        /// the auto-detected `last_import_at`) and ask the adapter to
        /// re-emit every candidate record. Does NOT bypass the store's
        /// `raw_hash` dedup — re-emitting an unchanged record is still
        /// a no-op upsert (PR-#15 fast-path).
        #[arg(long, conflicts_with = "since")]
        full: bool,
        /// RFC3339 timestamp (e.g. `2026-04-01T00:00:00Z`) — only
        /// records modified after this point get imported. When neither
        /// `--full` nor `--since` is given, the importer falls back to
        /// `sources.last_import_at` for incremental imports.
        #[arg(long)]
        since: Option<String>,
        /// Print what would be imported instead of writing.
        #[arg(long)]
        dry_run: bool,
        /// Skip running the embedding worker after the import.
        #[arg(long)]
        no_embed: bool,
        /// Optional path override (e.g. mem0 SQLite file when the default
        /// `~/.mem0/db.sqlite` is wrong).
        #[arg(long)]
        path: Option<PathBuf>,
    },

    /// Search across all imported records.
    Search {
        /// Free-text query.
        query: String,
        /// Restrict to one source (adapter id).
        #[arg(long)]
        source: Option<String>,
        /// Restrict to a specific source instance. Meaningful only when
        /// `--source` is also set; the SQL key is `(adapter, instance)`.
        #[arg(long)]
        instance: Option<String>,
        /// Restrict to one Kind: fact | preference | feedback | reference | episode | skill | unknown.
        #[arg(long)]
        kind: Option<String>,
        /// Restrict to one Scope: user | project | session | ephemeral.
        #[arg(long)]
        scope: Option<String>,
        /// RFC3339 lower bound on records.created_at (inclusive). E.g.
        /// `2026-04-01T00:00:00Z`.
        #[arg(long)]
        since: Option<String>,
        /// RFC3339 upper bound on records.created_at (inclusive).
        #[arg(long)]
        until: Option<String>,
        /// Result limit.
        #[arg(long, default_value_t = 10)]
        limit: u32,
        /// Modality: fulltext | vector | hybrid (default = hybrid).
        #[arg(long, default_value = "hybrid")]
        mode: String,
        /// Emit JSON instead of a human-readable table.
        #[arg(long)]
        json: bool,
    },

    /// Manage embedding models.
    #[command(subcommand)]
    Model(ModelCmd),

    /// Export records as JSONL or CSV.
    Export {
        /// Output file path (default: stdout).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Format: `jsonl` (one AnamnesisRecord per line) or `csv`.
        #[arg(long, default_value = "jsonl")]
        format: String,
        /// Restrict to one source (adapter id).
        #[arg(long)]
        source: Option<String>,
    },

    /// Run as an MCP server. Default = stdio; `--sse <port>` binds the
    /// HTTP/JSON-RPC transport on `127.0.0.1:<port>` (use `0` for an
    /// ephemeral port — the chosen port is printed to stderr).
    Serve {
        /// Bind the HTTP/JSON-RPC transport on `127.0.0.1:<port>`.
        /// Requires the `sse` cargo feature (on by default).
        #[arg(long)]
        sse: Option<u16>,
        /// Pre-shared bearer token for HTTP mode. If omitted a fresh
        /// 64-char token is generated and printed to stderr on startup.
        /// Stdio mode ignores this flag.
        #[arg(long, env = "ANAMNESIS_MCP_TOKEN")]
        token: Option<String>,
    },

    /// Verify database integrity. With --repair, rebuild the FTS index and
    /// re-queue any chunks that have no embeddings under the active model.
    Verify {
        /// Try to fix issues that are auto-repairable.
        #[arg(long)]
        repair: bool,
    },

    /// Run pending schema migrations (no-op after init).
    Migrate,
}

#[derive(Subcommand, Debug)]
enum SourceCmd {
    /// Register a new source.
    ///
    /// File-based adapters take `--path`. URL-based adapters (e.g.
    /// `generic-mcp`) take `--url` and optionally `--token-env`.
    Add {
        /// Adapter name (e.g. `claude-code`, `mem0`, `codex`, `generic-mcp`).
        adapter: String,
        /// Instance discriminator (optional).
        #[arg(long)]
        instance: Option<String>,
        /// Filesystem path, if the adapter takes one.
        #[arg(long, conflicts_with = "url")]
        path: Option<PathBuf>,
        /// Upstream URL, for URL-based adapters like `generic-mcp`
        /// (e.g. `http://127.0.0.1:7878`).
        #[arg(long)]
        url: Option<String>,
        /// Name of the environment variable holding the bearer token
        /// for URL-based adapters. The value is resolved at import time,
        /// not at registration time — only the variable *name* is stored
        /// in the registry, never the token itself.
        #[arg(long, requires = "url")]
        token_env: Option<String>,
    },
    /// List configured sources.
    List,
    /// Remove a registered source.
    Remove {
        /// `adapter` or `adapter:instance`.
        target: String,
    },
}

#[derive(Subcommand, Debug)]
enum ModelCmd {
    /// List curated models + which one is active.
    List,
    /// Switch the active embedding model. Re-queues all chunks for embed.
    Use {
        /// Curated key (e.g. `default`, `tiny`, `en`, `multi-strong`).
        key: String,
        /// Skip running the embedding worker after the switch.
        #[arg(long)]
        no_embed: bool,
    },
    /// Pre-download a model into the cache without changing the active one.
    Install {
        /// Curated key.
        key: String,
    },
    /// Re-embed every chunk under the current active model.
    Rebuild {
        /// Skip running the worker; only re-enqueue jobs.
        #[arg(long)]
        no_embed: bool,
    },
}

fn init_tracing(level: &str) {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(level))
        .unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).with_target(false).init();
}

fn resolve_data_dir(override_path: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = override_path {
        return Ok(p);
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
    Err(anyhow!("neither HOME nor USERPROFILE is set"))
}

fn db_path(data_dir: &std::path::Path) -> PathBuf {
    data_dir.join("anamnesis.sqlite")
}

fn models_dir(data_dir: &std::path::Path) -> PathBuf {
    data_dir.join("models")
}

fn resolve_config_path(override_path: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = override_path {
        return Ok(p);
    }
    let home = dirs_home()?;
    Ok(anamnesis_core::Config::default_path(&home))
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level);
    let data_dir = resolve_data_dir(cli.data_dir)?;
    let config_path = resolve_config_path(cli.config)?;
    let config = anamnesis_core::Config::load(&config_path)
        .with_context(|| format!("load {}", config_path.display()))?;
    tracing::debug!(path = %config_path.display(), "loaded config");

    match cli.command {
        Command::Init { model } => {
            // Precedence: --model > config.embedding.model > registry default.
            let chosen = model
                .clone()
                .or_else(|| Some(config.embedding.model.clone()))
                .filter(|m| !m.is_empty());
            cmd_init(&data_dir, chosen.as_deref())
        }
        Command::Status { json } => cmd_status(&data_dir, json),
        Command::Discover => cmd_discover().await,
        Command::Source(sub) => cmd_source(&data_dir, sub),
        Command::Import {
            target,
            full,
            since,
            dry_run,
            no_embed,
            path,
        } => {
            cmd_import(
                &data_dir,
                &target,
                full,
                since.as_deref(),
                dry_run,
                no_embed,
                path.as_deref(),
            )
            .await
        }
        Command::Search {
            query,
            source,
            instance,
            kind,
            scope,
            since,
            until,
            limit,
            mode,
            json,
        } => {
            cmd_search(
                &data_dir,
                &query,
                source.as_deref(),
                instance.as_deref(),
                kind.as_deref(),
                scope.as_deref(),
                since.as_deref(),
                until.as_deref(),
                limit,
                &mode,
                json,
            )
            .await
        }
        Command::Model(sub) => cmd_model(&data_dir, sub).await,
        Command::Serve { sse, token } => {
            cmd_serve(&data_dir, sse, token, config.server.allow_admin_tools).await
        }
        Command::Export {
            out,
            format,
            source,
        } => cmd_export(&data_dir, out.as_deref(), &format, source.as_deref()),
        Command::Verify { repair } => cmd_verify(&data_dir, repair),
        Command::Migrate => {
            let _ = Store::open(db_path(&data_dir))?;
            println!("migrations applied");
            Ok(())
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// serve — embed the MCP server in the CLI process (same code as the
// dedicated `anamnesis-mcp` binary, but one less binary for users to wire up).
// ─────────────────────────────────────────────────────────────────────────────

async fn cmd_serve(
    data_dir: &std::path::Path,
    sse: Option<u16>,
    token: Option<String>,
    allow_admin_tools: bool,
) -> Result<()> {
    let store = Store::open(db_path(data_dir))?;
    let active_model = store.active_model().ok().flatten();
    let provider = open_active_provider_optional(data_dir, &store, active_model.as_deref());
    let server =
        anamnesis_mcp_server::AnamnesisServer::new(store, provider, data_dir.to_path_buf())
            .with_admin_tools(allow_admin_tools);

    match sse {
        Some(port) => {
            eprintln!(
                "anamnesis serve (http) — active model: {}, admin tools: {}",
                active_model.as_deref().unwrap_or("<unset>"),
                if allow_admin_tools {
                    "ENABLED"
                } else {
                    "disabled"
                },
            );
            if allow_admin_tools {
                eprintln!(
                    "  ⚠ admin tools enabled — `import_source` is callable over MCP. \
                     Run only with trusted clients."
                );
            }
            audit(data_dir).record(anamnesis_core::AuditEntry::new(
                "serve.start",
                serde_json::json!({
                    "transport": "http",
                    "port": port,
                    "active_model": active_model,
                    "allow_admin_tools": allow_admin_tools,
                }),
            ));
            run_sse(server, port, token).await
        }
        None => {
            eprintln!(
                "anamnesis serve (stdio) — active model: {}, admin tools: {}",
                active_model.as_deref().unwrap_or("<unset>"),
                if allow_admin_tools {
                    "ENABLED"
                } else {
                    "disabled"
                },
            );
            if allow_admin_tools {
                eprintln!(
                    "  ⚠ admin tools enabled — `import_source` is callable. \
                     Run only with trusted clients."
                );
            }
            audit(data_dir).record(anamnesis_core::AuditEntry::new(
                "serve.start",
                serde_json::json!({
                    "transport": "stdio",
                    "active_model": active_model,
                    "allow_admin_tools": allow_admin_tools,
                }),
            ));
            anamnesis_mcp_server::stdio::run(server).await
        }
    }
}

#[cfg(feature = "sse")]
async fn run_sse(
    server: anamnesis_mcp_server::AnamnesisServer,
    port: u16,
    token: Option<String>,
) -> Result<()> {
    let config = anamnesis_mcp_server::sse::HttpServerConfig { port, token };
    anamnesis_mcp_server::sse::run(server, config).await
}

#[cfg(not(feature = "sse"))]
async fn run_sse(
    _server: anamnesis_mcp_server::AnamnesisServer,
    _port: u16,
    _token: Option<String>,
) -> Result<()> {
    Err(anyhow!(
        "this `anamnesis` build lacks the `sse` cargo feature; \
         rebuild with `--features sse` (on by default)."
    ))
}

#[cfg(feature = "local-fastembed")]
fn open_active_provider_optional(
    data_dir: &std::path::Path,
    _store: &Store,
    active_model: Option<&str>,
) -> Option<Box<dyn anamnesis_core::EmbeddingProvider>> {
    let key = active_model?.split(':').nth(1)?;
    match anamnesis_embedder::LocalFastembedProvider::new(key, models_dir(data_dir)) {
        Ok(p) => Some(Box::new(p)),
        Err(e) => {
            tracing::warn!(
                model = key,
                error = %e,
                "failed to open active embedding model; serve will degrade to FTS-only"
            );
            None
        }
    }
}

#[cfg(not(feature = "local-fastembed"))]
fn open_active_provider_optional(
    _data_dir: &std::path::Path,
    _store: &Store,
    _active_model: Option<&str>,
) -> Option<Box<dyn anamnesis_core::EmbeddingProvider>> {
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// init / status / discover
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_init(data_dir: &std::path::Path, model: Option<&str>) -> Result<()> {
    std::fs::create_dir_all(data_dir)
        .with_context(|| format!("create data dir {}", data_dir.display()))?;
    let store = Store::open(db_path(data_dir))
        .with_context(|| format!("open {}", db_path(data_dir).display()))?;

    let key = model.unwrap_or_else(|| registry::default_model().key);
    if registry::by_key(key).is_none() {
        return Err(anyhow!(
            "unknown model key {key:?} — available: {}",
            registry::available().join(", ")
        ));
    }
    let model_id = format!("local:{key}:1");
    store.set_active_model(&model_id)?;

    println!("initialized at {}", data_dir.display());
    println!("active embedding model: {model_id}");
    Ok(())
}

fn cmd_status(data_dir: &std::path::Path, json: bool) -> Result<()> {
    let db = db_path(data_dir);
    if !db.exists() {
        if json {
            let payload = serde_json::json!({
                "initialized": false,
                "data_dir": data_dir.display().to_string(),
                "db_path": db.display().to_string(),
            });
            println!("{}", serde_json::to_string_pretty(&payload)?);
        } else {
            println!(
                "no database found at {} — run `anamnesis init`",
                db.display()
            );
        }
        return Ok(());
    }
    let store = Store::open(&db)?;
    let stats = store.stats()?;
    let active = store.active_model()?;
    // Round-16: surface per-source counts + freshness right on
    // `status`, not just on `source list`. Operators landing on `status`
    // for the first time should see "this source hasn't imported in 30
    // days" without having to know about a second subcommand.
    let per_source = store.list_sources_with_counts()?;
    let now = chrono::Utc::now().timestamp();
    if json {
        let payload = serde_json::json!({
            "initialized": true,
            "data_dir": data_dir.display().to_string(),
            "models_dir": models_dir(data_dir).display().to_string(),
            "schema_version": anamnesis_core::SCHEMA_VERSION,
            "active_model": active,
            "stats": {
                "sources": stats.sources,
                "records": stats.records,
                "chunks": stats.chunks,
                "jobs_pending": stats.jobs_pending,
                "jobs_failed": stats.jobs_failed,
            },
            "sources": per_source.iter().map(|r| {
                let freshness = source_freshness(r.source.last_import_at, now);
                serde_json::json!({
                    "adapter": r.source.adapter,
                    "instance": if r.source.instance.is_empty() {
                        serde_json::Value::Null
                    } else {
                        serde_json::Value::String(r.source.instance.clone())
                    },
                    "location": r.source.location,
                    "added_at": r.source.added_at,
                    "last_import_at": r.source.last_import_at,
                    "record_count": r.record_count,
                    "chunk_count": r.chunk_count,
                    // Round-16 additions — let consumers branch on
                    // staleness without re-computing it from `now`.
                    "freshness": freshness.label,
                    "age_seconds": freshness.age_seconds,
                })
            }).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }
    println!("data_dir        : {}", data_dir.display());
    println!("models dir      : {}", models_dir(data_dir).display());
    println!("schema          : v{}", anamnesis_core::SCHEMA_VERSION);
    println!(
        "active model    : {}",
        active.as_deref().unwrap_or("<unset>")
    );
    println!("sources         : {}", stats.sources);
    println!("records         : {}", stats.records);
    println!("chunks          : {}", stats.chunks);
    println!("jobs pending    : {}", stats.jobs_pending);
    println!("jobs failed     : {}", stats.jobs_failed);

    // Per-source health table. Always emitted (even when empty) so the
    // operator sees the "no sources yet — run `anamnesis source add`"
    // affordance right after `init`.
    println!();
    if per_source.is_empty() {
        println!("sources by health:");
        println!("  (no sources registered — try `anamnesis discover` or `anamnesis source add`)");
    } else {
        println!(
            "{:<14} {:<14} {:<8} {:<8} {:<16} {}",
            "adapter", "instance", "records", "chunks", "last_import", "status"
        );
        for r in &per_source {
            let freshness = source_freshness(r.source.last_import_at, now);
            println!(
                "{:<14} {:<14} {:<8} {:<8} {:<16} {}",
                r.source.adapter,
                if r.source.instance.is_empty() {
                    "-".to_string()
                } else {
                    r.source.instance.clone()
                },
                r.record_count,
                r.chunk_count,
                freshness.age_human,
                freshness.label,
            );
        }
    }
    Ok(())
}

/// Human-readable freshness summary for a source, derived from
/// `last_import_at` (Unix seconds, `None` for never-imported sources).
///
/// `label` is one of `"never-imported" | "fresh" | "stale"`:
///   - `never-imported`: `last_import_at is None` (registered but no
///     import has landed)
///   - `fresh`: imported within the last 24 hours
///   - `stale`: imported more than 24 hours ago
///
/// `age_seconds` is `None` for never-imported, otherwise the gap from
/// `last_import_at` to `now`. `age_human` is the same gap rounded to a
/// short string: `<1m`, `5m`, `3h`, `2d`, `30d+`.
struct Freshness {
    label: &'static str,
    age_seconds: Option<i64>,
    age_human: String,
}

fn source_freshness(last_import_at: Option<i64>, now: i64) -> Freshness {
    match last_import_at {
        None => Freshness {
            label: "never-imported",
            age_seconds: None,
            age_human: "<never>".into(),
        },
        Some(t) => {
            let age = now.saturating_sub(t).max(0);
            let label = if age < 24 * 3600 { "fresh" } else { "stale" };
            Freshness {
                label,
                age_seconds: Some(age),
                age_human: human_age_short(age),
            }
        }
    }
}

/// Compact human-readable age: `<1m`, `5m`, `3h`, `2d`, `30d+`.
fn human_age_short(age_seconds: i64) -> String {
    if age_seconds < 60 {
        "<1m".into()
    } else if age_seconds < 3600 {
        format!("{}m", age_seconds / 60)
    } else if age_seconds < 24 * 3600 {
        format!("{}h", age_seconds / 3600)
    } else if age_seconds < 30 * 24 * 3600 {
        format!("{}d", age_seconds / (24 * 3600))
    } else {
        "30d+".into()
    }
}

async fn cmd_discover() -> Result<()> {
    let discovery = Discovery::new()
        .register(Box::new(ClaudeCodeDetector::new()))
        .register(Box::new(Mem0SqliteDetector::new()))
        .register(Box::new(CodexDetector::new()))
        .register(Box::new(LettaSqliteDetector::new()))
        .register(Box::new(HermesDetector::new()))
        .register(Box::new(OpenClawDetector::new()))
        .register(Box::new(GhastDetector::new()));
    let found = discovery.detect_all(&DetectOpts::default()).await;
    if found.is_empty() {
        println!("no known memory sources found at default locations");
        return Ok(());
    }
    println!(
        "{:<14} {:<8} {:<48} {}",
        "adapter", "conf", "location", "note"
    );
    for s in &found {
        let conf = match s.confidence {
            anamnesis_core::Confidence::High => "high",
            anamnesis_core::Confidence::Medium => "medium",
            anamnesis_core::Confidence::Low => "low",
        };
        println!(
            "{:<14} {:<8} {:<48} {}",
            s.adapter,
            conf,
            s.location,
            s.note.as_deref().unwrap_or("")
        );
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// source add / list / remove
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_source(data_dir: &std::path::Path, sub: SourceCmd) -> Result<()> {
    let store = Store::open(db_path(data_dir))?;
    match sub {
        SourceCmd::Add {
            adapter,
            instance,
            path,
            url,
            token_env,
        } => {
            // Round-17 (§19.3 PR-1): generic-mcp is now a first-class
            // CLI source — `source add generic-mcp --url ... [--token-env
            // ENV]` registers the upstream MCP HTTP server and lets the
            // subsequent `import generic-mcp:<instance>` pull from it
            // without any test-only construction code. The token *name*,
            // never the token *value*, lives in the registry (resolved
            // at import time via the operator's env).
            let location: Option<String> = match (path.as_ref(), url.as_ref()) {
                (Some(_), Some(_)) => {
                    // clap's `conflicts_with` should prevent this, but
                    // belt-and-braces.
                    return Err(anyhow!("--path and --url are mutually exclusive"));
                }
                (Some(p), None) => Some(p.display().to_string()),
                (None, Some(u)) => Some(u.clone()),
                (None, None) => None,
            };
            let config_json: Option<String> = token_env
                .as_deref()
                .map(|env| serde_json::json!({ "token_env": env }).to_string());
            // Sanity: for URL-based adapters, refuse to register without
            // a URL. This keeps `import generic-mcp:<i>` from later
            // failing with the confusing "no default path" error.
            if adapter == anamnesis_adapter_generic_mcp::ADAPTER_ID && url.is_none() {
                return Err(anyhow!(
                    "source add generic-mcp requires --url <upstream-mcp-url>"
                ));
            }
            store.register_source(
                &adapter,
                instance.as_deref(),
                location.as_deref(),
                config_json.as_deref(),
            )?;
            println!(
                "registered: {adapter}{}{}",
                instance
                    .as_deref()
                    .map(|i| format!(":{i}"))
                    .unwrap_or_default(),
                location.map(|l| format!(" @ {l}")).unwrap_or_default(),
            );
            Ok(())
        }
        SourceCmd::List => {
            // Round-9: show per-source counts alongside last_import so
            // operators can spot "registered but empty" sources at a
            // glance — same signal MCP agents get from list_sources.
            let rows = store.list_sources_with_counts()?;
            if rows.is_empty() {
                println!("no sources registered");
            } else {
                println!(
                    "{:<14} {:<14} {:<8} {:<8} {:<20} {}",
                    "adapter", "instance", "records", "chunks", "last_import", "location"
                );
                for r in rows {
                    let last = r
                        .source
                        .last_import_at
                        .map(|t| {
                            chrono::DateTime::<chrono::Utc>::from_timestamp(t, 0)
                                .map(|d| d.format("%Y-%m-%dT%H:%MZ").to_string())
                                .unwrap_or_else(|| t.to_string())
                        })
                        .unwrap_or_else(|| "<never>".into());
                    println!(
                        "{:<14} {:<14} {:<8} {:<8} {:<20} {}",
                        r.source.adapter,
                        r.source.instance,
                        r.record_count,
                        r.chunk_count,
                        last,
                        r.source.location.unwrap_or_default(),
                    );
                }
            }
            Ok(())
        }
        SourceCmd::Remove { target } => {
            let (adapter, instance) = split_target(&target);
            store.deregister_source(adapter, instance)?;
            println!("removed: {target}");
            Ok(())
        }
    }
}

fn split_target(t: &str) -> (&str, Option<&str>) {
    match t.split_once(':') {
        Some((a, i)) => (a, Some(i)),
        None => (t, None),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// import
// ─────────────────────────────────────────────────────────────────────────────

async fn cmd_import(
    data_dir: &std::path::Path,
    target: &str,
    full: bool,
    since_arg: Option<&str>,
    dry_run: bool,
    no_embed: bool,
    path_override: Option<&std::path::Path>,
) -> Result<()> {
    let (adapter_id, instance) = split_target(target);

    // Reject unknown adapters before doing any registry / filesystem work
    // so the error message is "not wired" rather than the more confusing
    // "no default path".
    if !is_known_adapter(adapter_id) {
        return Err(anyhow!(
            "adapter {adapter_id:?} not wired; supported: claude-code, codex, mem0, letta, hermes, openclaw, ghast, generic-mcp"
        ));
    }

    // Round-19 (§-1.5 PR-4a): resolve the effective `ScanOpts`.
    //
    //   --full         → forced full scan, `since = None`.
    //   --since <ts>   → explicit incremental bound (RFC3339 / chrono parse).
    //   neither        → default to the source's registry `last_import_at`
    //                    (auto-incremental). On a fresh source, this is
    //                    `None` and the run is effectively a full scan.
    //
    // `--full` and `--since` are mutually exclusive at the clap layer.
    let scan_opts = resolve_scan_opts(data_dir, adapter_id, instance, full, since_arg)?;
    if !dry_run {
        match (full, scan_opts.since.as_ref()) {
            (true, _) => eprintln!("import: --full → ignoring any incremental window"),
            (false, Some(t)) => eprintln!(
                "import: incremental since {t} (override with --full for a complete re-scan)"
            ),
            (false, None) => {}
        }
    }

    // PR-B (BLUEPRINT §18.4 F5): the source registry is the canonical
    // truth for "where does X live". Resolution order:
    //
    //   1. --path P    → trusted override; we'll register/overwrite P
    //                    so the registry catches up to the explicit user
    //                    intent.
    //   2. registry    → use the location the user registered earlier via
    //                    `source add`.
    //   3. fallback    → adapter default path; auto-registered on success
    //                    so the next `import` is no longer ambiguous.
    //
    // We never silently fall back from a registered (but missing) path
    // to the adapter default — that would mask a misconfiguration; the
    // adapter's health check will report the failure instead.
    let store_for_lookup = Store::open(db_path(data_dir))?;
    let registered = store_for_lookup.get_source(adapter_id, instance)?;
    let registered_location = registered.as_ref().and_then(|r| r.location.clone());
    let registered_config = registered.as_ref().and_then(|r| r.config_json.clone());
    drop(store_for_lookup);

    // Round-17 (§19.3 PR-1): generic-mcp is URL-based, not path-based.
    // It does NOT use `default_path_for` (URLs have no useful default —
    // the operator must register one via `source add generic-mcp --url`)
    // and it does NOT pass a PathBuf to `run_import`.
    if adapter_id == anamnesis_adapter_generic_mcp::ADAPTER_ID {
        if path_override.is_some() {
            return Err(anyhow!(
                "generic-mcp is URL-based; use `anamnesis source add generic-mcp --url <url>` \
                 instead of `--path`"
            ));
        }
        let url = registered_location.ok_or_else(|| {
            anyhow!(
                "generic-mcp source {target:?} is not registered. Run `anamnesis source add \
             generic-mcp{instance_suffix} --url <upstream-mcp-url> [--token-env ENV]` first.",
                target = target,
                instance_suffix = instance
                    .as_ref()
                    .map(|i| format!(":{i}"))
                    .unwrap_or_default(),
            )
        })?;
        let token = resolve_generic_mcp_token(registered_config.as_deref())?;
        let adapter = anamnesis_adapter_generic_mcp::generic_mcp_adapter(
            url.clone(),
            token.as_deref(),
            instance,
        );
        return run_import(data_dir, &adapter, dry_run, no_embed, None, true, scan_opts).await;
    }

    let (location, source_was_explicit) = match path_override {
        Some(p) => (p.to_path_buf(), true),
        None => match registered_location {
            Some(loc) => (PathBuf::from(loc), true),
            None => (default_path_for(adapter_id)?, false),
        },
    };

    match adapter_id {
        anamnesis_adapter_claude_code::ADAPTER_ID => {
            let adapter = ClaudeCodeAdapter::new(ClaudeCodeConfig {
                projects_root: location.clone(),
                instance: instance.map(str::to_owned),
            });
            run_import(
                data_dir,
                &adapter,
                dry_run,
                no_embed,
                Some(&location),
                source_was_explicit,
                scan_opts,
            )
            .await
        }
        anamnesis_adapter_mem0::ADAPTER_ID => {
            let adapter = mem0_sqlite_adapter(location.clone(), instance);
            run_import(
                data_dir,
                &adapter,
                dry_run,
                no_embed,
                Some(&location),
                source_was_explicit,
                scan_opts,
            )
            .await
        }
        anamnesis_adapter_codex::ADAPTER_ID => {
            let adapter = codex_adapter(location.clone(), instance);
            run_import(
                data_dir,
                &adapter,
                dry_run,
                no_embed,
                Some(&location),
                source_was_explicit,
                scan_opts,
            )
            .await
        }
        anamnesis_adapter_letta::ADAPTER_ID => {
            let adapter = letta_adapter(location.clone(), instance);
            run_import(
                data_dir,
                &adapter,
                dry_run,
                no_embed,
                Some(&location),
                source_was_explicit,
                scan_opts,
            )
            .await
        }
        anamnesis_adapter_hermes::ADAPTER_ID => {
            let adapter = hermes_adapter(location.clone(), instance);
            run_import(
                data_dir,
                &adapter,
                dry_run,
                no_embed,
                Some(&location),
                source_was_explicit,
                scan_opts,
            )
            .await
        }
        anamnesis_adapter_openclaw::ADAPTER_ID => {
            let adapter = openclaw_adapter(location.clone(), instance);
            run_import(
                data_dir,
                &adapter,
                dry_run,
                no_embed,
                Some(&location),
                source_was_explicit,
                scan_opts,
            )
            .await
        }
        anamnesis_adapter_ghast::ADAPTER_ID => {
            let adapter = ghast_adapter(location.clone(), instance);
            run_import(
                data_dir,
                &adapter,
                dry_run,
                no_embed,
                Some(&location),
                source_was_explicit,
                scan_opts,
            )
            .await
        }
        other => Err(anyhow!(
            "adapter {other:?} not wired; supported: claude-code, codex, mem0, letta, hermes, openclaw, ghast, generic-mcp"
        )),
    }
}

/// Resolve the effective `ScanOpts` for a CLI `import` invocation.
///
/// Order of precedence:
///   1. `--full`             → `ScanOpts { since: None, full: true }`
///   2. `--since "<ts>"`     → `ScanOpts { since: Some(parsed), full: false }`
///   3. neither → look up the source's registry `last_import_at` and use
///      it as the increment bound. On a fresh source (`last_import_at`
///      is `None`) this is effectively a full scan.
///
/// `--full` and `--since` are mutually exclusive at the clap layer;
/// belt-and-braces check here too.
fn resolve_scan_opts(
    data_dir: &std::path::Path,
    adapter_id: &str,
    instance: Option<&str>,
    full: bool,
    since_arg: Option<&str>,
) -> Result<anamnesis_core::adapter::ScanOpts> {
    use anamnesis_core::adapter::ScanOpts;
    if full && since_arg.is_some() {
        return Err(anyhow!("--full and --since are mutually exclusive"));
    }
    if full {
        return Ok(ScanOpts {
            since: None,
            full: true,
        });
    }
    if let Some(s) = since_arg {
        let parsed = chrono::DateTime::parse_from_rfc3339(s)
            .map_err(|e| anyhow!("--since must be RFC3339 (e.g. 2026-04-01T00:00:00Z): {e}"))?
            .with_timezone(&chrono::Utc);
        return Ok(ScanOpts {
            since: Some(parsed),
            full: false,
        });
    }
    // Fall back to the source's registered `last_import_at`.
    let store = Store::open(db_path(data_dir))?;
    let row = store.get_source(adapter_id, instance)?;
    let since = row
        .and_then(|r| r.last_import_at)
        .and_then(|t| chrono::DateTime::<chrono::Utc>::from_timestamp(t, 0));
    Ok(ScanOpts { since, full: false })
}

/// Read the bearer token for a generic-mcp source from the operator's
/// environment, looking up the env var name stored in
/// `sources.config_json` (`{"token_env": "ANAMNESIS_FOO_TOKEN"}`).
///
/// Returns `Ok(None)` when no `token_env` was registered (the upstream
/// allows unauthenticated access). Errors when the env var name is
/// registered but the variable is unset — that's a misconfiguration the
/// operator probably wants to know about *before* the import hits 401.
fn resolve_generic_mcp_token(config_json: Option<&str>) -> Result<Option<String>> {
    let Some(raw) = config_json else {
        return Ok(None);
    };
    let parsed: serde_json::Value = serde_json::from_str(raw)
        .map_err(|e| anyhow!("source.config_json is not valid JSON: {e}"))?;
    let Some(env) = parsed.get("token_env").and_then(|v| v.as_str()) else {
        return Ok(None);
    };
    match std::env::var(env) {
        Ok(v) if !v.is_empty() => Ok(Some(v)),
        Ok(_) => Err(anyhow!(
            "generic-mcp source's token_env={env:?} is set but empty"
        )),
        Err(_) => Err(anyhow!(
            "generic-mcp source requires env var {env:?} to be set with the bearer token"
        )),
    }
}

/// Whether `cmd_import` knows how to drive this adapter id. Used as the
/// up-front gate so unknown adapters get a clear "not wired" error before
/// any registry / filesystem work happens.
fn is_known_adapter(adapter_id: &str) -> bool {
    matches!(
        adapter_id,
        anamnesis_adapter_claude_code::ADAPTER_ID
            | anamnesis_adapter_mem0::ADAPTER_ID
            | anamnesis_adapter_codex::ADAPTER_ID
            | anamnesis_adapter_letta::ADAPTER_ID
            | anamnesis_adapter_hermes::ADAPTER_ID
            | anamnesis_adapter_openclaw::ADAPTER_ID
            | anamnesis_adapter_ghast::ADAPTER_ID
            | anamnesis_adapter_generic_mcp::ADAPTER_ID
    )
}

/// Adapter default discovery paths — used when neither `--path` nor a
/// registered location is available. Keep in sync with each adapter's
/// detector. Callers must gate on `is_known_adapter` first.
fn default_path_for(adapter_id: &str) -> Result<PathBuf> {
    match adapter_id {
        anamnesis_adapter_claude_code::ADAPTER_ID => home_join(&[".claude", "projects"]),
        anamnesis_adapter_mem0::ADAPTER_ID => home_join(&[".mem0", "db.sqlite"]),
        anamnesis_adapter_codex::ADAPTER_ID => home_join(&[".codex"]),
        anamnesis_adapter_letta::ADAPTER_ID => home_join(&[".letta", "letta.db"]),
        anamnesis_adapter_hermes::ADAPTER_ID => home_join(&[".hermes"]),
        anamnesis_adapter_openclaw::ADAPTER_ID => home_join(&[".openclaw"]),
        anamnesis_adapter_ghast::ADAPTER_ID => home_join(&["Documents", "ghast_desktop"]),
        other => Err(anyhow!("no default path for adapter {other:?}")),
    }
}

async fn run_import<A: anamnesis_core::adapter::MemoryAdapter>(
    data_dir: &std::path::Path,
    adapter: &A,
    dry_run: bool,
    no_embed: bool,
    canonical_location: Option<&std::path::Path>,
    source_was_explicit: bool,
    scan_opts: anamnesis_core::adapter::ScanOpts,
) -> Result<()> {
    // Round-18 (§-1.5 PR-3): the side effects of an import — preserving
    // registry `config_json`, stamping `last_import_at`, appending to
    // `audit.log` — used to live inline here. They now live in
    // `ImportService` so the MCP `tool_import_source` path produces the
    // exact same system-state delta. The CLI keeps the local pieces
    // that MCP shouldn't do: a) the human-readable progress line, b)
    // running the embedding worker (which can take minutes — fine for
    // CLI, but would block an MCP JSON-RPC request indefinitely).
    //
    // Round-19 (§-1.5 PR-4a): `scan_opts` now flows from the CLI
    // resolver (--full / --since / auto-from-`last_import_at`) into
    // `ImportOptions.scan_opts` and through to `adapter.scan(opts)`.
    let store = Store::open(db_path(data_dir))?;
    let service = ImportService::new(&store, audit(data_dir));
    let summary = service
        .import(
            adapter,
            ImportOptions {
                dry_run,
                canonical_location: canonical_location.map(|p| p.display().to_string()),
                source_was_explicit,
                scan_opts,
            },
        )
        .await
        .map_err(|e| anyhow!("import: {e}"))?;

    if dry_run {
        println!("dry-run: would import {} raw record(s)", summary.raw_seen);
        return Ok(());
    }

    println!(
        "import done: {} raw, {} upserted, {} chunks, {} errors",
        summary.raw_seen, summary.records_upserted, summary.chunks_written, summary.errors
    );

    if !no_embed {
        run_embed_worker(&store).await?;
    }
    Ok(())
}

fn audit(data_dir: &std::path::Path) -> anamnesis_core::Audit {
    anamnesis_core::Audit::new(data_dir)
}

// ─────────────────────────────────────────────────────────────────────────────
// export
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_export(
    data_dir: &std::path::Path,
    out: Option<&std::path::Path>,
    format: &str,
    source: Option<&str>,
) -> Result<()> {
    let store = Store::open(db_path(data_dir))?;
    // IMPORTANT: drop the connection guard BEFORE calling store.get_record
    // below. Store wraps Connection in parking_lot::Mutex internally; both
    // store.conn() and store.get_record() lock the same mutex, and
    // parking_lot is not re-entrant → would deadlock under load.
    let ids: Vec<String> = {
        let (where_clause, params): (String, Vec<rusqlite::types::Value>) = match source {
            Some(s) => (
                "WHERE adapter = ?1".to_string(),
                vec![rusqlite::types::Value::Text(s.to_string())],
            ),
            None => (String::new(), vec![]),
        };
        let sql = format!("SELECT id FROM records {where_clause} ORDER BY created_at ASC");
        let conn = store.conn();
        let mut stmt = conn.prepare(&sql)?;
        let collected: rusqlite::Result<Vec<String>> = stmt
            .query_map(rusqlite::params_from_iter(params), |r| {
                r.get::<_, String>(0)
            })?
            .collect();
        collected?
    }; // stmt + conn dropped here, mutex released before get_record below

    let mut writer: Box<dyn std::io::Write> = match out {
        Some(p) => Box::new(std::fs::File::create(p)?),
        None => Box::new(std::io::stdout()),
    };

    match format {
        "jsonl" => export_jsonl(&store, &ids, &mut writer)?,
        "csv" => export_csv(&store, &ids, &mut writer)?,
        other => return Err(anyhow!("unsupported format: {other} (try jsonl or csv)")),
    }
    eprintln!("exported {} record(s)", ids.len());
    audit(data_dir).record(anamnesis_core::AuditEntry::new(
        "export",
        serde_json::json!({
            "format": format,
            "source": source,
            "out": out.map(|p| p.display().to_string()),
            "records": ids.len(),
        }),
    ));
    Ok(())
}

fn export_jsonl(store: &Store, ids: &[String], writer: &mut dyn std::io::Write) -> Result<()> {
    for id in ids {
        if let Some(rec) = store.get_record(&anamnesis_core::RecordId(id.clone()))? {
            let line = serde_json::to_string(&rec)?;
            writeln!(writer, "{line}")?;
        }
    }
    Ok(())
}

fn export_csv(store: &Store, ids: &[String], writer: &mut dyn std::io::Write) -> Result<()> {
    writeln!(
        writer,
        "id,adapter,instance,kind,scope,created_at,native_id,native_path,content"
    )?;
    for id in ids {
        if let Some(rec) = store.get_record(&anamnesis_core::RecordId(id.clone()))? {
            let row = format!(
                "{id},{adapter},{instance},{kind},{scope},{created},{nid},{npath},{content}",
                id = csv_field(&rec.id.0),
                adapter = csv_field(&rec.source.adapter),
                instance = csv_field(rec.source.instance.as_deref().unwrap_or("")),
                kind = csv_field(&format!("{:?}", rec.kind).to_lowercase()),
                scope = csv_field(&format!("{:?}", rec.scope).to_lowercase()),
                created = rec.created_at.timestamp(),
                nid = csv_field(&rec.provenance.native_id),
                npath = csv_field(rec.provenance.native_path.as_deref().unwrap_or("")),
                content = csv_field(&rec.content),
            );
            writeln!(writer, "{row}")?;
        }
    }
    Ok(())
}

fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        let escaped = s.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        s.to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// verify
// ─────────────────────────────────────────────────────────────────────────────

fn cmd_verify(data_dir: &std::path::Path, repair: bool) -> Result<()> {
    let store = Store::open(db_path(data_dir))?;
    let mut problems = 0u64;

    // All raw-conn diagnostic queries are scoped tightly so the
    // parking_lot Mutex guard never overlaps with calls back into Store
    // (active_model / rebuild_embedding_jobs would otherwise deadlock).

    // 1. SQLite integrity_check.
    let integrity: String = {
        let conn = store.conn();
        conn.query_row("PRAGMA integrity_check(1)", [], |r| r.get(0))?
    };
    if integrity == "ok" {
        println!("integrity_check : ok");
    } else {
        println!("integrity_check : {integrity}");
        problems += 1;
    }

    // 2. records → record_chunks consistency.
    let (records_count, records_with_chunks): (i64, i64) = {
        let conn = store.conn();
        let rc: i64 = conn.query_row("SELECT COUNT(1) FROM records", [], |r| r.get(0))?;
        let rwc: i64 = conn.query_row(
            "SELECT COUNT(1) FROM records r WHERE EXISTS (SELECT 1 FROM record_chunks c WHERE c.record_id = r.id)",
            [],
            |r| r.get(0),
        )?;
        (rc, rwc)
    };
    let orphan_records = records_count - records_with_chunks;
    if orphan_records == 0 {
        println!("orphan records  : 0");
    } else {
        println!("orphan records  : {orphan_records} (no chunks)");
        problems += 1;
    }

    // 3. FTS index vs record_chunks row count.
    let (chunks_count, fts_count): (i64, i64) = {
        let conn = store.conn();
        let cc: i64 = conn.query_row("SELECT COUNT(1) FROM record_chunks", [], |r| r.get(0))?;
        let fc: i64 = conn.query_row("SELECT COUNT(1) FROM chunks_fts", [], |r| r.get(0))?;
        (cc, fc)
    };
    if chunks_count == fts_count {
        println!("FTS index       : ok ({chunks_count} rows)");
    } else {
        println!("FTS index       : drift ({chunks_count} chunks vs {fts_count} FTS rows)");
        problems += 1;
        if repair {
            println!("FTS index       : rebuilding…");
            let conn = store.conn();
            conn.execute("DELETE FROM chunks_fts", [])?;
            conn.execute(
                "INSERT INTO chunks_fts(rowid, content) SELECT rowid, content FROM record_chunks",
                [],
            )?;
            println!("FTS index       : rebuilt");
        }
    }

    // 4. embeddings vs active model — count chunks that lack an embedding
    //    under the current model.
    if let Some(active) = store.active_model()? {
        let missing: i64 = {
            let conn = store.conn();
            conn.query_row(
                "SELECT COUNT(1) FROM record_chunks c \
                 WHERE NOT EXISTS (SELECT 1 FROM chunk_embeddings e \
                    WHERE e.chunk_id = c.id AND e.model_id = ?1)",
                [&active],
                |r| r.get(0),
            )?
        };
        println!("missing embeds  : {missing} (model: {active})");
        if missing > 0 && repair {
            let n = store.rebuild_embedding_jobs(&active)?;
            println!("missing embeds  : re-queued {n} embedding job(s)");
        }
    } else {
        println!("missing embeds  : skipped (no active model)");
    }

    if problems == 0 {
        println!("status          : healthy");
    } else if repair {
        println!("status          : repair attempted on {problems} issue(s)");
    } else {
        println!("status          : {problems} issue(s) found (run with --repair)");
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Kind / Scope parsers (shared by search filters)
// ─────────────────────────────────────────────────────────────────────────────

fn parse_kind(s: &str) -> Result<anamnesis_core::Kind> {
    use anamnesis_core::Kind;
    Ok(match s {
        "fact" => Kind::Fact,
        "preference" => Kind::Preference,
        "feedback" => Kind::Feedback,
        "reference" => Kind::Reference,
        "episode" => Kind::Episode,
        "skill" => Kind::Skill,
        "unknown" => Kind::Unknown,
        other => return Err(anyhow!("unknown kind: {other}")),
    })
}

fn parse_scope(s: &str) -> Result<anamnesis_core::Scope> {
    use anamnesis_core::Scope;
    Ok(match s {
        "user" => Scope::User,
        "project" => Scope::Project,
        "session" => Scope::Session,
        "ephemeral" => Scope::Ephemeral,
        other => return Err(anyhow!("unknown scope: {other}")),
    })
}

fn home_join(parts: &[&str]) -> Result<PathBuf> {
    let mut p = dirs_home()?;
    for part in parts {
        p = p.join(part);
    }
    Ok(p)
}

// ─────────────────────────────────────────────────────────────────────────────
// search
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
async fn cmd_search(
    data_dir: &std::path::Path,
    query: &str,
    source: Option<&str>,
    instance: Option<&str>,
    kind: Option<&str>,
    scope: Option<&str>,
    since: Option<&str>,
    until: Option<&str>,
    limit: u32,
    mode_str: &str,
    json: bool,
) -> Result<()> {
    let store = Store::open(db_path(data_dir))?;
    let mode = match mode_str {
        "fulltext" => SearchMode::Fulltext,
        "vector" => SearchMode::Vector,
        _ => SearchMode::Hybrid,
    };
    let kind_filter = match kind {
        Some(k) => Some(parse_kind(k)?),
        None => None,
    };
    let scope_filter = match scope {
        Some(s) => Some(parse_scope(s)?),
        None => None,
    };
    // Round-22 (§-1.5 PR-5): RFC3339 → unix-seconds for the
    // `SearchFilter.time_from` / `time_to` SQL pushdown. Clap-level
    // validation (`Option<String>`) keeps the wire ergonomic; the
    // adapter-level parse error here is what users see if they hand-
    // type a malformed timestamp.
    let time_from = match since {
        Some(s) => Some(
            chrono::DateTime::parse_from_rfc3339(s)
                .map_err(|e| anyhow!("--since must be RFC3339 (e.g. 2026-04-01T00:00:00Z): {e}"))?
                .with_timezone(&chrono::Utc)
                .timestamp(),
        ),
        None => None,
    };
    let time_to = match until {
        Some(s) => Some(
            chrono::DateTime::parse_from_rfc3339(s)
                .map_err(|e| anyhow!("--until must be RFC3339 (e.g. 2026-04-30T23:59:59Z): {e}"))?
                .with_timezone(&chrono::Utc)
                .timestamp(),
        ),
        None => None,
    };

    // Embedding provider needed for Vector/Hybrid modes.
    let provider = match mode {
        SearchMode::Fulltext => None,
        _ => Some(open_active_provider(data_dir, &store)?),
    };

    // PR-C: build the SQL-level filter from the same CLI knobs the
    // post-filter used to consume. We turn `Kind` / `Scope` back into
    // their lower-case string form to match how the store writes them.
    let store_filter = anamnesis_store::SearchFilter {
        source: source.map(str::to_owned),
        instance: instance.map(str::to_owned),
        kind: kind_filter.map(|k| format!("{k:?}").to_lowercase()),
        scope: scope_filter.map(|s| format!("{s:?}").to_lowercase()),
        time_from,
        time_to,
    };

    let hits = run_search(&store, query, &store_filter, limit, mode, provider.as_ref()).await?;

    let packed = pack(
        &store,
        &hits,
        &ContextBudget {
            max_records: limit as usize,
            ..ContextBudget::default()
        },
    )?;

    let filtered: Vec<_> = packed
        .into_iter()
        .filter(|p| source.is_none_or(|src| p.record.source.adapter == src))
        .filter(|p| kind_filter.is_none_or(|k| p.record.kind == k))
        .filter(|p| scope_filter.is_none_or(|s| p.record.scope == s))
        .collect();

    audit(data_dir).record(anamnesis_core::AuditEntry::new(
        "search",
        serde_json::json!({
            "query": query,
            "source": source,
            "kind": kind,
            "scope": scope,
            "mode": mode_str,
            "limit": limit,
            "hits": filtered.len(),
        }),
    ));

    if json {
        let payload = serde_json::json!({
            "query": query,
            "mode": mode_str,
            // Round-8: same expanded wire format as the MCP server so
            // CLI and MCP consumers can rely on identical JSON shapes.
            "results": filtered.iter().map(|p| {
                let best = p.matched_chunks.first();
                serde_json::json!({
                    "record_id": p.record.id.0,
                    "trace_id": p.record.id.0,
                    "chunk_id": best.map(|c| c.chunk_id.clone()),
                    "adapter": p.record.source.adapter,
                    "instance": p.record.source.instance,
                    "kind": format!("{:?}", p.record.kind).to_lowercase(),
                    "scope": format!("{:?}", p.record.scope).to_lowercase(),
                    "score": p.score,
                    "rrf_score": p.score,
                    "fts_score": best.and_then(|c| c.fts_score),
                    "vector_score": best.and_then(|c| c.vector_score),
                    "from_fts": best.map(|c| c.from_fts).unwrap_or(false),
                    "from_vec": best.map(|c| c.from_vec).unwrap_or(false),
                    "snippet": best.map(|c| c.content.clone()).unwrap_or_default(),
                    "native_path": p.record.provenance.native_path,
                    "created_at": p.record.created_at.timestamp(),
                    "updated_at": p.record.updated_at.map(|t| t.timestamp()),
                })
            }).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else if filtered.is_empty() {
        println!("no results");
    } else {
        // Round-12: human-readable card mirrors the JSON wire format
        // (PR-#16) — same fields, same names, same semantics. CLI
        // operators see what MCP agents see; nothing is invented or
        // recomputed.
        for (i, p) in filtered.iter().enumerate() {
            let best = p.matched_chunks.first();
            let kind = format!("{:?}", p.record.kind).to_lowercase();
            let scope = format!("{:?}", p.record.scope).to_lowercase();

            // Line 1: rank, RRF score, adapter[:instance], kind/scope.
            let inst = p
                .record
                .source
                .instance
                .as_deref()
                .filter(|s| !s.is_empty())
                .map(|s| format!(":{s}"))
                .unwrap_or_default();
            println!(
                "[{:>2}] rrf={:.3}  {}{}  ({kind}, {scope})",
                i + 1,
                p.score,
                p.record.source.adapter,
                inst,
            );

            // Line 2: per-modality score breakdown + timestamps. Same
            // raw values the JSON exposes; null modality scores rendered
            // as `-` so the column line stays parseable visually.
            let fts = best
                .and_then(|c| c.fts_score)
                .map(|s| format!("{s:.3}"))
                .unwrap_or_else(|| "-".into());
            let vec = best
                .and_then(|c| c.vector_score)
                .map(|s| format!("{s:.3}"))
                .unwrap_or_else(|| "-".into());
            let created = p.record.created_at.format("%Y-%m-%dT%H:%MZ");
            let updated = p
                .record
                .updated_at
                .map(|t| t.format("%Y-%m-%dT%H:%MZ").to_string())
                .unwrap_or_else(|| "-".into());
            println!("     fts={fts}  vec={vec}  created={created}  updated={updated}");

            // Line 3: trace ids — exactly what an agent / a follow-up
            // CLI invocation would feed into `trace_provenance` /
            // `get_record`. Surface both record_id and chunk_id so the
            // operator can copy-paste either.
            let chunk_id = best
                .map(|c| c.chunk_id.clone())
                .unwrap_or_else(|| "-".into());
            println!(
                "     record_id={}  chunk_id={}  trace_id={}",
                p.record.id.0, chunk_id, p.record.id.0,
            );

            // Line 4: native_path (full, so operators can `cat $path`).
            println!(
                "     native_path={}",
                p.record.provenance.native_path.as_deref().unwrap_or("-"),
            );

            // Line 5: snippet (truncated on char boundary to stay terminal-safe).
            if let Some(c) = best {
                let snippet = c.content.replace('\n', " ");
                let snippet = if snippet.chars().count() > 180 {
                    let mut s: String = snippet.chars().take(180).collect();
                    s.push('…');
                    s
                } else {
                    snippet
                };
                println!("     snippet: {snippet}");
            }
            println!();
        }
    }
    Ok(())
}

async fn run_search(
    store: &Store,
    query: &str,
    filter: &anamnesis_store::SearchFilter,
    limit: u32,
    mode: SearchMode,
    provider: Option<&ProviderHandle>,
) -> Result<Vec<anamnesis_search::RankedChunk>> {
    let opts = HybridOpts {
        limit,
        candidate_pool: (limit * 4).max(limit),
        mode,
    };
    // PR-C: `search_filtered` pushes the filter into the SQL recall
    // stage so the candidate pool can't be dominated by a majority
    // adapter. The old post-RRF filter (still applied below as a safety
    // net) becomes a no-op rather than the only line of defense.
    match provider {
        Some(handle) => match handle {
            #[cfg(feature = "local-fastembed")]
            ProviderHandle::Local(p) => Ok(HybridSearcher::new(p.as_ref())
                .search_filtered(store, query, filter, &opts)
                .await?),
            ProviderHandle::None => Ok(HybridSearcher::<DummyProvider>::fulltext_only()
                .search_filtered(store, query, filter, &opts)
                .await?),
        },
        None => Ok(HybridSearcher::<DummyProvider>::fulltext_only()
            .search_filtered(store, query, filter, &opts)
            .await?),
    }
}

/// Type-erased provider handle so the CLI can branch on feature flags.
enum ProviderHandle {
    #[cfg(feature = "local-fastembed")]
    Local(Box<anamnesis_embedder::LocalFastembedProvider>),
    /// No provider available — present only so `match` has a None arm
    /// usable at compile time without fastembed.
    #[allow(dead_code)]
    None,
}

/// Dummy provider for fulltext_only generics — never actually instantiated.
struct DummyProvider;
#[async_trait::async_trait]
impl anamnesis_core::EmbeddingProvider for DummyProvider {
    fn model_id(&self) -> anamnesis_core::ModelId {
        anamnesis_core::ModelId::new("dummy", "dummy", 1)
    }
    fn dim(&self) -> u16 {
        1
    }
    async fn embed_batch(
        &self,
        _texts: &[&str],
        _task: anamnesis_core::EmbeddingTask,
    ) -> anamnesis_core::Result<Vec<Vec<f32>>> {
        Err(anamnesis_core::Error::Other("dummy provider".into()))
    }
}

#[cfg(feature = "local-fastembed")]
fn open_active_provider(data_dir: &std::path::Path, store: &Store) -> Result<ProviderHandle> {
    let active = store.active_model()?.ok_or_else(|| {
        anyhow!(
            "no active embedding model set; run `anamnesis init` or `anamnesis model use <key>`"
        )
    })?;
    let key = parse_model_key(&active)?;
    let provider = anamnesis_embedder::LocalFastembedProvider::new(&key, models_dir(data_dir))
        .map_err(|e| anyhow!("open local embedding model {key}: {e}"))?;
    Ok(ProviderHandle::Local(Box::new(provider)))
}

#[cfg(not(feature = "local-fastembed"))]
fn open_active_provider(_data_dir: &std::path::Path, _store: &Store) -> Result<ProviderHandle> {
    Err(anyhow!(
        "this anamnesis build lacks `local-fastembed` feature; rebuild with `--features local-fastembed`"
    ))
}

fn parse_model_key(model_id: &str) -> Result<String> {
    // model_id format: "<provider>:<key>:<version>"
    let parts: Vec<&str> = model_id.split(':').collect();
    if parts.len() != 3 {
        return Err(anyhow!("malformed active model id: {model_id}"));
    }
    Ok(parts[1].to_string())
}

// ─────────────────────────────────────────────────────────────────────────────
// embedding worker — used after import or model switch
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "local-fastembed")]
async fn run_embed_worker(store: &Store) -> Result<()> {
    let Some(active) = store.active_model()? else {
        return Ok(());
    };
    let key = parse_model_key(&active)?;
    let data_dir_guess = home_join(&[".local", "share", "anamnesis"]).unwrap_or_default();
    // The CLI keeps models under {data_dir}/models — get data_dir from
    // the store path. For simplicity, we always use the standard location;
    // the model is downloaded once and re-used.
    let cache = data_dir_guess.join("models");
    let provider = anamnesis_embedder::LocalFastembedProvider::new(&key, &cache)
        .map_err(|e| anyhow!("open local model for embedding worker: {e}"))?;
    let worker = anamnesis_embedder::EmbeddingWorker::new(&provider);
    let summary = worker
        .drain(store)
        .await
        .map_err(|e| anyhow!("worker drain: {e}"))?;
    println!(
        "embedding worker: {} processed, {} failed",
        summary.processed, summary.failed
    );
    Ok(())
}

#[cfg(not(feature = "local-fastembed"))]
async fn run_embed_worker(_store: &Store) -> Result<()> {
    println!("local-fastembed feature disabled; skipping embedding worker");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// model list / use / install / rebuild
// ─────────────────────────────────────────────────────────────────────────────

async fn cmd_model(data_dir: &std::path::Path, sub: ModelCmd) -> Result<()> {
    let store = Store::open(db_path(data_dir))?;
    let active = store.active_model()?;
    let active_key = active.as_deref().and_then(|id| parse_model_key(id).ok());
    match sub {
        ModelCmd::List => {
            println!(
                "{:<14} {:<8} {:<32} {:<8} {:<6}",
                "key", "active", "hf_id", "size_mb", "dim"
            );
            for m in registry::REGISTRY {
                let marker = if Some(m.key.to_string()) == active_key {
                    "yes"
                } else {
                    ""
                };
                println!(
                    "{:<14} {:<8} {:<32} {:<8} {:<6}",
                    m.key,
                    marker,
                    m.hf_id,
                    if m.approx_size_mb == 0 {
                        "cloud".to_string()
                    } else {
                        format!("{}", m.approx_size_mb)
                    },
                    m.dim
                );
            }
            Ok(())
        }
        ModelCmd::Use { key, no_embed } => {
            let m = registry::by_key(&key).ok_or_else(|| {
                anyhow!(
                    "unknown model key {key:?}; available: {}",
                    registry::available().join(", ")
                )
            })?;
            if !m.is_local {
                return Err(anyhow!(
                    "model {key:?} is a cloud provider; not supported in Phase 1"
                ));
            }
            let new_id = format!("local:{}:{}", m.key, 1);
            store.set_active_model(&new_id)?;
            let n = store.rebuild_embedding_jobs(&new_id)?;
            println!("active model now: {new_id} ({n} chunks re-queued)");
            if !no_embed && n > 0 {
                let store = Store::open(db_path(data_dir))?;
                run_embed_worker(&store).await?;
            }
            Ok(())
        }
        ModelCmd::Install { key } => {
            let m = registry::by_key(&key).ok_or_else(|| anyhow!("unknown model key {key:?}"))?;
            if !m.is_local {
                return Err(anyhow!(
                    "model {key:?} is a cloud provider; nothing to download"
                ));
            }
            install_model(data_dir, &key)?;
            println!(
                "installed model {key} to {}",
                models_dir(data_dir).display()
            );
            Ok(())
        }
        ModelCmd::Rebuild { no_embed } => {
            let active = store
                .active_model()?
                .ok_or_else(|| anyhow!("no active embedding model set"))?;
            let n = store.rebuild_embedding_jobs(&active)?;
            println!("re-queued {n} chunks under {active}");
            if !no_embed && n > 0 {
                let store = Store::open(db_path(data_dir))?;
                run_embed_worker(&store).await?;
            }
            Ok(())
        }
    }
}

#[cfg(feature = "local-fastembed")]
fn install_model(data_dir: &std::path::Path, key: &str) -> Result<()> {
    // Constructing the provider triggers a download into models_dir.
    let _ = anamnesis_embedder::LocalFastembedProvider::new(key, models_dir(data_dir))
        .map_err(|e| anyhow!("install {key}: {e}"))?;
    Ok(())
}

#[cfg(not(feature = "local-fastembed"))]
fn install_model(_data_dir: &std::path::Path, _key: &str) -> Result<()> {
    Err(anyhow!(
        "this build lacks `local-fastembed`; rebuild with `--features local-fastembed`"
    ))
}

#[cfg(test)]
mod freshness_tests {
    use super::{human_age_short, source_freshness};

    #[test]
    fn freshness_never_imported_when_last_import_is_none() {
        let f = source_freshness(None, 1_000_000);
        assert_eq!(f.label, "never-imported");
        assert!(f.age_seconds.is_none());
        assert_eq!(f.age_human, "<never>");
    }

    #[test]
    fn freshness_fresh_within_24h() {
        let now = 1_000_000_i64;
        // Just imported.
        let f = source_freshness(Some(now), now);
        assert_eq!(f.label, "fresh");
        assert_eq!(f.age_seconds, Some(0));
        assert_eq!(f.age_human, "<1m");

        // 23h59m ago → still fresh.
        let f = source_freshness(Some(now - (24 * 3600 - 1)), now);
        assert_eq!(f.label, "fresh");
    }

    #[test]
    fn freshness_stale_after_24h_boundary() {
        let now = 1_000_000_i64;
        // Exactly 24h → boundary lands on stale.
        let f = source_freshness(Some(now - 24 * 3600), now);
        assert_eq!(f.label, "stale");
        assert_eq!(f.age_seconds, Some(24 * 3600));
        assert_eq!(f.age_human, "1d");

        // 7 days ago.
        let f = source_freshness(Some(now - 7 * 24 * 3600), now);
        assert_eq!(f.label, "stale");
        assert_eq!(f.age_human, "7d");
    }

    #[test]
    fn freshness_clamps_negative_age_to_zero() {
        // A clock skew that makes `last_import_at > now` must not
        // underflow or report negative age. Round to "<1m"/fresh.
        let f = source_freshness(Some(2_000_000), 1_000_000);
        assert_eq!(f.label, "fresh");
        assert_eq!(f.age_seconds, Some(0));
        assert_eq!(f.age_human, "<1m");
    }

    #[test]
    fn human_age_short_buckets() {
        assert_eq!(human_age_short(0), "<1m");
        assert_eq!(human_age_short(30), "<1m");
        assert_eq!(human_age_short(59), "<1m");
        assert_eq!(human_age_short(60), "1m");
        assert_eq!(human_age_short(125), "2m");
        assert_eq!(human_age_short(3600), "1h");
        assert_eq!(human_age_short(7200), "2h");
        assert_eq!(human_age_short(24 * 3600), "1d");
        assert_eq!(human_age_short(7 * 24 * 3600), "7d");
        assert_eq!(human_age_short(29 * 24 * 3600), "29d");
        assert_eq!(human_age_short(30 * 24 * 3600), "30d+");
        assert_eq!(human_age_short(365 * 24 * 3600), "30d+");
    }
}
