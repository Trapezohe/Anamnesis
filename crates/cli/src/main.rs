//! `anamnesis` CLI entry point.

#![forbid(unsafe_code)]
// Aligned table headers are clearer than building format strings with
// inlined idents — clippy's literal-format-arg lint isn't load-bearing here.
#![allow(clippy::print_literal)]

use std::path::PathBuf;

use anamnesis_adapter_claude_code::{ClaudeCodeAdapter, ClaudeCodeConfig, ClaudeCodeDetector};
use anamnesis_adapter_mem0::{sqlite_adapter as mem0_sqlite_adapter, Mem0SqliteDetector};
use anamnesis_core::discovery::{DetectOpts, Discovery};
use anamnesis_embedder::registry;
use anamnesis_importer::ImportRunner;
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
    Status,

    /// Scan default paths for known memory sources (read-only).
    Discover,

    /// Manage configured memory sources.
    #[command(subcommand)]
    Source(SourceCmd),

    /// Run an import job for one source.
    Import {
        /// Adapter name, optionally `adapter:instance`.
        target: String,
        /// Full re-scan, ignoring dedup hashes.
        #[arg(long)]
        full: bool,
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
        /// Restrict to one source.
        #[arg(long)]
        source: Option<String>,
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

    /// Export records as JSONL or CSV (not yet implemented).
    Export {
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value = "jsonl")]
        format: String,
    },

    /// Run as an MCP server (not yet implemented).
    Serve {
        #[arg(long)]
        sse: Option<u16>,
    },

    /// Verify database integrity and rebuild indexes (not yet implemented).
    Verify {
        #[arg(long)]
        repair: bool,
    },

    /// Run pending schema migrations (no-op after init).
    Migrate,
}

#[derive(Subcommand, Debug)]
enum SourceCmd {
    /// Register a new source.
    Add {
        /// Adapter name (e.g. `claude-code`, `mem0`).
        adapter: String,
        /// Instance discriminator (optional).
        #[arg(long)]
        instance: Option<String>,
        /// Filesystem path, if the adapter takes one.
        #[arg(long)]
        path: Option<PathBuf>,
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
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .context("HOME not set")
}

fn db_path(data_dir: &std::path::Path) -> PathBuf {
    data_dir.join("anamnesis.sqlite")
}

fn models_dir(data_dir: &std::path::Path) -> PathBuf {
    data_dir.join("models")
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level);
    let data_dir = resolve_data_dir(cli.data_dir)?;

    match cli.command {
        Command::Init { model } => cmd_init(&data_dir, model.as_deref()),
        Command::Status => cmd_status(&data_dir),
        Command::Discover => cmd_discover().await,
        Command::Source(sub) => cmd_source(&data_dir, sub),
        Command::Import {
            target,
            full,
            dry_run,
            no_embed,
            path,
        } => cmd_import(&data_dir, &target, full, dry_run, no_embed, path.as_deref()).await,
        Command::Search {
            query,
            source,
            limit,
            mode,
            json,
        } => cmd_search(&data_dir, &query, source.as_deref(), limit, &mode, json).await,
        Command::Model(sub) => cmd_model(&data_dir, sub).await,
        Command::Export { .. } | Command::Serve { .. } | Command::Verify { .. } => {
            eprintln!("not yet implemented in Phase 1 — coming in Phase 2+");
            std::process::exit(2);
        }
        Command::Migrate => {
            let _ = Store::open(db_path(&data_dir))?;
            println!("migrations applied");
            Ok(())
        }
    }
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

fn cmd_status(data_dir: &std::path::Path) -> Result<()> {
    let db = db_path(data_dir);
    if !db.exists() {
        println!(
            "no database found at {} — run `anamnesis init`",
            db.display()
        );
        return Ok(());
    }
    let store = Store::open(&db)?;
    let stats = store.stats()?;
    let active = store.active_model()?;
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
    Ok(())
}

async fn cmd_discover() -> Result<()> {
    let discovery = Discovery::new()
        .register(Box::new(ClaudeCodeDetector::new()))
        .register(Box::new(Mem0SqliteDetector::new()));
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
        } => {
            let location = path.as_ref().map(|p| p.display().to_string());
            store.register_source(&adapter, instance.as_deref(), location.as_deref(), None)?;
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
            let rows = store.list_sources()?;
            if rows.is_empty() {
                println!("no sources registered");
            } else {
                println!("{:<14} {:<14} {}", "adapter", "instance", "location");
                for (a, i, loc) in rows {
                    println!("{:<14} {:<14} {}", a, i, loc.unwrap_or_default());
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
    _full: bool,
    dry_run: bool,
    no_embed: bool,
    path_override: Option<&std::path::Path>,
) -> Result<()> {
    let (adapter_id, instance) = split_target(target);
    match adapter_id {
        anamnesis_adapter_claude_code::ADAPTER_ID => {
            let projects_root = path_override
                .map(PathBuf::from)
                .map_or_else(|| home_join(&[".claude", "projects"]), Ok)?;
            let adapter = ClaudeCodeAdapter::new(ClaudeCodeConfig {
                projects_root,
                instance: instance.map(str::to_owned),
            });
            run_import(data_dir, &adapter, dry_run, no_embed).await
        }
        anamnesis_adapter_mem0::ADAPTER_ID => {
            let db_path_for_mem0 = path_override
                .map(PathBuf::from)
                .map_or_else(|| home_join(&[".mem0", "db.sqlite"]), Ok)?;
            let adapter = mem0_sqlite_adapter(db_path_for_mem0, instance);
            run_import(data_dir, &adapter, dry_run, no_embed).await
        }
        other => Err(anyhow!(
            "adapter {other:?} not wired; supported: claude-code, mem0"
        )),
    }
}

async fn run_import<A: anamnesis_core::adapter::MemoryAdapter>(
    data_dir: &std::path::Path,
    adapter: &A,
    dry_run: bool,
    no_embed: bool,
) -> Result<()> {
    if dry_run {
        use anamnesis_core::adapter::ScanOpts;
        use futures::StreamExt;
        let mut stream = adapter.scan(ScanOpts::default());
        let mut seen = 0usize;
        while let Some(item) = stream.next().await {
            if item.is_ok() {
                seen += 1;
            }
        }
        println!("dry-run: would import {seen} raw record(s)");
        return Ok(());
    }

    let mut store = Store::open(db_path(data_dir))?;
    let summary = ImportRunner::new(adapter).run(&mut store).await?;
    println!(
        "import done: {} raw, {} upserted, {} chunks, {} errors",
        summary.raw_seen, summary.records_upserted, summary.chunks_written, summary.errors
    );

    if !no_embed {
        run_embed_worker(&mut store).await?;
    }
    Ok(())
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

async fn cmd_search(
    data_dir: &std::path::Path,
    query: &str,
    source: Option<&str>,
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

    // Embedding provider needed for Vector/Hybrid modes.
    let provider = match mode {
        SearchMode::Fulltext => None,
        _ => Some(open_active_provider(data_dir, &store)?),
    };

    let hits = run_search(&store, query, limit, mode, provider.as_ref()).await?;

    let packed = pack(
        &store,
        &hits,
        &ContextBudget {
            max_records: limit as usize,
            ..ContextBudget::default()
        },
    )?;

    let filtered: Vec<_> = if let Some(src) = source {
        packed
            .into_iter()
            .filter(|p| p.record.source.adapter == src)
            .collect()
    } else {
        packed
    };

    if json {
        let payload = serde_json::json!({
            "query": query,
            "mode": mode_str,
            "results": filtered.iter().map(|p| serde_json::json!({
                "record_id": p.record.id.0,
                "adapter": p.record.source.adapter,
                "instance": p.record.source.instance,
                "kind": format!("{:?}", p.record.kind).to_lowercase(),
                "scope": format!("{:?}", p.record.scope).to_lowercase(),
                "score": p.score,
                "snippet": p.matched_chunks.first().map(|c| c.content.clone()).unwrap_or_default(),
                "native_path": p.record.provenance.native_path,
            })).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else if filtered.is_empty() {
        println!("no results");
    } else {
        for (i, p) in filtered.iter().enumerate() {
            println!(
                "[{:>2}] {:>5.3}  {}  ({}, {:?})",
                i + 1,
                p.score,
                p.record.source.adapter,
                p.record.provenance.native_path.as_deref().unwrap_or("?"),
                p.record.kind,
            );
            if let Some(c) = p.matched_chunks.first() {
                let snippet = c.content.replace('\n', " ");
                let snippet = if snippet.len() > 180 {
                    format!("{}…", &snippet[..180])
                } else {
                    snippet
                };
                println!("       {snippet}");
            }
        }
    }
    Ok(())
}

async fn run_search(
    store: &Store,
    query: &str,
    limit: u32,
    mode: SearchMode,
    provider: Option<&ProviderHandle>,
) -> Result<Vec<anamnesis_search::RankedChunk>> {
    let opts = HybridOpts {
        limit,
        candidate_pool: (limit * 4).max(limit),
        mode,
    };
    match provider {
        Some(handle) => match handle {
            #[cfg(feature = "local-fastembed")]
            ProviderHandle::Local(p) => Ok(HybridSearcher::new(p.as_ref())
                .search(store, query, &opts)
                .await?),
            ProviderHandle::None => Ok(HybridSearcher::<DummyProvider>::fulltext_only()
                .search(store, query, &opts)
                .await?),
        },
        None => Ok(HybridSearcher::<DummyProvider>::fulltext_only()
            .search(store, query, &opts)
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
async fn run_embed_worker(store: &mut Store) -> Result<()> {
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
async fn run_embed_worker(_store: &mut Store) -> Result<()> {
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
                let mut store = Store::open(db_path(data_dir))?;
                run_embed_worker(&mut store).await?;
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
                let mut store = Store::open(db_path(data_dir))?;
                run_embed_worker(&mut store).await?;
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
