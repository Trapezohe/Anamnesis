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
        /// Restrict to one source (adapter id).
        #[arg(long)]
        source: Option<String>,
        /// Restrict to one Kind: fact | preference | feedback | reference | episode | skill | unknown.
        #[arg(long)]
        kind: Option<String>,
        /// Restrict to one Scope: user | project | session | ephemeral.
        #[arg(long)]
        scope: Option<String>,
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

    /// Run as an MCP server. Default mode = stdio; future flag --sse for
    /// the network transport (Phase 4).
    Serve {
        /// Reserved for the SSE transport (not yet wired in this CLI;
        /// use the anamnesis-mcp binary's --sse flag instead).
        #[arg(long)]
        sse: Option<u16>,
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
            dry_run,
            no_embed,
            path,
        } => cmd_import(&data_dir, &target, full, dry_run, no_embed, path.as_deref()).await,
        Command::Search {
            query,
            source,
            kind,
            scope,
            limit,
            mode,
            json,
        } => {
            cmd_search(
                &data_dir,
                &query,
                source.as_deref(),
                kind.as_deref(),
                scope.as_deref(),
                limit,
                &mode,
                json,
            )
            .await
        }
        Command::Model(sub) => cmd_model(&data_dir, sub).await,
        Command::Serve { sse } => cmd_serve(&data_dir, sse).await,
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

async fn cmd_serve(data_dir: &std::path::Path, sse: Option<u16>) -> Result<()> {
    if sse.is_some() {
        return Err(anyhow!(
            "SSE transport is not yet wired into `anamnesis serve` — see Phase 4. \
             Use the dedicated `anamnesis-mcp --sse` binary in the meantime."
        ));
    }
    let store = Store::open(db_path(data_dir))?;
    let active_model = store.active_model().ok().flatten();
    let provider = open_active_provider_optional(data_dir, &store, active_model.as_deref());
    let server =
        anamnesis_mcp_server::AnamnesisServer::new(store, provider, data_dir.to_path_buf());
    eprintln!(
        "anamnesis serve (stdio) — active model: {}",
        active_model.as_deref().unwrap_or("<unset>")
    );
    anamnesis_mcp_server::stdio::run(server).await
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
    let sources = store.list_sources()?;
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
            "sources": sources.iter().map(|(a, i, loc)| serde_json::json!({
                "adapter": a,
                "instance": if i.is_empty() { serde_json::Value::Null } else { serde_json::Value::String(i.clone()) },
                "location": loc,
            })).collect::<Vec<_>>(),
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
    let conn = store.conn();
    let (where_clause, params): (String, Vec<rusqlite::types::Value>) = match source {
        Some(s) => (
            "WHERE adapter = ?1".to_string(),
            vec![rusqlite::types::Value::Text(s.to_string())],
        ),
        None => (String::new(), vec![]),
    };
    let sql = format!("SELECT id FROM records {where_clause} ORDER BY created_at ASC");
    let mut stmt = conn.prepare(&sql)?;
    let ids: Vec<String> = stmt
        .query_map(rusqlite::params_from_iter(params), |r| {
            r.get::<_, String>(0)
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

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
    let conn = store.conn();
    let mut problems = 0u64;

    // 1. SQLite integrity_check.
    let integrity: String = conn.query_row("PRAGMA integrity_check(1)", [], |r| r.get(0))?;
    if integrity == "ok" {
        println!("integrity_check : ok");
    } else {
        println!("integrity_check : {integrity}");
        problems += 1;
    }

    // 2. records → record_chunks consistency.
    let records_count: i64 = conn.query_row("SELECT COUNT(1) FROM records", [], |r| r.get(0))?;
    let records_with_chunks: i64 = conn.query_row(
        "SELECT COUNT(1) FROM records r WHERE EXISTS (SELECT 1 FROM record_chunks c WHERE c.record_id = r.id)",
        [],
        |r| r.get(0),
    )?;
    let orphan_records = records_count - records_with_chunks;
    if orphan_records == 0 {
        println!("orphan records  : 0");
    } else {
        println!("orphan records  : {orphan_records} (no chunks)");
        problems += 1;
    }

    // 3. FTS index vs record_chunks row count.
    let chunks_count: i64 =
        conn.query_row("SELECT COUNT(1) FROM record_chunks", [], |r| r.get(0))?;
    let fts_count: i64 = conn.query_row("SELECT COUNT(1) FROM chunks_fts", [], |r| r.get(0))?;
    if chunks_count == fts_count {
        println!("FTS index       : ok ({chunks_count} rows)");
    } else {
        println!("FTS index       : drift ({chunks_count} chunks vs {fts_count} FTS rows)");
        problems += 1;
        if repair {
            println!("FTS index       : rebuilding…");
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
        let missing: i64 = conn.query_row(
            "SELECT COUNT(1) FROM record_chunks c \
             WHERE NOT EXISTS (SELECT 1 FROM chunk_embeddings e \
                WHERE e.chunk_id = c.id AND e.model_id = ?1)",
            [&active],
            |r| r.get(0),
        )?;
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
    kind: Option<&str>,
    scope: Option<&str>,
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

    let filtered: Vec<_> = packed
        .into_iter()
        .filter(|p| source.is_none_or(|src| p.record.source.adapter == src))
        .filter(|p| kind_filter.is_none_or(|k| p.record.kind == k))
        .filter(|p| scope_filter.is_none_or(|s| p.record.scope == s))
        .collect();

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
