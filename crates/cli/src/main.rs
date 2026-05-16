//! `anamnesis` CLI entry point.
//!
//! Phase-0 scope: skeleton — every subcommand parses but only `init` and
//! `status` do real work. The rest print a "not yet implemented" stub so
//! the surface area is visible while Phase 1 fills it in.

#![forbid(unsafe_code)]

use std::path::PathBuf;

use anyhow::{Context, Result};
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
    Init,

    /// Show database stats and source health.
    Status,

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
        /// Only show what would be imported.
        #[arg(long)]
        dry_run: bool,
    },

    /// Search across all imported records.
    Search {
        /// Free-text query.
        query: String,
        /// Restrict to one source.
        #[arg(long)]
        source: Option<String>,
        /// Result limit.
        #[arg(long, default_value_t = 20)]
        limit: u32,
    },

    /// Export records as JSONL or CSV.
    Export {
        /// Output path (defaults to stdout).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Format: jsonl | csv.
        #[arg(long, default_value = "jsonl")]
        format: String,
    },

    /// Run as an MCP server (stdio by default).
    Serve {
        /// Use SSE on the given port instead of stdio.
        #[arg(long)]
        sse: Option<u16>,
    },

    /// Verify database integrity and rebuild indexes.
    Verify {
        /// Attempt to repair issues found.
        #[arg(long)]
        repair: bool,
    },

    /// Run pending schema migrations.
    Migrate,
}

#[derive(Subcommand, Debug)]
enum SourceCmd {
    /// Register a new source.
    Add {
        /// Adapter name (e.g. `claude-code`, `mem0`).
        adapter: String,
        /// Instance discriminator.
        #[arg(long)]
        instance: Option<String>,
        /// Filesystem path, if applicable.
        #[arg(long)]
        path: Option<PathBuf>,
    },
    /// List configured sources and their health.
    List,
    /// Remove a registered source.
    Remove {
        /// `adapter` or `adapter:instance`.
        target: String,
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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level);
    let data_dir = resolve_data_dir(cli.data_dir)?;

    match cli.command {
        Command::Init => cmd_init(&data_dir),
        Command::Status => cmd_status(&data_dir),
        Command::Source(_)
        | Command::Import { .. }
        | Command::Search { .. }
        | Command::Export { .. }
        | Command::Serve { .. }
        | Command::Verify { .. }
        | Command::Migrate => {
            eprintln!("not yet implemented in Phase 0 — coming in Phase 1+");
            std::process::exit(2);
        }
    }
}

fn cmd_init(data_dir: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(data_dir)
        .with_context(|| format!("create data dir {}", data_dir.display()))?;
    let db_path = data_dir.join("anamnesis.sqlite");
    let _store = anamnesis_store::Store::open(&db_path)
        .with_context(|| format!("open {}", db_path.display()))?;
    println!("initialized at {}", data_dir.display());
    Ok(())
}

fn cmd_status(data_dir: &std::path::Path) -> Result<()> {
    let db_path = data_dir.join("anamnesis.sqlite");
    if !db_path.exists() {
        println!(
            "no database found at {} — run `anamnesis init`",
            db_path.display()
        );
        return Ok(());
    }
    let store = anamnesis_store::Store::open(&db_path)?;
    let count: i64 = store
        .conn()
        .query_row("SELECT COUNT(1) FROM records", [], |r| r.get(0))?;
    println!("data_dir : {}", data_dir.display());
    println!("records  : {count}");
    println!("schema   : v{}", anamnesis_core::SCHEMA_VERSION);
    Ok(())
}
