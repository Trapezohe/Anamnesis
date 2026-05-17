//! Anamnesis adapter for **Letta** (formerly MemGPT).
//!
//! Letta is a stateful-agent framework with first-class long-term
//! memory. In self-hosted mode it persists to SQLite at
//! `~/.letta/letta.db`. This adapter reads that store **read-only**
//! and pulls every row from the `block` table (Letta's "core memory"
//! — short, durable, always-in-context chunks like `persona` /
//! `human` / user-defined labeled blocks) as one `AnamnesisRecord`
//! per row with `Kind::Fact` and `Scope::User`.
//!
//! ## What this PR (§-2.3 P0) covers
//!
//! - **`block` table only.** Letta's `archival_passages` (long-term
//!   memory with embeddings) and `messages` (conversation log) are
//!   deliberately out of scope until we have a real Letta install to
//!   validate their schemas against.
//! - **SQLite mode only.** Letta's Postgres production mode is a
//!   follow-up adapter (`adapter-letta-pg`) when there's a real user
//!   need.
//! - **Schema-tolerant.** `PRAGMA table_info(block)` introspection
//!   means new columns Letta adds in future migrations get captured
//!   into `letta_extra` rather than dropped.
//!
//! ## Mapping (per §-2.5 checklist)
//!
//! - `block.id` → `provenance.native_id` (instance-prefixed `{instance}|{id}`).
//! - `block.value` → `AnamnesisRecord.content`.
//! - `block.label` → `metadata.letta_label`.
//! - `block.description` → `metadata.letta_description`.
//! - `block.template_name` → `metadata.letta_template`.
//! - `block.metadata_` (JSON) → `metadata.letta_metadata`, opaque
//!   parsed (raw fallback when not JSON).
//! - `block.created_at` / `block.updated_at` → `record.created_at` /
//!   `record.updated_at`. RFC3339 or epoch-seconds-as-string.
//! - Unknown columns → `metadata.letta_extra` (per §-2.5 step 7).
//!
//! Per §-1.2.2, this adapter never writes back to Letta — open is
//! `SQLITE_OPEN_READ_ONLY`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod detector;
pub mod normalizer;
pub mod scanner;

use std::path::PathBuf;
use std::sync::Arc;

use anamnesis_core::adapter::{HealthStatus, MemoryAdapter, RawRecord, ScanOpts};
use anamnesis_core::error::Result;
use anamnesis_core::model::{AnamnesisRecord, SourceDescriptor};
use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};

pub use detector::LettaSqliteDetector;
pub use scanner::LettaBlockRow;

/// Stable adapter identifier.
pub const ADAPTER_ID: &str = "letta";

/// Adapter configuration. SQLite mode only — Postgres lands later.
#[derive(Debug, Clone)]
pub struct LettaConfig {
    /// Path to the Letta `letta.db` SQLite file. Default in detector:
    /// `~/.letta/letta.db`.
    pub path: PathBuf,
    /// Instance discriminator (e.g. the machine name). Defaults to
    /// `"self-hosted"` in `synth_native_id` when `None`.
    pub instance: Option<String>,
}

/// The adapter.
pub struct LettaAdapter {
    config: Arc<LettaConfig>,
}

impl LettaAdapter {
    /// Build a new adapter from explicit config.
    pub fn new(config: LettaConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

#[async_trait]
impl MemoryAdapter for LettaAdapter {
    fn descriptor(&self) -> SourceDescriptor {
        SourceDescriptor {
            adapter: ADAPTER_ID.into(),
            instance: self.config.instance.clone(),
            version: env!("CARGO_PKG_VERSION").into(),
        }
    }

    fn scan<'a>(&'a self, opts: ScanOpts) -> BoxStream<'a, Result<RawRecord>> {
        // Round-26 (§-2.3 PR-Letta-1): honor `opts.since` / `opts.full`
        // up-front. Letta `block` rows have `updated_at` / `created_at`
        // strings; we filter row-by-row after the read (the table is
        // O(blocks_per_user) — usually a handful — so a Rust-side
        // filter is fine; a SQL pushdown is overkill).
        let cfg = (*self.config).clone();
        let raws = collect_raws(&cfg, &opts);
        Box::pin(stream::iter(raws).map(Ok))
    }

    fn normalize(&self, raw: RawRecord) -> Result<Vec<AnamnesisRecord>> {
        normalizer::normalize(raw, self.config.instance.as_deref())
    }

    async fn health(&self) -> HealthStatus {
        HealthStatus {
            ok: self.config.path.exists(),
            detail: if self.config.path.exists() {
                format!("letta sqlite at {}", self.config.path.display())
            } else {
                format!("letta sqlite not found: {}", self.config.path.display())
            },
        }
    }
}

fn collect_raws(cfg: &LettaConfig, opts: &ScanOpts) -> Vec<RawRecord> {
    let rows = match scanner::read_all_blocks(&cfg.path) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(
                error = %e,
                path = %cfg.path.display(),
                "letta sqlite read failed; emitting zero records"
            );
            return Vec::new();
        }
    };
    rows.into_iter()
        .filter(|row| passes_since_filter(row, opts))
        .map(|row| normalizer::raw_from_block_at_path(&row, &cfg.path, cfg.instance.as_deref()))
        .collect()
}

/// Mirrors mem0's `passes_since_filter`: `opts.full` short-circuits to
/// "include all", `opts.since == None` likewise. Otherwise compare
/// `updated_at` (preferred) then `created_at`; if both unparseable,
/// conservatively include (false negatives would drop user data;
/// the importer's raw_hash fast-path makes false positives free).
fn passes_since_filter(row: &LettaBlockRow, opts: &ScanOpts) -> bool {
    if opts.full {
        return true;
    }
    let Some(threshold) = opts.since else {
        return true;
    };
    let parse = |s: &Option<String>| s.as_deref().and_then(parse_letta_ts);
    match (parse(&row.updated_at), parse(&row.created_at)) {
        (Some(u), _) => u > threshold,
        (None, Some(c)) => c > threshold,
        (None, None) => true,
    }
}

fn parse_letta_ts(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&chrono::Utc));
    }
    if let Ok(epoch) = s.parse::<i64>() {
        return chrono::DateTime::<chrono::Utc>::from_timestamp(epoch, 0);
    }
    None
}

/// Convenience constructor mirroring `mem0::sqlite_adapter`.
pub fn letta_adapter(path: impl Into<PathBuf>, instance: Option<&str>) -> LettaAdapter {
    LettaAdapter::new(LettaConfig {
        path: path.into(),
        instance: instance.map(str::to_owned),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anamnesis_core::Kind;
    use rusqlite::Connection;
    use std::fs;

    fn tmp_dir() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("anamnesis-letta-{n}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn seed(path: &std::path::Path) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            r#"CREATE TABLE block (
                id TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                label TEXT,
                description TEXT,
                metadata_ TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            INSERT INTO block VALUES
              ('p', 'I am Sam.',   'persona', 'self-view', NULL, '2024-01-01T00:00:00Z', NULL),
              ('h', 'User likes Rust.', 'human', 'user model', NULL, '2026-04-01T00:00:00Z', '2026-04-15T00:00:00Z'),
              ('c', 'Custom block.', 'note', NULL, '{"v":1}', '2025-06-01T00:00:00Z', NULL);
            "#,
        )
        .unwrap();
    }

    #[tokio::test]
    async fn descriptor_carries_instance() {
        let a = letta_adapter("/tmp/x", Some("laptop"));
        let d = a.descriptor();
        assert_eq!(d.adapter, "letta");
        assert_eq!(d.instance.as_deref(), Some("laptop"));
    }

    #[tokio::test]
    async fn health_reports_path_existence() {
        let a = letta_adapter("/tmp/no-such", None);
        let h = a.health().await;
        assert!(!h.ok);
        assert!(h.detail.contains("not found"));
    }

    #[tokio::test]
    async fn missing_db_yields_empty_stream() {
        let a = letta_adapter("/tmp/never-exists.db", None);
        let n = a.scan(ScanOpts::default()).collect::<Vec<_>>().await.len();
        assert_eq!(n, 0);
    }

    #[tokio::test]
    async fn scan_then_normalize_produces_fact_records() {
        let dir = tmp_dir();
        let db = dir.join("letta.db");
        seed(&db);
        let a = letta_adapter(&db, Some("local"));
        let raws: Vec<_> = a
            .scan(ScanOpts::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(raws.len(), 3);
        let mut facts = 0;
        for raw in raws {
            for r in a.normalize(raw).unwrap() {
                if r.kind == Kind::Fact && r.source.adapter == "letta" {
                    facts += 1;
                }
            }
        }
        assert_eq!(facts, 3);
    }

    /// `since` filter — preferring `updated_at` when present.
    /// Fixture has 3 rows:
    ///   p (created 2024-01, no updated) — DROP if since = 2025
    ///   h (created 2026-04, updated 2026-04-15) — KEEP
    ///   c (created 2025-06, no updated) — KEEP if since = 2025
    #[tokio::test]
    async fn scan_since_filters_by_updated_or_created() {
        let dir = tmp_dir();
        let db = dir.join("letta.db");
        seed(&db);
        let a = letta_adapter(&db, Some("local"));
        let cutoff = chrono::DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let raws: Vec<_> = a
            .scan(ScanOpts {
                since: Some(cutoff),
                full: false,
            })
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(raws.len(), 2, "p (2024) should be dropped");
        let ids: Vec<&str> = raws.iter().map(|r| r.native_id.as_str()).collect();
        assert!(ids.iter().any(|i| i.ends_with("|h")));
        assert!(ids.iter().any(|i| i.ends_with("|c")));
    }

    #[tokio::test]
    async fn scan_full_overrides_since() {
        let dir = tmp_dir();
        let db = dir.join("letta.db");
        seed(&db);
        let a = letta_adapter(&db, Some("local"));
        let cutoff = chrono::DateTime::parse_from_rfc3339("2030-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let raws: Vec<_> = a
            .scan(ScanOpts {
                since: Some(cutoff),
                full: true,
            })
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(raws.len(), 3);
    }

    /// Re-scanning the same DB produces identical record ids (the
    /// `Adapter contract test` invariant for idempotence).
    #[tokio::test]
    async fn idempotent_across_scans() {
        let dir = tmp_dir();
        let db = dir.join("letta.db");
        seed(&db);
        let a = letta_adapter(&db, Some("local"));
        let run = || async {
            let mut ids: Vec<_> = a
                .scan(ScanOpts::default())
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .filter_map(|r| r.ok())
                .flat_map(|raw| a.normalize(raw).unwrap())
                .map(|r| r.id.0)
                .collect();
            ids.sort();
            ids
        };
        assert_eq!(run().await, run().await);
    }
}
