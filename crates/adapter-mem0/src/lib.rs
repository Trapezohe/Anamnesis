//! Anamnesis adapter for mem0.
//!
//! Phase 2: SQLite mode (read self-hosted mem0 db.sqlite directly).
//! API mode (`mode = "api"`) lands later.
//!
//! Mapping (BLUEPRINT §6.9):
//!   - `memory` column      → AnamnesisRecord.content
//!   - `id` column          → provenance.native_id
//!   - `user_id`/agent/run  → metadata.mem0_*
//!   - mem0 `metadata` JSON → merged into metadata (best-effort)
//!   - default kind         → Kind::Fact (mem0 has no kind taxonomy)
//!   - default scope        → Scope::User
//!
//! mem0's source vectors (if present) are NOT carried into the embedding
//! field; per BLUEPRINT §6.6.1 source vectors stay out of retrieval, and
//! the importer's raw_artifacts persistence keeps them only as
//! provenance metadata.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod detector;
pub mod normalizer;
pub mod scanner;

use std::path::PathBuf;
use std::sync::Arc;

use anamnesis_core::adapter::{HealthStatus, MemoryAdapter, RawRecord, ScanOpts};
use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::{AnamnesisRecord, SourceDescriptor};
use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};

pub use detector::Mem0SqliteDetector;
pub use scanner::Mem0Row;

/// Stable adapter identifier — referenced from many places.
pub const ADAPTER_ID: &str = "mem0";

/// Adapter configuration. Phase 2 supports the SQLite variant only.
#[derive(Debug, Clone)]
pub enum Mem0Config {
    /// Read mem0's self-hosted SQLite store directly.
    Sqlite {
        /// Path to the SQLite file.
        path: PathBuf,
        /// Instance discriminator (defaults to `"self-hosted"`).
        instance: Option<String>,
    },
    /// Reserved for Phase 2.x — cloud REST API. Not yet wired.
    Api {
        /// API base URL.
        base_url: String,
        /// Environment variable holding the API key.
        api_key_env: String,
        /// Instance discriminator.
        instance: Option<String>,
    },
}

impl Mem0Config {
    fn instance(&self) -> Option<&str> {
        match self {
            Self::Sqlite { instance, .. } | Self::Api { instance, .. } => instance.as_deref(),
        }
    }
}

/// The adapter.
pub struct Mem0Adapter {
    config: Arc<Mem0Config>,
}

impl Mem0Adapter {
    /// Build a new adapter.
    pub fn new(config: Mem0Config) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

#[async_trait]
impl MemoryAdapter for Mem0Adapter {
    fn descriptor(&self) -> SourceDescriptor {
        SourceDescriptor {
            adapter: ADAPTER_ID.into(),
            instance: self.config.instance().map(str::to_owned),
            version: env!("CARGO_PKG_VERSION").into(),
        }
    }

    fn scan<'a>(&'a self, opts: ScanOpts) -> BoxStream<'a, Result<RawRecord>> {
        // Round-20 (§-1.5 PR-4b): honor `opts.since` / `opts.full`.
        // mem0 SQLite has both `created_at` and `updated_at` (TEXT,
        // typically ISO 8601 or epoch seconds). The since-filter
        // compares against `updated_at.unwrap_or(created_at)` so a
        // record that mutated post-import isn't missed. Filtering is
        // row-level (Rust-side) because mem0 doesn't normalize the
        // timestamp column type — SQL `WHERE created_at >= ?` would be
        // unsafe across the ISO/epoch variants the scanner already
        // tolerates.
        let cfg = self.config.clone();
        let raws = collect_raw_records(&cfg, &opts);
        Box::pin(stream::iter(raws).map(Ok))
    }

    fn normalize(&self, raw: RawRecord) -> Result<Vec<AnamnesisRecord>> {
        normalizer::normalize(raw, self.config.instance())
    }

    async fn health(&self) -> HealthStatus {
        match self.config.as_ref() {
            Mem0Config::Sqlite { path, .. } => HealthStatus {
                ok: path.exists(),
                detail: format!("sqlite path: {}", path.display()),
            },
            Mem0Config::Api {
                base_url,
                api_key_env,
                ..
            } => HealthStatus {
                ok: std::env::var(api_key_env).is_ok(),
                detail: format!("api base: {base_url} (key env: {api_key_env}) — Phase 2.x"),
            },
        }
    }
}

fn collect_raw_records(cfg: &Mem0Config, opts: &ScanOpts) -> Vec<RawRecord> {
    match cfg {
        Mem0Config::Sqlite { path, instance } => match scanner::read_all(path) {
            Ok(rows) => rows
                .iter()
                .filter(|r| passes_since_filter(r, opts))
                .map(|r| normalizer::raw_from_row(r, instance.as_deref()))
                .collect(),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    path = %path.display(),
                    "mem0 sqlite read failed; emitting zero records"
                );
                Vec::new()
            }
        },
        Mem0Config::Api { .. } => {
            tracing::warn!("mem0 api mode is Phase 2.x; emitting zero records");
            Vec::new()
        }
    }
}

/// Whether a mem0 row passes the `opts.since` window. `opts.full == true`
/// always returns true (= "ignore since"). `opts.since == None` always
/// returns true. Otherwise, the row passes when **either** of:
///
///   * `updated_at` (preferred — captures post-creation edits) is
///     parseable AND strictly greater than the threshold, OR
///   * `created_at` is parseable AND strictly greater than the threshold,
///
/// is true. Unparseable timestamps conservatively INCLUDE the row (the
/// importer's raw_hash fast-path makes a false positive a no-op upsert;
/// a false negative would silently drop user data — see §-1.6.4).
fn passes_since_filter(row: &scanner::Mem0Row, opts: &ScanOpts) -> bool {
    if opts.full {
        return true;
    }
    let Some(threshold) = opts.since else {
        return true;
    };
    let updated = row.updated_at.as_deref().and_then(parse_mem0_timestamp);
    let created = row.created_at.as_deref().and_then(parse_mem0_timestamp);
    match (updated, created) {
        (Some(u), _) => u > threshold,
        (None, Some(c)) => c > threshold,
        (None, None) => true, // unparseable → include
    }
}

/// Parse mem0's `created_at` / `updated_at` strings, which historically
/// could be either RFC3339 (`"2026-05-01T12:00:00Z"`) or epoch seconds
/// stringified (`"1714564800"`). Tries RFC3339 first, then i64-as-epoch.
/// Returns `None` if neither parse succeeds.
fn parse_mem0_timestamp(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&chrono::Utc));
    }
    if let Ok(epoch) = s.parse::<i64>() {
        return chrono::DateTime::<chrono::Utc>::from_timestamp(epoch, 0);
    }
    None
}

/// Convenience: build an adapter from a SQLite path.
pub fn sqlite_adapter(path: impl Into<PathBuf>, instance: Option<&str>) -> Mem0Adapter {
    Mem0Adapter::new(Mem0Config::Sqlite {
        path: path.into(),
        instance: instance.map(str::to_owned),
    })
}

/// Mem0 API mode is not wired yet; returns a clear error.
#[allow(dead_code)]
fn api_not_supported() -> Error {
    Error::Adapter {
        adapter: ADAPTER_ID.into(),
        message: "mem0 API mode is Phase 2.x; use sqlite mode for now".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anamnesis_core::Kind;
    use futures::StreamExt;
    use rusqlite::Connection;
    use std::fs;

    fn tmp_dir() -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("anamnesis-mem0-adapter-{nonce}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn seed_db(path: &std::path::Path) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            "CREATE TABLE memories(
                id TEXT PRIMARY KEY,
                memory TEXT NOT NULL,
                user_id TEXT,
                created_at TEXT
            );",
        )
        .unwrap();
        for (id, mem) in [
            ("a", "user prefers vim"),
            ("b", "never mock the database"),
            ("c", "deployments happen on fridays"),
        ] {
            conn.execute(
                "INSERT INTO memories(id, memory, user_id, created_at) VALUES(?1,?2,?3,?4)",
                rusqlite::params![id, mem, "u1", "1700000000"],
            )
            .unwrap();
        }
    }

    #[tokio::test]
    async fn descriptor_carries_instance() {
        let a = sqlite_adapter("/tmp/x", Some("self-hosted"));
        let d = a.descriptor();
        assert_eq!(d.adapter, ADAPTER_ID);
        assert_eq!(d.instance.as_deref(), Some("self-hosted"));
    }

    #[tokio::test]
    async fn scan_emits_one_raw_per_memory() {
        let dir = tmp_dir();
        let db = dir.join("db.sqlite");
        seed_db(&db);
        let a = sqlite_adapter(&db, Some("self-hosted"));
        let raws: Vec<_> = a
            .scan(ScanOpts::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(raws.len(), 3);
    }

    #[tokio::test]
    async fn scan_then_normalize_produces_fact_records() {
        let dir = tmp_dir();
        let db = dir.join("db.sqlite");
        seed_db(&db);
        let a = sqlite_adapter(&db, Some("self-hosted"));
        let raws: Vec<_> = a
            .scan(ScanOpts::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        let mut facts = 0;
        for raw in raws {
            for r in a.normalize(raw).unwrap() {
                if r.kind == Kind::Fact {
                    facts += 1;
                }
            }
        }
        assert_eq!(facts, 3);
    }

    #[tokio::test]
    async fn missing_db_yields_empty_stream() {
        let a = sqlite_adapter("/tmp/never-exists.sqlite", None);
        let n = a.scan(ScanOpts::default()).collect::<Vec<_>>().await.len();
        assert_eq!(n, 0);
    }

    #[tokio::test]
    async fn health_reports_path_existence() {
        let a = sqlite_adapter("/tmp/no-such-db", None);
        let h = a.health().await;
        assert!(!h.ok);
        assert!(h.detail.contains("sqlite path"));
    }

    #[tokio::test]
    async fn api_mode_reports_key_env_check() {
        let a = Mem0Adapter::new(Mem0Config::Api {
            base_url: "https://api.mem0.ai".into(),
            api_key_env: "ANAMNESIS_MEM0_FAKE_KEY".into(),
            instance: None,
        });
        let h = a.health().await;
        assert!(!h.ok);
        assert!(h.detail.contains("Phase 2.x"));
    }

    #[tokio::test]
    async fn idempotent_across_scans() {
        let dir = tmp_dir();
        let db = dir.join("db.sqlite");
        seed_db(&db);
        let a = sqlite_adapter(&db, Some("self-hosted"));
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
        let a_ids = run().await;
        let b_ids = run().await;
        assert_eq!(a_ids, b_ids);
    }

    /// Round-20 (§-1.5 PR-4b): build a fixture where rows have
    /// different `created_at` values (mix of epoch + ISO 8601 to exercise
    /// both parsers), then scan with `opts.since` and assert filtering.
    fn seed_db_mixed_timestamps(path: &std::path::Path) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            "CREATE TABLE memories(
                id TEXT PRIMARY KEY,
                memory TEXT NOT NULL,
                user_id TEXT,
                created_at TEXT,
                updated_at TEXT
            );",
        )
        .unwrap();
        // 1700000000 = 2023-11-14T22:13:20Z (well before cutoff)
        // 1750000000 = 2025-06-15T17:33:20Z (cutoff)
        // 1800000000 = 2027-01-15T08:00:00Z (well after cutoff)
        let rows = [
            ("old-epoch", "old via epoch", "1700000000", None),
            ("old-iso", "old via iso", "2024-01-01T00:00:00Z", None),
            ("new-epoch", "new via epoch", "1800000000", None),
            ("new-iso", "new via iso", "2026-12-01T00:00:00Z", None),
            // updated_at supersedes created_at: created_at OLD but
            // updated_at NEW → should be included.
            (
                "updated-old-to-new",
                "edited recently",
                "1700000000",
                Some("2026-11-01T00:00:00Z"),
            ),
        ];
        for (id, mem, ca, ua) in rows {
            conn.execute(
                "INSERT INTO memories(id, memory, user_id, created_at, updated_at) \
                 VALUES(?1,?2,?3,?4,?5)",
                rusqlite::params![id, mem, "u1", ca, ua],
            )
            .unwrap();
        }
    }

    #[tokio::test]
    async fn scan_since_filters_rows_by_created_or_updated_at() {
        let dir = tmp_dir();
        let db = dir.join("db.sqlite");
        seed_db_mixed_timestamps(&db);
        let a = sqlite_adapter(&db, None);
        let cutoff = chrono::DateTime::<chrono::Utc>::from_timestamp(1_750_000_000, 0).unwrap();
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
        // Should keep: new-epoch, new-iso, updated-old-to-new (3).
        // Drop: old-epoch, old-iso (2).
        assert_eq!(
            raws.len(),
            3,
            "expected 3 post-cutoff rows; got {} — native_ids: {:?}",
            raws.len(),
            raws.iter().map(|r| &r.native_id).collect::<Vec<_>>()
        );
        // native_id format: `"{instance}|{id}"` where instance defaults
        // to `"self-hosted"` when none is provided (see normalizer).
        let ids: Vec<&str> = raws.iter().map(|r| r.native_id.as_str()).collect();
        assert!(ids.contains(&"self-hosted|new-epoch"), "ids={ids:?}");
        assert!(ids.contains(&"self-hosted|new-iso"), "ids={ids:?}");
        assert!(
            ids.contains(&"self-hosted|updated-old-to-new"),
            "ids={ids:?}"
        );
        assert!(!ids.contains(&"self-hosted|old-epoch"));
        assert!(!ids.contains(&"self-hosted|old-iso"));
    }

    #[tokio::test]
    async fn scan_full_overrides_since_filter() {
        let dir = tmp_dir();
        let db = dir.join("db.sqlite");
        seed_db_mixed_timestamps(&db);
        let a = sqlite_adapter(&db, None);
        let cutoff = chrono::DateTime::<chrono::Utc>::from_timestamp(1_750_000_000, 0).unwrap();
        let raws: Vec<_> = a
            .scan(ScanOpts {
                since: Some(cutoff),
                full: true, // override → all 5
            })
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(raws.len(), 5);
    }

    #[test]
    fn parse_mem0_timestamp_handles_both_formats() {
        // RFC 3339 round-trip.
        let rfc = super::parse_mem0_timestamp("2026-05-01T12:00:00Z").unwrap();
        assert_eq!(rfc.to_rfc3339(), "2026-05-01T12:00:00+00:00");
        // Epoch seconds round-trip.
        let epoch = super::parse_mem0_timestamp("1714564800").unwrap();
        assert_eq!(epoch.timestamp(), 1714564800);
        // Garbage stays None.
        assert!(super::parse_mem0_timestamp("not-a-time").is_none());
        assert!(super::parse_mem0_timestamp("").is_none());
    }
}
