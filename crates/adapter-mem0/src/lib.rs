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

    fn scan<'a>(&'a self, _opts: ScanOpts) -> BoxStream<'a, Result<RawRecord>> {
        let cfg = self.config.clone();
        let raws = collect_raw_records(&cfg);
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

fn collect_raw_records(cfg: &Mem0Config) -> Vec<RawRecord> {
    match cfg {
        Mem0Config::Sqlite { path, instance } => match scanner::read_all(path) {
            Ok(rows) => rows
                .iter()
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
}
