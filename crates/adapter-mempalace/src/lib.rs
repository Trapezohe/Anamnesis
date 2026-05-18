//! Anamnesis adapter for **MemPalace** (mempalace/mempalace, AGPLv3 upstream).
//!
//! MemPalace persists state under `~/.mempalace/`:
//!
//! ```text
//! ~/.mempalace/
//! ├── identity.txt              # L0 identity (plain text)         → Preference / User
//! ├── config.json               # config (not ingested)
//! └── palace/
//!     └── chroma.sqlite3        # ChromaDB persistent client
//!         ├── collection mempalace_drawers  # mined memory chunks  → Episode    / Project
//!         └── collection mempalace_closets  # searchable index     → Reference  / Project
//! ```
//!
//! Drawer documents live in `embedding_metadata` under
//! `key='chroma:document'`. User-supplied metadata (`wing`, `room`,
//! `source_file`, `filed_at`, …) lives there too under each key.
//!
//! **Registry sentinels** (drawers with `room = '_registry'` or
//! `ingest_mode = 'registry'`) are MemPalace bookkeeping rows — one per
//! source file — that we filter out at scan time.
//!
//! **Licensing**: MemPalace is AGPLv3 upstream. This adapter only **reads**
//! ChromaDB data files MemPalace writes and does **not** link to or execute
//! MemPalace or chromadb code (we read the SQLite directly), so it ships
//! under Anamnesis's Apache-2.0.
//!
//! Per §-1.2.2 the adapter is read-only.

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

pub use detector::MempalaceDetector;
pub use scanner::{MempalaceDrawer, MempalaceIdentity, MempalaceScan};

/// Stable adapter identifier.
pub const ADAPTER_ID: &str = "mempalace";

/// Adapter configuration.
#[derive(Debug, Clone)]
pub struct MempalaceConfig {
    /// MemPalace home dir (default: `~/.mempalace/`).
    pub home_dir: PathBuf,
    /// Instance discriminator (defaults to `"local"` in id synthesis).
    pub instance: Option<String>,
}

/// The adapter.
pub struct MempalaceAdapter {
    config: Arc<MempalaceConfig>,
}

impl MempalaceAdapter {
    /// Build from explicit config.
    pub fn new(config: MempalaceConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

#[async_trait]
impl MemoryAdapter for MempalaceAdapter {
    fn descriptor(&self) -> SourceDescriptor {
        SourceDescriptor {
            adapter: ADAPTER_ID.into(),
            instance: self.config.instance.clone(),
            version: env!("CARGO_PKG_VERSION").into(),
        }
    }

    fn scan<'a>(&'a self, opts: ScanOpts) -> BoxStream<'a, Result<RawRecord>> {
        let cfg = (*self.config).clone();
        let raws = collect_raws(&cfg, &opts);
        Box::pin(stream::iter(raws).map(Ok))
    }

    fn normalize(&self, raw: RawRecord) -> Result<Vec<AnamnesisRecord>> {
        normalizer::normalize(raw, self.config.instance.as_deref())
    }

    async fn health(&self) -> HealthStatus {
        if !self.config.home_dir.is_dir() {
            return HealthStatus {
                ok: false,
                detail: format!(
                    "mempalace home dir not found: {}",
                    self.config.home_dir.display()
                ),
            };
        }
        let s = scanner::scan_mempalace(&self.config.home_dir);
        let mut detail = format!(
            "mempalace home: {} (identities={}, drawers={})",
            self.config.home_dir.display(),
            s.identities.len(),
            s.drawers.len(),
        );
        if let Some(err) = s.chroma_error {
            detail.push_str(&format!(" — chroma read error: {err}"));
        }
        HealthStatus { ok: true, detail }
    }
}

fn collect_raws(cfg: &MempalaceConfig, opts: &ScanOpts) -> Vec<RawRecord> {
    let scan = scanner::scan_mempalace(&cfg.home_dir);
    let mut out = Vec::with_capacity(scan.total());
    for i in &scan.identities {
        if passes_since(i.mtime_unix, opts) {
            out.push(normalizer::raw_from_identity(i, cfg.instance.as_deref()));
        }
    }
    for d in &scan.drawers {
        // Drawers don't always have a Chroma `created_at`; if missing, treat
        // as "always passes" — the row-level since-filter is best-effort.
        let ts = d
            .created_unix
            .or_else(|| scanner::filed_at_unix(&d.metadata));
        if passes_since(ts, opts) {
            out.push(normalizer::raw_from_drawer(d, cfg.instance.as_deref()));
        }
    }
    out
}

fn passes_since(ts_unix: Option<i64>, opts: &ScanOpts) -> bool {
    if opts.full {
        return true;
    }
    let Some(threshold) = opts.since else {
        return true;
    };
    match ts_unix {
        Some(t) => t > threshold.timestamp(),
        None => true,
    }
}

/// Convenience constructor.
pub fn mempalace_adapter(home_dir: impl Into<PathBuf>, instance: Option<&str>) -> MempalaceAdapter {
    MempalaceAdapter::new(MempalaceConfig {
        home_dir: home_dir.into(),
        instance: instance.map(str::to_owned),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anamnesis_core::Kind;
    use rusqlite::{params, Connection};
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static MP_LIB_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp_dir() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = MP_LIB_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "anamnesis-mempalace-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn seed(home: &std::path::Path) {
        fs::write(home.join("identity.txt"), "I am Atlas, an AI for Alice.").unwrap();
        fs::create_dir_all(home.join("palace")).unwrap();
        let db = home.join("palace/chroma.sqlite3");
        let conn = Connection::open(&db).unwrap();
        conn.execute_batch(
            "CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT, dimension INTEGER);
             CREATE TABLE segments (id TEXT PRIMARY KEY, collection TEXT, scope TEXT);
             CREATE TABLE embeddings (
                 id INTEGER PRIMARY KEY,
                 segment_id TEXT,
                 embedding_id TEXT,
                 seq_id BLOB,
                 created_at INTEGER
             );
             CREATE TABLE embedding_metadata (
                 id INTEGER,
                 key TEXT,
                 string_value TEXT,
                 int_value INTEGER,
                 float_value REAL,
                 bool_value INTEGER
             );",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO collections (id, name, dimension) VALUES (?, ?, ?)",
            params!["coll-drawers", "mempalace_drawers", 384],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO segments (id, collection, scope) VALUES (?, ?, ?)",
            params!["seg-drawers", "coll-drawers", "METADATA"],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO embeddings (id, segment_id, embedding_id, created_at) \
             VALUES (?, ?, ?, ?)",
            params![
                1,
                "seg-drawers",
                "drawer_default_general_aaa",
                1_730_000_000_i64
            ],
        )
        .unwrap();
        for (k, sv) in [
            ("chroma:document", "user prefers dark mode"),
            ("wing", "default"),
            ("room", "general"),
            ("source_file", "/repo/CLAUDE.md"),
            ("filed_at", "2026-05-01T10:00:00Z"),
        ] {
            conn.execute(
                "INSERT INTO embedding_metadata (id, key, string_value) VALUES (?, ?, ?)",
                params![1, k, sv],
            )
            .unwrap();
        }
    }

    #[tokio::test]
    async fn descriptor_carries_instance() {
        let a = mempalace_adapter("/tmp/x", Some("laptop"));
        let d = a.descriptor();
        assert_eq!(d.adapter, "mempalace");
        assert_eq!(d.instance.as_deref(), Some("laptop"));
    }

    #[tokio::test]
    async fn health_false_when_home_missing() {
        let a = mempalace_adapter("/tmp/never-here-mp", None);
        let h = a.health().await;
        assert!(!h.ok);
    }

    #[tokio::test]
    async fn scan_yields_identity_and_drawer() {
        let dir = tmp_dir();
        seed(&dir);
        let a = mempalace_adapter(&dir, Some("local"));
        let raws: Vec<_> = a
            .scan(ScanOpts::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        // 1 identity + 1 drawer = 2.
        assert_eq!(raws.len(), 2);
        let mut kinds = std::collections::HashSet::new();
        for raw in raws {
            for r in a.normalize(raw).unwrap() {
                kinds.insert(r.kind);
            }
        }
        assert!(kinds.contains(&Kind::Preference));
        assert!(kinds.contains(&Kind::Episode));
    }

    #[tokio::test]
    async fn scan_full_overrides_since() {
        let dir = tmp_dir();
        seed(&dir);
        let a = mempalace_adapter(&dir, None);
        let cutoff = chrono::DateTime::parse_from_rfc3339("2099-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let dropped: Vec<_> = a
            .scan(ScanOpts {
                since: Some(cutoff),
                full: false,
            })
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(dropped.len(), 0);
        let full: Vec<_> = a
            .scan(ScanOpts {
                since: Some(cutoff),
                full: true,
            })
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(full.len(), 2);
    }

    #[tokio::test]
    async fn idempotent_across_scans() {
        let dir = tmp_dir();
        seed(&dir);
        let a = mempalace_adapter(&dir, Some("laptop"));
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
