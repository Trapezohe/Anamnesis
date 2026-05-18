//! Anamnesis adapter for **Memori** (MemoriLabs/Memori, Apache-2.0).
//!
//! Memori is "agent-native memory infrastructure" that turns LLM
//! conversations into structured durable state. It supports multiple
//! storage backends; this adapter targets the SQLite backend (most
//! common for local installs).
//!
//! Memori writes data into a fixed set of `memori_*` tables. Coverage:
//!
//! | Table                             | Anamnesis Kind | Scope    |
//! |-----------------------------------|----------------|----------|
//! | `memori_entity_fact`              | `Fact`         | User     |
//! | `memori_process_attribute`        | `Reference`    | Project  |
//! | `memori_conversation_message`     | `Episode`      | Session  |
//! | `memori_conversation` (`summary`) | `Episode`      | Session  |
//! | `memori_knowledge_graph` (triple) | `Fact`         | User     |
//!
//! Per §-1.2.2 the adapter is read-only — we open with
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

pub use detector::MemoriDetector;
pub use scanner::{
    MemoriConversationMessage, MemoriConversationSummary, MemoriEntityFact, MemoriKgTriple,
    MemoriProcessAttribute, MemoriScan,
};

/// Stable adapter identifier.
pub const ADAPTER_ID: &str = "memori";

/// Adapter configuration.
#[derive(Debug, Clone)]
pub struct MemoriConfig {
    /// Path to the Memori SQLite file.
    pub db_path: PathBuf,
    /// Instance discriminator (defaults to `"local"` in id synthesis).
    pub instance: Option<String>,
}

/// The adapter.
pub struct MemoriAdapter {
    config: Arc<MemoriConfig>,
}

impl MemoriAdapter {
    /// Build from explicit config.
    pub fn new(config: MemoriConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

#[async_trait]
impl MemoryAdapter for MemoriAdapter {
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
        if !self.config.db_path.is_file() {
            return HealthStatus {
                ok: false,
                detail: format!("memori db not found: {}", self.config.db_path.display()),
            };
        }
        let s = scanner::scan_memori(&self.config.db_path);
        let mut detail = format!(
            "memori db: {} (entity_facts={}, process_attrs={}, messages={}, summaries={}, kg={})",
            self.config.db_path.display(),
            s.entity_facts.len(),
            s.process_attrs.len(),
            s.messages.len(),
            s.summaries.len(),
            s.kg_triples.len(),
        );
        if let Some(err) = s.schema_error {
            detail.push_str(&format!(" — schema note: {err}"));
        }
        HealthStatus { ok: true, detail }
    }
}

fn collect_raws(cfg: &MemoriConfig, opts: &ScanOpts) -> Vec<RawRecord> {
    let scan = scanner::scan_memori(&cfg.db_path);
    let mut out = Vec::with_capacity(scan.total());
    for f in &scan.entity_facts {
        let ts = f
            .date_last_time
            .as_deref()
            .and_then(scanner::parse_memori_time)
            .or_else(|| {
                f.date_created
                    .as_deref()
                    .and_then(scanner::parse_memori_time)
            });
        if passes_since(ts, opts) {
            out.push(normalizer::raw_from_entity_fact(f, cfg.instance.as_deref()));
        }
    }
    for a in &scan.process_attrs {
        let ts = a
            .date_last_time
            .as_deref()
            .and_then(scanner::parse_memori_time)
            .or_else(|| {
                a.date_created
                    .as_deref()
                    .and_then(scanner::parse_memori_time)
            });
        if passes_since(ts, opts) {
            out.push(normalizer::raw_from_process_attr(
                a,
                cfg.instance.as_deref(),
            ));
        }
    }
    for m in &scan.messages {
        let ts = m
            .date_created
            .as_deref()
            .and_then(scanner::parse_memori_time);
        if passes_since(ts, opts) {
            out.push(normalizer::raw_from_message(m, cfg.instance.as_deref()));
        }
    }
    for s in &scan.summaries {
        let ts = s
            .date_created
            .as_deref()
            .and_then(scanner::parse_memori_time);
        if passes_since(ts, opts) {
            out.push(normalizer::raw_from_summary(s, cfg.instance.as_deref()));
        }
    }
    for t in &scan.kg_triples {
        let ts = t
            .date_last_time
            .as_deref()
            .and_then(scanner::parse_memori_time)
            .or_else(|| {
                t.date_created
                    .as_deref()
                    .and_then(scanner::parse_memori_time)
            });
        if passes_since(ts, opts) {
            out.push(normalizer::raw_from_kg_triple(t, cfg.instance.as_deref()));
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
pub fn memori_adapter(db_path: impl Into<PathBuf>, instance: Option<&str>) -> MemoriAdapter {
    MemoriAdapter::new(MemoriConfig {
        db_path: db_path.into(),
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

    static MEMORI_LIB_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp_db() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = MEMORI_LIB_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "anamnesis-memori-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir.join("memori.db")
    }

    fn seed(db: &std::path::Path) {
        let conn = Connection::open(db).unwrap();
        conn.execute_batch(
            "CREATE TABLE memori_entity (id INTEGER PRIMARY KEY, uuid TEXT, external_id TEXT);
             CREATE TABLE memori_process (id INTEGER PRIMARY KEY, uuid TEXT, external_id TEXT);
             CREATE TABLE memori_session (id INTEGER PRIMARY KEY, uuid TEXT, entity_id INTEGER, process_id INTEGER);
             CREATE TABLE memori_conversation (id INTEGER PRIMARY KEY, uuid TEXT, session_id INTEGER, summary TEXT, date_created TEXT);
             CREATE TABLE memori_conversation_message (
                 id INTEGER PRIMARY KEY, uuid TEXT, conversation_id INTEGER,
                 role TEXT, type TEXT, content TEXT, date_created TEXT
             );
             CREATE TABLE memori_entity_fact (
                 id INTEGER PRIMARY KEY, uuid TEXT, entity_id INTEGER,
                 content TEXT, num_times INTEGER, date_last_time TEXT, date_created TEXT
             );",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memori_entity (id, uuid, external_id) VALUES (?, ?, ?)",
            params![1, "ent-uuid", "user-123"],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memori_session (id, uuid, entity_id) VALUES (?, ?, ?)",
            params![100, "sess-uuid", 1],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memori_conversation (id, uuid, session_id, summary, date_created) \
             VALUES (?, ?, ?, ?, ?)",
            params![
                1000,
                "conv-uuid",
                100,
                "User asked about Paris.",
                "2026-05-01 10:00:00"
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memori_conversation_message \
             (uuid, conversation_id, role, content, date_created) \
             VALUES (?, ?, ?, ?, ?)",
            params![
                "msg-1",
                1000,
                "user",
                "I live in Paris",
                "2026-05-01 10:00:00"
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memori_entity_fact \
             (uuid, entity_id, content, num_times, date_last_time, date_created) \
             VALUES (?, ?, ?, ?, ?, ?)",
            params![
                "fact-1",
                1,
                "user lives in Paris",
                1,
                "2026-05-01 10:00:00",
                "2026-04-01 10:00:00",
            ],
        )
        .unwrap();
    }

    #[tokio::test]
    async fn descriptor_carries_instance() {
        let a = memori_adapter("/tmp/x", Some("laptop"));
        let d = a.descriptor();
        assert_eq!(d.adapter, "memori");
        assert_eq!(d.instance.as_deref(), Some("laptop"));
    }

    #[tokio::test]
    async fn health_false_when_db_missing() {
        let a = memori_adapter("/tmp/never-here-memori.db", None);
        let h = a.health().await;
        assert!(!h.ok);
    }

    #[tokio::test]
    async fn scan_yields_expected_kinds() {
        let db = tmp_db();
        seed(&db);
        let a = memori_adapter(&db, Some("local"));
        let raws: Vec<_> = a
            .scan(ScanOpts::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        // 1 fact + 1 message + 1 summary = 3.
        assert_eq!(raws.len(), 3);
        let mut kinds = std::collections::HashSet::new();
        for raw in raws {
            for r in a.normalize(raw).unwrap() {
                kinds.insert(r.kind);
            }
        }
        assert!(kinds.contains(&Kind::Fact));
        assert!(kinds.contains(&Kind::Episode));
    }

    #[tokio::test]
    async fn scan_full_overrides_since() {
        let db = tmp_db();
        seed(&db);
        let a = memori_adapter(&db, None);
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
        assert_eq!(full.len(), 3);
    }

    #[tokio::test]
    async fn idempotent_across_scans() {
        let db = tmp_db();
        seed(&db);
        let a = memori_adapter(&db, Some("laptop"));
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
