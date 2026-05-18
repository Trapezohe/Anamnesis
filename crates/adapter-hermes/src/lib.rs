//! Anamnesis adapter for **Hermes Agent** (Nous Research, MIT).
//!
//! Hermes keeps cross-session context in three places under its data
//! directory (default `~/.hermes/` on unix-likes,
//! `%LOCALAPPDATA%\hermes` on Windows):
//!
//!   * `MEMORY.md` — environment info / past lessons → `Kind::Reference`
//!   * `USER.md`   — user preferences / work style    → `Kind::Preference`
//!   * SQLite session DB(s) — conversation log         → `Kind::Episode`
//!
//! This adapter reads all three read-only (§-1.2.2). The SQLite path
//! introspects every user table via `PRAGMA table_info` and pulls any
//! table that exposes a content-shaped column (`content`, `message`,
//! `text`, `body`, …) — Hermes' session schema hasn't been frozen and
//! v0.7+ supports six pluggable memory backends, so adapter-side
//! schema-tolerance is load-bearing.
//!
//! ## Mapping
//!
//! - `MEMORY.md` → 1 `Reference` record, `Scope::User`.
//! - `USER.md`   → 1 `Preference` record, `Scope::User`.
//! - Each session row → 1 `Episode` record, `Scope::Session`,
//!   `metadata.hermes_role` + `metadata.hermes_table` preserved.
//! - Unknown SQLite columns → `metadata.hermes_extra`.

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

pub use detector::HermesDetector;
pub use scanner::{HermesMarkdownBlock, HermesScan, HermesSessionRow};

/// Stable adapter identifier.
pub const ADAPTER_ID: &str = "hermes";

/// Configuration for the Hermes adapter.
#[derive(Debug, Clone)]
pub struct HermesConfig {
    /// Hermes data directory (default in detector: `~/.hermes/`).
    pub data_dir: PathBuf,
    /// Instance discriminator (e.g. machine name). Defaults to
    /// `"default"` in the id-synthesis helpers when `None`.
    pub instance: Option<String>,
}

/// The adapter.
pub struct HermesAdapter {
    config: Arc<HermesConfig>,
}

impl HermesAdapter {
    /// Build a new adapter from explicit config.
    pub fn new(config: HermesConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

#[async_trait]
impl MemoryAdapter for HermesAdapter {
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
        HealthStatus {
            ok: self.config.data_dir.is_dir(),
            detail: if self.config.data_dir.is_dir() {
                format!("hermes data dir: {}", self.config.data_dir.display())
            } else {
                format!(
                    "hermes data dir not found: {}",
                    self.config.data_dir.display()
                )
            },
        }
    }
}

fn collect_raws(cfg: &HermesConfig, opts: &ScanOpts) -> Vec<RawRecord> {
    let scan = scanner::scan_hermes_dir(&cfg.data_dir);
    let mut out = Vec::with_capacity(scan.total());

    for block in &scan.markdown {
        if passes_since_for_markdown(block, opts) {
            out.push(normalizer::raw_from_markdown(
                block,
                cfg.instance.as_deref(),
            ));
        }
    }
    for (db_path, row) in &scan.session_rows {
        if passes_since_for_session(row, opts) {
            out.push(normalizer::raw_from_session(
                db_path,
                row,
                cfg.instance.as_deref(),
            ));
        }
    }
    out
}

fn passes_since_for_markdown(block: &HermesMarkdownBlock, opts: &ScanOpts) -> bool {
    if opts.full {
        return true;
    }
    let Some(threshold) = opts.since else {
        return true;
    };
    match block.mtime_unix {
        Some(t) => t > threshold.timestamp(),
        None => true, // unparseable / unsupported → conservatively include
    }
}

fn passes_since_for_session(row: &HermesSessionRow, opts: &ScanOpts) -> bool {
    if opts.full {
        return true;
    }
    let Some(threshold) = opts.since else {
        return true;
    };
    match row.timestamp {
        Some(t) => t > threshold.timestamp(),
        None => true,
    }
}

/// Convenience constructor.
pub fn hermes_adapter(data_dir: impl Into<PathBuf>, instance: Option<&str>) -> HermesAdapter {
    HermesAdapter::new(HermesConfig {
        data_dir: data_dir.into(),
        instance: instance.map(str::to_owned),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anamnesis_core::Kind;
    use rusqlite::Connection;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static HERMES_LIB_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp_dir() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = HERMES_LIB_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "anamnesis-hermes-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn seed(dir: &std::path::Path) {
        fs::write(dir.join("MEMORY.md"), "system on macOS").unwrap();
        fs::write(dir.join("USER.md"), "prefers Rust").unwrap();
        let conn = Connection::open(dir.join("sessions.db")).unwrap();
        conn.execute_batch(
            r#"CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                role TEXT,
                content TEXT NOT NULL,
                created_at TEXT
            );
            INSERT INTO messages VALUES
              ('m1', 'user',      'hi',        '2024-01-01T00:00:00Z'),
              ('m2', 'assistant', 'hello back','2026-04-01T00:00:00Z');"#,
        )
        .unwrap();
    }

    #[tokio::test]
    async fn descriptor_carries_instance() {
        let a = hermes_adapter("/tmp/x", Some("laptop"));
        let d = a.descriptor();
        assert_eq!(d.adapter, "hermes");
        assert_eq!(d.instance.as_deref(), Some("laptop"));
    }

    #[tokio::test]
    async fn health_false_when_dir_missing() {
        let a = hermes_adapter("/tmp/never-here-hermes", None);
        let h = a.health().await;
        assert!(!h.ok);
    }

    #[tokio::test]
    async fn scan_yields_all_three_record_kinds() {
        let dir = tmp_dir();
        seed(&dir);
        let a = hermes_adapter(&dir, Some("laptop"));
        let raws: Vec<_> = a
            .scan(ScanOpts::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        // 2 md + 2 session = 4 raws.
        assert_eq!(raws.len(), 4);

        let mut kinds = std::collections::HashSet::new();
        for raw in raws {
            for r in a.normalize(raw).unwrap() {
                kinds.insert(r.kind);
            }
        }
        assert!(kinds.contains(&Kind::Reference));
        assert!(kinds.contains(&Kind::Preference));
        assert!(kinds.contains(&Kind::Episode));
    }

    #[tokio::test]
    async fn scan_since_filters_session_by_timestamp() {
        let dir = tmp_dir();
        seed(&dir);
        let a = hermes_adapter(&dir, Some("laptop"));
        // Cutoff = 2025-01-01. m1 (2024) drops, m2 (2026) stays.
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
        // Markdown blocks have no mtime in the test fixture (just-written
        // files in tmpdir have mtime ~ now > cutoff, so they survive).
        // Session row m1 drops, m2 stays.
        let session_payloads: Vec<_> = raws
            .iter()
            .filter(|r| {
                r.payload.get("payload_kind").and_then(|v| v.as_str())
                    == Some(normalizer::PAYLOAD_KIND_SESSION)
            })
            .collect();
        assert_eq!(session_payloads.len(), 1);
    }

    #[tokio::test]
    async fn idempotent_across_scans() {
        let dir = tmp_dir();
        seed(&dir);
        let a = hermes_adapter(&dir, Some("laptop"));
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

    #[tokio::test]
    async fn empty_dir_yields_zero_records_without_erroring() {
        let dir = tmp_dir();
        // dir exists but has nothing in it
        let a = hermes_adapter(&dir, None);
        let raws: Vec<_> = a
            .scan(ScanOpts::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(raws.len(), 0);
    }
}
