//! Detect a Memori install.
//!
//! Memori doesn't have a fixed default-path convention (the user passes a
//! SQLAlchemy `conn` factory at construction). The most common paths we
//! probe in priority order:
//!
//!   1. `~/.memori/memori.db`
//!   2. `~/.memori.db`
//!
//! Anything outside these — including `./memori.db` in the user's cwd —
//! requires explicit `anamnesis source add memori --path <file>`.

use std::path::PathBuf;

use anamnesis_core::discovery::{Confidence, DetectOpts, DetectedSource, SourceDetector};
use anamnesis_core::error::Result;
use async_trait::async_trait;

/// `SourceDetector` for Memori SQLite installs.
#[derive(Debug, Default)]
pub struct MemoriDetector {
    home_override: Option<PathBuf>,
}

impl MemoriDetector {
    /// New detector reading `$HOME`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the home root — tests use this.
    pub fn with_home(mut self, home: PathBuf) -> Self {
        self.home_override = Some(home);
        self
    }

    fn home(&self) -> Option<PathBuf> {
        self.home_override
            .clone()
            .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
            .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
    }
}

#[async_trait]
impl SourceDetector for MemoriDetector {
    fn adapter_id(&self) -> &'static str {
        crate::ADAPTER_ID
    }

    async fn detect(&self, _opts: &DetectOpts) -> Result<Vec<DetectedSource>> {
        let Some(home) = self.home() else {
            return Ok(vec![]);
        };
        let candidates = [home.join(".memori/memori.db"), home.join(".memori.db")];
        for candidate in candidates {
            if !candidate.is_file() {
                continue;
            }
            let scan = crate::scanner::scan_memori(&candidate);
            let total = scan.total();
            // If the file exists but isn't a Memori schema, skip silently —
            // it's somebody else's SQLite that just happens to share the name.
            if scan.schema_error.is_some() && total == 0 {
                continue;
            }
            let confidence = if total > 0 {
                Confidence::High
            } else {
                Confidence::Low
            };
            return Ok(vec![DetectedSource {
                adapter: crate::ADAPTER_ID.into(),
                instance: None,
                location: candidate.display().to_string(),
                local_path: Some(candidate),
                confidence,
                estimated_records: Some(total as u64),
                note: Some(
                    "Memori SQLite (entity_facts + process_attrs + messages + summaries + KG)"
                        .into(),
                ),
            }]);
        }
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::{params, Connection};
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static MEMORI_DET_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp_dir() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = MEMORI_DET_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "anamnesis-memori-det-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn seed_memori_db(db_path: &std::path::Path) {
        let conn = Connection::open(db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE memori_entity (id INTEGER PRIMARY KEY, uuid TEXT, external_id TEXT);
             CREATE TABLE memori_entity_fact (
                 id INTEGER PRIMARY KEY, uuid TEXT, entity_id INTEGER,
                 content TEXT, num_times INTEGER, date_last_time TEXT, date_created TEXT
             );
             CREATE TABLE memori_conversation (id INTEGER PRIMARY KEY, uuid TEXT, session_id INTEGER, summary TEXT, date_created TEXT);",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memori_entity (id, uuid, external_id) VALUES (?, ?, ?)",
            params![1, "ent-uuid", "user-123"],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memori_entity_fact (uuid, entity_id, content, num_times, date_last_time, date_created) \
             VALUES (?, ?, ?, ?, ?, ?)",
            params!["fact-1", 1, "user lives in Paris", 1, "2026-05-01 10:00:00", "2026-04-01 10:00:00"],
        )
        .unwrap();
    }

    #[tokio::test]
    async fn empty_home_yields_no_detection() {
        let home = tmp_dir();
        let det = MemoriDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert!(found.is_empty());
    }

    #[tokio::test]
    async fn detects_canonical_dotdir_high_confidence() {
        let home = tmp_dir();
        fs::create_dir_all(home.join(".memori")).unwrap();
        seed_memori_db(&home.join(".memori/memori.db"));
        let det = MemoriDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].confidence, Confidence::High);
        assert!(found[0].estimated_records.unwrap_or(0) >= 1);
    }

    #[tokio::test]
    async fn detects_dotfile_at_home_root() {
        let home = tmp_dir();
        seed_memori_db(&home.join(".memori.db"));
        let det = MemoriDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].confidence, Confidence::High);
    }

    #[tokio::test]
    async fn unrelated_sqlite_with_same_name_is_skipped() {
        let home = tmp_dir();
        // A SQLite file at the canonical path but not actually Memori.
        let conn = Connection::open(home.join(".memori.db")).unwrap();
        conn.execute_batch("CREATE TABLE not_memori (x INTEGER);")
            .unwrap();
        drop(conn);
        let det = MemoriDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert!(found.is_empty());
    }
}
