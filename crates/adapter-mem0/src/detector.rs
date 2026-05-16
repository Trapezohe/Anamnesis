//! Detector for self-hosted mem0 SQLite databases.

use std::path::PathBuf;

use anamnesis_core::discovery::{Confidence, DetectOpts, DetectedSource, SourceDetector};
use anamnesis_core::error::Result;
use async_trait::async_trait;
use rusqlite::Connection;

/// Detector for `~/.mem0/db.sqlite` (or equivalent self-hosted layout).
pub struct Mem0SqliteDetector {
    /// Explicit override path — production resolves `$HOME/.mem0/db.sqlite`.
    pub override_path: Option<PathBuf>,
}

impl Mem0SqliteDetector {
    /// Production constructor; resolves the default path at detect time.
    pub fn new() -> Self {
        Self {
            override_path: None,
        }
    }

    /// Test constructor — point at an explicit SQLite file.
    pub fn with_path(path: impl Into<PathBuf>) -> Self {
        Self {
            override_path: Some(path.into()),
        }
    }
}

impl Default for Mem0SqliteDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SourceDetector for Mem0SqliteDetector {
    fn adapter_id(&self) -> &'static str {
        crate::ADAPTER_ID
    }

    async fn detect(&self, opts: &DetectOpts) -> Result<Vec<DetectedSource>> {
        let path = self.resolve_path(opts);
        if !path.exists() {
            return Ok(Vec::new());
        }
        match probe_memories_table(&path) {
            Ok(Some(rows)) => Ok(vec![DetectedSource {
                adapter: crate::ADAPTER_ID.into(),
                instance: Some("self-hosted".into()),
                location: path.display().to_string(),
                local_path: Some(path),
                confidence: Confidence::High,
                estimated_records: Some(rows),
                note: Some(format!("memories table present, ~{rows} row(s)")),
            }]),
            Ok(None) => Ok(vec![DetectedSource {
                adapter: crate::ADAPTER_ID.into(),
                instance: Some("self-hosted".into()),
                location: path.display().to_string(),
                local_path: Some(path),
                confidence: Confidence::Low,
                estimated_records: Some(0),
                note: Some("SQLite file found but no `memories` table".into()),
            }]),
            Err(e) => Err(anamnesis_core::Error::Adapter {
                adapter: crate::ADAPTER_ID.into(),
                message: format!("probe {}: {e}", path.display()),
            }),
        }
    }
}

impl Mem0SqliteDetector {
    fn resolve_path(&self, opts: &DetectOpts) -> PathBuf {
        if let Some(p) = &self.override_path {
            return p.clone();
        }
        let home = opts
            .home_override
            .clone()
            .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("/"));
        home.join(".mem0").join("db.sqlite")
    }
}

/// Returns `Ok(Some(count))` if `memories` table exists, `Ok(None)` if not,
/// `Err` on SQLite open failure.
fn probe_memories_table(path: &std::path::Path) -> rusqlite::Result<Option<u64>> {
    let conn = Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    )?;
    let exists: i64 = conn.query_row(
        "SELECT COUNT(1) FROM sqlite_master WHERE type='table' AND name='memories'",
        [],
        |r| r.get(0),
    )?;
    if exists == 0 {
        return Ok(None);
    }
    let count: i64 = conn
        .query_row("SELECT COUNT(1) FROM memories", [], |r| r.get(0))
        .unwrap_or(0);
    Ok(Some(count as u64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use std::fs;

    fn tmp_dir() -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("anamnesis-mem0-det-{nonce}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn write_memories_db(path: &std::path::Path, rows: usize) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            "CREATE TABLE memories (id TEXT PRIMARY KEY, memory TEXT NOT NULL, user_id TEXT, created_at INTEGER);",
        )
        .unwrap();
        for i in 0..rows {
            conn.execute(
                "INSERT INTO memories(id, memory, user_id, created_at) VALUES(?1, ?2, ?3, ?4)",
                rusqlite::params![format!("m{i}"), format!("memory #{i}"), "u1", 0],
            )
            .unwrap();
        }
    }

    #[tokio::test]
    async fn returns_empty_when_path_missing() {
        let d = Mem0SqliteDetector::with_path("/nonexistent/path/db.sqlite");
        let found = d.detect(&DetectOpts::default()).await.unwrap();
        assert!(found.is_empty());
    }

    #[tokio::test]
    async fn high_confidence_when_memories_table_exists() {
        let dir = tmp_dir();
        let db = dir.join("db.sqlite");
        write_memories_db(&db, 3);
        let d = Mem0SqliteDetector::with_path(&db);
        let found = d.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].confidence, Confidence::High);
        assert_eq!(found[0].estimated_records, Some(3));
        assert_eq!(found[0].adapter, "mem0");
    }

    #[tokio::test]
    async fn low_confidence_when_sqlite_lacks_memories_table() {
        let dir = tmp_dir();
        let db = dir.join("db.sqlite");
        let conn = Connection::open(&db).unwrap();
        conn.execute_batch("CREATE TABLE other(x INTEGER);")
            .unwrap();
        let d = Mem0SqliteDetector::with_path(&db);
        let found = d.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found[0].confidence, Confidence::Low);
        assert_eq!(found[0].estimated_records, Some(0));
    }

    #[tokio::test]
    async fn respects_home_override() {
        let home = tmp_dir();
        let mem0_dir = home.join(".mem0");
        std::fs::create_dir_all(&mem0_dir).unwrap();
        write_memories_db(&mem0_dir.join("db.sqlite"), 5);
        let d = Mem0SqliteDetector::new();
        let opts = DetectOpts {
            home_override: Some(home),
            ..Default::default()
        };
        let found = d.detect(&opts).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].estimated_records, Some(5));
    }

    #[tokio::test]
    async fn adapter_id_is_stable() {
        assert_eq!(Mem0SqliteDetector::new().adapter_id(), "mem0");
    }
}
