//! Detect a Letta self-hosted SQLite install.
//!
//! Looks for `~/.letta/letta.db` (Letta's default in dev mode) and
//! confirms it has a `block` table — that's enough to know an
//! Anamnesis user will get useful records from `import letta`.

use std::path::{Path, PathBuf};

use anamnesis_core::discovery::{Confidence, DetectOpts, DetectedSource, SourceDetector};
use anamnesis_core::error::Result;
use async_trait::async_trait;
use rusqlite::{Connection, OpenFlags};

/// `SourceDetector` impl for Letta SQLite installations.
#[derive(Debug, Default)]
pub struct LettaSqliteDetector {
    /// Optional override of the user's home root (used in tests so we
    /// don't accidentally read a real Letta install).
    home_override: Option<PathBuf>,
}

impl LettaSqliteDetector {
    /// New detector that reads `$HOME`.
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
impl SourceDetector for LettaSqliteDetector {
    fn adapter_id(&self) -> &'static str {
        crate::ADAPTER_ID
    }

    async fn detect(&self, _opts: &DetectOpts) -> Result<Vec<DetectedSource>> {
        let Some(home) = self.home() else {
            return Ok(vec![]);
        };
        let candidate = home.join(".letta").join("letta.db");
        if !candidate.exists() {
            return Ok(vec![]);
        }
        let confidence = match has_block_table(&candidate) {
            Ok(true) => Confidence::High,
            Ok(false) => Confidence::Low, // file exists but schema unfamiliar
            Err(_) => Confidence::Low,    // unreadable
        };
        Ok(vec![DetectedSource {
            adapter: crate::ADAPTER_ID.into(),
            instance: None,
            location: candidate.display().to_string(),
            local_path: Some(candidate),
            confidence,
            estimated_records: None,
            note: Some("Letta self-hosted SQLite store".into()),
        }])
    }
}

fn has_block_table(path: &Path) -> rusqlite::Result<bool> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    let n: i64 = conn.query_row(
        "SELECT COUNT(1) FROM sqlite_master WHERE type='table' AND name='block'",
        [],
        |r| r.get(0),
    )?;
    Ok(n > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_dir() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("anamnesis-letta-det-{n}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[tokio::test]
    async fn detector_returns_empty_when_no_letta_dir() {
        let home = tmp_dir();
        let det = LettaSqliteDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert!(found.is_empty());
    }

    #[tokio::test]
    async fn detector_reports_high_confidence_for_block_table() {
        let home = tmp_dir();
        fs::create_dir_all(home.join(".letta")).unwrap();
        let dbp = home.join(".letta/letta.db");
        let conn = Connection::open(&dbp).unwrap();
        conn.execute_batch("CREATE TABLE block (id TEXT PRIMARY KEY, value TEXT);")
            .unwrap();

        let det = LettaSqliteDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].adapter, "letta");
        assert_eq!(found[0].confidence, Confidence::High);
        // Path-separator-agnostic check so the test passes on
        // Windows (`\letta.db`) and unix-likes (`/letta.db`) alike.
        let p = found[0].local_path.as_ref().unwrap();
        assert_eq!(p.file_name().and_then(|s| s.to_str()), Some("letta.db"));
        assert_eq!(
            p.parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str()),
            Some(".letta")
        );
    }

    #[tokio::test]
    async fn detector_low_confidence_when_block_table_missing() {
        let home = tmp_dir();
        fs::create_dir_all(home.join(".letta")).unwrap();
        let dbp = home.join(".letta/letta.db");
        Connection::open(&dbp).unwrap(); // empty DB, no `block` table

        let det = LettaSqliteDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].confidence, Confidence::Low);
    }
}
