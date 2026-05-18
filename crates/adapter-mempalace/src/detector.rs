//! Detect a MemPalace install at `~/.mempalace/`.
//!
//! High confidence: `palace/chroma.sqlite3` exists.
//! Low confidence:  `~/.mempalace/` exists but the palace DB is absent
//!                  (user has run `mempalace init` but not yet mined).

use std::path::PathBuf;

use anamnesis_core::discovery::{Confidence, DetectOpts, DetectedSource, SourceDetector};
use anamnesis_core::error::Result;
use async_trait::async_trait;

/// `SourceDetector` impl for MemPalace installations.
#[derive(Debug, Default)]
pub struct MempalaceDetector {
    home_override: Option<PathBuf>,
}

impl MempalaceDetector {
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
impl SourceDetector for MempalaceDetector {
    fn adapter_id(&self) -> &'static str {
        crate::ADAPTER_ID
    }

    async fn detect(&self, _opts: &DetectOpts) -> Result<Vec<DetectedSource>> {
        let Some(home) = self.home() else {
            return Ok(vec![]);
        };
        let root = home.join(".mempalace");
        if !root.is_dir() {
            return Ok(vec![]);
        }

        let palace_db = root.join("palace").join("chroma.sqlite3");
        let identity = root.join("identity.txt");
        let has_palace = palace_db.is_file();
        let has_identity = identity.is_file();
        if !has_palace && !has_identity {
            return Ok(vec![]);
        }

        let scan = crate::scanner::scan_mempalace(&root);
        let total = scan.total();
        let confidence = if total > 0 {
            Confidence::High
        } else {
            Confidence::Low
        };
        let mut note = "MemPalace (identity.txt + ChromaDB-backed drawers)".to_string();
        if let Some(err) = scan.chroma_error {
            note.push_str(&format!(" — chroma read error: {err}"));
        }

        Ok(vec![DetectedSource {
            adapter: crate::ADAPTER_ID.into(),
            instance: None,
            location: root.display().to_string(),
            local_path: Some(root),
            confidence,
            estimated_records: Some(total as u64),
            note: Some(note),
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static MP_DET_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp_dir() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = MP_DET_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "anamnesis-mempalace-det-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[tokio::test]
    async fn empty_home_yields_no_detection() {
        let home = tmp_dir();
        let det = MempalaceDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert!(found.is_empty());
    }

    #[tokio::test]
    async fn mempalace_dir_with_identity_only_is_high() {
        let home = tmp_dir();
        fs::create_dir_all(home.join(".mempalace")).unwrap();
        fs::write(home.join(".mempalace/identity.txt"), "I am Atlas").unwrap();
        let det = MempalaceDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].confidence, Confidence::High);
        assert!(found[0].estimated_records.unwrap_or(0) >= 1);
    }

    #[tokio::test]
    async fn mempalace_dir_without_artifacts_returns_empty() {
        let home = tmp_dir();
        fs::create_dir_all(home.join(".mempalace")).unwrap();
        let det = MempalaceDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        // No identity.txt and no chroma.sqlite3 → not a real install.
        assert!(found.is_empty());
    }
}
