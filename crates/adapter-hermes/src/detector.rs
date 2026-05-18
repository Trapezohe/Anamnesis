//! Detect a local Hermes Agent install.
//!
//! Hermes (Nous Research, MIT) places its data under `~/.hermes/` on
//! unix-likes (and `%LOCALAPPDATA%\hermes` on Windows — out of scope
//! for the initial detector). The presence of either `MEMORY.md`,
//! `USER.md`, or any `.db`/`.sqlite` file in that directory is enough
//! to count as a positive detection.

use std::path::{Path, PathBuf};

use anamnesis_core::discovery::{Confidence, DetectOpts, DetectedSource, SourceDetector};
use anamnesis_core::error::Result;
use async_trait::async_trait;

/// `SourceDetector` impl for Hermes Agent installations.
#[derive(Debug, Default)]
pub struct HermesDetector {
    home_override: Option<PathBuf>,
}

impl HermesDetector {
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
impl SourceDetector for HermesDetector {
    fn adapter_id(&self) -> &'static str {
        crate::ADAPTER_ID
    }

    async fn detect(&self, _opts: &DetectOpts) -> Result<Vec<DetectedSource>> {
        let Some(home) = self.home() else {
            return Ok(vec![]);
        };
        let dir = home.join(".hermes");
        if !dir.is_dir() {
            return Ok(vec![]);
        }
        let confidence = if has_canonical_files(&dir) {
            Confidence::High
        } else {
            Confidence::Low
        };
        Ok(vec![DetectedSource {
            adapter: crate::ADAPTER_ID.into(),
            instance: None,
            location: dir.display().to_string(),
            local_path: Some(dir),
            confidence,
            estimated_records: None,
            note: Some("Hermes Agent data dir".into()),
        }])
    }
}

/// Whether the directory looks like a real Hermes install. We look
/// for the canonical markdown files OR any sqlite-shaped file; either
/// is enough to upgrade confidence from Low → High.
fn has_canonical_files(dir: &Path) -> bool {
    if dir.join("MEMORY.md").is_file() || dir.join("USER.md").is_file() {
        return true;
    }
    let Ok(read) = std::fs::read_dir(dir) else {
        return false;
    };
    for entry in read.flatten() {
        let p = entry.path();
        if !p.is_file() {
            continue;
        }
        let Some(ext) = p.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if matches!(ext.to_lowercase().as_str(), "db" | "sqlite" | "sqlite3") {
            return true;
        }
    }
    false
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
        let p = std::env::temp_dir().join(format!("anamnesis-hermes-det-{n}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[tokio::test]
    async fn detector_returns_empty_when_no_hermes_dir() {
        let home = tmp_dir();
        let det = HermesDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert!(found.is_empty());
    }

    #[tokio::test]
    async fn detector_high_confidence_when_memory_md_present() {
        let home = tmp_dir();
        let hermes = home.join(".hermes");
        fs::create_dir_all(&hermes).unwrap();
        fs::write(hermes.join("MEMORY.md"), "system").unwrap();

        let det = HermesDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].adapter, "hermes");
        assert_eq!(found[0].confidence, Confidence::High);
    }

    #[tokio::test]
    async fn detector_high_confidence_for_db_only_install() {
        let home = tmp_dir();
        let hermes = home.join(".hermes");
        fs::create_dir_all(&hermes).unwrap();
        fs::write(hermes.join("sessions.db"), "").unwrap();
        let det = HermesDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found[0].confidence, Confidence::High);
    }

    #[tokio::test]
    async fn detector_low_confidence_when_dir_exists_but_empty() {
        let home = tmp_dir();
        let hermes = home.join(".hermes");
        fs::create_dir_all(&hermes).unwrap();
        let det = HermesDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].confidence, Confidence::Low);
    }
}
