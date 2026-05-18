//! Detect a TencentDB Agent Memory (TDAI) install at
//! `~/.openclaw/memory-tdai/`.

use std::path::PathBuf;

use anamnesis_core::discovery::{Confidence, DetectOpts, DetectedSource, SourceDetector};
use anamnesis_core::error::Result;
use async_trait::async_trait;

/// `SourceDetector` impl for TDAI installations.
#[derive(Debug, Default)]
pub struct TdaiDetector {
    home_override: Option<PathBuf>,
}

impl TdaiDetector {
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
impl SourceDetector for TdaiDetector {
    fn adapter_id(&self) -> &'static str {
        crate::ADAPTER_ID
    }

    async fn detect(&self, _opts: &DetectOpts) -> Result<Vec<DetectedSource>> {
        let Some(home) = self.home() else {
            return Ok(vec![]);
        };
        let dir = home.join(".openclaw").join("memory-tdai");
        if !dir.is_dir() {
            return Ok(vec![]);
        }
        let scan = crate::scanner::scan_tdai(&dir);
        let total = scan.total();
        let confidence = if total > 0 {
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
            estimated_records: Some(total as u64),
            note: Some("TencentDB Agent Memory (4-tier under ~/.openclaw/memory-tdai/)".into()),
        }])
    }
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
        let p = std::env::temp_dir().join(format!("anamnesis-tdai-det-{n}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[tokio::test]
    async fn empty_home_yields_no_detection() {
        let home = tmp_dir();
        let det = TdaiDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert!(found.is_empty());
    }

    #[tokio::test]
    async fn detects_high_confidence_with_data() {
        let home = tmp_dir();
        let dir = home.join(".openclaw/memory-tdai");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("persona.md"), "persona").unwrap();
        let det = TdaiDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].confidence, Confidence::High);
        assert_eq!(found[0].estimated_records, Some(1));
    }

    #[tokio::test]
    async fn detects_low_confidence_when_empty() {
        let home = tmp_dir();
        let dir = home.join(".openclaw/memory-tdai");
        fs::create_dir_all(&dir).unwrap();
        let det = TdaiDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found[0].confidence, Confidence::Low);
    }
}
