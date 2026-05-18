//! Detect an OpenClaw install at `~/.openclaw/`.

use std::path::PathBuf;

use anamnesis_core::discovery::{Confidence, DetectOpts, DetectedSource, SourceDetector};
use anamnesis_core::error::Result;
use async_trait::async_trait;

/// `SourceDetector` impl for OpenClaw installations.
#[derive(Debug, Default)]
pub struct OpenClawDetector {
    home_override: Option<PathBuf>,
}

impl OpenClawDetector {
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
impl SourceDetector for OpenClawDetector {
    fn adapter_id(&self) -> &'static str {
        crate::ADAPTER_ID
    }

    async fn detect(&self, _opts: &DetectOpts) -> Result<Vec<DetectedSource>> {
        let Some(home) = self.home() else {
            return Ok(vec![]);
        };
        let dir = home.join(".openclaw");
        if !dir.is_dir() {
            return Ok(vec![]);
        }
        // Confidence::High if we see ANY canonical OpenClaw artifact.
        let canonical = dir.join("openclaw.json").is_file()
            || dir.join("workspace/AGENTS.md").is_file()
            || dir.join("workspace/SOUL.md").is_file()
            || dir.join("workspace/TOOLS.md").is_file()
            || dir.join("workspace/skills").is_dir();
        let confidence = if canonical {
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
            note: Some("OpenClaw data dir".into()),
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static OPENCLAW_DET_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp_dir() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = OPENCLAW_DET_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "anamnesis-openclaw-det-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[tokio::test]
    async fn detector_returns_empty_when_no_openclaw_dir() {
        let home = tmp_dir();
        let det = OpenClawDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert!(found.is_empty());
    }

    #[tokio::test]
    async fn detector_high_confidence_when_openclaw_json_present() {
        let home = tmp_dir();
        let oc = home.join(".openclaw");
        fs::create_dir_all(&oc).unwrap();
        fs::write(oc.join("openclaw.json"), "{}").unwrap();
        let det = OpenClawDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].adapter, "openclaw");
        assert_eq!(found[0].confidence, Confidence::High);
    }

    #[tokio::test]
    async fn detector_high_confidence_for_skills_dir_only_install() {
        let home = tmp_dir();
        let oc = home.join(".openclaw");
        fs::create_dir_all(oc.join("workspace/skills")).unwrap();
        let det = OpenClawDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found[0].confidence, Confidence::High);
    }

    #[tokio::test]
    async fn detector_low_confidence_for_empty_dir() {
        let home = tmp_dir();
        let oc = home.join(".openclaw");
        fs::create_dir_all(&oc).unwrap();
        let det = OpenClawDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].confidence, Confidence::Low);
    }
}
