//! Detect an OpenViking install at `~/.openviking/`.
//!
//! Default layout (pip-install reference + Docker reference setup):
//!
//! ```text
//! ~/.openviking/
//! ├── ov.conf                # config
//! └── data/                  # workspace (`storage.workspace` in ov.conf)
//!     └── local/<account_id>/...
//! ```
//!
//! We probe (in priority order):
//!   1. `~/.openviking/data/`  (default workspace)
//!   2. `~/.openviking/`       (in case workspace was overridden to root)

use std::path::PathBuf;

use anamnesis_core::discovery::{Confidence, DetectOpts, DetectedSource, SourceDetector};
use anamnesis_core::error::Result;
use async_trait::async_trait;

/// `SourceDetector` impl for OpenViking installations.
#[derive(Debug, Default)]
pub struct OpenVikingDetector {
    home_override: Option<PathBuf>,
}

impl OpenVikingDetector {
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
impl SourceDetector for OpenVikingDetector {
    fn adapter_id(&self) -> &'static str {
        crate::ADAPTER_ID
    }

    async fn detect(&self, _opts: &DetectOpts) -> Result<Vec<DetectedSource>> {
        let Some(home) = self.home() else {
            return Ok(vec![]);
        };
        let root = home.join(".openviking");
        if !root.is_dir() {
            return Ok(vec![]);
        }
        let candidates = [root.join("data"), root.clone()];
        for candidate in candidates {
            if !candidate.is_dir() {
                continue;
            }
            let scan = crate::scanner::scan_openviking(&candidate);
            let total = scan.total();
            if total == 0 && !candidate.join("local").is_dir() && candidate != root.join("data") {
                // Probing `~/.openviking` root with neither data/ nor local/ is
                // a false positive — bail.
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
                note: Some("OpenViking VikingFS (resources/user/agent/session × L0/L1/L2)".into()),
            }]);
        }
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static OV_DET_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp_dir() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = OV_DET_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "anamnesis-ov-det-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[tokio::test]
    async fn empty_home_yields_no_detection() {
        let home = tmp_dir();
        let det = OpenVikingDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert!(found.is_empty());
    }

    #[tokio::test]
    async fn detects_data_workspace_high_confidence() {
        let home = tmp_dir();
        let acct = home.join(".openviking/data/local/acct/resources/x");
        fs::create_dir_all(&acct).unwrap();
        fs::write(acct.join("note.md"), "body").unwrap();
        let det = OpenVikingDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].confidence, Confidence::High);
        assert!(found[0].estimated_records.unwrap_or(0) > 0);
    }

    #[tokio::test]
    async fn detects_low_confidence_when_workspace_empty() {
        let home = tmp_dir();
        fs::create_dir_all(home.join(".openviking/data")).unwrap();
        let det = OpenVikingDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].confidence, Confidence::Low);
    }
}
