//! Detect a Memary install.
//!
//! Memary has no fixed home-dir convention (the streamlit example runs
//! out of `streamlit_app/data/`). We probe two paths:
//!
//!   1. `~/.memary/`
//!   2. `~/.memary/data/`
//!
//! Anything else needs an explicit `anamnesis source add memary --path <dir>`.

use std::path::PathBuf;

use anamnesis_core::discovery::{Confidence, DetectOpts, DetectedSource, SourceDetector};
use anamnesis_core::error::Result;
use async_trait::async_trait;

/// `SourceDetector` for Memary installs.
#[derive(Debug, Default)]
pub struct MemaryDetector {
    home_override: Option<PathBuf>,
}

impl MemaryDetector {
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
impl SourceDetector for MemaryDetector {
    fn adapter_id(&self) -> &'static str {
        crate::ADAPTER_ID
    }

    async fn detect(&self, _opts: &DetectOpts) -> Result<Vec<DetectedSource>> {
        let Some(home) = self.home() else {
            return Ok(vec![]);
        };
        let candidates = [home.join(".memary/data"), home.join(".memary")];
        for candidate in candidates {
            if !candidate.is_dir() {
                continue;
            }
            let scan = crate::scanner::scan_memary(&candidate);
            if scan.total() == 0 && scan.parse_errors.is_empty() {
                continue;
            }
            let confidence = if scan.total() > 0 {
                Confidence::High
            } else {
                Confidence::Low
            };
            let mut note = format!(
                "Memary local cache (stream={}, tallies={}, chat={}, personas={})",
                scan.stream_entries.len(),
                scan.entity_tallies.len(),
                scan.chat_messages.len(),
                scan.personas.len(),
            );
            if !scan.parse_errors.is_empty() {
                note.push_str(&format!(" — {} parse error(s)", scan.parse_errors.len()));
            }
            return Ok(vec![DetectedSource {
                adapter: crate::ADAPTER_ID.into(),
                instance: None,
                location: candidate.display().to_string(),
                local_path: Some(candidate),
                confidence,
                estimated_records: Some(scan.total() as u64),
                note: Some(note),
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

    static MEMARY_DET_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp_dir() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = MEMARY_DET_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "anamnesis-memary-det-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn seed(dir: &std::path::Path) {
        fs::write(
            dir.join("memory_stream.json"),
            r#"[{"entity":"Alice","date":"2026-05-01T10:00:00"}]"#,
        )
        .unwrap();
    }

    #[tokio::test]
    async fn empty_home_yields_no_detection() {
        let home = tmp_dir();
        let det = MemaryDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert!(found.is_empty());
    }

    #[tokio::test]
    async fn detects_data_subdir_high_confidence() {
        let home = tmp_dir();
        let data = home.join(".memary/data");
        fs::create_dir_all(&data).unwrap();
        seed(&data);
        let det = MemaryDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].confidence, Confidence::High);
    }

    #[tokio::test]
    async fn detects_dotdir_root_when_data_absent() {
        let home = tmp_dir();
        let root = home.join(".memary");
        fs::create_dir_all(&root).unwrap();
        seed(&root);
        let det = MemaryDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].confidence, Confidence::High);
    }
}
