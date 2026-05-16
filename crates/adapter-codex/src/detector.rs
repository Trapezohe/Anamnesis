//! Detector for `~/.codex/`.

use std::path::PathBuf;

use anamnesis_core::discovery::{Confidence, DetectOpts, DetectedSource, SourceDetector};
use anamnesis_core::error::Result;
use async_trait::async_trait;

use crate::scanner::scan_root;

/// Codex detector.
pub struct CodexDetector {
    /// Optional override; production uses `$HOME/.codex/`.
    pub override_root: Option<PathBuf>,
}

impl CodexDetector {
    /// Production constructor.
    pub fn new() -> Self {
        Self {
            override_root: None,
        }
    }

    /// Test constructor.
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self {
            override_root: Some(root.into()),
        }
    }
}

impl Default for CodexDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SourceDetector for CodexDetector {
    fn adapter_id(&self) -> &'static str {
        crate::ADAPTER_ID
    }

    async fn detect(&self, opts: &DetectOpts) -> Result<Vec<DetectedSource>> {
        let root = self.resolve_root(opts);
        if !root.exists() {
            return Ok(Vec::new());
        }
        let files = match scan_root(&root) {
            Ok(f) => f,
            Err(e) => {
                return Err(anamnesis_core::Error::Adapter {
                    adapter: crate::ADAPTER_ID.into(),
                    message: format!("scan {}: {e}", root.display()),
                })
            }
        };
        let n = files.len() as u64;
        let (conf, note) = if n == 0 {
            (
                Confidence::Medium,
                "codex/ exists but no session files".into(),
            )
        } else {
            (Confidence::High, format!("{n} session file(s) found"))
        };
        Ok(vec![DetectedSource {
            adapter: crate::ADAPTER_ID.into(),
            instance: Some("default".into()),
            location: root.display().to_string(),
            local_path: Some(root),
            confidence: conf,
            estimated_records: Some(n),
            note: Some(note),
        }])
    }
}

impl CodexDetector {
    fn resolve_root(&self, opts: &DetectOpts) -> PathBuf {
        if let Some(p) = &self.override_root {
            return p.clone();
        }
        let home = opts
            .home_override
            .clone()
            .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("/"));
        home.join(".codex")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    static NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn tmp_dir() -> PathBuf {
        let n = NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let pid = std::process::id();
        let p = std::env::temp_dir().join(format!("anamnesis-codex-det-{pid}-{n}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[tokio::test]
    async fn returns_empty_when_root_missing() {
        let d = CodexDetector::with_root("/definitely/not/a/path");
        let found = d.detect(&DetectOpts::default()).await.unwrap();
        assert!(found.is_empty());
    }

    #[tokio::test]
    async fn medium_when_root_empty() {
        let dir = tmp_dir();
        let d = CodexDetector::with_root(&dir);
        let found = d.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found[0].confidence, Confidence::Medium);
        assert_eq!(found[0].estimated_records, Some(0));
    }

    #[tokio::test]
    async fn high_when_session_files_present() {
        let dir = tmp_dir();
        fs::create_dir_all(dir.join("sessions")).unwrap();
        fs::write(dir.join("sessions").join("a.jsonl"), "{}").unwrap();
        fs::write(dir.join("sessions").join("b.json"), "{}").unwrap();
        let d = CodexDetector::with_root(&dir);
        let found = d.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found[0].confidence, Confidence::High);
        assert_eq!(found[0].estimated_records, Some(2));
    }

    #[tokio::test]
    async fn respects_home_override() {
        let home = tmp_dir();
        std::fs::create_dir_all(home.join(".codex/sessions")).unwrap();
        std::fs::write(home.join(".codex/sessions").join("z.jsonl"), "{}").unwrap();
        let d = CodexDetector::new();
        let opts = DetectOpts {
            home_override: Some(home),
            ..Default::default()
        };
        let found = d.detect(&opts).await.unwrap();
        assert_eq!(found[0].estimated_records, Some(1));
    }
}
