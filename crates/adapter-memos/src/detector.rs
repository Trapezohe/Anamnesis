//! Detect a MemOS install.
//!
//! MemOS uses `Path.cwd() / .memos` by default (unless `MEMOS_BASE_PATH` env
//! override). We probe two locations for autodetection:
//!
//!   1. `~/.memos/`
//!   2. `$MEMOS_BASE_PATH/.memos/` (if env set)
//!
//! Detection works by walking the candidate root looking for any
//! `textual_memory.json` file (the canonical MemOS dump artifact).

use std::path::PathBuf;

use anamnesis_core::discovery::{Confidence, DetectOpts, DetectedSource, SourceDetector};
use anamnesis_core::error::Result;
use async_trait::async_trait;

/// `SourceDetector` for MemOS installs.
#[derive(Debug, Default)]
pub struct MemosDetector {
    home_override: Option<PathBuf>,
    env_override: Option<PathBuf>,
}

impl MemosDetector {
    /// New detector reading `$HOME` and `$MEMOS_BASE_PATH`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the home root — tests use this.
    pub fn with_home(mut self, home: PathBuf) -> Self {
        self.home_override = Some(home);
        self
    }

    /// Override `$MEMOS_BASE_PATH` — tests use this to avoid touching the
    /// process env.
    pub fn with_memos_base(mut self, base: PathBuf) -> Self {
        self.env_override = Some(base);
        self
    }

    fn home(&self) -> Option<PathBuf> {
        self.home_override
            .clone()
            .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
            .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
    }

    fn memos_base(&self) -> Option<PathBuf> {
        self.env_override
            .clone()
            .or_else(|| std::env::var_os("MEMOS_BASE_PATH").map(PathBuf::from))
    }
}

#[async_trait]
impl SourceDetector for MemosDetector {
    fn adapter_id(&self) -> &'static str {
        crate::ADAPTER_ID
    }

    async fn detect(&self, _opts: &DetectOpts) -> Result<Vec<DetectedSource>> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Some(home) = self.home() {
            candidates.push(home.join(".memos"));
        }
        if let Some(base) = self.memos_base() {
            candidates.push(base.join(".memos"));
        }
        for candidate in candidates {
            if !candidate.is_dir() {
                continue;
            }
            let scan = crate::scanner::scan_memos(&candidate);
            // No MemCube dirs and no parse errors → almost certainly not a
            // MemOS root. Skip; let downstream candidates win.
            if scan.cube_dirs.is_empty() && scan.parse_errors.is_empty() {
                continue;
            }
            let total = scan.total();
            let confidence = if total > 0 {
                Confidence::High
            } else {
                Confidence::Low
            };
            let mut note = format!(
                "MemOS ({} MemCube{} discovered)",
                scan.cube_dirs.len(),
                if scan.cube_dirs.len() == 1 { "" } else { "s" }
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
                estimated_records: Some(total as u64),
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

    static MEMOS_DET_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp_dir() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = MEMOS_DET_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "anamnesis-memos-det-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn seed_cube(home: &std::path::Path, cube_name: &str) {
        let cube = home.join(".memos").join(cube_name);
        fs::create_dir_all(&cube).unwrap();
        let payload = serde_json::json!([
            {
                "id": format!("{cube_name}-item-1"),
                "memory": "user prefers Rust",
                "metadata": {"memory_type": "UserMemory", "status": "activated"}
            }
        ]);
        fs::write(cube.join("textual_memory.json"), payload.to_string()).unwrap();
    }

    #[tokio::test]
    async fn empty_home_yields_no_detection() {
        let home = tmp_dir();
        let det = MemosDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert!(found.is_empty());
    }

    #[tokio::test]
    async fn detects_canonical_dotdir_with_cube_high_confidence() {
        let home = tmp_dir();
        seed_cube(&home, "cube-1");
        let det = MemosDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].confidence, Confidence::High);
        assert!(found[0].estimated_records.unwrap_or(0) >= 1);
    }

    #[tokio::test]
    async fn detects_memos_base_path_override() {
        let base = tmp_dir();
        seed_cube(&base, "cube-x");
        // Empty home — only the env-based override should fire.
        let home_empty = tmp_dir();
        let det = MemosDetector::new()
            .with_home(home_empty)
            .with_memos_base(base);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].confidence, Confidence::High);
    }
}
