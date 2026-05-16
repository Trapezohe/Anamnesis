//! Detector for Claude Code memory installations.

use std::path::PathBuf;

use anamnesis_core::discovery::{Confidence, DetectOpts, DetectedSource, SourceDetector};
use anamnesis_core::error::Result;
use async_trait::async_trait;

use crate::scanner::{count_records, scan_projects_root};

/// Detector for `~/.claude/projects/`.
pub struct ClaudeCodeDetector {
    /// Optional override path (set by tests; production uses `$HOME`).
    pub override_root: Option<PathBuf>,
}

impl ClaudeCodeDetector {
    /// Production constructor — resolves `$HOME/.claude/projects` at detect time.
    pub fn new() -> Self {
        Self {
            override_root: None,
        }
    }

    /// Test constructor — point at an explicit projects root.
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self {
            override_root: Some(root.into()),
        }
    }
}

impl Default for ClaudeCodeDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SourceDetector for ClaudeCodeDetector {
    fn adapter_id(&self) -> &'static str {
        crate::ADAPTER_ID
    }

    async fn detect(&self, opts: &DetectOpts) -> Result<Vec<DetectedSource>> {
        let root = self.resolve_root(opts);
        if !root.exists() {
            return Ok(Vec::new());
        }
        // Scan returns Ok(empty) when root vanishes between the check and
        // the open — treat that as "nothing to import" rather than an error.
        let scans = match scan_projects_root(&root) {
            Ok(s) => s,
            Err(e) => {
                return Err(anamnesis_core::Error::Adapter {
                    adapter: crate::ADAPTER_ID.into(),
                    message: format!("scan {}: {e}", root.display()),
                });
            }
        };
        if scans.is_empty() {
            // Directory exists but no project subdirs — medium confidence so the
            // CLI shows it but doesn't auto-select.
            return Ok(vec![DetectedSource {
                adapter: crate::ADAPTER_ID.into(),
                instance: Some("default".into()),
                location: root.display().to_string(),
                local_path: Some(root),
                confidence: Confidence::Medium,
                estimated_records: Some(0),
                note: Some("projects/ exists but is empty".into()),
            }]);
        }
        let (mem, jsonl) = count_records(&scans);
        let note = format!(
            "{} project(s), {mem} memory file(s), {jsonl} session file(s)",
            scans.len(),
        );
        Ok(vec![DetectedSource {
            adapter: crate::ADAPTER_ID.into(),
            instance: Some("default".into()),
            location: root.display().to_string(),
            local_path: Some(root),
            confidence: Confidence::High,
            estimated_records: Some(mem + jsonl),
            note: Some(note),
        }])
    }
}

impl ClaudeCodeDetector {
    fn resolve_root(&self, opts: &DetectOpts) -> PathBuf {
        if let Some(p) = &self.override_root {
            return p.clone();
        }
        let home = opts
            .home_override
            .clone()
            .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("/"));
        home.join(".claude").join("projects")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_dir() -> std::path::PathBuf {
        let base = std::env::temp_dir();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = base.join(format!("anamnesis-detector-{nonce}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[tokio::test]
    async fn returns_empty_when_root_missing() {
        let d = ClaudeCodeDetector::with_root("/definitely/not/a/path");
        let found = d.detect(&DetectOpts::default()).await.unwrap();
        assert!(found.is_empty());
    }

    #[tokio::test]
    async fn medium_confidence_when_root_exists_but_no_projects() {
        let root = tmp_dir();
        let d = ClaudeCodeDetector::with_root(&root);
        let found = d.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].confidence, Confidence::Medium);
        assert_eq!(found[0].estimated_records, Some(0));
    }

    #[tokio::test]
    async fn high_confidence_with_realistic_layout() {
        let root = tmp_dir();
        let proj = root.join("project-hash");
        fs::create_dir_all(&proj).unwrap();
        fs::write(proj.join("session-1.jsonl"), "{}").unwrap();
        fs::write(proj.join("session-2.jsonl"), "{}").unwrap();
        fs::create_dir_all(proj.join("memory")).unwrap();
        fs::write(
            proj.join("memory").join("user_role.md"),
            "---\nname: x\n---\n",
        )
        .unwrap();
        fs::write(proj.join("memory").join("MEMORY.md"), "index").unwrap();

        let d = ClaudeCodeDetector::with_root(&root);
        let found = d.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        let s = &found[0];
        assert_eq!(s.confidence, Confidence::High);
        // 2 jsonl + 1 memory (MEMORY.md excluded) = 3
        assert_eq!(s.estimated_records, Some(3));
        assert!(s.note.as_deref().unwrap().contains("1 project"));
    }

    #[tokio::test]
    async fn respects_home_override_when_no_explicit_root() {
        let root_home = tmp_dir();
        std::fs::create_dir_all(root_home.join(".claude").join("projects")).unwrap();
        let d = ClaudeCodeDetector::new();
        let opts = DetectOpts {
            home_override: Some(root_home.clone()),
            ..Default::default()
        };
        let found = d.detect(&opts).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(
            found[0].local_path.as_deref().unwrap(),
            root_home.join(".claude").join("projects"),
        );
    }

    #[tokio::test]
    async fn adapter_id_is_stable() {
        let d = ClaudeCodeDetector::new();
        assert_eq!(d.adapter_id(), "claude-code");
    }
}
