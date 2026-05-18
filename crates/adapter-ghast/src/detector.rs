//! Detect a ghast install (source repo OR app-support dir).

use std::path::PathBuf;

use anamnesis_core::discovery::{Confidence, DetectOpts, DetectedSource, SourceDetector};
use anamnesis_core::error::Result;
use async_trait::async_trait;

/// `SourceDetector` impl for ghast.
#[derive(Debug, Default)]
pub struct GhastDetector {
    home_override: Option<PathBuf>,
}

impl GhastDetector {
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
impl SourceDetector for GhastDetector {
    fn adapter_id(&self) -> &'static str {
        crate::ADAPTER_ID
    }

    async fn detect(&self, _opts: &DetectOpts) -> Result<Vec<DetectedSource>> {
        let Some(home) = self.home() else {
            return Ok(vec![]);
        };
        let mut found = Vec::new();

        // App-support dir on macOS / Linux.
        let app_support = home
            .join("Library")
            .join("Application Support")
            .join("ghast");
        if app_support.is_dir() {
            // Always Low — we can detect the install but the user data
            // is encrypted; user has to combine with the source repo for
            // useful content. The adapter's health() spells this out.
            found.push(DetectedSource {
                adapter: crate::ADAPTER_ID.into(),
                instance: None,
                location: app_support.display().to_string(),
                local_path: Some(app_support),
                confidence: Confidence::Low,
                estimated_records: None,
                note: Some(
                    "ghast user data dir detected but ghast.db is encrypted; \
                     point this adapter at the ghast source repo instead for now"
                        .into(),
                ),
            });
        }

        // Common source-repo locations: `~/Documents/ghast_desktop`,
        // `~/Desktop/ghast_desktop`, `~/ghast_desktop`.
        let candidates = [
            home.join("Documents").join("ghast_desktop"),
            home.join("Desktop").join("ghast_desktop"),
            home.join("ghast_desktop"),
        ];
        for c in candidates {
            if !c.is_dir() {
                continue;
            }
            let canonical_repo =
                c.join("prompts").is_dir() || c.join("resources/bundled-skills").is_dir();
            if !canonical_repo {
                continue;
            }
            found.push(DetectedSource {
                adapter: crate::ADAPTER_ID.into(),
                instance: None,
                location: c.display().to_string(),
                local_path: Some(c),
                confidence: Confidence::High,
                estimated_records: None,
                note: Some("ghast source repo (prompts + bundled-skills)".into()),
            });
        }

        Ok(found)
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
        let p = std::env::temp_dir().join(format!("anamnesis-ghast-det-{n}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[tokio::test]
    async fn empty_home_yields_no_detections() {
        let home = tmp_dir();
        let det = GhastDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert!(found.is_empty());
    }

    #[tokio::test]
    async fn detects_source_repo_high_confidence() {
        let home = tmp_dir();
        let repo = home.join("Documents/ghast_desktop");
        fs::create_dir_all(repo.join("prompts/coding")).unwrap();
        fs::create_dir_all(repo.join("resources/bundled-skills/memory")).unwrap();
        let det = GhastDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].confidence, Confidence::High);
    }

    #[tokio::test]
    async fn detects_app_support_low_confidence() {
        let home = tmp_dir();
        fs::create_dir_all(home.join("Library/Application Support/ghast")).unwrap();
        let det = GhastDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].confidence, Confidence::Low);
        assert!(found[0].note.as_deref().unwrap_or("").contains("encrypted"));
    }

    #[tokio::test]
    async fn detects_both_when_both_exist() {
        let home = tmp_dir();
        fs::create_dir_all(home.join("Library/Application Support/ghast")).unwrap();
        let repo = home.join("Documents/ghast_desktop");
        fs::create_dir_all(repo.join("prompts/coding")).unwrap();
        let det = GhastDetector::new().with_home(home);
        let found = det.detect(&DetectOpts::default()).await.unwrap();
        assert_eq!(found.len(), 2);
    }
}
