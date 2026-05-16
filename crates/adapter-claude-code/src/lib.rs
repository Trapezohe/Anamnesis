//! Anamnesis adapter for Claude Code.
//!
//! Data sources (planned, see `docs/BLUEPRINT.md §6.8`):
//!
//!   ~/.claude/projects/<hash>/*.jsonl          — conversation history
//!   ~/.claude/projects/<hash>/memory/MEMORY.md — index (not imported)
//!   ~/.claude/projects/<hash>/memory/*.md      — typed memory files
//!
//! Mapping (planned):
//!   - `memory/*.md` frontmatter `type` field → `Kind`
//!     * user      → Kind::Fact      (scope = User)
//!     * feedback  → Kind::Feedback
//!     * project   → Kind::Fact      (scope = Project)
//!     * reference → Kind::Reference
//!   - One JSONL session → one Episode record (summary + key turns)
//!
//! Phase 0: stub only. The real scanner + normalizer land in Phase 1.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::path::PathBuf;

use anamnesis_core::adapter::{HealthStatus, MemoryAdapter, RawRecord, ScanOpts};
use anamnesis_core::error::Result;
use anamnesis_core::model::{AnamnesisRecord, SourceDescriptor};
use async_trait::async_trait;
use futures::stream::{self, BoxStream};

/// Configuration for the Claude Code adapter.
#[derive(Debug, Clone)]
pub struct ClaudeCodeConfig {
    /// Root directory containing per-project subfolders.
    pub projects_root: PathBuf,
    /// Optional instance discriminator (e.g. `"default"`).
    pub instance: Option<String>,
}

/// The adapter.
pub struct ClaudeCodeAdapter {
    config: ClaudeCodeConfig,
}

impl ClaudeCodeAdapter {
    /// Build a new adapter from config.
    pub fn new(config: ClaudeCodeConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl MemoryAdapter for ClaudeCodeAdapter {
    fn descriptor(&self) -> SourceDescriptor {
        SourceDescriptor {
            adapter: "claude-code".into(),
            instance: self.config.instance.clone(),
            version: env!("CARGO_PKG_VERSION").into(),
        }
    }

    fn scan<'a>(&'a self, _opts: ScanOpts) -> BoxStream<'a, Result<RawRecord>> {
        // Phase 0: empty stream. Phase 1 walks `projects_root` and emits raw
        // records for each conversation file and each typed memory markdown.
        Box::pin(stream::empty())
    }

    fn normalize(&self, _raw: RawRecord) -> Result<Vec<AnamnesisRecord>> {
        // Phase 0: never called because `scan` is empty.
        Ok(Vec::new())
    }

    async fn health(&self) -> HealthStatus {
        let exists = self.config.projects_root.exists();
        HealthStatus {
            ok: exists,
            detail: if exists {
                format!("projects_root: {}", self.config.projects_root.display())
            } else {
                format!(
                    "projects_root not found: {}",
                    self.config.projects_root.display()
                )
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn descriptor_is_stable() {
        let a = ClaudeCodeAdapter::new(ClaudeCodeConfig {
            projects_root: "/tmp/nonexistent".into(),
            instance: Some("default".into()),
        });
        let d = a.descriptor();
        assert_eq!(d.adapter, "claude-code");
        assert_eq!(d.instance.as_deref(), Some("default"));
    }
}
