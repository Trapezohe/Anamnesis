//! Anamnesis adapter for Claude Code.
//!
//! Data sources (see `docs/BLUEPRINT.md §6.8`):
//!
//!   ~/.claude/projects/<hash>/*.jsonl          — conversation history
//!   ~/.claude/projects/<hash>/memory/MEMORY.md — index (NOT imported)
//!   ~/.claude/projects/<hash>/memory/*.md      — typed memory files
//!
//! Mapping rules:
//!   - `memory/*.md` frontmatter `type` → `Kind` / `Scope`
//!       * user      → Kind::Fact      / Scope::User
//!       * feedback  → Kind::Feedback  / Scope::User
//!       * project   → Kind::Fact      / Scope::Project
//!       * reference → Kind::Reference / Scope::User
//!   - Each JSONL session → one `Kind::Episode` record (Scope::Session).
//!
//! Module layout:
//!   detector    — `SourceDetector` impl (metadata-only discovery)
//!   scanner     — filesystem walker (no content reads)
//!   frontmatter — minimal YAML frontmatter parser
//!   normalizer  — `RawRecord` → `AnamnesisRecord`

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod detector;
pub mod frontmatter;
pub mod normalizer;
pub mod scanner;

use std::path::PathBuf;
use std::sync::Arc;

use anamnesis_core::adapter::{HealthStatus, MemoryAdapter, RawRecord, ScanOpts};
use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::{AnamnesisRecord, SourceDescriptor};
use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};

pub use detector::ClaudeCodeDetector;

/// Stable adapter identifier — referenced from many places.
pub const ADAPTER_ID: &str = "claude-code";

/// Configuration for the Claude Code adapter.
#[derive(Debug, Clone)]
pub struct ClaudeCodeConfig {
    /// Root directory containing per-project subfolders.
    pub projects_root: PathBuf,
    /// Optional instance discriminator.
    pub instance: Option<String>,
}

/// The adapter.
pub struct ClaudeCodeAdapter {
    config: Arc<ClaudeCodeConfig>,
}

impl ClaudeCodeAdapter {
    /// Build a new adapter from config.
    pub fn new(config: ClaudeCodeConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

#[async_trait]
impl MemoryAdapter for ClaudeCodeAdapter {
    fn descriptor(&self) -> SourceDescriptor {
        SourceDescriptor {
            adapter: ADAPTER_ID.into(),
            instance: self.config.instance.clone(),
            version: env!("CARGO_PKG_VERSION").into(),
        }
    }

    fn scan<'a>(&'a self, _opts: ScanOpts) -> BoxStream<'a, Result<RawRecord>> {
        let cfg = self.config.clone();
        let raws = collect_raw_records(&cfg);
        Box::pin(stream::iter(raws).map(Ok))
    }

    fn normalize(&self, raw: RawRecord) -> Result<Vec<AnamnesisRecord>> {
        normalizer::normalize(raw, self.config.instance.as_deref())
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

/// Walk every project under `projects_root` and produce one `RawRecord`
/// per memory file + per session file. Files that can't be read are
/// skipped (the caller logs to `import_errors`).
fn collect_raw_records(cfg: &ClaudeCodeConfig) -> Vec<RawRecord> {
    let scans = match scanner::scan_projects_root(&cfg.projects_root) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                error = %e,
                root = %cfg.projects_root.display(),
                "scan_projects_root failed; emitting zero records"
            );
            return Vec::new();
        }
    };
    let mut out = Vec::new();
    for proj in scans {
        for mem in proj.memory_files {
            match std::fs::read_to_string(&mem) {
                Ok(body) => out.push(normalizer::raw_memory(&mem, body, cfg.instance.as_deref())),
                Err(e) => {
                    tracing::warn!(
                        path = %mem.display(),
                        error = %e,
                        "skipping unreadable memory file"
                    );
                }
            }
        }
        for sess in proj.jsonl_files {
            match std::fs::read_to_string(&sess) {
                Ok(body) => out.push(normalizer::raw_session(
                    &sess,
                    body,
                    cfg.instance.as_deref(),
                )),
                Err(e) => {
                    tracing::warn!(
                        path = %sess.display(),
                        error = %e,
                        "skipping unreadable session file"
                    );
                }
            }
        }
    }
    out
}

/// Convenience: read a single memory file into a `RawRecord` (used by
/// the importer when re-importing one file outside the streaming scan).
pub fn read_memory_file(path: &std::path::Path, instance: Option<&str>) -> Result<RawRecord> {
    let body = std::fs::read_to_string(path).map_err(|e| Error::Adapter {
        adapter: ADAPTER_ID.into(),
        message: format!("read {}: {e}", path.display()),
    })?;
    Ok(normalizer::raw_memory(path, body, instance))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anamnesis_core::adapter::MemoryAdapter;
    use anamnesis_core::Kind;
    use futures::StreamExt;
    use std::fs;

    static NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn tmp_dir() -> std::path::PathBuf {
        let n = NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let pid = std::process::id();
        let p = std::env::temp_dir().join(format!("anamnesis-adapter-{pid}-{n}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn touch(p: &std::path::Path, content: &str) {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(p, content).unwrap();
    }

    fn realistic_fixture() -> std::path::PathBuf {
        let root = tmp_dir();
        let proj = root.join("project-abc");
        touch(
            &proj.join("memory").join("user_role.md"),
            "---\nname: senior-dev\ndescription: 10y rust\nmetadata:\n  type: user\n---\n\nuser is senior",
        );
        touch(
            &proj.join("memory").join("feedback_tests.md"),
            "---\nname: no-mocks\nmetadata:\n  type: feedback\n---\n\nuse real DB",
        );
        touch(&proj.join("memory").join("MEMORY.md"), "index");
        touch(
            &proj.join("session-1.jsonl"),
            "{\"role\":\"user\",\"content\":\"hi\"}\n{\"role\":\"assistant\",\"content\":\"hello\"}\n",
        );
        root
    }

    #[tokio::test]
    async fn descriptor_is_stable() {
        let a = ClaudeCodeAdapter::new(ClaudeCodeConfig {
            projects_root: "/tmp/nonexistent".into(),
            instance: Some("default".into()),
        });
        let d = a.descriptor();
        assert_eq!(d.adapter, ADAPTER_ID);
        assert_eq!(d.instance.as_deref(), Some("default"));
    }

    #[tokio::test]
    async fn scan_empty_when_root_missing() {
        let a = ClaudeCodeAdapter::new(ClaudeCodeConfig {
            projects_root: "/tmp/definitely-not-here".into(),
            instance: None,
        });
        let count = a.scan(ScanOpts::default()).collect::<Vec<_>>().await.len();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn scan_emits_memory_and_session_artifacts() {
        let root = realistic_fixture();
        let a = ClaudeCodeAdapter::new(ClaudeCodeConfig {
            projects_root: root,
            instance: Some("default".into()),
        });
        let items: Vec<_> = a
            .scan(ScanOpts::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(items.len(), 3, "2 memory + 1 session (MEMORY.md excluded)");
        let kinds: Vec<&str> = items
            .iter()
            .map(|r| r.payload["payload_kind"].as_str().unwrap())
            .collect();
        assert_eq!(kinds.iter().filter(|k| **k == "memory_md").count(), 2,);
        assert_eq!(kinds.iter().filter(|k| **k == "session_jsonl").count(), 1,);
    }

    #[tokio::test]
    async fn scan_then_normalize_produces_correct_record_kinds() {
        let root = realistic_fixture();
        let a = ClaudeCodeAdapter::new(ClaudeCodeConfig {
            projects_root: root,
            instance: Some("default".into()),
        });
        let mut user = 0;
        let mut feedback = 0;
        let mut episode = 0;
        let raws: Vec<_> = a
            .scan(ScanOpts::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        for raw in raws {
            for record in a.normalize(raw).unwrap() {
                match record.kind {
                    Kind::Fact => user += 1,
                    Kind::Feedback => feedback += 1,
                    Kind::Episode => episode += 1,
                    _ => {}
                }
            }
        }
        assert_eq!(user, 1, "user_role.md should produce Kind::Fact");
        assert_eq!(feedback, 1);
        assert_eq!(episode, 1);
    }

    async fn collect_ids(adapter: &ClaudeCodeAdapter) -> Vec<anamnesis_core::RecordId> {
        let raws: Vec<_> = adapter
            .scan(ScanOpts::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        let mut ids = Vec::new();
        for raw in raws {
            for record in adapter.normalize(raw).unwrap() {
                ids.push(record.id);
            }
        }
        ids.sort_by(|a, b| a.0.cmp(&b.0));
        ids
    }

    #[tokio::test]
    async fn import_is_idempotent_across_scan_runs() {
        let root = realistic_fixture();
        let a = ClaudeCodeAdapter::new(ClaudeCodeConfig {
            projects_root: root,
            instance: Some("default".into()),
        });
        let a_ids = collect_ids(&a).await;
        let b_ids = collect_ids(&a).await;
        assert_eq!(a_ids, b_ids, "two scans must produce identical record ids");
    }

    #[tokio::test]
    async fn health_reports_path_existence() {
        let a = ClaudeCodeAdapter::new(ClaudeCodeConfig {
            projects_root: "/tmp/never".into(),
            instance: None,
        });
        let h = a.health().await;
        assert!(!h.ok);
        assert!(h.detail.contains("not found"));
    }
}
