//! Anamnesis adapter for the OpenAI Codex CLI.
//!
//! Codex stores conversation history under `~/.codex/` (sessions and/or
//! conversations subdirectories, depending on version). Layout has
//! changed across Codex versions, so this adapter is intentionally
//! permissive: every `.json` and `.jsonl` file under the configured
//! root becomes one `Kind::Episode` record.
//!
//! Frontmatter / structured-memory like Claude Code's `memory/*.md`
//! does not exist in Codex, so there's no per-memory-type taxonomy to
//! map. Future Codex releases that add structured fields can extend
//! the normalizer.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod detector;
pub mod normalizer;
pub mod scanner;

use std::path::PathBuf;
use std::sync::Arc;

use anamnesis_core::adapter::{HealthStatus, MemoryAdapter, RawRecord, ScanOpts};
use anamnesis_core::error::Result;
use anamnesis_core::model::{AnamnesisRecord, SourceDescriptor};
use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};

pub use detector::CodexDetector;

/// Stable adapter identifier.
pub const ADAPTER_ID: &str = "codex";

/// Adapter configuration.
#[derive(Debug, Clone)]
pub struct CodexConfig {
    /// Root directory to walk (typically `~/.codex/`).
    pub root: PathBuf,
    /// Optional instance discriminator.
    pub instance: Option<String>,
}

/// The adapter.
pub struct CodexAdapter {
    config: Arc<CodexConfig>,
}

impl CodexAdapter {
    /// Build from config.
    pub fn new(config: CodexConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

#[async_trait]
impl MemoryAdapter for CodexAdapter {
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
        let exists = self.config.root.exists();
        HealthStatus {
            ok: exists,
            detail: if exists {
                format!("codex root: {}", self.config.root.display())
            } else {
                format!("codex root not found: {}", self.config.root.display())
            },
        }
    }
}

fn collect_raw_records(cfg: &CodexConfig) -> Vec<RawRecord> {
    match scanner::scan_root(&cfg.root) {
        Ok(files) => files
            .iter()
            .filter_map(|path| match std::fs::read_to_string(path) {
                Ok(body) => Some(normalizer::raw_session(path, body, cfg.instance.as_deref())),
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "skipping unreadable codex session file"
                    );
                    None
                }
            })
            .collect(),
        Err(e) => {
            tracing::warn!(
                error = %e,
                root = %cfg.root.display(),
                "scan_root failed; emitting zero records"
            );
            Vec::new()
        }
    }
}

/// Convenience constructor.
pub fn codex_adapter(root: impl Into<PathBuf>, instance: Option<&str>) -> CodexAdapter {
    CodexAdapter::new(CodexConfig {
        root: root.into(),
        instance: instance.map(str::to_owned),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    static NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn tmp_dir() -> PathBuf {
        let n = NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let pid = std::process::id();
        let p = std::env::temp_dir().join(format!("anamnesis-codex-{pid}-{n}"));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[tokio::test]
    async fn descriptor_is_stable() {
        let a = codex_adapter("/tmp/no-such", Some("default"));
        let d = a.descriptor();
        assert_eq!(d.adapter, "codex");
        assert_eq!(d.instance.as_deref(), Some("default"));
    }

    #[tokio::test]
    async fn scan_empty_when_root_missing() {
        let a = codex_adapter("/tmp/never-here", None);
        let n = a.scan(ScanOpts::default()).collect::<Vec<_>>().await.len();
        assert_eq!(n, 0);
    }

    #[tokio::test]
    async fn scan_finds_json_and_jsonl_files() {
        let dir = tmp_dir();
        fs::create_dir_all(dir.join("sessions")).unwrap();
        fs::write(
            dir.join("sessions").join("s1.jsonl"),
            "{\"role\":\"user\",\"content\":\"hello codex\"}\n",
        )
        .unwrap();
        fs::write(
            dir.join("sessions").join("s2.json"),
            "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
        )
        .unwrap();
        fs::write(dir.join("sessions").join("ignore.txt"), "no").unwrap();

        let a = codex_adapter(&dir, None);
        let raws: Vec<_> = a
            .scan(ScanOpts::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(raws.len(), 2);
    }

    #[tokio::test]
    async fn scan_then_normalize_yields_episode_records() {
        use anamnesis_core::Kind;
        let dir = tmp_dir();
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("s.jsonl"),
            "{\"role\":\"user\",\"content\":\"first\"}\n{\"role\":\"assistant\",\"content\":\"second\"}\n",
        )
        .unwrap();
        let a = codex_adapter(&dir, Some("default"));
        let raws: Vec<_> = a
            .scan(ScanOpts::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        let recs: Vec<_> = raws
            .into_iter()
            .flat_map(|r| a.normalize(r).unwrap())
            .collect();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].kind, Kind::Episode);
        assert_eq!(recs[0].source.adapter, "codex");
    }
}
