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

    fn scan<'a>(&'a self, opts: ScanOpts) -> BoxStream<'a, Result<RawRecord>> {
        // Round-20 (§-1.5 PR-4b): honor `opts.since` / `opts.full`.
        // Same pattern as `adapter-claude-code`: walk the tree up-front
        // (cheap), filter by file mtime BEFORE reading the body, and
        // read the body lazily inside the async closure. `opts.full`
        // bypasses the `since` filter; mtime read failures
        // conservatively INCLUDE the file (the importer's raw_hash
        // fast-path makes a false positive a no-op upsert; a false
        // negative would silently drop user data).
        let cfg = (*self.config).clone();
        Box::pin(stream_raw_records(cfg, opts).map(Ok))
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

/// Round-20 (§-1.5 PR-4b): true streaming scan of codex session files
/// that honors `opts.since` / `opts.full` exactly the same way the
/// claude-code adapter does. Walk happens up-front (cheap); mtime check
/// runs BEFORE the body read; body is read lazily inside the async
/// closure.
fn stream_raw_records(cfg: CodexConfig, opts: ScanOpts) -> BoxStream<'static, RawRecord> {
    let files = match scanner::scan_root(&cfg.root) {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!(
                error = %e,
                root = %cfg.root.display(),
                "scan_root failed; emitting zero records"
            );
            return Box::pin(stream::iter(Vec::<RawRecord>::new()));
        }
    };

    let since = if opts.full { None } else { opts.since };
    let instance = cfg.instance.clone();
    let stream = stream::iter(files).filter_map(move |path| {
        let instance = instance.clone();
        async move {
            if !passes_since_filter(&path, since) {
                return None;
            }
            match std::fs::read_to_string(&path) {
                Ok(body) => Some(normalizer::raw_session(&path, body, instance.as_deref())),
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "skipping unreadable codex session file"
                    );
                    None
                }
            }
        }
    });
    Box::pin(stream)
}

/// Whether the file at `path` is "newer than the threshold" for an
/// incremental scan. `since == None` (default / `--full`) means "no
/// filter, always include". Metadata-read failures conservatively
/// INCLUDE the file — see `adapter-claude-code` for the same rationale.
fn passes_since_filter(
    path: &std::path::Path,
    since: Option<chrono::DateTime<chrono::Utc>>,
) -> bool {
    let Some(threshold) = since else { return true };
    let Ok(meta) = std::fs::metadata(path) else {
        return true;
    };
    let Ok(modified) = meta.modified() else {
        return true;
    };
    chrono::DateTime::<chrono::Utc>::from(modified) > threshold
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

    /// Round-20 (§-1.5 PR-4b): mtime filter.
    #[tokio::test]
    async fn scan_since_filters_files_by_mtime() {
        use filetime::FileTime;
        let dir = tmp_dir();
        fs::create_dir_all(&dir).unwrap();
        let old_p = dir.join("old.jsonl");
        let new_p = dir.join("new.jsonl");
        fs::write(&old_p, "{\"role\":\"user\",\"content\":\"old\"}\n").unwrap();
        fs::write(&new_p, "{\"role\":\"user\",\"content\":\"new\"}\n").unwrap();
        filetime::set_file_mtime(&old_p, FileTime::from_unix_time(1_700_000_000, 0)).unwrap();
        let cutoff = chrono::DateTime::<chrono::Utc>::from_timestamp(1_750_000_000, 0).unwrap();

        let a = codex_adapter(&dir, None);
        let raws: Vec<_> = a
            .scan(ScanOpts {
                since: Some(cutoff),
                full: false,
            })
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(raws.len(), 1, "old file should be filtered out");
        assert!(raws[0]
            .native_path
            .as_deref()
            .unwrap_or("")
            .ends_with("new.jsonl"));
    }

    /// `opts.full` overrides `opts.since`.
    #[tokio::test]
    async fn scan_full_overrides_since() {
        use filetime::FileTime;
        let dir = tmp_dir();
        fs::create_dir_all(&dir).unwrap();
        let old_p = dir.join("old.jsonl");
        let new_p = dir.join("new.jsonl");
        fs::write(&old_p, "{\"role\":\"user\",\"content\":\"old\"}\n").unwrap();
        fs::write(&new_p, "{\"role\":\"user\",\"content\":\"new\"}\n").unwrap();
        filetime::set_file_mtime(&old_p, FileTime::from_unix_time(1_700_000_000, 0)).unwrap();
        let cutoff = chrono::DateTime::<chrono::Utc>::from_timestamp(1_750_000_000, 0).unwrap();

        let a = codex_adapter(&dir, None);
        let raws: Vec<_> = a
            .scan(ScanOpts {
                since: Some(cutoff),
                full: true, // override
            })
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(raws.len(), 2, "--full must include both files");
    }
}
