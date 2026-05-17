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
pub mod session;

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

    fn scan<'a>(&'a self, opts: ScanOpts) -> BoxStream<'a, Result<RawRecord>> {
        // Round-19 (§-1.5 PR-4a): stream files lazily and honor
        // `opts.since` / `opts.full`. We still pre-walk the directory
        // tree (the walk itself is cheap; what was expensive was reading
        // every file into memory before yielding the first record).
        // True per-file laziness happens inside `stream_raw_records`,
        // which yields one `RawRecord` at a time and only reads the
        // file body on demand.
        let cfg = (*self.config).clone();
        Box::pin(stream_raw_records(cfg, opts).map(Ok))
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

/// Whether the file at `path` is "newer than the threshold" for an
/// incremental scan. `since == None` (the default / `--full` case) means
/// "no filter, always include".
///
/// On a metadata-read failure we conservatively INCLUDE the file
/// (return `true`): the importer's per-record raw_hash fast-path is a
/// safety net — a re-emitted unchanged record is a no-op upsert. False
/// positives are cheap; a false negative would silently drop user data.
fn passes_since_filter(
    path: &std::path::Path,
    since: Option<chrono::DateTime<chrono::Utc>>,
) -> bool {
    let Some(threshold) = since else { return true };
    match file_mtime(path) {
        Some(mtime) => mtime > threshold,
        None => {
            tracing::debug!(
                path = %path.display(),
                "no mtime available; conservatively including in incremental scan"
            );
            true
        }
    }
}

/// Walk every project under `projects_root` and **stream** one
/// `RawRecord` per memory / session file. Files that can't be read are
/// skipped (the caller logs to `import_errors`). Lazy IO — the file
/// body is read inside the per-item closure, not up-front.
///
/// Round-19 (§-1.5 PR-4a): if `opts.since` is set, files whose mtime is
/// at or before `since` are skipped without reading the body.
/// `opts.full` overrides this back to "yield everything".
fn stream_raw_records(cfg: ClaudeCodeConfig, opts: ScanOpts) -> BoxStream<'static, RawRecord> {
    let scans = match scanner::scan_projects_root(&cfg.projects_root) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                error = %e,
                root = %cfg.projects_root.display(),
                "scan_projects_root failed; emitting zero records"
            );
            return Box::pin(stream::iter(Vec::<RawRecord>::new()));
        }
    };

    // Flatten into a single (kind, path) list while preserving the
    // existing order (memory files first per project, then sessions).
    // PR-4b will push this flattening inside the scanner itself; for
    // PR-4a we only fix the IO-per-file laziness.
    enum FileKind {
        Memory,
        Session,
    }
    let mut work: Vec<(FileKind, std::path::PathBuf)> = Vec::new();
    for proj in scans {
        for mem in proj.memory_files {
            work.push((FileKind::Memory, mem));
        }
        for sess in proj.jsonl_files {
            work.push((FileKind::Session, sess));
        }
    }

    // Apply `since` filter via per-file mtime BEFORE reading bodies.
    let since = if opts.full { None } else { opts.since };
    let instance = cfg.instance.clone();
    let stream = stream::iter(work).filter_map(move |(kind, path)| {
        let instance = instance.clone();
        async move {
            if !passes_since_filter(&path, since) {
                return None;
            }
            match std::fs::read_to_string(&path) {
                Ok(body) => {
                    let mtime = file_mtime(&path);
                    let raw = match kind {
                        FileKind::Memory => {
                            normalizer::raw_memory(&path, body, mtime, instance.as_deref())
                        }
                        FileKind::Session => {
                            normalizer::raw_session(&path, &body, mtime, instance.as_deref())
                        }
                    };
                    Some(raw)
                }
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "skipping unreadable file"
                    );
                    None
                }
            }
        }
    });
    Box::pin(stream)
}

/// Read a file's modification time as `DateTime<Utc>`. Returns `None`
/// when `metadata()` fails or the platform doesn't expose mtime — the
/// normalizer falls back to `captured_at` in that case.
fn file_mtime(path: &std::path::Path) -> Option<chrono::DateTime<chrono::Utc>> {
    let meta = std::fs::metadata(path).ok()?;
    let m = meta.modified().ok()?;
    Some(chrono::DateTime::<chrono::Utc>::from(m))
}

/// Convenience: read a single memory file into a `RawRecord` (used by
/// the importer when re-importing one file outside the streaming scan).
pub fn read_memory_file(path: &std::path::Path, instance: Option<&str>) -> Result<RawRecord> {
    let body = std::fs::read_to_string(path).map_err(|e| Error::Adapter {
        adapter: ADAPTER_ID.into(),
        message: format!("read {}: {e}", path.display()),
    })?;
    let mtime = file_mtime(path);
    Ok(normalizer::raw_memory(path, body, mtime, instance))
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

    /// Round-19 (§-1.5 PR-4a): the adapter must skip files whose mtime
    /// is at or before `opts.since`. Build a fixture with two memory
    /// files, force one's mtime into the past, then scan with `since`
    /// set between them. Only the newer file should be emitted.
    #[tokio::test]
    async fn scan_since_filters_files_by_mtime() {
        use filetime::FileTime;
        let root = tmp_dir();
        let proj = root.join("proj-pr4");

        touch(
            &proj.join("memory").join("old.md"),
            "---\ntype: fact\n---\nold content",
        );
        touch(
            &proj.join("memory").join("new.md"),
            "---\ntype: fact\n---\nnew content",
        );

        // Force the old file's mtime to a known-past timestamp.
        let old_path = proj.join("memory").join("old.md");
        filetime::set_file_mtime(&old_path, FileTime::from_unix_time(1_700_000_000, 0)).unwrap();

        // Cutoff sits AFTER the old file but BEFORE the new file.
        // (`new.md` was just written, so its mtime is "now"; old.md was
        // pushed back to ~2023-11-14.)
        let cutoff = chrono::DateTime::<chrono::Utc>::from_timestamp(1_750_000_000, 0).unwrap();

        let adapter = ClaudeCodeAdapter::new(ClaudeCodeConfig {
            projects_root: root,
            instance: Some("default".into()),
        });

        let raws: Vec<_> = adapter
            .scan(ScanOpts {
                since: Some(cutoff),
                full: false,
            })
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        assert_eq!(
            raws.len(),
            1,
            "since-filter should drop the old.md file; got: {raws:?}"
        );
        assert!(
            raws[0]
                .native_path
                .as_deref()
                .unwrap_or("")
                .ends_with("new.md"),
            "the surviving record must be new.md; got native_path={:?}",
            raws[0].native_path,
        );
    }

    /// `opts.full = true` must override `opts.since` — the contract that
    /// keeps `--full` honest.
    #[tokio::test]
    async fn scan_full_overrides_since_filter() {
        use filetime::FileTime;
        let root = tmp_dir();
        let proj = root.join("proj-pr4-full");
        touch(
            &proj.join("memory").join("old.md"),
            "---\ntype: fact\n---\nold",
        );
        touch(
            &proj.join("memory").join("new.md"),
            "---\ntype: fact\n---\nnew",
        );
        let old_path = proj.join("memory").join("old.md");
        filetime::set_file_mtime(&old_path, FileTime::from_unix_time(1_700_000_000, 0)).unwrap();

        let cutoff = chrono::DateTime::<chrono::Utc>::from_timestamp(1_750_000_000, 0).unwrap();
        let adapter = ClaudeCodeAdapter::new(ClaudeCodeConfig {
            projects_root: root,
            instance: Some("default".into()),
        });

        let raws: Vec<_> = adapter
            .scan(ScanOpts {
                since: Some(cutoff),
                full: true, // → ignore `since`
            })
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        assert_eq!(
            raws.len(),
            2,
            "--full must override --since; expected both files, got: {raws:?}"
        );
    }
}
