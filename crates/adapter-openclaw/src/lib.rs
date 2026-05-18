//! Anamnesis adapter for **OpenClaw** (openclaw/openclaw, TypeScript/Node).
//!
//! Layout (default — `agents.defaults.workspace` may rebase it):
//!
//! ```text
//! ~/.openclaw/
//! ├── openclaw.json                       — top-level config        → Kind::Reference
//! └── workspace/
//!     ├── AGENTS.md                       — agents config preamble  → Kind::Reference
//!     ├── SOUL.md                         — agent persona           → Kind::Reference
//!     ├── TOOLS.md                        — tool list               → Kind::Reference
//!     ├── skills/
//!     │   └── <name>/SKILL.md             — one per skill           → Kind::Skill
//!     └── sessions/
//!         └── *.{json,jsonl,ndjson}       — one per session file    → Kind::Episode
//! ```
//!
//! Per §-1.2.2 the adapter is read-only. `ScanOpts.since` /
//! `ScanOpts.full` (PR-4 contract) are honored on every record kind
//! via file mtime.

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

pub use detector::OpenClawDetector;
pub use scanner::{OpenClawConfigFile, OpenClawScan, OpenClawSessionBlob, OpenClawSkill};

/// Stable adapter identifier.
pub const ADAPTER_ID: &str = "openclaw";

/// Adapter configuration.
#[derive(Debug, Clone)]
pub struct OpenClawConfig {
    /// OpenClaw data directory (default in detector: `~/.openclaw/`).
    pub data_dir: PathBuf,
    /// Instance discriminator. Defaults to `"default"` in
    /// id-synthesis helpers when `None`.
    pub instance: Option<String>,
}

/// The adapter.
pub struct OpenClawAdapter {
    config: Arc<OpenClawConfig>,
}

impl OpenClawAdapter {
    /// Build from explicit config.
    pub fn new(config: OpenClawConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

#[async_trait]
impl MemoryAdapter for OpenClawAdapter {
    fn descriptor(&self) -> SourceDescriptor {
        SourceDescriptor {
            adapter: ADAPTER_ID.into(),
            instance: self.config.instance.clone(),
            version: env!("CARGO_PKG_VERSION").into(),
        }
    }

    fn scan<'a>(&'a self, opts: ScanOpts) -> BoxStream<'a, Result<RawRecord>> {
        let cfg = (*self.config).clone();
        let raws = collect_raws(&cfg, &opts);
        Box::pin(stream::iter(raws).map(Ok))
    }

    fn normalize(&self, raw: RawRecord) -> Result<Vec<AnamnesisRecord>> {
        normalizer::normalize(raw, self.config.instance.as_deref())
    }

    async fn health(&self) -> HealthStatus {
        HealthStatus {
            ok: self.config.data_dir.is_dir(),
            detail: if self.config.data_dir.is_dir() {
                format!("openclaw data dir: {}", self.config.data_dir.display())
            } else {
                format!(
                    "openclaw data dir not found: {}",
                    self.config.data_dir.display()
                )
            },
        }
    }
}

fn collect_raws(cfg: &OpenClawConfig, opts: &ScanOpts) -> Vec<RawRecord> {
    let scan = scanner::scan_openclaw_dir(&cfg.data_dir);
    let mut out = Vec::with_capacity(scan.total());

    for cf in &scan.configs {
        if passes_since(cf.mtime_unix, opts) {
            out.push(normalizer::raw_from_config(cf, cfg.instance.as_deref()));
        }
    }
    for s in &scan.skills {
        if passes_since(s.mtime_unix, opts) {
            out.push(normalizer::raw_from_skill(s, cfg.instance.as_deref()));
        }
    }
    for sess in &scan.sessions {
        if passes_since(sess.mtime_unix, opts) {
            out.push(normalizer::raw_from_session(sess, cfg.instance.as_deref()));
        }
    }
    out
}

fn passes_since(mtime_unix: Option<i64>, opts: &ScanOpts) -> bool {
    if opts.full {
        return true;
    }
    let Some(threshold) = opts.since else {
        return true;
    };
    match mtime_unix {
        Some(t) => t > threshold.timestamp(),
        // No mtime: conservatively include — false positive is a no-op
        // upsert via raw_hash fast-path, false negative drops user data.
        None => true,
    }
}

/// Convenience constructor.
pub fn openclaw_adapter(data_dir: impl Into<PathBuf>, instance: Option<&str>) -> OpenClawAdapter {
    OpenClawAdapter::new(OpenClawConfig {
        data_dir: data_dir.into(),
        instance: instance.map(str::to_owned),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anamnesis_core::Kind;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static OPENCLAW_LIB_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp_dir() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = OPENCLAW_LIB_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "anamnesis-openclaw-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn seed(dir: &std::path::Path) {
        let ws = dir.join("workspace");
        let skills = ws.join("skills");
        let sess = ws.join("sessions");
        fs::create_dir_all(skills.join("write-code")).unwrap();
        fs::create_dir_all(&sess).unwrap();
        fs::write(dir.join("openclaw.json"), "{}").unwrap();
        fs::write(ws.join("AGENTS.md"), "agents config").unwrap();
        fs::write(ws.join("SOUL.md"), "system persona").unwrap();
        fs::write(skills.join("write-code/SKILL.md"), "produce rust").unwrap();
        fs::write(sess.join("a.jsonl"), "{\"k\":1}\n").unwrap();
    }

    #[tokio::test]
    async fn descriptor_carries_instance() {
        let a = openclaw_adapter("/tmp/x", Some("laptop"));
        let d = a.descriptor();
        assert_eq!(d.adapter, "openclaw");
        assert_eq!(d.instance.as_deref(), Some("laptop"));
    }

    #[tokio::test]
    async fn health_false_when_dir_missing() {
        let a = openclaw_adapter("/tmp/never-here-openclaw", None);
        let h = a.health().await;
        assert!(!h.ok);
    }

    #[tokio::test]
    async fn scan_yields_three_kinds() {
        let dir = tmp_dir();
        seed(&dir);
        let a = openclaw_adapter(&dir, Some("laptop"));
        let raws: Vec<_> = a
            .scan(ScanOpts::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        // 3 configs (openclaw.json + AGENTS + SOUL) + 1 skill + 1 session = 5
        assert_eq!(raws.len(), 5);
        let mut kinds = std::collections::HashSet::new();
        for raw in raws {
            for r in a.normalize(raw).unwrap() {
                kinds.insert(r.kind);
            }
        }
        assert!(kinds.contains(&Kind::Reference));
        assert!(kinds.contains(&Kind::Skill));
        assert!(kinds.contains(&Kind::Episode));
    }

    #[tokio::test]
    async fn scan_full_overrides_since_filter() {
        let dir = tmp_dir();
        seed(&dir);
        let a = openclaw_adapter(&dir, Some("laptop"));
        // Cutoff far in the future → with `full=false` everything would drop.
        let cutoff = chrono::DateTime::parse_from_rfc3339("2099-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let raws_after: Vec<_> = a
            .scan(ScanOpts {
                since: Some(cutoff),
                full: false,
            })
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        // All files just written → mtime ~ now → since=2099 drops all.
        assert_eq!(raws_after.len(), 0);
        let raws_full: Vec<_> = a
            .scan(ScanOpts {
                since: Some(cutoff),
                full: true, // override
            })
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(raws_full.len(), 5);
    }

    #[tokio::test]
    async fn idempotent_across_scans() {
        let dir = tmp_dir();
        seed(&dir);
        let a = openclaw_adapter(&dir, Some("laptop"));
        let run = || async {
            let mut ids: Vec<_> = a
                .scan(ScanOpts::default())
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .filter_map(|r| r.ok())
                .flat_map(|raw| a.normalize(raw).unwrap())
                .map(|r| r.id.0)
                .collect();
            ids.sort();
            ids
        };
        assert_eq!(run().await, run().await);
    }
}
