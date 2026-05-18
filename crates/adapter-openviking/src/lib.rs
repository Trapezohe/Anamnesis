//! Anamnesis adapter for **OpenViking** (volcengine/OpenViking, AGPLv3
//! upstream; this adapter only **reads** the data files OpenViking writes
//! and does **not** link to or execute OpenViking code, so it ships under
//! Anamnesis's own Apache-2.0 license — see workspace LICENSE / NOTICE).
//!
//! Storage model (per `docs/en/concepts/04-viking-uri.md` &
//! `docs/en/concepts/05-storage.md`):
//!
//! ```text
//! ~/.openviking/                         # config root
//! ├── ov.conf
//! └── data/                              # `storage.workspace` (default `./data`)
//!     └── local/<account_id>/
//!         ├── resources/<project>/...    → Reference  / User
//!         ├── user/<uid>/memories/
//!         │   ├── profile.md             → Preference / User
//!         │   ├── preferences/*.md       → Preference / User
//!         │   ├── entities/*.md          → Fact       / User
//!         │   └── events/*.md            → Episode    / User
//!         ├── agent/<aid>/
//!         │   ├── memories/
//!         │   │   ├── cases/*.md         → Episode    / Project
//!         │   │   ├── patterns/*.md      → Reference  / Project
//!         │   │   ├── tools/*.md         → Reference  / Project
//!         │   │   └── skills/*.md        → Reference  / Project
//!         │   ├── skills/<name>/SKILL.md → Skill      / Project
//!         │   └── instructions/*.md      → Reference  / Project
//!         └── session/<sid>/
//!             ├── .abstract.md           → Episode    / Session
//!             ├── .overview.md           → Episode    / Session
//!             └── messages.jsonl         → Episode    / Session (one record / line)
//! ```
//!
//! Per §-1.2.2 the adapter is read-only.

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

pub use detector::OpenVikingDetector;
pub use scanner::{OpenVikingFileRecord, OpenVikingMessage, OpenVikingScan, OpenVikingScope};

/// Stable adapter identifier.
pub const ADAPTER_ID: &str = "openviking";

/// Adapter configuration.
#[derive(Debug, Clone)]
pub struct OpenVikingConfig {
    /// AGFS workspace dir (default: `~/.openviking/data/`).
    pub workspace_dir: PathBuf,
    /// Instance discriminator (defaults to `"local"` in id synthesis).
    pub instance: Option<String>,
}

/// The adapter.
pub struct OpenVikingAdapter {
    config: Arc<OpenVikingConfig>,
}

impl OpenVikingAdapter {
    /// Build from explicit config.
    pub fn new(config: OpenVikingConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

#[async_trait]
impl MemoryAdapter for OpenVikingAdapter {
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
        if !self.config.workspace_dir.is_dir() {
            return HealthStatus {
                ok: false,
                detail: format!(
                    "openviking workspace not found: {}",
                    self.config.workspace_dir.display()
                ),
            };
        }
        let s = scanner::scan_openviking(&self.config.workspace_dir);
        HealthStatus {
            ok: true,
            detail: format!(
                "openviking workspace: {} (files={}, messages={})",
                self.config.workspace_dir.display(),
                s.files.len(),
                s.messages.len(),
            ),
        }
    }
}

fn collect_raws(cfg: &OpenVikingConfig, opts: &ScanOpts) -> Vec<RawRecord> {
    let scan = scanner::scan_openviking(&cfg.workspace_dir);
    let mut out = Vec::with_capacity(scan.total());
    for f in &scan.files {
        if passes_since(f.mtime_unix, opts) {
            out.push(normalizer::raw_from_file(f, cfg.instance.as_deref()));
        }
    }
    for m in &scan.messages {
        if passes_since(m.mtime_unix, opts) {
            out.push(normalizer::raw_from_message(m, cfg.instance.as_deref()));
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
        None => true,
    }
}

/// Convenience constructor.
pub fn openviking_adapter(
    workspace_dir: impl Into<PathBuf>,
    instance: Option<&str>,
) -> OpenVikingAdapter {
    OpenVikingAdapter::new(OpenVikingConfig {
        workspace_dir: workspace_dir.into(),
        instance: instance.map(str::to_owned),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anamnesis_core::Kind;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static OV_LIB_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp_dir() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = OV_LIB_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "anamnesis-ov-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn seed(workspace: &std::path::Path) {
        let acct = workspace.join("local/acct-1");
        fs::create_dir_all(acct.join("resources/docs")).unwrap();
        fs::write(acct.join("resources/docs/api.md"), "api ref").unwrap();
        fs::create_dir_all(acct.join("user/u/memories/preferences")).unwrap();
        fs::write(acct.join("user/u/memories/profile.md"), "i am pm").unwrap();
        fs::write(
            acct.join("user/u/memories/preferences/dark.md"),
            "dark mode",
        )
        .unwrap();
        fs::create_dir_all(acct.join("agent/a/skills/search")).unwrap();
        fs::write(acct.join("agent/a/skills/search/SKILL.md"), "search").unwrap();
        fs::create_dir_all(acct.join("session/s1")).unwrap();
        fs::write(
            acct.join("session/s1/messages.jsonl"),
            "{\"id\":\"m1\",\"role\":\"user\",\"parts\":[{\"type\":\"text\",\"text\":\"hi\"}]}\n",
        )
        .unwrap();
    }

    #[tokio::test]
    async fn descriptor_carries_instance() {
        let a = openviking_adapter("/tmp/x", Some("laptop"));
        let d = a.descriptor();
        assert_eq!(d.adapter, "openviking");
        assert_eq!(d.instance.as_deref(), Some("laptop"));
    }

    #[tokio::test]
    async fn health_false_when_workspace_missing() {
        let a = openviking_adapter("/tmp/never-here-ov", None);
        let h = a.health().await;
        assert!(!h.ok);
    }

    #[tokio::test]
    async fn scan_yields_all_known_kinds() {
        let dir = tmp_dir();
        seed(&dir);
        let a = openviking_adapter(&dir, Some("local"));
        let raws: Vec<_> = a
            .scan(ScanOpts::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        // 1 resource + 1 profile + 1 pref + 1 skill + 1 message = 5
        assert_eq!(raws.len(), 5);
        let mut kinds = std::collections::HashSet::new();
        for raw in raws {
            for r in a.normalize(raw).unwrap() {
                kinds.insert(r.kind);
            }
        }
        assert!(kinds.contains(&Kind::Reference));
        assert!(kinds.contains(&Kind::Preference));
        assert!(kinds.contains(&Kind::Skill));
        assert!(kinds.contains(&Kind::Episode));
    }

    #[tokio::test]
    async fn scan_full_overrides_since() {
        let dir = tmp_dir();
        seed(&dir);
        let a = openviking_adapter(&dir, None);
        let cutoff = chrono::DateTime::parse_from_rfc3339("2099-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let dropped: Vec<_> = a
            .scan(ScanOpts {
                since: Some(cutoff),
                full: false,
            })
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(dropped.len(), 0);
        let full: Vec<_> = a
            .scan(ScanOpts {
                since: Some(cutoff),
                full: true,
            })
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(full.len(), 5);
    }

    #[tokio::test]
    async fn idempotent_across_scans() {
        let dir = tmp_dir();
        seed(&dir);
        let a = openviking_adapter(&dir, Some("laptop"));
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
