//! Anamnesis adapter for **ghast** (Electron AI companion).
//!
//! ## What this adapter reads
//!
//! `~/Documents/ghast_desktop/` (or wherever the user cloned ghast):
//!
//! ```text
//! ghast_desktop/
//! ├── prompts/<role>/*.md                      → Kind::Reference
//! └── resources/bundled-skills/<skill>/
//!     ├── SKILL.md                             → Kind::Skill
//!     └── REFERENCES.md / README.md / NOTES.md → Kind::Reference
//! ```
//!
//! ## What this adapter does NOT yet read
//!
//! The ghast user-profile database at
//! `~/Library/Application Support/ghast/profiles/<id>/ghast.db` is
//! **encrypted at rest** (sqlite3-multiple-ciphers — not plain
//! SQLite). ghast hasn't yet published a key-export contract, so the
//! adapter:
//!
//!   * detects the encrypted file and surfaces its path in
//!     `health().detail` so the user knows there's MORE data to
//!     migrate once an export path lands,
//!   * never tries to decrypt blind — that would be a bad-citizen
//!     move and would silently fail anyway,
//!   * scans only the prompts + skills, which are useful in their
//!     own right (system prompts + skill definitions are durable
//!     reference / capability records the agent re-uses).
//!
//! When ghast adds a `ghast export` command (plain JSONL/SQLite) or
//! exposes an MCP server with `resources/list`, a follow-up round
//! wires that in — likely as `adapter-ghast-export` so the
//! source-repo reader stays clean.
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

pub use detector::GhastDetector;
pub use scanner::{GhastPromptFile, GhastScan, GhastSkill};

/// Stable adapter identifier.
pub const ADAPTER_ID: &str = "ghast";

/// Adapter configuration.
#[derive(Debug, Clone)]
pub struct GhastConfig {
    /// Root directory: either the source repo (recommended) or the
    /// app-support dir. Either is fine — the scanner probes for
    /// `prompts/` and `resources/bundled-skills/` independently and
    /// for an encrypted `ghast.db` independently.
    pub root: PathBuf,
    /// Instance discriminator. Defaults to `"local"` in id-synthesis.
    pub instance: Option<String>,
}

/// The adapter.
pub struct GhastAdapter {
    config: Arc<GhastConfig>,
}

impl GhastAdapter {
    /// Build from explicit config.
    pub fn new(config: GhastConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

#[async_trait]
impl MemoryAdapter for GhastAdapter {
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
        let root = &self.config.root;
        if !root.is_dir() {
            return HealthStatus {
                ok: false,
                detail: format!("ghast root not found: {}", root.display()),
            };
        }
        let scan = scanner::scan_ghast(root);
        let prompts = scan.prompts.len();
        let skills = scan.skills.len();
        let mut detail = format!(
            "ghast root: {} ({prompts} prompts, {skills} skill files)",
            root.display()
        );
        if let Some(db) = &scan.encrypted_profile_db {
            detail.push_str(&format!(
                "; encrypted profile DB at {} is detected but cannot be \
                 read (waiting on ghast `export` / MCP-server path).",
                db.display()
            ));
        }
        HealthStatus { ok: true, detail }
    }
}

fn collect_raws(cfg: &GhastConfig, opts: &ScanOpts) -> Vec<RawRecord> {
    let scan = scanner::scan_ghast(&cfg.root);

    if let Some(db) = &scan.encrypted_profile_db {
        tracing::warn!(
            adapter = ADAPTER_ID,
            path = %db.display(),
            "ghast encrypted profile DB detected; not ingested (no key-export path yet). \
             prompts + bundled-skills are still imported."
        );
    }

    let mut out = Vec::with_capacity(scan.total());
    for p in &scan.prompts {
        if passes_since(p.mtime_unix, opts) {
            out.push(normalizer::raw_from_prompt(p, cfg.instance.as_deref()));
        }
    }
    for s in &scan.skills {
        if passes_since(s.mtime_unix, opts) {
            out.push(normalizer::raw_from_skill(s, cfg.instance.as_deref()));
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
pub fn ghast_adapter(root: impl Into<PathBuf>, instance: Option<&str>) -> GhastAdapter {
    GhastAdapter::new(GhastConfig {
        root: root.into(),
        instance: instance.map(str::to_owned),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anamnesis_core::Kind;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static GHAST_LIB_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp_dir() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = GHAST_LIB_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "anamnesis-ghast-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn seed(root: &std::path::Path) {
        fs::create_dir_all(root.join("prompts/coding")).unwrap();
        fs::create_dir_all(root.join("resources/bundled-skills/memory-management")).unwrap();
        fs::write(
            root.join("prompts/coding/default.md"),
            "default coding prompt",
        )
        .unwrap();
        fs::write(
            root.join("resources/bundled-skills/memory-management/SKILL.md"),
            "skill body",
        )
        .unwrap();
        fs::write(
            root.join("resources/bundled-skills/memory-management/REFERENCES.md"),
            "refs",
        )
        .unwrap();
    }

    #[tokio::test]
    async fn descriptor_carries_instance() {
        let a = ghast_adapter("/tmp/x", Some("laptop"));
        let d = a.descriptor();
        assert_eq!(d.adapter, "ghast");
        assert_eq!(d.instance.as_deref(), Some("laptop"));
    }

    #[tokio::test]
    async fn health_warns_when_root_missing() {
        let a = ghast_adapter("/tmp/never-here-ghast", None);
        let h = a.health().await;
        assert!(!h.ok);
        assert!(h.detail.contains("not found"));
    }

    #[tokio::test]
    async fn scan_yields_prompt_and_skill_records() {
        let root = tmp_dir();
        seed(&root);
        let a = ghast_adapter(&root, Some("laptop"));
        let raws: Vec<_> = a
            .scan(ScanOpts::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        // 1 prompt + 2 skill files = 3
        assert_eq!(raws.len(), 3);
        let mut kinds = std::collections::HashSet::new();
        for raw in raws {
            for r in a.normalize(raw).unwrap() {
                kinds.insert(r.kind);
            }
        }
        assert!(kinds.contains(&Kind::Reference));
        assert!(kinds.contains(&Kind::Skill));
    }

    #[tokio::test]
    async fn health_mentions_encrypted_profile_db_when_present() {
        let root = tmp_dir();
        seed(&root);
        let profile = root.join("profiles/some-id");
        fs::create_dir_all(&profile).unwrap();
        fs::write(profile.join("ghast.db"), b"\xff encrypted bytes").unwrap();
        let a = ghast_adapter(&root, None);
        let h = a.health().await;
        assert!(h.ok);
        assert!(
            h.detail.contains("encrypted"),
            "health detail should mention the encrypted profile DB; got: {}",
            h.detail
        );
    }

    #[tokio::test]
    async fn scan_full_overrides_since_filter() {
        let root = tmp_dir();
        seed(&root);
        let a = ghast_adapter(&root, None);
        let cutoff = chrono::DateTime::parse_from_rfc3339("2099-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let raws_drop_all: Vec<_> = a
            .scan(ScanOpts {
                since: Some(cutoff),
                full: false,
            })
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(raws_drop_all.len(), 0);

        let raws_full: Vec<_> = a
            .scan(ScanOpts {
                since: Some(cutoff),
                full: true,
            })
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(raws_full.len(), 3);
    }

    #[tokio::test]
    async fn idempotent_across_scans() {
        let root = tmp_dir();
        seed(&root);
        let a = ghast_adapter(&root, Some("laptop"));
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
