//! Anamnesis adapter for **TencentDB Agent Memory** (TDAI,
//! github.com/Tencent/TencentDB-Agent-Memory, MIT).
//!
//! TDAI is an OpenClaw plugin that persists 4-tier hierarchical
//! memory under `~/.openclaw/memory-tdai/`:
//!
//!   * **L0** raw conversation references — `refs/*.md` → `Kind::Episode`,
//!     `Scope::Session`
//!   * **L1** atomic facts — `*.jsonl` (one fact per line) → `Kind::Fact`,
//!     `Scope::User`
//!   * **L2** scenario blocks — plain `*.md` → `Kind::Reference`,
//!     `Scope::User`
//!   * **L3** user persona — `persona.md` → `Kind::Preference`,
//!     `Scope::User`
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

pub use detector::TdaiDetector;
pub use scanner::{TdaiL0Ref, TdaiL1Fact, TdaiL2Scenario, TdaiL3Persona, TdaiScan};

/// Stable adapter identifier.
pub const ADAPTER_ID: &str = "tdai";

/// Adapter configuration.
#[derive(Debug, Clone)]
pub struct TdaiConfig {
    /// TDAI data dir (default: `~/.openclaw/memory-tdai/`).
    pub data_dir: PathBuf,
    /// Instance discriminator (defaults to `"local"` in id synthesis).
    pub instance: Option<String>,
}

/// The adapter.
pub struct TdaiAdapter {
    config: Arc<TdaiConfig>,
}

impl TdaiAdapter {
    /// Build from explicit config.
    pub fn new(config: TdaiConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

#[async_trait]
impl MemoryAdapter for TdaiAdapter {
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
        if !self.config.data_dir.is_dir() {
            return HealthStatus {
                ok: false,
                detail: format!(
                    "tdai data dir not found: {}",
                    self.config.data_dir.display()
                ),
            };
        }
        let s = scanner::scan_tdai(&self.config.data_dir);
        HealthStatus {
            ok: true,
            detail: format!(
                "tdai data dir: {} (L0={}, L1={}, L2={}, L3={})",
                self.config.data_dir.display(),
                s.l0_refs.len(),
                s.l1_facts.len(),
                s.l2_scenarios.len(),
                s.l3_personas.len(),
            ),
        }
    }
}

fn collect_raws(cfg: &TdaiConfig, opts: &ScanOpts) -> Vec<RawRecord> {
    let scan = scanner::scan_tdai(&cfg.data_dir);
    let mut out = Vec::with_capacity(scan.total());
    for r in &scan.l0_refs {
        if passes_since(r.mtime_unix, opts) {
            out.push(normalizer::raw_from_l0(r, cfg.instance.as_deref()));
        }
    }
    for f in &scan.l1_facts {
        if passes_since(f.mtime_unix, opts) {
            out.push(normalizer::raw_from_l1(f, cfg.instance.as_deref()));
        }
    }
    for s in &scan.l2_scenarios {
        if passes_since(s.mtime_unix, opts) {
            out.push(normalizer::raw_from_l2(s, cfg.instance.as_deref()));
        }
    }
    for p in &scan.l3_personas {
        if passes_since(p.mtime_unix, opts) {
            out.push(normalizer::raw_from_l3(p, cfg.instance.as_deref()));
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
pub fn tdai_adapter(data_dir: impl Into<PathBuf>, instance: Option<&str>) -> TdaiAdapter {
    TdaiAdapter::new(TdaiConfig {
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

    static TDAI_LIB_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp_dir() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = TDAI_LIB_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "anamnesis-tdai-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn seed(dir: &std::path::Path) {
        fs::create_dir_all(dir.join("refs")).unwrap();
        fs::write(dir.join("persona.md"), "I am a senior engineer.").unwrap();
        fs::write(dir.join("refs/conv-1.md"), "raw conversation").unwrap();
        fs::write(
            dir.join("facts.jsonl"),
            "{\"f\":\"likes rust\"}\n{\"f\":\"hates mocks\"}\n",
        )
        .unwrap();
        fs::write(dir.join("scenario.md"), "scenario body").unwrap();
    }

    #[tokio::test]
    async fn descriptor_carries_instance() {
        let a = tdai_adapter("/tmp/x", Some("laptop"));
        let d = a.descriptor();
        assert_eq!(d.adapter, "tdai");
        assert_eq!(d.instance.as_deref(), Some("laptop"));
    }

    #[tokio::test]
    async fn health_false_when_dir_missing() {
        let a = tdai_adapter("/tmp/never-here-tdai", None);
        let h = a.health().await;
        assert!(!h.ok);
    }

    #[tokio::test]
    async fn scan_yields_all_four_tiers() {
        let dir = tmp_dir();
        seed(&dir);
        let a = tdai_adapter(&dir, Some("local"));
        let raws: Vec<_> = a
            .scan(ScanOpts::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        // 1 persona + 1 L0 + 2 L1 + 1 L2 = 5
        assert_eq!(raws.len(), 5);
        let mut kinds = std::collections::HashSet::new();
        for raw in raws {
            for r in a.normalize(raw).unwrap() {
                kinds.insert(r.kind);
            }
        }
        assert!(kinds.contains(&Kind::Episode));
        assert!(kinds.contains(&Kind::Fact));
        assert!(kinds.contains(&Kind::Reference));
        assert!(kinds.contains(&Kind::Preference));
    }

    #[tokio::test]
    async fn scan_full_overrides_since() {
        let dir = tmp_dir();
        seed(&dir);
        let a = tdai_adapter(&dir, None);
        let cutoff = chrono::DateTime::parse_from_rfc3339("2099-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let raws_dropped: Vec<_> = a
            .scan(ScanOpts {
                since: Some(cutoff),
                full: false,
            })
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(raws_dropped.len(), 0);
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
        assert_eq!(raws_full.len(), 5);
    }

    #[tokio::test]
    async fn idempotent_across_scans() {
        let dir = tmp_dir();
        seed(&dir);
        let a = tdai_adapter(&dir, Some("laptop"));
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
