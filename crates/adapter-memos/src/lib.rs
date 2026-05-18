//! Anamnesis adapter for **MemOS** (MemTensor/MemOS, Apache-2.0).
//!
//! MemOS organizes durable memory as "MemCubes" — each cube is a directory
//! that, when dumped, contains a flat layout:
//!
//! ```text
//! <cube_dir>/
//! ├── config.json                  # cube config
//! ├── textual_memory.json          # textual memory items (we read this)
//! ├── activation_memory.pickle     # KV cache, binary — skipped
//! └── parametric_memory.adapter    # LoRA weights, binary — skipped
//! ```
//!
//! The adapter walks a MemOS root (default `~/.memos/`), finds every
//! `textual_memory.json`, and emits one `AnamnesisRecord` per textual item.
//!
//! `memory_type` → Anamnesis classification:
//!
//! | MemOS `memory_type`     | Kind        | Scope     |
//! |-------------------------|-------------|-----------|
//! | `WorkingMemory`         | Reference   | Ephemeral |
//! | `LongTermMemory`        | Fact        | User      |
//! | `UserMemory`            | Preference  | User      |
//! | `PreferenceMemory`      | Preference  | User      |
//! | `OuterMemory`           | Reference   | User      |
//! | `ToolSchemaMemory`      | Skill       | Project   |
//! | `SkillMemory`           | Skill       | Project   |
//! | `ToolTrajectoryMemory`  | Episode     | Project   |
//! | `RawFileMemory`         | Reference   | Project   |
//! | (any other / missing)   | Reference   | User      |
//!
//! Flat-backend free-form `type` heuristics (`fact`, `event`, `opinion`,
//! `procedure`) are also honored as a secondary mapping.
//!
//! Tombstones (`status` ∈ {`archived`, `deleted`, `resolving`}) are skipped.
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

pub use detector::MemosDetector;
pub use scanner::{MemosScan, MemosTextItem};

/// Stable adapter identifier.
pub const ADAPTER_ID: &str = "memos";

/// Adapter configuration.
#[derive(Debug, Clone)]
pub struct MemosConfig {
    /// MemOS root dir (default: `~/.memos/`).
    pub root_dir: PathBuf,
    /// Instance discriminator (defaults to `"local"` in id synthesis).
    pub instance: Option<String>,
}

/// The adapter.
pub struct MemosAdapter {
    config: Arc<MemosConfig>,
}

impl MemosAdapter {
    /// Build from explicit config.
    pub fn new(config: MemosConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

#[async_trait]
impl MemoryAdapter for MemosAdapter {
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
        if !self.config.root_dir.is_dir() {
            return HealthStatus {
                ok: false,
                detail: format!("memos root not found: {}", self.config.root_dir.display()),
            };
        }
        let s = scanner::scan_memos(&self.config.root_dir);
        let mut detail = format!(
            "memos root: {} (cubes={}, items={})",
            self.config.root_dir.display(),
            s.cube_dirs.len(),
            s.items.len(),
        );
        if !s.parse_errors.is_empty() {
            detail.push_str(&format!(" — {} parse error(s)", s.parse_errors.len()));
        }
        HealthStatus { ok: true, detail }
    }
}

fn collect_raws(cfg: &MemosConfig, opts: &ScanOpts) -> Vec<RawRecord> {
    let scan = scanner::scan_memos(&cfg.root_dir);
    let mut out = Vec::with_capacity(scan.total());
    for i in &scan.items {
        let ts = i
            .updated_at
            .as_deref()
            .and_then(scanner::parse_memos_time)
            .or_else(|| i.created_at.as_deref().and_then(scanner::parse_memos_time));
        if passes_since(ts, opts) {
            out.push(normalizer::raw_from_item(i, cfg.instance.as_deref()));
        }
    }
    out
}

fn passes_since(ts_unix: Option<i64>, opts: &ScanOpts) -> bool {
    if opts.full {
        return true;
    }
    let Some(threshold) = opts.since else {
        return true;
    };
    match ts_unix {
        Some(t) => t > threshold.timestamp(),
        None => true,
    }
}

/// Convenience constructor.
pub fn memos_adapter(root_dir: impl Into<PathBuf>, instance: Option<&str>) -> MemosAdapter {
    MemosAdapter::new(MemosConfig {
        root_dir: root_dir.into(),
        instance: instance.map(str::to_owned),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anamnesis_core::Kind;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static MEMOS_LIB_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp_dir() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = MEMOS_LIB_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "anamnesis-memos-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn seed_root(root: &std::path::Path) {
        let cube = root.join("cube-1");
        fs::create_dir_all(&cube).unwrap();
        let payload = serde_json::json!([
            {
                "id": "i-1",
                "memory": "user prefers Rust",
                "metadata": {
                    "memory_type": "UserMemory",
                    "user_id": "u-1",
                    "session_id": "s-1",
                    "source": "conversation",
                    "status": "activated",
                    "updated_at": "2026-05-01T10:00:00"
                }
            },
            {
                "id": "i-2",
                "memory": "Paris is the capital",
                "metadata": {
                    "memory_type": "LongTermMemory",
                    "status": "activated",
                    "updated_at": "2026-05-02T10:00:00"
                }
            }
        ]);
        fs::write(cube.join("textual_memory.json"), payload.to_string()).unwrap();
    }

    #[tokio::test]
    async fn descriptor_carries_instance() {
        let a = memos_adapter("/tmp/x", Some("laptop"));
        let d = a.descriptor();
        assert_eq!(d.adapter, "memos");
        assert_eq!(d.instance.as_deref(), Some("laptop"));
    }

    #[tokio::test]
    async fn health_false_when_root_missing() {
        let a = memos_adapter("/tmp/never-here-memos", None);
        let h = a.health().await;
        assert!(!h.ok);
    }

    #[tokio::test]
    async fn scan_yields_expected_kinds() {
        let dir = tmp_dir();
        seed_root(&dir);
        let a = memos_adapter(&dir, Some("local"));
        let raws: Vec<_> = a
            .scan(ScanOpts::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        assert_eq!(raws.len(), 2);
        let mut kinds = std::collections::HashSet::new();
        for raw in raws {
            for r in a.normalize(raw).unwrap() {
                kinds.insert(r.kind);
            }
        }
        assert!(kinds.contains(&Kind::Preference));
        assert!(kinds.contains(&Kind::Fact));
    }

    #[tokio::test]
    async fn scan_full_overrides_since() {
        let dir = tmp_dir();
        seed_root(&dir);
        let a = memos_adapter(&dir, None);
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
        assert_eq!(full.len(), 2);
    }

    #[tokio::test]
    async fn idempotent_across_scans() {
        let dir = tmp_dir();
        seed_root(&dir);
        let a = memos_adapter(&dir, Some("laptop"));
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
