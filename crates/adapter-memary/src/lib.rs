//! Anamnesis adapter for **Memary** (kingjulio8238/Memary, MIT).
//!
//! Memary's primary knowledge graph lives in Neo4j (a remote server). This
//! adapter ingests Memary's **local cache files** — the persistent
//! filesystem state Memary writes alongside the Neo4j layer:
//!
//! ```text
//! <data_dir>/
//! ├── memory_stream.json           # [{entity, date}]          → Reference / Project
//! ├── entity_knowledge_store.json  # [{entity, count, date}]   → Reference / Project
//! ├── past_chat.json               # LlamaIndex ChatMessage[]  → Episode   / Session
//! ├── system_persona.txt           # plain text                → Reference / Project
//! └── user_persona.txt             # plain text                → Preference / User
//! ```
//!
//! To ingest the Neo4j knowledge graph itself, run Memary's own MCP server
//! and use Anamnesis's `generic-mcp` adapter instead (per §-2.4).
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

pub use detector::MemaryDetector;
pub use scanner::{
    MemaryChatMessage, MemaryEntityTally, MemaryPersona, MemaryScan, MemaryStreamEntry,
};

/// Stable adapter identifier.
pub const ADAPTER_ID: &str = "memary";

/// Adapter configuration.
#[derive(Debug, Clone)]
pub struct MemaryConfig {
    /// Memary data directory (default: `~/.memary/data/` or `~/.memary/`).
    pub data_dir: PathBuf,
    /// Instance discriminator (defaults to `"local"` in id synthesis).
    pub instance: Option<String>,
}

/// The adapter.
pub struct MemaryAdapter {
    config: Arc<MemaryConfig>,
}

impl MemaryAdapter {
    /// Build from explicit config.
    pub fn new(config: MemaryConfig) -> Self {
        Self {
            config: Arc::new(config),
        }
    }
}

#[async_trait]
impl MemoryAdapter for MemaryAdapter {
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
                    "memary data dir not found: {}",
                    self.config.data_dir.display()
                ),
            };
        }
        let s = scanner::scan_memary(&self.config.data_dir);
        let mut detail = format!(
            "memary data dir: {} (stream={}, tallies={}, chat={}, personas={})",
            self.config.data_dir.display(),
            s.stream_entries.len(),
            s.entity_tallies.len(),
            s.chat_messages.len(),
            s.personas.len(),
        );
        if !s.parse_errors.is_empty() {
            detail.push_str(&format!(" — {} parse error(s)", s.parse_errors.len()));
        }
        HealthStatus { ok: true, detail }
    }
}

fn collect_raws(cfg: &MemaryConfig, opts: &ScanOpts) -> Vec<RawRecord> {
    let scan = scanner::scan_memary(&cfg.data_dir);
    let mut out = Vec::with_capacity(scan.total());
    for e in &scan.stream_entries {
        let ts = e.date.as_deref().and_then(scanner::parse_memary_time);
        if passes_since(ts, opts) {
            out.push(normalizer::raw_from_stream(e, cfg.instance.as_deref()));
        }
    }
    for t in &scan.entity_tallies {
        let ts = t.date.as_deref().and_then(scanner::parse_memary_time);
        if passes_since(ts, opts) {
            out.push(normalizer::raw_from_tally(t, cfg.instance.as_deref()));
        }
    }
    // Chat messages don't carry per-row timestamps in past_chat.json — pass
    // them all through; the since-filter would just drop them silently.
    for c in &scan.chat_messages {
        if passes_since(None, opts) {
            out.push(normalizer::raw_from_chat(c, cfg.instance.as_deref()));
        }
    }
    for p in &scan.personas {
        if passes_since(p.mtime_unix, opts) {
            out.push(normalizer::raw_from_persona(p, cfg.instance.as_deref()));
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
pub fn memary_adapter(data_dir: impl Into<PathBuf>, instance: Option<&str>) -> MemaryAdapter {
    MemaryAdapter::new(MemaryConfig {
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

    static MEMARY_LIB_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp_dir() -> PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = MEMARY_LIB_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "anamnesis-memary-{n}-{pid}-{seq}",
            pid = std::process::id()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn seed(dir: &std::path::Path) {
        fs::write(
            dir.join("memory_stream.json"),
            r#"[{"entity":"Alice","date":"2026-05-01T10:00:00"}]"#,
        )
        .unwrap();
        fs::write(
            dir.join("entity_knowledge_store.json"),
            r#"[{"entity":"Alice","count":3,"date":"2026-05-01T10:05:00"}]"#,
        )
        .unwrap();
        fs::write(
            dir.join("past_chat.json"),
            r#"[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]"#,
        )
        .unwrap();
        fs::write(dir.join("system_persona.txt"), "I am helpful").unwrap();
        fs::write(dir.join("user_persona.txt"), "i prefer rust").unwrap();
    }

    #[tokio::test]
    async fn descriptor_carries_instance() {
        let a = memary_adapter("/tmp/x", Some("laptop"));
        let d = a.descriptor();
        assert_eq!(d.adapter, "memary");
        assert_eq!(d.instance.as_deref(), Some("laptop"));
    }

    #[tokio::test]
    async fn health_false_when_dir_missing() {
        let a = memary_adapter("/tmp/never-here-memary", None);
        let h = a.health().await;
        assert!(!h.ok);
    }

    #[tokio::test]
    async fn scan_yields_expected_kinds() {
        let dir = tmp_dir();
        seed(&dir);
        let a = memary_adapter(&dir, Some("local"));
        let raws: Vec<_> = a
            .scan(ScanOpts::default())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        // 1 stream + 1 tally + 2 chat + 2 personas = 6
        assert_eq!(raws.len(), 6);
        let mut kinds = std::collections::HashSet::new();
        for raw in raws {
            for r in a.normalize(raw).unwrap() {
                kinds.insert(r.kind);
            }
        }
        assert!(kinds.contains(&Kind::Reference));
        assert!(kinds.contains(&Kind::Episode));
        assert!(kinds.contains(&Kind::Preference));
    }

    #[tokio::test]
    async fn scan_full_overrides_since() {
        let dir = tmp_dir();
        seed(&dir);
        let a = memary_adapter(&dir, None);
        let cutoff = chrono::DateTime::parse_from_rfc3339("2099-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        // Even without `full`, chat msgs lack per-row dates so they pass —
        // only the dated rows (stream/tally/personas) get dropped.
        let dropped_partial: Vec<_> = a
            .scan(ScanOpts {
                since: Some(cutoff),
                full: false,
            })
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        // chat (2) survive the since filter (None ts → passes); the others drop.
        assert_eq!(dropped_partial.len(), 2);

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
        assert_eq!(full.len(), 6);
    }

    #[tokio::test]
    async fn idempotent_across_scans() {
        let dir = tmp_dir();
        seed(&dir);
        let a = memary_adapter(&dir, Some("laptop"));
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
