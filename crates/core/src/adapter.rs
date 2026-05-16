//! The `MemoryAdapter` trait every source connector implements.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::model::{AnamnesisRecord, SourceDescriptor};

/// Options passed to `MemoryAdapter::scan`.
#[derive(Debug, Clone, Default)]
pub struct ScanOpts {
    /// If set, only records modified since this time are returned.
    pub since: Option<DateTime<Utc>>,
    /// If true, skip the dedup hash check and re-emit everything.
    pub full: bool,
}

/// Options passed to `MemoryAdapter::watch`.
#[derive(Debug, Clone, Default)]
pub struct WatchOpts {
    /// Polling interval, if the adapter falls back to polling.
    pub poll_interval: Option<std::time::Duration>,
}

/// A raw record as produced by an adapter scan, before normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawRecord {
    /// Source-native id.
    pub native_id: String,
    /// Optional path or DB reference.
    pub native_path: Option<String>,
    /// Opaque adapter-specific payload.
    pub payload: serde_json::Value,
    /// When the record was captured.
    pub captured_at: DateTime<Utc>,
}

/// A change event surfaced by `MemoryAdapter::watch`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RawDelta {
    /// A new or updated raw record.
    Upsert(RawRecord),
    /// A record was removed from the source.
    Delete {
        /// Native id of the removed record.
        native_id: String,
    },
}

/// Adapter health status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Is the adapter usable right now?
    pub ok: bool,
    /// Human-readable detail.
    pub detail: String,
}

/// The contract every memory source connector implements.
#[async_trait]
pub trait MemoryAdapter: Send + Sync {
    /// Self-describing metadata.
    fn descriptor(&self) -> SourceDescriptor;

    /// Stream raw records from the source.
    fn scan<'a>(&'a self, opts: ScanOpts) -> BoxStream<'a, Result<RawRecord>>;

    /// Optional incremental watcher. Default: not supported.
    fn watch<'a>(&'a self, _opts: WatchOpts) -> Option<BoxStream<'a, Result<RawDelta>>> {
        None
    }

    /// Normalize a raw record into one or more canonical records.
    ///
    /// Returning multiple records is allowed when one source row maps to
    /// several memories (e.g. one conversation → multiple extracted facts).
    fn normalize(&self, raw: RawRecord) -> Result<Vec<AnamnesisRecord>>;

    /// Cheap credential / path / connectivity check.
    async fn health(&self) -> HealthStatus;
}
