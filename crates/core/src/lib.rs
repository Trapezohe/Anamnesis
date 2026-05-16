//! Anamnesis core: domain types, adapter trait, and query model.
//!
//! This crate has **no IO**. It defines the contract every other crate
//! (`store`, `cli`, `mcp-server`, `adapter-*`) implements or consumes.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod adapter;
pub mod embedding;
pub mod error;
pub mod model;
pub mod query;

pub use adapter::{HealthStatus, MemoryAdapter, RawDelta, RawRecord, ScanOpts, WatchOpts};
pub use embedding::{EmbeddingProvider, EmbeddingTask, ModelId};
pub use error::{Error, Result};
pub use model::{
    AnamnesisRecord, Embedding, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
pub use query::{Query, SearchMode, TimeRange};
