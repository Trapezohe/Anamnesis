//! Anamnesis embedding providers.
//!
//! Phase-1 scope:
//!   - `EmbeddingProvider` trait re-export from `anamnesis-core`
//!   - Curated 5-model `registry` — `default` / `tiny` / `en` /
//!     `multi-strong` / `cloud-voyage` (data only, zero ML deps)
//!   - Local provider via `fastembed-rs` (Task #4)
//!   - Model cache lives under `$XDG_DATA_HOME/anamnesis/models/`; binary
//!     stays small. See `docs/BLUEPRINT.md §16.7`.
//!
//! The cloud provider (`voyage`) implementation lands in Phase 2 alongside
//! the mem0 adapter.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod registry;
pub mod worker;

#[cfg(feature = "local-fastembed")]
pub mod local;

#[cfg(feature = "local-fastembed")]
pub use local::LocalFastembedProvider;

pub use anamnesis_core::embedding::{EmbeddingProvider, EmbeddingTask, ModelId};
pub use registry::{available, by_key, default_model, local_only, CuratedModel, REGISTRY};
pub use worker::{DrainSummary, EmbeddingWorker};

/// Crate version, exposed for diagnostics and `anamnesis status` output.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_version_is_pinned() {
        assert!(!CRATE_VERSION.is_empty());
        assert!(CRATE_VERSION.starts_with("0."));
    }
}
