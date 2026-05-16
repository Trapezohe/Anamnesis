//! Anamnesis embedding providers.
//!
//! Phase-1 scope:
//!   - `EmbeddingProvider` trait (added in Task #2 via `anamnesis-core`,
//!     re-exported here in Task #4)
//!   - Local provider via `fastembed-rs` (Task #4)
//!   - Curated 5-model registry — `default` / `tiny` / `en` / `multi-strong` /
//!     `cloud-voyage` (Task #3)
//!   - Model cache lives under `$XDG_DATA_HOME/anamnesis/models/`; binary stays
//!     small. See `docs/BLUEPRINT.md §16.7`.
//!
//! The cloud provider (`voyage`) implementation lands in Phase 2 alongside the
//! mem0 adapter.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub use anamnesis_core::embedding::{EmbeddingProvider, EmbeddingTask, ModelId};

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
