//! Crate-wide error type. Adapter and store crates re-use or wrap this.

use thiserror::Error;

/// Top-level error type for Anamnesis core.
#[derive(Debug, Error)]
pub enum Error {
    /// A record failed schema validation during normalization.
    #[error("invalid record: {0}")]
    InvalidRecord(String),

    /// Adapter ran into an unrecoverable problem while scanning.
    #[error("adapter error ({adapter}): {message}")]
    Adapter {
        /// Adapter identifier (e.g. "claude-code").
        adapter: String,
        /// Human-readable message.
        message: String,
    },

    /// Schema version mismatch that cannot be auto-migrated.
    #[error("schema version mismatch: found {found}, expected {expected}")]
    SchemaVersion {
        /// Version found on disk / in the input.
        found: u32,
        /// Version this binary supports.
        expected: u32,
    },

    /// IO error (file, network, etc).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// Serde JSON error.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    /// Catch-all for adapter-specific or downstream errors.
    #[error("{0}")]
    Other(String),
}

/// Convenience `Result` alias.
pub type Result<T> = std::result::Result<T, Error>;
