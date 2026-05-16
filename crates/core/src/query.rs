//! Query model used by both CLI and MCP server.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::model::{Kind, Scope};

/// Search backends.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// SQLite FTS5 only.
    Fulltext,
    /// Vector kNN only (requires embeddings).
    Vector,
    /// FTS + vector with reciprocal rank fusion.
    #[default]
    Hybrid,
}

/// Inclusive time range filter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeRange {
    /// Lower bound (inclusive).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<DateTime<Utc>>,
    /// Upper bound (inclusive).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to: Option<DateTime<Utc>>,
}

/// Cross-source query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    /// Free-text query (FTS / vector).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Restrict to a specific adapter (e.g. `"claude-code"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Restrict by kind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<Kind>,
    /// Restrict by scope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<Scope>,
    /// Time window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_range: Option<TimeRange>,
    /// Search mode.
    #[serde(default)]
    pub mode: SearchMode,
    /// Maximum results returned.
    #[serde(default = "default_limit")]
    pub limit: u32,
}

fn default_limit() -> u32 {
    20
}

impl Default for Query {
    fn default() -> Self {
        Self {
            text: None,
            source: None,
            kind: None,
            scope: None,
            time_range: None,
            mode: SearchMode::default(),
            limit: default_limit(),
        }
    }
}
