//! Query model used by both CLI and MCP server.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::model::{Kind, Scope};

/// Round 103 (PR-78y): shared splitter for any CLI/MCP filter
/// arg that takes either a single value (`--source mem0`) or a
/// comma-separated OR list (`--source mem0,claude-code`).
///
/// Rules — same as R102's `parse_audit_actions` (which now
/// delegates here) so every CSV filter on every surface behaves
/// the same way:
///   * `None` or `Some("")` → `vec![]` (no filter).
///   * Split on `,`, trim each token, drop empties — so
///     `"mem0, , claude-code ,"` becomes `["mem0",
///     "claude-code"]`.
///   * Tokens are case-sensitive — adapter ids, action names,
///     etc. are stored verbatim in the store/audit log, so a
///     typo like `Mem0` should not be silently widened.
///   * Order is preserved but otherwise irrelevant — the
///     downstream filter is OR.
///
/// Living in core keeps CLI + MCP byte-identical: both feed
/// their raw arg through this helper and build their filter
/// `Vec<String>` from the result.
pub fn parse_csv_filter(spec: Option<&str>) -> Vec<String> {
    spec.unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_owned)
        .collect()
}

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

#[cfg(test)]
mod tests {
    use super::*;

    /// `parse_csv_filter` is the single source of truth for
    /// every CSV-style filter token list (R102 actions, R103
    /// source ids, future surfaces). Same contract as R102's
    /// `parse_audit_actions`: empty input is no filter; commas
    /// split; whitespace trims; empty tokens drop; case
    /// preserved exactly.
    #[test]
    fn parse_csv_filter_normalises_comma_separated_input() {
        assert_eq!(parse_csv_filter(None), Vec::<String>::new());
        assert_eq!(parse_csv_filter(Some("")), Vec::<String>::new());
        assert_eq!(
            parse_csv_filter(Some("mem0")),
            vec!["mem0".to_string()],
            "single value still parses"
        );
        assert_eq!(
            parse_csv_filter(Some("mem0,claude-code")),
            vec!["mem0".to_string(), "claude-code".to_string()]
        );
        assert_eq!(
            parse_csv_filter(Some(" mem0 , , claude-code ,")),
            vec!["mem0".to_string(), "claude-code".to_string()],
            "trim whitespace, drop empty tokens"
        );
        // Case-sensitive — typo'd `Mem0` ≠ stored `mem0` so a
        // mis-cased token returns no rows instead of silently
        // matching everything.
        assert_eq!(
            parse_csv_filter(Some("Mem0")),
            vec!["Mem0".to_string()],
            "case is preserved exactly"
        );
        // Order is preserved (callers asserting order can
        // expect it), even though downstream filters are OR.
        assert_eq!(
            parse_csv_filter(Some("c,a,b")),
            vec!["c".to_string(), "a".to_string(), "b".to_string()]
        );
    }
}
