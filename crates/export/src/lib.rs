//! Round 140 (PR-78bi): shared exporters for Anamnesis records.
//!
//! R138-R139 introduced `mem0-sqlite` and `letta-sqlite` exporters
//! inline in the CLI. R140 lifts the four supported formats
//! (`jsonl`, `csv`, `mem0-sqlite`, `letta-sqlite`) into a workspace
//! crate so both the CLI's `anamnesis export` and the new
//! admin-gated MCP `export_memories` tool share one implementation.
//!
//! Single source of truth for:
//! - the format catalogue (`ExportFormat`),
//! - the filter shape (`ExportFilter`),
//! - the round-trip metadata convention (Anamnesis provenance keys
//!   on every SQLite export so re-imports preserve lineage).
//!
//! Privacy / safety inherits from the CLI contract this code came
//! from: no `cargo bench`-style work, no live `~/.mem0/history.db`
//! overwrite (callers MUST guard `out.exists()` at the entry point —
//! see `ExportError::OutputAlreadyExists`).

#![deny(unsafe_code)]
#![warn(missing_docs)]

use std::path::{Path, PathBuf};

use anamnesis_core::RecordId;
use anamnesis_store::Store;
use thiserror::Error;

/// Wire-shape format identifier shared by CLI / MCP entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// One [`anamnesis_core::AnamnesisRecord`] per line, JSON-encoded.
    Jsonl,
    /// Flat tabular dump of the operator-decision-ready columns.
    Csv,
    /// Fresh SQLite DB with mem0's canonical `memories` table
    /// (R138 PR-78bg). Round-trip friendly: provenance keys live in
    /// `metadata` JSON under `anamnesis_*`.
    Mem0Sqlite,
    /// Fresh SQLite DB with Letta's canonical `block` table (R139
    /// PR-78bh). Letta-origin records reconstruct their native
    /// `label` / `description` / `template_name` from metadata;
    /// foreign-origin gets a stable `anamnesis/<adapter>` label.
    LettaSqlite,
}

impl ExportFormat {
    /// Parse the operator-typed format token. Stable wire vocabulary.
    pub fn parse(token: &str) -> Result<Self, ExportError> {
        match token {
            "jsonl" => Ok(Self::Jsonl),
            "csv" => Ok(Self::Csv),
            "mem0-sqlite" => Ok(Self::Mem0Sqlite),
            "letta-sqlite" => Ok(Self::LettaSqlite),
            other => Err(ExportError::UnknownFormat(other.to_owned())),
        }
    }

    /// Inverse of [`Self::parse`] — for echoing back in audit logs
    /// and MCP responses.
    pub fn as_token(self) -> &'static str {
        match self {
            Self::Jsonl => "jsonl",
            Self::Csv => "csv",
            Self::Mem0Sqlite => "mem0-sqlite",
            Self::LettaSqlite => "letta-sqlite",
        }
    }

    /// SQLite-output formats MUST materialise to a path; they can't
    /// stream to stdout (rusqlite needs a path) and a typo would
    /// otherwise risk clobbering the operator's real upstream DB.
    pub fn requires_out_path(self) -> bool {
        matches!(self, Self::Mem0Sqlite | Self::LettaSqlite)
    }
}

/// Filter for [`select_record_ids`]. All fields are optional; an
/// empty filter selects every live record in the store.
///
/// `source` / `instance` use the existing R104/R115 comma-separated
/// OR grammar (parsed via `anamnesis_core::parse_csv_filter`).
/// `kind` is a single Kind discriminator (`fact` / `preference` /
/// `episode` / `feedback` / `skill` / `reference` / `unknown`).
#[derive(Debug, Clone, Default)]
pub struct ExportFilter {
    /// Adapter id or comma-separated OR list. `None` matches all.
    pub source: Option<String>,
    /// Instance discriminator or comma-separated OR list. `None`
    /// matches all instances.
    pub instance: Option<String>,
    /// Single `Kind` token. `None` matches all kinds.
    pub kind: Option<String>,
}

/// Errors the exporter surface emits. CLI / MCP wrap these into
/// their own user-facing error messages.
#[derive(Debug, Error)]
pub enum ExportError {
    /// Format token didn't match the known set.
    #[error("unsupported format: {0} (try jsonl, csv, mem0-sqlite, or letta-sqlite)")]
    UnknownFormat(String),
    /// SQLite-output format was asked for without `--out`.
    #[error("--format {format} requires --out <path>: SQLite output cannot stream to stdout")]
    OutPathRequired {
        /// The format token that requires `--out`.
        format: &'static str,
    },
    /// The target file already exists. We refuse to overwrite so
    /// a typo can't clobber an upstream `~/.mem0/history.db` or
    /// `~/.letta/letta.db`.
    #[error("refusing to overwrite existing file {0}; pick a fresh --out path")]
    OutputAlreadyExists(PathBuf),
    /// Anamnesis store I/O failed.
    #[error("store: {0}")]
    Store(#[from] anamnesis_store::StoreError),
    /// SQLite write failed.
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// File system write failed.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialisation failed.
    #[error("serde: {0}")]
    Json(#[from] serde_json::Error),
}

/// Outcome of an export call — bounded metadata the caller can
/// echo to the operator / MCP client / audit log.
#[derive(Debug, Clone)]
pub struct ExportOutcome {
    /// Format that was written.
    pub format: ExportFormat,
    /// Output path (when one was written) — `None` for `jsonl` /
    /// `csv` writing to stdout.
    pub out: Option<PathBuf>,
    /// Number of records successfully written.
    pub records: u64,
    /// Output file size in bytes. `None` for stdout sinks.
    pub bytes: Option<u64>,
}

/// Resolve the set of `RecordId`s an export call should serialise,
/// honouring source/instance/kind filters. Returns ids ordered by
/// `created_at ASC` (then `id ASC` as a stable tiebreaker), the
/// same convention `cmd_export` has used since the original R0
/// shape.
///
/// Releases the store's `parking_lot` connection guard before
/// returning — callers must NOT hold an outer guard while invoking
/// downstream `store.get_record()` calls (the underlying
/// `parking_lot::Mutex` is not re-entrant).
pub fn select_record_ids(store: &Store, filter: &ExportFilter) -> Result<Vec<String>, ExportError> {
    let sources = anamnesis_core::parse_csv_filter(filter.source.as_deref());
    let instances = anamnesis_core::parse_csv_filter(filter.instance.as_deref());
    let kind = filter.kind.as_deref().map(str::to_owned);

    let mut where_parts: Vec<String> = Vec::new();
    let mut params: Vec<rusqlite::types::Value> = Vec::new();
    if !sources.is_empty() {
        let placeholders = vec!["?"; sources.len()].join(", ");
        where_parts.push(format!("adapter IN ({placeholders})"));
        for s in &sources {
            params.push(rusqlite::types::Value::Text(s.clone()));
        }
    }
    if !instances.is_empty() {
        let placeholders = vec!["?"; instances.len()].join(", ");
        where_parts.push(format!("instance IN ({placeholders})"));
        for i in &instances {
            params.push(rusqlite::types::Value::Text(i.clone()));
        }
    }
    if let Some(k) = kind {
        where_parts.push("kind = ?".to_string());
        params.push(rusqlite::types::Value::Text(k));
    }
    let where_clause = if where_parts.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", where_parts.join(" AND "))
    };
    let sql = format!("SELECT id FROM records {where_clause} ORDER BY created_at ASC, id ASC");

    let conn = store.conn();
    let mut stmt = conn.prepare(&sql)?;
    let collected: rusqlite::Result<Vec<String>> = stmt
        .query_map(rusqlite::params_from_iter(params), |r| {
            r.get::<_, String>(0)
        })?
        .collect();
    Ok(collected?)
}

/// Validate the SQLite-output safety contract: an `out` path is
/// supplied and doesn't already exist. CLI and MCP entry points
/// must call this BEFORE doing any work so the operator sees
/// the failure mode before partial state lands on disk.
pub fn validate_sqlite_output(
    format: ExportFormat,
    out: Option<&Path>,
) -> Result<&Path, ExportError> {
    let Some(p) = out else {
        return Err(ExportError::OutPathRequired {
            format: format.as_token(),
        });
    };
    if p.exists() {
        return Err(ExportError::OutputAlreadyExists(p.to_path_buf()));
    }
    Ok(p)
}

/// Render the export's [`ExportFilter`] as a one-line summary for
/// audit-log / MCP-response consumption. Sample:
/// `"source filter: mem0,letta; instance filter: prod; kind filter: fact"`.
/// Empty fields render as `"all sources"` / etc.
pub fn render_filter_summary(filter: &ExportFilter) -> String {
    let src = filter
        .source
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(|s| format!("source filter: {s}"))
        .unwrap_or_else(|| "source filter: all sources".into());
    let inst = filter
        .instance
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(|s| format!("instance filter: {s}"))
        .unwrap_or_else(|| "instance filter: all instances".into());
    let kind = filter
        .kind
        .as_deref()
        .filter(|s| !s.is_empty())
        .map(|s| format!("kind filter: {s}"))
        .unwrap_or_else(|| "kind filter: all kinds".into());
    format!("{src}; {inst}; {kind}")
}

mod sqlite_exporters;
mod text_exporters;

pub use sqlite_exporters::{export_letta_sqlite, export_mem0_sqlite};
pub use text_exporters::{export_csv, export_jsonl};

/// High-level entry point: select records by filter, then route to
/// the format-specific writer. CLI / MCP both call this.
///
/// Returns an [`ExportOutcome`] with bounded metadata (no record
/// content). When `format.requires_out_path()` is true, this fn
/// enforces the SQLite-output safety contract; for `jsonl` / `csv`
/// the caller supplies its own `writer` (stdout for CLI default,
/// fresh file for `--out`).
pub fn run_export(
    store: &Store,
    filter: &ExportFilter,
    format: ExportFormat,
    out: Option<&Path>,
    writer: Option<&mut dyn std::io::Write>,
) -> Result<ExportOutcome, ExportError> {
    let ids = select_record_ids(store, filter)?;
    match format {
        ExportFormat::Jsonl | ExportFormat::Csv => {
            // We accept either an explicit `writer` (CLI stdout sink)
            // or an `out` path (CLI / MCP `--out` file). Materialise
            // the file into a local that outlives the borrow so the
            // dyn-Write reference stays valid for the whole call.
            let mut owned_file: std::fs::File;
            let writer_ref: &mut dyn std::io::Write = if let Some(w) = writer {
                w
            } else if let Some(p) = out {
                owned_file = std::fs::File::create(p)?;
                &mut owned_file
            } else {
                // No writer, no out — caller wanted us to materialise
                // nothing? That's a misuse. CLI / MCP normalise this
                // before calling us, but assert with a clear error.
                return Err(ExportError::OutPathRequired {
                    format: format.as_token(),
                });
            };
            match format {
                ExportFormat::Jsonl => export_jsonl(store, &ids, writer_ref)?,
                ExportFormat::Csv => export_csv(store, &ids, writer_ref)?,
                _ => unreachable!(),
            }
            let bytes = out.and_then(|p| std::fs::metadata(p).ok().map(|m| m.len()));
            Ok(ExportOutcome {
                format,
                out: out.map(Path::to_path_buf),
                records: ids.len() as u64,
                bytes,
            })
        }
        ExportFormat::Mem0Sqlite | ExportFormat::LettaSqlite => {
            let p = validate_sqlite_output(format, out)?;
            match format {
                ExportFormat::Mem0Sqlite => export_mem0_sqlite(store, &ids, p)?,
                ExportFormat::LettaSqlite => export_letta_sqlite(store, &ids, p)?,
                _ => unreachable!(),
            }
            let bytes = std::fs::metadata(p).ok().map(|m| m.len());
            Ok(ExportOutcome {
                format,
                out: Some(p.to_path_buf()),
                records: ids.len() as u64,
                bytes,
            })
        }
    }
}

/// Helper for callers that just want the ids; same logic as
/// `select_record_ids` but kept as a separate symbol for clarity.
#[doc(hidden)]
pub fn _select_record_ids(
    store: &Store,
    filter: &ExportFilter,
) -> Result<Vec<RecordId>, ExportError> {
    Ok(select_record_ids(store, filter)?
        .into_iter()
        .map(RecordId)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_parse_known_tokens() {
        for (token, expected) in [
            ("jsonl", ExportFormat::Jsonl),
            ("csv", ExportFormat::Csv),
            ("mem0-sqlite", ExportFormat::Mem0Sqlite),
            ("letta-sqlite", ExportFormat::LettaSqlite),
        ] {
            assert_eq!(ExportFormat::parse(token).unwrap(), expected);
            assert_eq!(expected.as_token(), token);
        }
    }

    #[test]
    fn format_parse_rejects_unknown() {
        let err = ExportFormat::parse("yaml").unwrap_err();
        assert!(matches!(err, ExportError::UnknownFormat(_)));
    }

    #[test]
    fn requires_out_path_only_for_sqlite_formats() {
        assert!(!ExportFormat::Jsonl.requires_out_path());
        assert!(!ExportFormat::Csv.requires_out_path());
        assert!(ExportFormat::Mem0Sqlite.requires_out_path());
        assert!(ExportFormat::LettaSqlite.requires_out_path());
    }

    #[test]
    fn validate_sqlite_output_rejects_missing_path() {
        let err = validate_sqlite_output(ExportFormat::Mem0Sqlite, None).unwrap_err();
        assert!(matches!(err, ExportError::OutPathRequired { .. }));
    }

    #[test]
    fn validate_sqlite_output_rejects_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("existing.sqlite");
        std::fs::write(&p, b"x").unwrap();
        let err = validate_sqlite_output(ExportFormat::LettaSqlite, Some(p.as_path())).unwrap_err();
        assert!(matches!(err, ExportError::OutputAlreadyExists(_)));
    }

    #[test]
    fn render_filter_summary_handles_defaults() {
        let summary = render_filter_summary(&ExportFilter::default());
        assert!(summary.contains("all sources"));
        assert!(summary.contains("all instances"));
        assert!(summary.contains("all kinds"));
    }

    #[test]
    fn render_filter_summary_echoes_filter_tokens() {
        let summary = render_filter_summary(&ExportFilter {
            source: Some("mem0,letta".into()),
            instance: Some("prod".into()),
            kind: Some("fact".into()),
        });
        assert!(summary.contains("source filter: mem0,letta"));
        assert!(summary.contains("instance filter: prod"));
        assert!(summary.contains("kind filter: fact"));
    }
}
