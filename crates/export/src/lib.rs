//! Shared exporters (`jsonl`, `csv`, `mem0-sqlite`, `letta-sqlite`).
//!
//! Single source of truth for the format catalogue, filter shape, and the
//! `anamnesis_*` provenance metadata on SQLite exports. Used by the CLI's
//! `anamnesis export` and the MCP `export_memories` tool.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use std::path::{Path, PathBuf};

use anamnesis_core::RecordId;
use anamnesis_store::Store;
use thiserror::Error;

/// Output format token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// One `AnamnesisRecord` per line, JSON-encoded.
    Jsonl,
    /// Flat tabular dump.
    Csv,
    /// Fresh SQLite DB with mem0's `memories` table.
    Mem0Sqlite,
    /// Fresh SQLite DB with Letta's `block` table.
    LettaSqlite,
}

impl ExportFormat {
    /// Parse the operator-typed token.
    pub fn parse(token: &str) -> Result<Self, ExportError> {
        match token {
            "jsonl" => Ok(Self::Jsonl),
            "csv" => Ok(Self::Csv),
            "mem0-sqlite" => Ok(Self::Mem0Sqlite),
            "letta-sqlite" => Ok(Self::LettaSqlite),
            other => Err(ExportError::UnknownFormat(other.to_owned())),
        }
    }

    /// Wire token (inverse of [`Self::parse`]).
    pub fn as_token(self) -> &'static str {
        match self {
            Self::Jsonl => "jsonl",
            Self::Csv => "csv",
            Self::Mem0Sqlite => "mem0-sqlite",
            Self::LettaSqlite => "letta-sqlite",
        }
    }

    /// SQLite formats can't stream → caller MUST supply `out`.
    pub fn requires_out_path(self) -> bool {
        matches!(self, Self::Mem0Sqlite | Self::LettaSqlite)
    }
}

/// Selection filter. `None` everywhere = all live records.
/// `source` / `instance` use the comma-separated OR grammar.
#[derive(Debug, Clone, Default)]
pub struct ExportFilter {
    /// Adapter id or CSV OR list.
    pub source: Option<String>,
    /// Instance id or CSV OR list.
    pub instance: Option<String>,
    /// Single Kind token.
    pub kind: Option<String>,
}

/// Exporter errors. CLI/MCP wrap into user-facing strings.
#[derive(Debug, Error)]
pub enum ExportError {
    /// Format token unknown.
    #[error("unsupported format: {0} (try jsonl, csv, mem0-sqlite, or letta-sqlite)")]
    UnknownFormat(String),
    /// SQLite format requested without `--out`.
    #[error("--format {format} requires --out <path>: SQLite output cannot stream to stdout")]
    OutPathRequired {
        /// The format that requires `--out`.
        format: &'static str,
    },
    /// Target path exists; refuse to overwrite (protects upstream DBs).
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

/// Bounded export metadata — for audit logs and MCP responses.
#[derive(Debug, Clone)]
pub struct ExportOutcome {
    /// Format written.
    pub format: ExportFormat,
    /// Output path, or `None` for stdout.
    pub out: Option<PathBuf>,
    /// Records written.
    pub records: u64,
    /// File size, or `None` for stdout.
    pub bytes: Option<u64>,
}

/// Select matching ids ordered `(created_at ASC, id ASC)`.
///
/// Drops the store connection guard before returning — caller must NOT
/// hold an outer guard while calling downstream `store.get_record()`
/// (parking_lot mutex is not re-entrant).
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

/// Enforce SQLite-output safety: `out` supplied AND doesn't exist.
/// Call BEFORE any work so failures surface before partial state.
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

/// One-line filter summary for audit/MCP. Empty fields render as `all *`.
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

/// Entry point: select + route to format writer. CLI/MCP both call this.
/// SQLite formats enforce `validate_sqlite_output`; text formats accept
/// an explicit `writer` (stdout) OR an `out` path (file).
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
            // Materialise file into a local that outlives the borrow.
            let mut owned_file: std::fs::File;
            let writer_ref: &mut dyn std::io::Write = if let Some(w) = writer {
                w
            } else if let Some(p) = out {
                owned_file = std::fs::File::create(p)?;
                &mut owned_file
            } else {
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

/// `select_record_ids` typed-id variant.
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
