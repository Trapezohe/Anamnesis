//! Text-format exporters (`jsonl`, `csv`).
//!
//! Lifted from `crates/cli/src/main.rs` so the same logic powers
//! both the CLI's `anamnesis export` and the MCP `export_memories`
//! tool. No behaviour change vs the original CLI implementation —
//! only the location of the code.

use std::io::Write;

use anamnesis_core::RecordId;
use anamnesis_store::Store;

use crate::ExportError;

/// One [`anamnesis_core::AnamnesisRecord`] per line, JSON-encoded.
/// Includes the full record (content, metadata, provenance, tags,
/// etc.) — the most lossless format we expose. Records that vanish
/// between `select_record_ids` and read time are silently skipped.
pub fn export_jsonl(
    store: &Store,
    ids: &[String],
    writer: &mut dyn Write,
) -> Result<(), ExportError> {
    for id in ids {
        if let Some(rec) = store.get_record(&RecordId(id.clone()))? {
            let line = serde_json::to_string(&rec)?;
            writeln!(writer, "{line}")?;
        }
    }
    Ok(())
}

/// Flat tabular dump of operator-decision-ready columns. Header
/// matches the original CLI shape so any existing script keeps
/// working unchanged.
pub fn export_csv(
    store: &Store,
    ids: &[String],
    writer: &mut dyn Write,
) -> Result<(), ExportError> {
    writeln!(
        writer,
        "id,adapter,instance,kind,scope,created_at,native_id,native_path,content"
    )?;
    for id in ids {
        if let Some(rec) = store.get_record(&RecordId(id.clone()))? {
            let row = format!(
                "{id},{adapter},{instance},{kind},{scope},{created},{nid},{npath},{content}",
                id = csv_field(&rec.id.0),
                adapter = csv_field(&rec.source.adapter),
                instance = csv_field(rec.source.instance.as_deref().unwrap_or("")),
                kind = csv_field(&format!("{:?}", rec.kind).to_lowercase()),
                scope = csv_field(&format!("{:?}", rec.scope).to_lowercase()),
                created = rec.created_at.timestamp(),
                nid = csv_field(&rec.provenance.native_id),
                npath = csv_field(rec.provenance.native_path.as_deref().unwrap_or("")),
                content = csv_field(&rec.content),
            );
            writeln!(writer, "{row}")?;
        }
    }
    Ok(())
}

/// Simple RFC-4180-ish escaping: quote + double-inner-quote when
/// the field contains a comma, quote, or newline. Same rule the
/// CLI used pre-R140.
fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        let escaped = s.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        s.to_string()
    }
}
