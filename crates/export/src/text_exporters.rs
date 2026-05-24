//! Text-format exporters (`jsonl`, `csv`). Shared by CLI + MCP.

use std::io::Write;

use anamnesis_core::RecordId;
use anamnesis_store::Store;

use crate::ExportError;

/// One full record per JSONL line. Records that vanish between
/// `select_record_ids` and read are skipped.
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

/// Flat tabular dump. Stable header for back-compat with existing scripts.
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

/// RFC-4180-ish: quote + double-inner-quote when field has `,` / `"` / `\n`.
fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        let escaped = s.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        s.to_string()
    }
}
