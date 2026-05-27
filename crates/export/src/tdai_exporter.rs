//! R154: `tdai-dir` round-trip exporter. Writes a fresh directory with a
//! single `anamnesis_facts.jsonl` — one JSON envelope per record. TDAI's
//! scanner reads each line as an L1 fact; the normalizer recognises the
//! `anamnesis_native_id` sentinel and restores identity + provenance.

use std::io::Write;
use std::path::Path;

use anamnesis_core::model::RecordId;
use anamnesis_store::Store;
use serde_json::Value;

use crate::sqlite_exporters::anamnesis_provenance_block;
use crate::ExportError;

/// File written inside the `tdai-dir` output directory.
pub const TDAI_FACTS_FILE: &str = "anamnesis_facts.jsonl";

/// Write `<out>/anamnesis_facts.jsonl`, one Anamnesis envelope per record.
/// `content` is a top-level field (the L1 fact text); the provenance block
/// rides alongside so re-import restores the original `anamnesis_native_id`.
pub fn export_tdai_dir(store: &Store, ids: &[String], out: &Path) -> Result<(), ExportError> {
    std::fs::create_dir_all(out)?;
    let mut file = std::fs::File::create(out.join(TDAI_FACTS_FILE))?;
    for id in ids {
        let Some(rec) = store.get_record(&RecordId(id.clone()))? else {
            continue;
        };
        let mut obj = anamnesis_provenance_block(&rec);
        obj.insert("content".into(), Value::String(rec.content.clone()));
        writeln!(file, "{}", serde_json::to_string(&Value::Object(obj))?)?;
    }
    Ok(())
}
