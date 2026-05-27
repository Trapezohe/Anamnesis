//! R155: `claude-code-dir` round-trip exporter. Writes a fresh projects
//! root with `anamnesis-export/memory/<slug>.md` files — markdown whose
//! frontmatter carries the provenance block. The claude-code scanner reads
//! them as memory files; the normalizer restores identity from the
//! `anamnesis_native_id` frontmatter key. `content` is the markdown body, so
//! it round-trips exactly; `kind`/`scope` are best-effort via `type`.

use std::io::Write;
use std::path::Path;

use anamnesis_core::model::{AnamnesisRecord, Kind, RecordId, Scope};
use anamnesis_store::Store;

use crate::ExportError;

/// Synthetic project directory created under the output root.
pub const CLAUDE_CODE_PROJECT_DIR: &str = "anamnesis-export";

/// Best-effort inverse of `map_memory_type`: a frontmatter `type` token for
/// a record's kind/scope, or `None` to omit `type` (→ Unknown/Ephemeral).
fn memory_type_token(rec: &AnamnesisRecord) -> Option<&'static str> {
    match (rec.kind, rec.scope) {
        (Kind::Feedback, _) => Some("feedback"),
        (Kind::Reference, _) => Some("reference"),
        (Kind::Preference, _) => Some("preference"),
        (Kind::Skill, _) => Some("skill"),
        (Kind::Fact, Scope::Project) => Some("project"),
        (Kind::Fact, _) => Some("user"),
        _ => None,
    }
}

/// Filesystem-safe slug from `name` metadata / first tag / record id.
fn slug(rec: &AnamnesisRecord) -> String {
    let base = rec
        .metadata
        .get("name")
        .and_then(|v| v.as_str())
        .or_else(|| rec.tags.first().map(String::as_str))
        .unwrap_or(&rec.id.0);
    let cleaned: String = base
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let cleaned = cleaned.trim_matches('-');
    let stem = if cleaned.is_empty() {
        "memory"
    } else {
        cleaned
    };
    // 128-bit blake3 of the full record id — collision-resistant enough that
    // distinct records get distinct filenames in practice.
    let hash = blake3::hash(rec.id.0.as_bytes()).to_hex();
    format!("{stem}-{}", &hash[..32])
}

/// One frontmatter scalar: collapse newlines so it stays single-line.
fn scalar(value: &str) -> String {
    value.replace(['\n', '\r'], " ")
}

/// Write `<out>/anamnesis-export/memory/<slug>.md` per record.
pub fn export_claude_code_dir(
    store: &Store,
    ids: &[String],
    out: &Path,
) -> Result<(), ExportError> {
    let memory_dir = out.join(CLAUDE_CODE_PROJECT_DIR).join("memory");
    std::fs::create_dir_all(&memory_dir)?;
    for id in ids {
        let Some(rec) = store.get_record(&RecordId(id.clone()))? else {
            continue;
        };
        let mut fm = String::from("---\n");
        if let Some(name) = rec.metadata.get("name").and_then(|v| v.as_str()) {
            fm.push_str(&format!("name: {}\n", scalar(name)));
        }
        if let Some(desc) = rec.metadata.get("description").and_then(|v| v.as_str()) {
            fm.push_str(&format!("description: {}\n", scalar(desc)));
        }
        if let Some(t) = memory_type_token(&rec) {
            fm.push_str(&format!("metadata:\n  type: {t}\n"));
        }
        fm.push_str(&format!(
            "anamnesis_native_id: {}\n",
            scalar(&rec.provenance.native_id)
        ));
        fm.push_str(&format!(
            "anamnesis_source_adapter: {}\n",
            scalar(&rec.source.adapter)
        ));
        if let Some(inst) = &rec.source.instance {
            fm.push_str(&format!("anamnesis_source_instance: {}\n", scalar(inst)));
        }
        fm.push_str(&format!(
            "anamnesis_kind: {}\n",
            format!("{:?}", rec.kind).to_lowercase()
        ));
        fm.push_str(&format!(
            "anamnesis_scope: {}\n",
            format!("{:?}", rec.scope).to_lowercase()
        ));
        fm.push_str(&format!(
            "anamnesis_raw_hash: {}\n",
            scalar(&rec.provenance.raw_hash)
        ));
        if let Some(parent) = &rec.provenance.derived_from {
            fm.push_str(&format!("anamnesis_derived_from: {}\n", scalar(&parent.0)));
        }
        if !rec.tags.is_empty() {
            fm.push_str(&format!(
                "anamnesis_tags: {}\n",
                scalar(&serde_json::to_string(&rec.tags)?)
            ));
        }
        fm.push_str("---\n");

        // Body follows the closing fence directly (no extra newlines) so the
        // normalizer's sentinel path recovers `content` byte-exactly.
        let mut file = std::fs::File::create(memory_dir.join(format!("{}.md", slug(&rec))))?;
        write!(file, "{fm}{}", rec.content)?;
    }
    Ok(())
}
