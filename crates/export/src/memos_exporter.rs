//! MemOS textual-MemCube exporter. Writes a fresh directory containing
//! `textual_memory.json` — the only file the MemOS adapter scanner reads.
//!
//! Caller MUST ensure `out` doesn't exist (CLI/MCP enforce via
//! `validate_dir_output`); we always create the directory ourselves.
//! Each item carries an `anamnesis_*` provenance block in `metadata`
//! so a future re-import preserves lineage; the `memory_type` round-trips
//! from `metadata.memos_memory_type` when set, otherwise we conservatively
//! map Anamnesis kind/scope back to a MemOS bucket.

use std::path::Path;

use anamnesis_core::model::{AnamnesisRecord, Kind, Scope};
use anamnesis_core::RecordId;
use anamnesis_store::Store;
use serde_json::Value;

use crate::ExportError;

/// Write a fresh `textual_memory.json` MemCube under `out_dir`.
/// Caller must validate `out_dir` doesn't already exist.
pub fn export_memos_dir(store: &Store, ids: &[String], out_dir: &Path) -> Result<(), ExportError> {
    std::fs::create_dir_all(out_dir)?;
    let mut items: Vec<Value> = Vec::with_capacity(ids.len());
    for id in ids {
        let Some(rec) = store.get_record(&RecordId(id.clone()))? else {
            continue;
        };
        let item = textual_item_for(&rec);
        items.push(item);
    }
    let payload = Value::Array(items);
    let mut path = out_dir.to_path_buf();
    path.push("textual_memory.json");
    let body = serde_json::to_string_pretty(&payload)?;
    std::fs::write(&path, body)?;
    Ok(())
}

/// Build one `TextualMemoryItem` from an Anamnesis record. The shape
/// matches `crates/adapter-memos/src/scanner.rs` parser expectations:
/// `id`, `memory`, plus a flat `metadata` map.
fn textual_item_for(rec: &AnamnesisRecord) -> Value {
    let mut metadata = anamnesis_provenance_block(rec);
    // `status` and `memory_type` are the two metadata fields MemOS keys
    // on; everything else is operator-readable.
    metadata.insert("status".into(), Value::String("activated".into()));
    metadata.insert(
        "memory_type".into(),
        Value::String(memos_memory_type_for(rec)),
    );
    metadata.insert(
        "tags".into(),
        Value::Array(rec.tags.iter().map(|t| Value::String(t.clone())).collect()),
    );
    metadata.insert(
        "created_at".into(),
        Value::String(rec.created_at.to_rfc3339()),
    );
    if let Some(ts) = rec.updated_at {
        metadata.insert("updated_at".into(), Value::String(ts.to_rfc3339()));
    }
    serde_json::json!({
        "id":       rec.id.0,
        "memory":   rec.content,
        "metadata": Value::Object(metadata),
    })
}

/// Pick the MemOS `memory_type` bucket. Round-trips the original value
/// when `metadata.memos_memory_type` is present (MemOS-origin records);
/// otherwise conservatively maps Anamnesis kind/scope back to a MemOS bucket
/// reachable by the importer.
fn memos_memory_type_for(rec: &AnamnesisRecord) -> String {
    if let Some(existing) = rec
        .metadata
        .get("memos_memory_type")
        .and_then(|v| v.as_str())
    {
        return existing.to_owned();
    }
    match (&rec.kind, &rec.scope) {
        (Kind::Preference, _) => "PreferenceMemory".into(),
        (Kind::Skill, _) => "SkillMemory".into(),
        (Kind::Episode, _) => "ToolTrajectoryMemory".into(),
        (Kind::Fact, _) => "LongTermMemory".into(),
        (Kind::Reference, Scope::Ephemeral) => "WorkingMemory".into(),
        (Kind::Reference, _) => "OuterMemory".into(),
        (Kind::Feedback, _) => "LongTermMemory".into(),
        (Kind::Unknown, _) => "LongTermMemory".into(),
    }
}

/// `anamnesis_*` provenance block layered onto MemOS metadata.
/// Same convention as the SQLite exporters so re-import preserves lineage.
fn anamnesis_provenance_block(rec: &AnamnesisRecord) -> serde_json::Map<String, Value> {
    let mut meta: serde_json::Map<String, Value> = rec.metadata.clone();
    meta.insert(
        "anamnesis_source_adapter".into(),
        Value::String(rec.source.adapter.clone()),
    );
    if let Some(inst) = &rec.source.instance {
        meta.insert(
            "anamnesis_source_instance".into(),
            Value::String(inst.clone()),
        );
    }
    meta.insert(
        "anamnesis_kind".into(),
        Value::String(format!("{:?}", rec.kind).to_lowercase()),
    );
    meta.insert(
        "anamnesis_scope".into(),
        Value::String(format!("{:?}", rec.scope).to_lowercase()),
    );
    meta.insert("anamnesis_tags".into(), serde_json::json!(rec.tags));
    meta.insert(
        "anamnesis_native_id".into(),
        Value::String(rec.provenance.native_id.clone()),
    );
    meta.insert(
        "anamnesis_raw_hash".into(),
        Value::String(rec.provenance.raw_hash.clone()),
    );
    if let Some(parent) = &rec.provenance.derived_from {
        meta.insert(
            "anamnesis_derived_from".into(),
            Value::String(parent.0.clone()),
        );
    }
    meta
}

#[cfg(test)]
mod tests {
    use super::*;
    use anamnesis_core::model::{
        AnamnesisRecord, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use chrono::Utc;

    fn rec(
        adapter: &str,
        native: &str,
        content: &str,
        kind: Kind,
        scope: Scope,
    ) -> AnamnesisRecord {
        AnamnesisRecord {
            id: RecordId::from_parts(adapter, None, native),
            source: SourceDescriptor {
                adapter: adapter.into(),
                instance: None,
                version: "0".into(),
            },
            content: content.into(),
            embedding: None,
            scope,
            kind,
            created_at: Utc::now(),
            updated_at: None,
            tags: vec!["unit-test".into()],
            metadata: Default::default(),
            provenance: Provenance {
                native_id: native.into(),
                native_path: None,
                captured_at: Utc::now(),
                raw_hash: "raw-hash-secret".into(),
                derived_from: None,
            },
            schema_version: SCHEMA_VERSION,
        }
    }

    #[test]
    fn memos_memory_type_round_trips_when_metadata_carries_origin() {
        let mut r = rec("memos", "n1", "x", Kind::Fact, Scope::User);
        r.metadata.insert(
            "memos_memory_type".into(),
            Value::String("WorkingMemory".into()),
        );
        assert_eq!(memos_memory_type_for(&r), "WorkingMemory");
    }

    #[test]
    fn memos_memory_type_falls_back_to_kind_scope_mapping() {
        assert_eq!(
            memos_memory_type_for(&rec("mem0", "n1", "x", Kind::Preference, Scope::User)),
            "PreferenceMemory"
        );
        assert_eq!(
            memos_memory_type_for(&rec("mem0", "n1", "x", Kind::Skill, Scope::Project)),
            "SkillMemory"
        );
        assert_eq!(
            memos_memory_type_for(&rec("mem0", "n1", "x", Kind::Reference, Scope::Ephemeral)),
            "WorkingMemory"
        );
        assert_eq!(
            memos_memory_type_for(&rec("mem0", "n1", "x", Kind::Reference, Scope::User)),
            "OuterMemory"
        );
        assert_eq!(
            memos_memory_type_for(&rec("mem0", "n1", "x", Kind::Fact, Scope::User)),
            "LongTermMemory"
        );
    }

    #[test]
    fn textual_item_carries_status_and_provenance_block() {
        let r = rec(
            "claude-code",
            "alpha",
            "memory body",
            Kind::Fact,
            Scope::User,
        );
        let item = textual_item_for(&r);
        assert_eq!(item["id"], r.id.0);
        assert_eq!(item["memory"], "memory body");
        let meta = &item["metadata"];
        assert_eq!(meta["status"], "activated");
        assert_eq!(meta["memory_type"], "LongTermMemory");
        assert_eq!(meta["anamnesis_source_adapter"], "claude-code");
        assert_eq!(meta["anamnesis_kind"], "fact");
        assert_eq!(meta["anamnesis_native_id"], "alpha");
        assert_eq!(meta["anamnesis_raw_hash"], "raw-hash-secret");
    }
}
