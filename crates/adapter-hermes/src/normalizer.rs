//! Normalize Hermes raw records into `AnamnesisRecord`s.

use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_core::RawRecord;
use chrono::{TimeZone, Utc};
use serde_json::{json, Value};
use std::path::Path;

use crate::scanner::{HermesMarkdownBlock, HermesSessionRow};
use crate::ADAPTER_ID;

/// Payload-kind discriminator for `MEMORY.md` raw records.
pub const PAYLOAD_KIND_MEMORY_MD: &str = "hermes_memory_md";
/// Payload-kind discriminator for `USER.md` raw records.
pub const PAYLOAD_KIND_USER_MD: &str = "hermes_user_md";
/// Payload-kind discriminator for one SQLite session-row raw record.
pub const PAYLOAD_KIND_SESSION: &str = "hermes_session_row";

/// Build a `RawRecord` from a Hermes markdown block. Kind is decided
/// by the source filename:
///   * `MEMORY.md` → `Reference` (environment state — agent's own
///     working memory of the system).
///   * `USER.md`   → `Preference` (user-stated preferences).
pub fn raw_from_markdown(block: &HermesMarkdownBlock, instance: Option<&str>) -> RawRecord {
    let payload_kind = if block.source_file == "MEMORY.md" {
        PAYLOAD_KIND_MEMORY_MD
    } else {
        PAYLOAD_KIND_USER_MD
    };
    let native_id = synth_md_id(instance, &block.source_file);
    RawRecord {
        native_id,
        native_path: Some(block.path.display().to_string()),
        payload: json!({
            "payload_kind": payload_kind,
            "source_file": block.source_file,
            "path": block.path.display().to_string(),
            "content": block.content,
            "mtime_unix": block.mtime_unix,
        }),
        captured_at: Utc::now(),
    }
}

/// Build a `RawRecord` from one Hermes session row. Kind is always
/// `Episode` — this is conversation log territory.
pub fn raw_from_session(
    db_path: &Path,
    row: &HermesSessionRow,
    instance: Option<&str>,
) -> RawRecord {
    let native_id = synth_session_id(instance, &row.id);
    RawRecord {
        native_id,
        native_path: Some(format!("{}#{}/{}", db_path.display(), row.table, row.id)),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_SESSION,
            "db_path": db_path.display().to_string(),
            "table": row.table,
            "id": row.id,
            "content": row.content,
            "role": row.role,
            "timestamp": row.timestamp,
            "extra": Value::Object(row.extra.clone()),
        }),
        captured_at: Utc::now(),
    }
}

fn synth_md_id(instance: Option<&str>, source_file: &str) -> String {
    let instance = instance.unwrap_or("default");
    format!("{instance}|md|{source_file}")
}

fn synth_session_id(instance: Option<&str>, id: &str) -> String {
    let instance = instance.unwrap_or("default");
    format!("{instance}|session|{id}")
}

/// Normalize one `RawRecord` (any Hermes payload_kind) into an
/// `AnamnesisRecord`. Always yields a 1-element Vec.
pub fn normalize(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let payload_kind = raw
        .payload
        .get("payload_kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("hermes: missing payload_kind".into()))?;
    match payload_kind {
        PAYLOAD_KIND_MEMORY_MD => normalize_md(raw, instance, Kind::Reference, "MEMORY.md"),
        PAYLOAD_KIND_USER_MD => normalize_md(raw, instance, Kind::Preference, "USER.md"),
        PAYLOAD_KIND_SESSION => normalize_session(raw, instance),
        other => Err(Error::InvalidRecord(format!(
            "hermes: unexpected payload_kind {other:?}"
        ))),
    }
}

fn normalize_md(
    raw: RawRecord,
    instance: Option<&str>,
    kind: Kind,
    label: &str,
) -> Result<Vec<AnamnesisRecord>> {
    let content = raw
        .payload
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("hermes: markdown content missing".into()))?
        .to_string();
    let path = raw
        .payload
        .get("path")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let mtime_unix = raw.payload.get("mtime_unix").and_then(|v| v.as_i64());

    let created_at = mtime_unix
        .and_then(|t| Utc.timestamp_opt(t, 0).single())
        .unwrap_or(raw.captured_at);

    let id_local = format!("md|{label}");
    let record_id = RecordId::from_parts(ADAPTER_ID, instance, &id_local);
    let raw_hash = blake3::hash(content.as_bytes()).to_hex().to_string();

    let mut metadata = serde_json::Map::new();
    metadata.insert("hermes_source_file".into(), Value::String(label.into()));

    Ok(vec![AnamnesisRecord {
        id: record_id,
        source: SourceDescriptor {
            adapter: ADAPTER_ID.into(),
            instance: instance.map(str::to_owned),
            version: env!("CARGO_PKG_VERSION").into(),
        },
        content,
        embedding: None,
        scope: Scope::User,
        kind,
        created_at,
        updated_at: None,
        tags: vec![],
        metadata,
        provenance: Provenance {
            native_id: raw.native_id.clone(),
            native_path: path,
            captured_at: raw.captured_at,
            raw_hash,
        },
        schema_version: SCHEMA_VERSION,
    }])
}

fn normalize_session(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let id = raw
        .payload
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("hermes: session row missing id".into()))?
        .to_string();
    let content = raw
        .payload
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("hermes: session row missing content".into()))?
        .to_string();
    let role = raw
        .payload
        .get("role")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let table = raw
        .payload
        .get("table")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let db_path = raw
        .payload
        .get("db_path")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let timestamp = raw.payload.get("timestamp").and_then(|v| v.as_i64());

    let created_at = timestamp
        .and_then(|t| Utc.timestamp_opt(t, 0).single())
        .unwrap_or(raw.captured_at);

    let local_id = format!("session|{table}|{id}");
    let record_id = RecordId::from_parts(ADAPTER_ID, instance, &local_id);
    let raw_hash = blake3::hash(content.as_bytes()).to_hex().to_string();

    let mut metadata = serde_json::Map::new();
    metadata.insert("hermes_table".into(), Value::String(table.into()));
    if let Some(r) = role {
        metadata.insert("hermes_role".into(), Value::String(r));
    }
    if let Some(extra) = raw.payload.get("extra").cloned() {
        if let Some(obj) = extra.as_object() {
            if !obj.is_empty() {
                metadata.insert("hermes_extra".into(), extra);
            }
        }
    }

    Ok(vec![AnamnesisRecord {
        id: record_id,
        source: SourceDescriptor {
            adapter: ADAPTER_ID.into(),
            instance: instance.map(str::to_owned),
            version: env!("CARGO_PKG_VERSION").into(),
        },
        content,
        embedding: None,
        scope: Scope::Session,
        kind: Kind::Episode,
        created_at,
        updated_at: None,
        tags: vec![],
        metadata,
        provenance: Provenance {
            native_id: raw.native_id.clone(),
            native_path: db_path,
            captured_at: raw.captured_at,
            raw_hash,
        },
        schema_version: SCHEMA_VERSION,
    }])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn md_block(name: &str, content: &str) -> HermesMarkdownBlock {
        HermesMarkdownBlock {
            source_file: name.into(),
            path: PathBuf::from(format!("/fake/.hermes/{name}")),
            content: content.into(),
            mtime_unix: Some(1_730_000_000),
        }
    }

    #[test]
    fn memory_md_normalizes_to_reference() {
        let block = md_block("MEMORY.md", "system on macOS, prefers Rust");
        let raw = raw_from_markdown(&block, Some("laptop"));
        let recs = normalize(raw, Some("laptop")).unwrap();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].kind, Kind::Reference);
        assert_eq!(recs[0].scope, Scope::User);
        assert_eq!(
            recs[0]
                .metadata
                .get("hermes_source_file")
                .and_then(|v| v.as_str()),
            Some("MEMORY.md")
        );
    }

    #[test]
    fn user_md_normalizes_to_preference() {
        let block = md_block("USER.md", "no mocks in tests");
        let raw = raw_from_markdown(&block, None);
        let recs = normalize(raw, None).unwrap();
        assert_eq!(recs[0].kind, Kind::Preference);
    }

    #[test]
    fn session_row_normalizes_to_episode() {
        let row = HermesSessionRow {
            id: "m1".into(),
            content: "hello there".into(),
            table: "messages".into(),
            role: Some("user".into()),
            timestamp: Some(1_730_000_000),
            extra: Default::default(),
        };
        let dbp = PathBuf::from("/fake/.hermes/sessions.db");
        let raw = raw_from_session(&dbp, &row, Some("laptop"));
        let recs = normalize(raw, Some("laptop")).unwrap();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].kind, Kind::Episode);
        assert_eq!(recs[0].scope, Scope::Session);
        assert_eq!(
            recs[0].metadata.get("hermes_role").and_then(|v| v.as_str()),
            Some("user")
        );
        assert_eq!(
            recs[0]
                .metadata
                .get("hermes_table")
                .and_then(|v| v.as_str()),
            Some("messages")
        );
        assert_eq!(recs[0].created_at.timestamp(), 1_730_000_000);
    }

    #[test]
    fn unknown_payload_kind_rejected() {
        let raw = RawRecord {
            native_id: "x".into(),
            native_path: None,
            payload: json!({"payload_kind": "hermes_NOT_real"}),
            captured_at: Utc::now(),
        };
        assert!(normalize(raw, None).is_err());
    }
}
