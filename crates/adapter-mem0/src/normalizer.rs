//! Normalize a `Mem0Row` into an `AnamnesisRecord`.
//!
//! Mapping (BLUEPRINT §6.9):
//!   - `memory`         → `AnamnesisRecord.content`
//!   - `id`             → `provenance.native_id`
//!   - `user_id`        → `metadata.mem0_user_id`, scope = User
//!   - `agent_id`       → `metadata.mem0_agent_id`
//!   - `run_id`         → `metadata.mem0_run_id`
//!   - `metadata`       → merged into `metadata` (parsed best-effort)
//!   - `created_at`     → `created_at` (best-effort parse)
//!   - Kind defaults to `Fact` (mem0 has no built-in kind taxonomy).

use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_core::RawRecord;
use chrono::{DateTime, TimeZone, Utc};

use crate::scanner::Mem0Row;

/// Payload tag.
pub const PAYLOAD_KIND_MEMORY: &str = "mem0_memory";

/// Build a `RawRecord` from a `Mem0Row`.
pub fn raw_from_row(row: &Mem0Row, instance: Option<&str>) -> RawRecord {
    let payload = serde_json::to_value(row).unwrap_or(serde_json::Value::Null);
    let payload = serde_json::json!({
        "payload_kind": PAYLOAD_KIND_MEMORY,
        "row": payload,
    });
    RawRecord {
        native_id: synth_native_id(instance, &row.id),
        native_path: None,
        payload,
        captured_at: Utc::now(),
    }
}

fn synth_native_id(instance: Option<&str>, id: &str) -> String {
    let instance = instance.unwrap_or("self-hosted");
    format!("{instance}|{id}")
}

/// Normalize a `RawRecord` produced from a mem0 row.
pub fn normalize(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let payload_kind = raw
        .payload
        .get("payload_kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("missing payload_kind".into()))?;
    if payload_kind != PAYLOAD_KIND_MEMORY {
        return Err(Error::InvalidRecord(format!(
            "unexpected payload_kind: {payload_kind}"
        )));
    }
    let row: Mem0Row = raw
        .payload
        .get("row")
        .cloned()
        .ok_or_else(|| Error::InvalidRecord("missing row".into()))
        .and_then(|v| {
            serde_json::from_value(v).map_err(|e| Error::InvalidRecord(format!("row decode: {e}")))
        })?;

    let id = RecordId::from_parts(crate::ADAPTER_ID, instance, &raw.native_id);
    let created_at = parse_timestamp(row.created_at.as_deref()).unwrap_or(raw.captured_at);
    let updated_at = parse_timestamp(row.updated_at.as_deref());
    let mut metadata = serde_json::Map::new();
    if let Some(u) = &row.user_id {
        metadata.insert("mem0_user_id".into(), serde_json::Value::String(u.clone()));
    }
    if let Some(a) = &row.agent_id {
        metadata.insert("mem0_agent_id".into(), serde_json::Value::String(a.clone()));
    }
    if let Some(r) = &row.run_id {
        metadata.insert("mem0_run_id".into(), serde_json::Value::String(r.clone()));
    }
    // Merge mem0 `metadata` JSON if it parses, otherwise drop in the
    // raw string under a sentinel key.
    if let Some(m) = &row.metadata_json {
        match serde_json::from_str::<serde_json::Value>(m) {
            Ok(serde_json::Value::Object(obj)) => {
                for (k, v) in obj {
                    metadata.entry(k).or_insert(v);
                }
            }
            _ => {
                metadata.insert(
                    "mem0_metadata_raw".into(),
                    serde_json::Value::String(m.clone()),
                );
            }
        }
    }
    // Tags: derive from `categories` if mem0 put any in metadata.
    let tags = metadata
        .get("categories")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();

    let raw_hash = blake3::hash(row.memory.as_bytes()).to_hex().to_string();

    Ok(vec![AnamnesisRecord {
        id,
        source: SourceDescriptor {
            adapter: crate::ADAPTER_ID.into(),
            instance: instance.map(str::to_owned),
            version: env!("CARGO_PKG_VERSION").into(),
        },
        content: row.memory,
        embedding: None,
        scope: Scope::User,
        kind: Kind::Fact,
        created_at,
        updated_at,
        tags,
        metadata,
        provenance: Provenance {
            native_id: raw.native_id,
            native_path: None,
            captured_at: raw.captured_at,
            raw_hash,
        },
        schema_version: SCHEMA_VERSION,
    }])
}

fn parse_timestamp(s: Option<&str>) -> Option<DateTime<Utc>> {
    let s = s?;
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&Utc));
    }
    if let Ok(epoch) = s.parse::<i64>() {
        return Utc.timestamp_opt(epoch, 0).single();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, mem: &str) -> Mem0Row {
        Mem0Row {
            id: id.into(),
            memory: mem.into(),
            user_id: Some("alice".into()),
            agent_id: Some("ag1".into()),
            run_id: Some("r1".into()),
            metadata_json: Some("{\"categories\":[\"editor\",\"shell\"]}".into()),
            created_at: Some("2026-05-01T00:00:00Z".into()),
            updated_at: None,
            extra: Default::default(),
        }
    }

    #[test]
    fn full_row_normalizes_to_kind_fact_user_scope() {
        let r = row("m1", "user prefers vim");
        let raw = raw_from_row(&r, Some("self-hosted"));
        let recs = normalize(raw, Some("self-hosted")).unwrap();
        assert_eq!(recs.len(), 1);
        let rec = &recs[0];
        assert_eq!(rec.kind, Kind::Fact);
        assert_eq!(rec.scope, Scope::User);
        assert_eq!(rec.content, "user prefers vim");
        assert_eq!(rec.source.adapter, "mem0");
        assert_eq!(rec.source.instance.as_deref(), Some("self-hosted"));
        assert_eq!(
            rec.metadata.get("mem0_user_id").and_then(|v| v.as_str()),
            Some("alice")
        );
        assert!(rec.tags.iter().any(|t| t == "editor"));
        assert!(rec.tags.iter().any(|t| t == "shell"));
    }

    #[test]
    fn missing_optional_fields_still_works() {
        let r = Mem0Row {
            id: "x".into(),
            memory: "hello".into(),
            user_id: None,
            agent_id: None,
            run_id: None,
            metadata_json: None,
            created_at: None,
            updated_at: None,
            extra: Default::default(),
        };
        let raw = raw_from_row(&r, None);
        let recs = normalize(raw, None).unwrap();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].content, "hello");
        assert!(recs[0].metadata.is_empty());
        assert!(recs[0].tags.is_empty());
    }

    #[test]
    fn malformed_metadata_json_kept_as_raw() {
        let r = Mem0Row {
            id: "x".into(),
            memory: "hi".into(),
            user_id: None,
            agent_id: None,
            run_id: None,
            metadata_json: Some("not json".into()),
            created_at: None,
            updated_at: None,
            extra: Default::default(),
        };
        let raw = raw_from_row(&r, None);
        let recs = normalize(raw, None).unwrap();
        assert_eq!(
            recs[0]
                .metadata
                .get("mem0_metadata_raw")
                .and_then(|v| v.as_str()),
            Some("not json")
        );
    }

    #[test]
    fn epoch_timestamp_parses_to_utc() {
        let r = Mem0Row {
            id: "x".into(),
            memory: "hi".into(),
            user_id: None,
            agent_id: None,
            run_id: None,
            metadata_json: None,
            created_at: Some("1700000000".into()),
            updated_at: None,
            extra: Default::default(),
        };
        let raw = raw_from_row(&r, None);
        let rec = &normalize(raw, None).unwrap()[0];
        assert_eq!(rec.created_at.timestamp(), 1700000000);
    }

    #[test]
    fn unknown_payload_kind_errors() {
        let raw = RawRecord {
            native_id: "x".into(),
            native_path: None,
            payload: serde_json::json!({"payload_kind": "wat", "row": {}}),
            captured_at: Utc::now(),
        };
        let err = normalize(raw, None).unwrap_err();
        assert!(format!("{err}").contains("unexpected payload_kind"));
    }

    #[test]
    fn raw_hash_is_deterministic_blake3_of_memory_text() {
        let r = row("m1", "hello world");
        let raw = raw_from_row(&r, None);
        let rec = &normalize(raw, None).unwrap()[0];
        let expected = blake3::hash("hello world".as_bytes()).to_hex().to_string();
        assert_eq!(rec.provenance.raw_hash, expected);
    }

    #[test]
    fn record_id_is_instance_scoped() {
        let r = row("m1", "x");
        let a = &normalize(raw_from_row(&r, Some("workspace-a")), Some("workspace-a")).unwrap()[0];
        let b = &normalize(raw_from_row(&r, Some("workspace-b")), Some("workspace-b")).unwrap()[0];
        assert_ne!(a.id, b.id);
    }
}
