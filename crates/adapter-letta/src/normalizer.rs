//! Normalize a Letta `block` row into one `AnamnesisRecord`.

use anamnesis_core::error::Result;
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_core::RawRecord;
use chrono::{DateTime, TimeZone, Utc};
use serde_json::{json, Value};
use std::path::Path;

use crate::scanner::LettaBlockRow;
use crate::ADAPTER_ID;

const PAYLOAD_KIND_BLOCK: &str = "letta_block";

/// Build a `RawRecord` from a Letta block row. Mirrors the mem0
/// adapter's `raw_from_row` shape: opaque payload preserved verbatim
/// so the importer's `raw_artifacts` can hold the original for
/// provenance.
pub fn raw_from_block(row: &LettaBlockRow, instance: Option<&str>) -> RawRecord {
    let mut payload = json!({
        "payload_kind": PAYLOAD_KIND_BLOCK,
        "id": row.id,
        "value": row.value,
        "label": row.label,
        "description": row.description,
        "template_name": row.template_name,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    });
    // Best-effort: opaque parse of Letta's metadata_ blob into the
    // payload tree; if it's not valid JSON, keep as a string so
    // provenance is still preserved.
    if let Some(raw) = &row.metadata_json {
        match serde_json::from_str::<Value>(raw) {
            Ok(v) => payload["letta_metadata"] = v,
            Err(_) => payload["letta_metadata_raw"] = Value::String(raw.clone()),
        }
    }
    if !row.extra.is_empty() {
        payload["letta_extra"] = Value::Object(row.extra.clone());
    }
    RawRecord {
        native_id: synth_native_id(instance, &row.id),
        native_path: Some(format!("letta://block/{}", row.id)),
        payload,
        captured_at: Utc::now(),
    }
}

/// Build a `RawRecord` for one block on disk — used by external
/// callers (`anamnesis-cli`) that already pulled the row.
pub fn raw_from_block_at_path(
    row: &LettaBlockRow,
    db_path: &Path,
    instance: Option<&str>,
) -> RawRecord {
    let mut raw = raw_from_block(row, instance);
    raw.native_path = Some(format!("{}#block/{}", db_path.display(), row.id));
    raw
}

fn synth_native_id(instance: Option<&str>, id: &str) -> String {
    let instance = instance.unwrap_or("self-hosted");
    format!("{instance}|{id}")
}

/// Normalize one `RawRecord` (carrying a Letta block payload) into
/// the canonical `AnamnesisRecord`. Returns a 1-element Vec — the
/// importer trait expects multi-result but Letta blocks are 1:1.
pub fn normalize(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    use anamnesis_core::error::Error;

    let payload_kind = raw
        .payload
        .get("payload_kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("letta: missing payload_kind".into()))?;
    if payload_kind != PAYLOAD_KIND_BLOCK {
        return Err(Error::InvalidRecord(format!(
            "letta: unexpected payload_kind {payload_kind:?}"
        )));
    }

    let id = raw
        .payload
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("letta: missing block id".into()))?;
    let content = raw
        .payload
        .get("value")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("letta: missing block value".into()))?
        .to_string();
    let label = raw
        .payload
        .get("label")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let description = raw
        .payload
        .get("description")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let template_name = raw
        .payload
        .get("template_name")
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    // Scope: `persona` / `human` core blocks describe the agent's
    // own state → Scope::User. Arbitrary user-labeled blocks are
    // also user-scope by default. Project / session scope shows up
    // for derived memory in future PRs (PR-6 extractor).
    let scope = Scope::User;

    // Kind: a Letta core block is durable structured information
    // about the user/agent — that's exactly Kind::Fact in our
    // taxonomy. Future PRs may downgrade certain labels to
    // Kind::Reference based on `description`.
    let kind = Kind::Fact;

    // Timestamps. Letta uses ISO-8601 strings in newer migrations;
    // older builds use TEXT epoch. Be permissive.
    let (created_at, updated_at) = parse_letta_timestamps(
        raw.payload.get("created_at").and_then(|v| v.as_str()),
        raw.payload.get("updated_at").and_then(|v| v.as_str()),
        raw.captured_at,
    );

    let raw_hash = blake3::hash(content.as_bytes()).to_hex().to_string();
    let record_id = RecordId::from_parts(ADAPTER_ID, instance, id);

    let mut metadata = serde_json::Map::<String, Value>::new();
    if let Some(l) = label {
        metadata.insert("letta_label".into(), Value::String(l));
    }
    if let Some(d) = description {
        metadata.insert("letta_description".into(), Value::String(d));
    }
    if let Some(t) = template_name {
        metadata.insert("letta_template".into(), Value::String(t));
    }
    if let Some(extras) = raw.payload.get("letta_extra").cloned() {
        metadata.insert("letta_extra".into(), extras);
    }
    if let Some(lmeta) = raw.payload.get("letta_metadata").cloned() {
        metadata.insert("letta_metadata".into(), lmeta);
    }

    let native_id = raw.native_id.clone();
    let native_path = raw.native_path.clone();

    Ok(vec![AnamnesisRecord {
        id: record_id,
        source: SourceDescriptor {
            adapter: ADAPTER_ID.into(),
            instance: instance.map(str::to_owned),
            version: env!("CARGO_PKG_VERSION").into(),
        },
        content,
        embedding: None,
        scope,
        kind,
        created_at,
        updated_at,
        tags: vec![],
        metadata,
        provenance: Provenance {
            native_id,
            native_path,
            captured_at: raw.captured_at,
            raw_hash,
        },
        schema_version: SCHEMA_VERSION,
    }])
}

/// Parse Letta's TEXT timestamps. Tries RFC3339 first then
/// epoch-seconds-as-string (older sqlite migrations stored it that
/// way). Falls back to `captured_at` (= import wall-clock) if both
/// fail — same convention as mem0 / codex adapters.
fn parse_letta_timestamps(
    created: Option<&str>,
    updated: Option<&str>,
    fallback: DateTime<Utc>,
) -> (DateTime<Utc>, Option<DateTime<Utc>>) {
    let created_at = created.and_then(parse_one).unwrap_or(fallback);
    let updated_at = updated.and_then(parse_one);
    (created_at, updated_at)
}

fn parse_one(s: &str) -> Option<DateTime<Utc>> {
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

    fn block(id: &str, value: &str) -> LettaBlockRow {
        LettaBlockRow {
            id: id.into(),
            value: value.into(),
            ..Default::default()
        }
    }

    #[test]
    fn raw_payload_shape_is_canonical() {
        let row = LettaBlockRow {
            label: Some("persona".into()),
            description: Some("the agent's self-view".into()),
            metadata_json: Some(r#"{"v":1}"#.into()),
            ..block("b1", "I am Sam")
        };
        let raw = raw_from_block(&row, Some("local"));
        assert_eq!(raw.native_id, "local|b1");
        assert_eq!(raw.native_path.as_deref(), Some("letta://block/b1"));
        assert_eq!(raw.payload["payload_kind"], "letta_block");
        assert_eq!(raw.payload["value"], "I am Sam");
        assert_eq!(raw.payload["label"], "persona");
        assert_eq!(raw.payload["letta_metadata"]["v"], 1);
    }

    #[test]
    fn normalize_produces_fact_record() {
        let row = LettaBlockRow {
            label: Some("persona".into()),
            description: Some("desc".into()),
            template_name: Some("default-persona".into()),
            metadata_json: Some(r#"{"v":1}"#.into()),
            extra: {
                let mut m = serde_json::Map::new();
                m.insert("limit".into(), serde_json::json!("2000"));
                m
            },
            ..block("b1", "I am Sam")
        };
        let raw = raw_from_block(&row, Some("local"));
        let recs = normalize(raw, Some("local")).unwrap();
        assert_eq!(recs.len(), 1);
        let r = &recs[0];
        assert_eq!(r.kind, Kind::Fact);
        assert_eq!(r.scope, Scope::User);
        assert_eq!(r.content, "I am Sam");
        assert_eq!(r.source.adapter, "letta");
        assert_eq!(r.source.instance.as_deref(), Some("local"));
        assert_eq!(
            r.metadata.get("letta_label").and_then(|v| v.as_str()),
            Some("persona")
        );
        assert_eq!(
            r.metadata.get("letta_description").and_then(|v| v.as_str()),
            Some("desc")
        );
        assert_eq!(
            r.metadata.get("letta_template").and_then(|v| v.as_str()),
            Some("default-persona")
        );
        assert!(r.metadata.contains_key("letta_extra"));
        assert!(r.metadata.contains_key("letta_metadata"));
        // raw_hash is deterministic blake3 of content.
        assert_eq!(r.provenance.raw_hash.len(), 64);
    }

    #[test]
    fn normalize_rejects_wrong_payload_kind() {
        let raw = RawRecord {
            native_id: "x".into(),
            native_path: None,
            payload: json!({"payload_kind": "letta_NOT_block"}),
            captured_at: Utc::now(),
        };
        assert!(normalize(raw, None).is_err());
    }

    #[test]
    fn timestamps_parse_rfc3339_and_epoch() {
        let row = LettaBlockRow {
            created_at: Some("2026-04-01T00:00:00Z".into()),
            updated_at: Some("1730000000".into()),
            ..block("b1", "x")
        };
        let raw = raw_from_block(&row, None);
        let r = &normalize(raw, None).unwrap()[0];
        assert_eq!(r.created_at.to_rfc3339(), "2026-04-01T00:00:00+00:00");
        assert_eq!(r.updated_at.unwrap().timestamp(), 1_730_000_000);
    }
}
