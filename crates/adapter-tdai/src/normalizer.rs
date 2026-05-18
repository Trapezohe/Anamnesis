//! Normalize TDAI raw records into `AnamnesisRecord`s.

use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_core::RawRecord;
use chrono::{TimeZone, Utc};
use serde_json::{json, Value};

use crate::scanner::{TdaiL0Ref, TdaiL1Fact, TdaiL2Scenario, TdaiL3Persona};
use crate::ADAPTER_ID;

/// Payload-kind discriminator: L0 raw conversation ref (`refs/*.md`).
pub const PAYLOAD_KIND_L0_REF: &str = "tdai_l0_ref";
/// Payload-kind discriminator: L1 atomic fact (one JSONL line).
pub const PAYLOAD_KIND_L1_FACT: &str = "tdai_l1_fact";
/// Payload-kind discriminator: L2 scenario block (markdown).
pub const PAYLOAD_KIND_L2_SCENARIO: &str = "tdai_l2_scenario";
/// Payload-kind discriminator: L3 user persona (`persona.md`).
pub const PAYLOAD_KIND_L3_PERSONA: &str = "tdai_l3_persona";

/// Build a `RawRecord` from an L0 ref. Normalizer → Kind::Episode, Scope::Session.
pub fn raw_from_l0(r: &TdaiL0Ref, instance: Option<&str>) -> RawRecord {
    let native_id = synth_id(instance, &format!("l0|{}", r.path.display()));
    RawRecord {
        native_id,
        native_path: Some(r.path.display().to_string()),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_L0_REF,
            "path": r.path.display().to_string(),
            "content": r.content,
            "mtime_unix": r.mtime_unix,
        }),
        captured_at: Utc::now(),
    }
}

/// Build a `RawRecord` from one L1 fact. Normalizer → Kind::Fact, Scope::User.
pub fn raw_from_l1(f: &TdaiL1Fact, instance: Option<&str>) -> RawRecord {
    let native_id = synth_id(
        instance,
        &format!("l1|{}|{}", f.source_path.display(), f.line_no),
    );
    RawRecord {
        native_id,
        native_path: Some(format!("{}#{}", f.source_path.display(), f.line_no)),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_L1_FACT,
            "source_path": f.source_path.display().to_string(),
            "line_no": f.line_no,
            "content": f.content,
            "mtime_unix": f.mtime_unix,
        }),
        captured_at: Utc::now(),
    }
}

/// Build a `RawRecord` from an L2 scenario. Normalizer → Kind::Reference, Scope::User.
pub fn raw_from_l2(s: &TdaiL2Scenario, instance: Option<&str>) -> RawRecord {
    let native_id = synth_id(instance, &format!("l2|{}", s.path.display()));
    RawRecord {
        native_id,
        native_path: Some(s.path.display().to_string()),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_L2_SCENARIO,
            "path": s.path.display().to_string(),
            "content": s.content,
            "mtime_unix": s.mtime_unix,
        }),
        captured_at: Utc::now(),
    }
}

/// Build a `RawRecord` from an L3 persona. Normalizer → Kind::Preference, Scope::User.
pub fn raw_from_l3(p: &TdaiL3Persona, instance: Option<&str>) -> RawRecord {
    let native_id = synth_id(instance, &format!("l3|{}", p.path.display()));
    RawRecord {
        native_id,
        native_path: Some(p.path.display().to_string()),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_L3_PERSONA,
            "path": p.path.display().to_string(),
            "content": p.content,
            "mtime_unix": p.mtime_unix,
        }),
        captured_at: Utc::now(),
    }
}

fn synth_id(instance: Option<&str>, local: &str) -> String {
    let instance = instance.unwrap_or("local");
    let hashed = blake3::hash(local.as_bytes()).to_hex();
    // Cap to 32 chars — the path-derived local id can be long.
    format!("{instance}|{}", &hashed[..32])
}

/// Normalize one `RawRecord` (any TDAI payload_kind) → `AnamnesisRecord`.
pub fn normalize(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let payload_kind = raw
        .payload
        .get("payload_kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("tdai: missing payload_kind".into()))?;
    match payload_kind {
        PAYLOAD_KIND_L0_REF => one(raw, instance, Kind::Episode, Scope::Session, "l0"),
        PAYLOAD_KIND_L1_FACT => one(raw, instance, Kind::Fact, Scope::User, "l1"),
        PAYLOAD_KIND_L2_SCENARIO => one(raw, instance, Kind::Reference, Scope::User, "l2"),
        PAYLOAD_KIND_L3_PERSONA => one(raw, instance, Kind::Preference, Scope::User, "l3"),
        other => Err(Error::InvalidRecord(format!(
            "tdai: unexpected payload_kind {other:?}"
        ))),
    }
}

fn one(
    raw: RawRecord,
    instance: Option<&str>,
    kind: Kind,
    scope: Scope,
    tier: &str,
) -> Result<Vec<AnamnesisRecord>> {
    let content = raw
        .payload
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord(format!("tdai: {tier} content missing")))?
        .to_string();
    let mtime_unix = raw.payload.get("mtime_unix").and_then(|v| v.as_i64());
    let created_at = mtime_unix
        .and_then(|t| Utc.timestamp_opt(t, 0).single())
        .unwrap_or(raw.captured_at);

    let local_id = raw.native_id.clone();
    let record_id = RecordId::from_parts(ADAPTER_ID, instance, &local_id);
    let raw_hash = blake3::hash(content.as_bytes()).to_hex().to_string();

    let mut metadata = serde_json::Map::new();
    metadata.insert("tdai_tier".into(), Value::String(tier.into()));
    if let Some(line_no) = raw.payload.get("line_no").and_then(|v| v.as_u64()) {
        metadata.insert("tdai_line_no".into(), Value::Number(line_no.into()));
    }
    if let Some(src) = raw.payload.get("source_path").and_then(|v| v.as_str()) {
        metadata.insert("tdai_source_path".into(), Value::String(src.into()));
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
        scope,
        kind,
        created_at,
        updated_at: None,
        tags: vec![],
        metadata,
        provenance: Provenance {
            native_id: raw.native_id.clone(),
            native_path: raw.native_path.clone(),
            captured_at: raw.captured_at,
            raw_hash,
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    }])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn l0(path: &str, body: &str) -> TdaiL0Ref {
        TdaiL0Ref {
            path: PathBuf::from(path),
            content: body.into(),
            mtime_unix: Some(1_730_000_000),
        }
    }

    #[test]
    fn l0_normalizes_to_episode_session() {
        let r = normalize(raw_from_l0(&l0("/fake/refs/a.md", "raw conv"), None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Episode);
        assert_eq!(r[0].scope, Scope::Session);
        assert_eq!(
            r[0].metadata.get("tdai_tier").and_then(|v| v.as_str()),
            Some("l0")
        );
    }

    #[test]
    fn l1_normalizes_to_fact_user() {
        let f = TdaiL1Fact {
            source_path: PathBuf::from("/fake/facts.jsonl"),
            line_no: 2,
            content: r#"{"fact":"likes rust"}"#.into(),
            mtime_unix: Some(1_730_000_000),
        };
        let r = normalize(raw_from_l1(&f, None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Fact);
        assert_eq!(r[0].scope, Scope::User);
        assert_eq!(
            r[0].metadata.get("tdai_line_no").and_then(|v| v.as_u64()),
            Some(2)
        );
        assert!(r[0]
            .provenance
            .native_path
            .as_deref()
            .unwrap()
            .ends_with("#2"));
    }

    #[test]
    fn l2_normalizes_to_reference() {
        let s = TdaiL2Scenario {
            path: PathBuf::from("/fake/scenario.md"),
            content: "scenario body".into(),
            mtime_unix: None,
        };
        let r = normalize(raw_from_l2(&s, None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Reference);
        assert_eq!(r[0].scope, Scope::User);
    }

    #[test]
    fn l3_normalizes_to_preference() {
        let p = TdaiL3Persona {
            path: PathBuf::from("/fake/persona.md"),
            content: "persona body".into(),
            mtime_unix: None,
        };
        let r = normalize(raw_from_l3(&p, None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Preference);
        assert_eq!(r[0].scope, Scope::User);
    }

    #[test]
    fn unknown_payload_kind_rejected() {
        let raw = RawRecord {
            native_id: "x".into(),
            native_path: None,
            payload: json!({"payload_kind": "tdai_NOPE"}),
            captured_at: Utc::now(),
        };
        assert!(normalize(raw, None).is_err());
    }
}
