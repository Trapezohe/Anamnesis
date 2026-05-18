//! Normalize OpenClaw raw records into `AnamnesisRecord`s.

use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_core::RawRecord;
use chrono::{TimeZone, Utc};
use serde_json::{json, Value};

use crate::scanner::{OpenClawConfigFile, OpenClawSessionBlob, OpenClawSkill};
use crate::ADAPTER_ID;

/// Payload-kind discriminator for OpenClaw config / preamble files
/// (AGENTS.md / SOUL.md / TOOLS.md / openclaw.json).
pub const PAYLOAD_KIND_CONFIG: &str = "openclaw_config";
/// Payload-kind discriminator for one OpenClaw skill (`SKILL.md`).
pub const PAYLOAD_KIND_SKILL: &str = "openclaw_skill";
/// Payload-kind discriminator for one OpenClaw session log blob.
pub const PAYLOAD_KIND_SESSION: &str = "openclaw_session";

/// Build a `RawRecord` for a config file (AGENTS / SOUL / TOOLS /
/// openclaw.json). The normalizer turns it into `Kind::Reference` —
/// configs describe HOW the agent works, which we treat as durable
/// reference material (`Scope::User`).
pub fn raw_from_config(cf: &OpenClawConfigFile, instance: Option<&str>) -> RawRecord {
    let native_id = synth_config_id(instance, &cf.kind);
    RawRecord {
        native_id,
        native_path: Some(cf.path.display().to_string()),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_CONFIG,
            "config_kind": cf.kind,
            "path": cf.path.display().to_string(),
            "content": cf.content,
            "mtime_unix": cf.mtime_unix,
        }),
        captured_at: Utc::now(),
    }
}

/// Build a `RawRecord` for one skill. Normalizer → `Kind::Skill`.
pub fn raw_from_skill(skill: &OpenClawSkill, instance: Option<&str>) -> RawRecord {
    let native_id = synth_skill_id(instance, &skill.name);
    RawRecord {
        native_id,
        native_path: Some(skill.path.display().to_string()),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_SKILL,
            "skill_name": skill.name,
            "path": skill.path.display().to_string(),
            "content": skill.content,
            "mtime_unix": skill.mtime_unix,
        }),
        captured_at: Utc::now(),
    }
}

/// Build a `RawRecord` for one session log file. Normalizer →
/// `Kind::Episode`, `Scope::Session`. The whole file body becomes one
/// record's `content` — chunk granularity is left to the chunker.
pub fn raw_from_session(blob: &OpenClawSessionBlob, instance: Option<&str>) -> RawRecord {
    let native_id = synth_session_id(instance, &blob.name);
    RawRecord {
        native_id,
        native_path: Some(blob.path.display().to_string()),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_SESSION,
            "file_name": blob.name,
            "path": blob.path.display().to_string(),
            "content": blob.content,
            "mtime_unix": blob.mtime_unix,
        }),
        captured_at: Utc::now(),
    }
}

fn synth_config_id(instance: Option<&str>, kind: &str) -> String {
    let instance = instance.unwrap_or("default");
    format!("{instance}|config|{kind}")
}

fn synth_skill_id(instance: Option<&str>, name: &str) -> String {
    let instance = instance.unwrap_or("default");
    format!("{instance}|skill|{name}")
}

fn synth_session_id(instance: Option<&str>, file: &str) -> String {
    let instance = instance.unwrap_or("default");
    format!("{instance}|session|{file}")
}

/// Normalize one `RawRecord` (any OpenClaw payload_kind) into an
/// `AnamnesisRecord`.
pub fn normalize(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let payload_kind = raw
        .payload
        .get("payload_kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("openclaw: missing payload_kind".into()))?;
    match payload_kind {
        PAYLOAD_KIND_CONFIG => normalize_config(raw, instance),
        PAYLOAD_KIND_SKILL => normalize_skill(raw, instance),
        PAYLOAD_KIND_SESSION => normalize_session(raw, instance),
        other => Err(Error::InvalidRecord(format!(
            "openclaw: unexpected payload_kind {other:?}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_record(
    raw: &RawRecord,
    instance: Option<&str>,
    local_id: &str,
    content: String,
    kind: Kind,
    scope: Scope,
    metadata: serde_json::Map<String, Value>,
    mtime_unix: Option<i64>,
) -> AnamnesisRecord {
    let created_at = mtime_unix
        .and_then(|t| Utc.timestamp_opt(t, 0).single())
        .unwrap_or(raw.captured_at);
    let record_id = RecordId::from_parts(ADAPTER_ID, instance, local_id);
    let raw_hash = blake3::hash(content.as_bytes()).to_hex().to_string();
    AnamnesisRecord {
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
    }
}

fn normalize_config(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let kind_label = raw
        .payload
        .get("config_kind")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let content = raw
        .payload
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("openclaw: config content missing".into()))?
        .to_string();
    let mtime_unix = raw.payload.get("mtime_unix").and_then(|v| v.as_i64());
    let local_id = format!("config|{kind_label}");
    let mut metadata = serde_json::Map::new();
    metadata.insert(
        "openclaw_config_kind".into(),
        Value::String(kind_label.into()),
    );
    Ok(vec![build_record(
        &raw,
        instance,
        &local_id,
        content,
        Kind::Reference,
        Scope::User,
        metadata,
        mtime_unix,
    )])
}

fn normalize_skill(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let name = raw
        .payload
        .get("skill_name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("openclaw: skill name missing".into()))?
        .to_string();
    let content = raw
        .payload
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("openclaw: skill content missing".into()))?
        .to_string();
    let mtime_unix = raw.payload.get("mtime_unix").and_then(|v| v.as_i64());
    let local_id = format!("skill|{name}");
    let mut metadata = serde_json::Map::new();
    metadata.insert("openclaw_skill_name".into(), Value::String(name));
    Ok(vec![build_record(
        &raw,
        instance,
        &local_id,
        content,
        Kind::Skill,
        Scope::User,
        metadata,
        mtime_unix,
    )])
}

fn normalize_session(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let file_name = raw
        .payload
        .get("file_name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("openclaw: session file_name missing".into()))?
        .to_string();
    let content = raw
        .payload
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("openclaw: session content missing".into()))?
        .to_string();
    let mtime_unix = raw.payload.get("mtime_unix").and_then(|v| v.as_i64());
    let local_id = format!("session|{file_name}");
    let mut metadata = serde_json::Map::new();
    metadata.insert("openclaw_session_file".into(), Value::String(file_name));
    Ok(vec![build_record(
        &raw,
        instance,
        &local_id,
        content,
        Kind::Episode,
        Scope::Session,
        metadata,
        mtime_unix,
    )])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn config_normalizes_to_reference_user() {
        let cf = OpenClawConfigFile {
            kind: "SOUL.md".into(),
            path: PathBuf::from("/fake/SOUL.md"),
            content: "system: rust engineer".into(),
            mtime_unix: Some(1_730_000_000),
        };
        let r = normalize(raw_from_config(&cf, Some("laptop")), Some("laptop")).unwrap();
        assert_eq!(r[0].kind, Kind::Reference);
        assert_eq!(r[0].scope, Scope::User);
        assert_eq!(
            r[0].metadata
                .get("openclaw_config_kind")
                .and_then(|v| v.as_str()),
            Some("SOUL.md")
        );
        assert_eq!(r[0].created_at.timestamp(), 1_730_000_000);
    }

    #[test]
    fn skill_normalizes_to_skill_kind() {
        let s = OpenClawSkill {
            name: "write-code".into(),
            path: PathBuf::from("/fake/skill.md"),
            content: "produce rust code".into(),
            mtime_unix: None,
        };
        let r = normalize(raw_from_skill(&s, None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Skill);
        assert_eq!(r[0].scope, Scope::User);
        assert_eq!(
            r[0].metadata
                .get("openclaw_skill_name")
                .and_then(|v| v.as_str()),
            Some("write-code")
        );
    }

    #[test]
    fn session_normalizes_to_episode_session() {
        let b = OpenClawSessionBlob {
            name: "2026-04-01.jsonl".into(),
            path: PathBuf::from("/fake/sessions/2026-04-01.jsonl"),
            content: "{\"k\":1}\n".into(),
            mtime_unix: Some(1_740_000_000),
        };
        let r = normalize(raw_from_session(&b, Some("laptop")), Some("laptop")).unwrap();
        assert_eq!(r[0].kind, Kind::Episode);
        assert_eq!(r[0].scope, Scope::Session);
        assert_eq!(
            r[0].metadata
                .get("openclaw_session_file")
                .and_then(|v| v.as_str()),
            Some("2026-04-01.jsonl")
        );
    }

    #[test]
    fn unknown_payload_kind_rejected() {
        let raw = RawRecord {
            native_id: "x".into(),
            native_path: None,
            payload: json!({"payload_kind": "openclaw_NOPE"}),
            captured_at: Utc::now(),
        };
        assert!(normalize(raw, None).is_err());
    }
}
