//! Normalize ghast raw records into `AnamnesisRecord`s.

use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_core::RawRecord;
use chrono::{TimeZone, Utc};
use serde_json::{json, Value};

use crate::scanner::{GhastPromptFile, GhastSkill};
use crate::ADAPTER_ID;

/// Payload-kind discriminator for ghast prompt files (`prompts/<role>/*.md`).
pub const PAYLOAD_KIND_PROMPT: &str = "ghast_prompt";
/// Payload-kind discriminator for ghast bundled-skill markdown files.
pub const PAYLOAD_KIND_SKILL: &str = "ghast_skill";

/// Build a `RawRecord` from a prompt file. Normalizer → `Kind::Reference`,
/// `Scope::User`.
pub fn raw_from_prompt(p: &GhastPromptFile, instance: Option<&str>) -> RawRecord {
    let native_id = synth_id(instance, &format!("prompt|{}|{}", p.role, p.name));
    RawRecord {
        native_id,
        native_path: Some(p.path.display().to_string()),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_PROMPT,
            "role": p.role,
            "name": p.name,
            "path": p.path.display().to_string(),
            "content": p.content,
            "mtime_unix": p.mtime_unix,
        }),
        captured_at: Utc::now(),
    }
}

/// Build a `RawRecord` from a skill markdown file. Normalizer →
/// `Kind::Skill` when `file_kind == "SKILL.md"`, otherwise
/// `Kind::Reference` (READMEs, REFERENCES.md, etc.).
pub fn raw_from_skill(s: &GhastSkill, instance: Option<&str>) -> RawRecord {
    let native_id = synth_id(instance, &format!("skill|{}|{}", s.name, s.file_kind));
    RawRecord {
        native_id,
        native_path: Some(s.path.display().to_string()),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_SKILL,
            "skill_name": s.name,
            "file_kind": s.file_kind,
            "path": s.path.display().to_string(),
            "content": s.content,
            "mtime_unix": s.mtime_unix,
        }),
        captured_at: Utc::now(),
    }
}

fn synth_id(instance: Option<&str>, local: &str) -> String {
    let instance = instance.unwrap_or("local");
    format!("{instance}|{local}")
}

/// Normalize one `RawRecord` (any ghast payload_kind) into an `AnamnesisRecord`.
pub fn normalize(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let payload_kind = raw
        .payload
        .get("payload_kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("ghast: missing payload_kind".into()))?;
    match payload_kind {
        PAYLOAD_KIND_PROMPT => normalize_prompt(raw, instance),
        PAYLOAD_KIND_SKILL => normalize_skill(raw, instance),
        other => Err(Error::InvalidRecord(format!(
            "ghast: unexpected payload_kind {other:?}"
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
        },
        schema_version: SCHEMA_VERSION,
    }
}

fn normalize_prompt(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let role = raw
        .payload
        .get("role")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let name = raw
        .payload
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let content = raw
        .payload
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("ghast: prompt content missing".into()))?
        .to_string();
    let mtime_unix = raw.payload.get("mtime_unix").and_then(|v| v.as_i64());
    let local_id = format!("prompt|{role}|{name}");
    let mut metadata = serde_json::Map::new();
    metadata.insert("ghast_role".into(), Value::String(role));
    metadata.insert("ghast_prompt_name".into(), Value::String(name));
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
    let skill_name = raw
        .payload
        .get("skill_name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let file_kind = raw
        .payload
        .get("file_kind")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let content = raw
        .payload
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("ghast: skill content missing".into()))?
        .to_string();
    let mtime_unix = raw.payload.get("mtime_unix").and_then(|v| v.as_i64());
    let local_id = format!("skill|{skill_name}|{file_kind}");
    // SKILL.md is the canonical skill definition — Kind::Skill.
    // README.md / REFERENCES.md / NOTES.md are supporting → Kind::Reference.
    let kind = if file_kind == "SKILL.md" {
        Kind::Skill
    } else {
        Kind::Reference
    };
    let mut metadata = serde_json::Map::new();
    metadata.insert("ghast_skill_name".into(), Value::String(skill_name));
    metadata.insert("ghast_skill_file".into(), Value::String(file_kind));
    Ok(vec![build_record(
        &raw,
        instance,
        &local_id,
        content,
        kind,
        Scope::User,
        metadata,
        mtime_unix,
    )])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn prompt_normalizes_to_reference() {
        let p = GhastPromptFile {
            role: "coding".into(),
            name: "default".into(),
            path: PathBuf::from("/fake/prompts/coding/default.md"),
            content: "default coding prompt".into(),
            mtime_unix: Some(1_730_000_000),
        };
        let r = normalize(raw_from_prompt(&p, Some("dev")), Some("dev")).unwrap();
        assert_eq!(r[0].kind, Kind::Reference);
        assert_eq!(r[0].scope, Scope::User);
        assert_eq!(
            r[0].metadata.get("ghast_role").and_then(|v| v.as_str()),
            Some("coding")
        );
        assert_eq!(r[0].created_at.timestamp(), 1_730_000_000);
    }

    #[test]
    fn skill_md_normalizes_to_skill_kind() {
        let s = GhastSkill {
            name: "memory-management".into(),
            file_kind: "SKILL.md".into(),
            path: PathBuf::from("/fake/skill.md"),
            content: "skill body".into(),
            mtime_unix: None,
        };
        let r = normalize(raw_from_skill(&s, None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Skill);
    }

    #[test]
    fn references_md_normalizes_to_reference_kind() {
        let s = GhastSkill {
            name: "memory-management".into(),
            file_kind: "REFERENCES.md".into(),
            path: PathBuf::from("/fake/refs.md"),
            content: "ref body".into(),
            mtime_unix: None,
        };
        let r = normalize(raw_from_skill(&s, None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Reference);
    }

    #[test]
    fn unknown_payload_kind_rejected() {
        let raw = RawRecord {
            native_id: "x".into(),
            native_path: None,
            payload: json!({"payload_kind": "ghast_NOPE"}),
            captured_at: Utc::now(),
        };
        assert!(normalize(raw, None).is_err());
    }
}
