//! Normalize OpenViking raw records into `AnamnesisRecord`s.

use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_core::RawRecord;
use chrono::{DateTime, TimeZone, Utc};
use serde_json::{json, Value};

use crate::scanner::{OpenVikingFileRecord, OpenVikingMessage};
use crate::ADAPTER_ID;

/// `payload_kind` discriminator for a file-shaped OpenViking record
/// (resources, user/agent memories, skill defs, instructions, session summaries).
pub const PAYLOAD_KIND_FILE: &str = "openviking_file";
/// `payload_kind` discriminator for a session-message JSONL line.
pub const PAYLOAD_KIND_MESSAGE: &str = "openviking_message";

/// Build a `RawRecord` from a file-shaped OpenViking record.
pub fn raw_from_file(f: &OpenVikingFileRecord, instance: Option<&str>) -> RawRecord {
    let native_id = synth_id(instance, &format!("file|{}", f.path.display()));
    let viking_uri = f
        .viking_uri
        .clone()
        .unwrap_or_else(|| f.path.display().to_string());
    RawRecord {
        native_id,
        native_path: Some(viking_uri.clone()),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_FILE,
            "path": f.path.display().to_string(),
            "viking_uri": viking_uri,
            "ov_scope": format!("{:?}", f.scope),
            "layer": f.layer,
            "content": f.content,
            "mtime_unix": f.mtime_unix,
        }),
        captured_at: Utc::now(),
    }
}

/// Build a `RawRecord` from one session message line.
pub fn raw_from_message(m: &OpenVikingMessage, instance: Option<&str>) -> RawRecord {
    let native_id = synth_id(
        instance,
        &format!("msg|{}|{}", m.source_path.display(), m.line_no),
    );
    RawRecord {
        native_id,
        native_path: Some(format!("{}#{}", m.source_path.display(), m.line_no)),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_MESSAGE,
            "source_path": m.source_path.display().to_string(),
            "line_no": m.line_no,
            "session_id": m.session_id,
            "raw_json": m.raw_json,
            "mtime_unix": m.mtime_unix,
        }),
        captured_at: Utc::now(),
    }
}

fn synth_id(instance: Option<&str>, local: &str) -> String {
    let instance = instance.unwrap_or("local");
    let hashed = blake3::hash(local.as_bytes()).to_hex();
    format!("{instance}|{}", &hashed[..32])
}

/// Normalize one `RawRecord` (any OpenViking payload_kind) → `AnamnesisRecord`(s).
///
/// Session messages still emit exactly one record per line — splitting happens
/// at scan time (`messages.jsonl` → one `OpenVikingMessage` per line).
pub fn normalize(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let payload_kind = raw
        .payload
        .get("payload_kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("openviking: missing payload_kind".into()))?;
    match payload_kind {
        PAYLOAD_KIND_FILE => normalize_file(raw, instance),
        PAYLOAD_KIND_MESSAGE => normalize_message(raw, instance),
        other => Err(Error::InvalidRecord(format!(
            "openviking: unexpected payload_kind {other:?}"
        ))),
    }
}

fn normalize_file(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let content = raw
        .payload
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("openviking: file.content missing".into()))?
        .to_string();
    let scope_str = raw
        .payload
        .get("ov_scope")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let (kind, scope) = kind_scope_for(scope_str);

    let mtime_unix = raw.payload.get("mtime_unix").and_then(|v| v.as_i64());
    let created_at = mtime_to_chrono(mtime_unix).unwrap_or(raw.captured_at);

    let local_id = raw.native_id.clone();
    let record_id = RecordId::from_parts(ADAPTER_ID, instance, &local_id);
    let raw_hash = blake3::hash(content.as_bytes()).to_hex().to_string();

    let mut metadata = serde_json::Map::new();
    metadata.insert(
        "openviking_scope".into(),
        Value::String(scope_str.to_string()),
    );
    if let Some(layer) = raw.payload.get("layer").and_then(|v| v.as_str()) {
        metadata.insert("openviking_layer".into(), Value::String(layer.to_string()));
    }
    if let Some(uri) = raw.payload.get("viking_uri").and_then(|v| v.as_str()) {
        metadata.insert("openviking_uri".into(), Value::String(uri.to_string()));
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
        },
        schema_version: SCHEMA_VERSION,
    }])
}

fn normalize_message(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let raw_json = raw
        .payload
        .get("raw_json")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("openviking: message.raw_json missing".into()))?
        .to_string();

    // Best-effort parse: pull `role`, `created_at`, and concatenate text parts.
    let parsed: Value = serde_json::from_str(&raw_json).unwrap_or(Value::Null);
    let role = parsed
        .get("role")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let msg_id = parsed.get("id").and_then(|v| v.as_str()).map(str::to_owned);
    let created_from_msg = parsed
        .get("created_at")
        .and_then(|v| v.as_str())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|d| d.with_timezone(&Utc));

    let text_concat = extract_text_parts(&parsed);
    // Prefer extracted text; fall back to the raw JSON line so the record is never empty.
    let content = if text_concat.is_empty() {
        raw_json.clone()
    } else {
        text_concat
    };

    let mtime_unix = raw.payload.get("mtime_unix").and_then(|v| v.as_i64());
    let created_at = created_from_msg
        .or_else(|| mtime_to_chrono(mtime_unix))
        .unwrap_or(raw.captured_at);

    let session_id = raw
        .payload
        .get("session_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let line_no = raw
        .payload
        .get("line_no")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let record_id = RecordId::from_parts(ADAPTER_ID, instance, &raw.native_id);
    let raw_hash = blake3::hash(content.as_bytes()).to_hex().to_string();

    let mut metadata = serde_json::Map::new();
    metadata.insert(
        "openviking_scope".into(),
        Value::String("SessionMessage".into()),
    );
    metadata.insert("openviking_role".into(), Value::String(role));
    metadata.insert("openviking_session_id".into(), Value::String(session_id));
    metadata.insert("openviking_line_no".into(), Value::Number(line_no.into()));
    if let Some(id) = msg_id {
        metadata.insert("openviking_message_id".into(), Value::String(id));
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
            native_path: raw.native_path.clone(),
            captured_at: raw.captured_at,
            raw_hash,
        },
        schema_version: SCHEMA_VERSION,
    }])
}

fn extract_text_parts(parsed: &Value) -> String {
    let Some(arr) = parsed.get("parts").and_then(|v| v.as_array()) else {
        return String::new();
    };
    let mut out = String::new();
    for p in arr {
        let ty = p.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if ty == "text" {
            if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(t);
            }
        }
    }
    out
}

fn mtime_to_chrono(mtime_unix: Option<i64>) -> Option<DateTime<Utc>> {
    mtime_unix.and_then(|t| Utc.timestamp_opt(t, 0).single())
}

fn kind_scope_for(scope_dbg: &str) -> (Kind, Scope) {
    // `scope_dbg` is `format!("{:?}", OpenVikingScope::...)`, e.g. "UserPreference".
    match scope_dbg {
        "Resource" => (Kind::Reference, Scope::User),
        "UserProfile" => (Kind::Preference, Scope::User),
        "UserPreference" => (Kind::Preference, Scope::User),
        "UserEntity" => (Kind::Fact, Scope::User),
        "UserEvent" => (Kind::Episode, Scope::User),
        "AgentCase" => (Kind::Episode, Scope::Project),
        "AgentPattern" => (Kind::Reference, Scope::Project),
        "AgentTool" => (Kind::Reference, Scope::Project),
        "AgentSkillMemory" => (Kind::Reference, Scope::Project),
        "AgentSkillDef" => (Kind::Skill, Scope::Project),
        "AgentInstruction" => (Kind::Reference, Scope::Project),
        "SessionSummary" => (Kind::Episode, Scope::Session),
        _ => (Kind::Unknown, Scope::Ephemeral),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scanner::OpenVikingScope;
    use std::path::PathBuf;

    fn file(scope: OpenVikingScope, layer: &'static str, body: &str) -> OpenVikingFileRecord {
        OpenVikingFileRecord {
            path: PathBuf::from("/fake/p.md"),
            viking_uri: Some("viking://x".into()),
            scope,
            layer: Some(layer),
            content: body.into(),
            mtime_unix: Some(1_730_000_000),
        }
    }

    #[test]
    fn resource_normalizes_to_reference_user() {
        let f = file(OpenVikingScope::Resource, "L1", "body");
        let r = normalize(raw_from_file(&f, None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Reference);
        assert_eq!(r[0].scope, Scope::User);
        assert_eq!(
            r[0].metadata
                .get("openviking_layer")
                .and_then(|v| v.as_str()),
            Some("L1")
        );
    }

    #[test]
    fn user_preference_normalizes_to_preference_user() {
        let f = file(OpenVikingScope::UserPreference, "L2", "uses dark mode");
        let r = normalize(raw_from_file(&f, None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Preference);
        assert_eq!(r[0].scope, Scope::User);
    }

    #[test]
    fn user_entity_normalizes_to_fact() {
        let f = file(OpenVikingScope::UserEntity, "L2", "alice");
        let r = normalize(raw_from_file(&f, None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Fact);
    }

    #[test]
    fn user_event_normalizes_to_episode_user() {
        let f = file(OpenVikingScope::UserEvent, "L2", "shipped");
        let r = normalize(raw_from_file(&f, None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Episode);
        assert_eq!(r[0].scope, Scope::User);
    }

    #[test]
    fn agent_case_normalizes_to_episode_project() {
        let f = file(OpenVikingScope::AgentCase, "L2", "case body");
        let r = normalize(raw_from_file(&f, None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Episode);
        assert_eq!(r[0].scope, Scope::Project);
    }

    #[test]
    fn agent_skill_def_normalizes_to_skill_project() {
        let f = file(OpenVikingScope::AgentSkillDef, "L1", "SKILL.md body");
        let r = normalize(raw_from_file(&f, None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Skill);
        assert_eq!(r[0].scope, Scope::Project);
    }

    #[test]
    fn message_with_text_part_concatenates_content() {
        let m = OpenVikingMessage {
            source_path: PathBuf::from("/fake/messages.jsonl"),
            line_no: 0,
            session_id: "sid-1".into(),
            raw_json: r#"{"id":"m1","role":"user","parts":[{"type":"text","text":"hi"},{"type":"text","text":"there"}],"created_at":"2026-05-17T10:00:00Z"}"#.into(),
            mtime_unix: Some(1_730_000_000),
        };
        let r = normalize(raw_from_message(&m, Some("inst")), Some("inst")).unwrap();
        assert_eq!(r[0].kind, Kind::Episode);
        assert_eq!(r[0].scope, Scope::Session);
        assert_eq!(r[0].content, "hi\nthere");
        assert_eq!(
            r[0].metadata
                .get("openviking_role")
                .and_then(|v| v.as_str()),
            Some("user")
        );
        // created_at parsed from the message's own field.
        assert_eq!(r[0].created_at.to_rfc3339(), "2026-05-17T10:00:00+00:00");
    }

    #[test]
    fn message_with_no_text_part_falls_back_to_raw_json() {
        let m = OpenVikingMessage {
            source_path: PathBuf::from("/fake/messages.jsonl"),
            line_no: 1,
            session_id: "sid-1".into(),
            raw_json:
                r#"{"id":"m2","role":"assistant","parts":[{"type":"tool","tool_name":"search"}]}"#
                    .into(),
            mtime_unix: None,
        };
        let r = normalize(raw_from_message(&m, None), None).unwrap();
        assert!(r[0].content.contains("\"tool_name\""));
    }

    #[test]
    fn unknown_payload_kind_rejected() {
        let raw = RawRecord {
            native_id: "x".into(),
            native_path: None,
            payload: json!({"payload_kind": "openviking_NOPE"}),
            captured_at: Utc::now(),
        };
        assert!(normalize(raw, None).is_err());
    }
}
