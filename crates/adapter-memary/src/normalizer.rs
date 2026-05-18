//! Normalize Memary raw records into `AnamnesisRecord`s.

use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_core::RawRecord;
use chrono::Utc;
use serde_json::{json, Value};

use crate::scanner::{
    parse_memary_time, unix_to_utc, MemaryChatMessage, MemaryEntityTally, MemaryPersona,
    MemaryStreamEntry,
};
use crate::ADAPTER_ID;

/// `payload_kind` discriminator: one entry from `memory_stream.json`.
pub const PAYLOAD_KIND_STREAM: &str = "memary_stream_entry";
/// `payload_kind` discriminator: one entry from `entity_knowledge_store.json`.
pub const PAYLOAD_KIND_TALLY: &str = "memary_entity_tally";
/// `payload_kind` discriminator: one entry from `past_chat.json`.
pub const PAYLOAD_KIND_CHAT: &str = "memary_chat_message";
/// `payload_kind` discriminator: a persona file (system or user).
pub const PAYLOAD_KIND_PERSONA: &str = "memary_persona";

/// Wrap a memory-stream entry.
pub fn raw_from_stream(e: &MemaryStreamEntry, instance: Option<&str>) -> RawRecord {
    let native_id = synth_id(
        instance,
        &format!("stream|{}|{}", e.source_path.display(), e.index),
    );
    RawRecord {
        native_id,
        native_path: Some(format!("{}#{}", e.source_path.display(), e.index)),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_STREAM,
            "source_path": e.source_path.display().to_string(),
            "index": e.index,
            "entity": e.entity,
            "date": e.date,
        }),
        captured_at: Utc::now(),
    }
}

/// Wrap an entity-knowledge tally.
pub fn raw_from_tally(t: &MemaryEntityTally, instance: Option<&str>) -> RawRecord {
    let native_id = synth_id(
        instance,
        &format!("tally|{}|{}", t.source_path.display(), t.index),
    );
    RawRecord {
        native_id,
        native_path: Some(format!("{}#{}", t.source_path.display(), t.index)),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_TALLY,
            "source_path": t.source_path.display().to_string(),
            "index": t.index,
            "entity": t.entity,
            "count": t.count,
            "date": t.date,
        }),
        captured_at: Utc::now(),
    }
}

/// Wrap a chat message.
pub fn raw_from_chat(c: &MemaryChatMessage, instance: Option<&str>) -> RawRecord {
    let native_id = synth_id(
        instance,
        &format!("chat|{}|{}", c.source_path.display(), c.index),
    );
    RawRecord {
        native_id,
        native_path: Some(format!("{}#{}", c.source_path.display(), c.index)),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_CHAT,
            "source_path": c.source_path.display().to_string(),
            "index": c.index,
            "role": c.role,
            "content": c.content,
        }),
        captured_at: Utc::now(),
    }
}

/// Wrap a persona file.
pub fn raw_from_persona(p: &MemaryPersona, instance: Option<&str>) -> RawRecord {
    let native_id = synth_id(
        instance,
        &format!("persona|{}|{}", p.persona_kind, p.path.display()),
    );
    RawRecord {
        native_id,
        native_path: Some(p.path.display().to_string()),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_PERSONA,
            "path": p.path.display().to_string(),
            "persona_kind": p.persona_kind,
            "content": p.content,
            "mtime_unix": p.mtime_unix,
        }),
        captured_at: Utc::now(),
    }
}

fn synth_id(instance: Option<&str>, local: &str) -> String {
    let instance = instance.unwrap_or("local");
    let hashed = blake3::hash(local.as_bytes()).to_hex();
    format!("{instance}|{}", &hashed[..32])
}

/// Normalize one Memary `RawRecord` → `AnamnesisRecord`.
pub fn normalize(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let payload_kind = raw
        .payload
        .get("payload_kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("memary: missing payload_kind".into()))?;
    match payload_kind {
        PAYLOAD_KIND_STREAM => normalize_stream(raw, instance),
        PAYLOAD_KIND_TALLY => normalize_tally(raw, instance),
        PAYLOAD_KIND_CHAT => normalize_chat(raw, instance),
        PAYLOAD_KIND_PERSONA => normalize_persona(raw, instance),
        other => Err(Error::InvalidRecord(format!(
            "memary: unexpected payload_kind {other:?}"
        ))),
    }
}

fn normalize_stream(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let entity = required_str(&raw.payload, "entity", "stream")?;
    let date_str = raw.payload.get("date").and_then(|v| v.as_str());
    let created_at = date_str
        .and_then(parse_memary_time)
        .and_then(unix_to_utc)
        .unwrap_or(raw.captured_at);

    finalize(
        raw,
        instance,
        Kind::Reference,
        Scope::Project,
        "stream",
        format!("entity mentioned: {entity}"),
        created_at,
    )
}

fn normalize_tally(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let entity = required_str(&raw.payload, "entity", "tally")?;
    let count = raw
        .payload
        .get("count")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let date_str = raw.payload.get("date").and_then(|v| v.as_str());
    let created_at = date_str
        .and_then(parse_memary_time)
        .and_then(unix_to_utc)
        .unwrap_or(raw.captured_at);

    finalize(
        raw,
        instance,
        Kind::Reference,
        Scope::Project,
        "tally",
        format!("entity tally: {entity} (count={count})"),
        created_at,
    )
}

fn normalize_chat(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let content = required_str(&raw.payload, "content", "chat")?;
    let created_at = raw.captured_at;
    finalize(
        raw,
        instance,
        Kind::Episode,
        Scope::Session,
        "chat",
        content,
        created_at,
    )
}

fn normalize_persona(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let content = required_str(&raw.payload, "content", "persona")?;
    let persona_kind = raw
        .payload
        .get("persona_kind")
        .and_then(|v| v.as_str())
        .unwrap_or("user");
    let (kind, scope) = match persona_kind {
        "user" => (Kind::Preference, Scope::User),
        // "system" persona — operator-tuned default behavior for the agent.
        _ => (Kind::Reference, Scope::Project),
    };
    let created_at = raw
        .payload
        .get("mtime_unix")
        .and_then(|v| v.as_i64())
        .and_then(unix_to_utc)
        .unwrap_or(raw.captured_at);

    finalize(raw, instance, kind, scope, "persona", content, created_at)
}

fn required_str(payload: &Value, key: &str, tier: &str) -> Result<String> {
    payload
        .get(key)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| Error::InvalidRecord(format!("memary: {tier}.{key} missing/empty")))
}

#[allow(clippy::too_many_arguments)]
fn finalize(
    raw: RawRecord,
    instance: Option<&str>,
    kind: Kind,
    scope: Scope,
    tier: &str,
    content: String,
    created_at: chrono::DateTime<chrono::Utc>,
) -> Result<Vec<AnamnesisRecord>> {
    let record_id = RecordId::from_parts(ADAPTER_ID, instance, &raw.native_id);
    let raw_hash = blake3::hash(content.as_bytes()).to_hex().to_string();
    let mut metadata = serde_json::Map::new();
    metadata.insert("memary_tier".into(), Value::String(tier.into()));
    for k in [
        "entity",
        "count",
        "role",
        "persona_kind",
        "source_path",
        "index",
    ] {
        if let Some(v) = raw.payload.get(k) {
            if !v.is_null() {
                metadata.insert(format!("memary_{k}"), v.clone());
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn stream() -> MemaryStreamEntry {
        MemaryStreamEntry {
            source_path: PathBuf::from("/fake/memory_stream.json"),
            index: 0,
            entity: "Alice".into(),
            date: Some("2026-05-01T10:00:00".into()),
        }
    }

    fn tally() -> MemaryEntityTally {
        MemaryEntityTally {
            source_path: PathBuf::from("/fake/entity_knowledge_store.json"),
            index: 0,
            entity: "Alice".into(),
            count: 3,
            date: Some("2026-05-01T10:00:00".into()),
        }
    }

    fn chat() -> MemaryChatMessage {
        MemaryChatMessage {
            source_path: PathBuf::from("/fake/past_chat.json"),
            index: 0,
            role: "user".into(),
            content: "hi".into(),
        }
    }

    fn persona(kind: &str, body: &str) -> MemaryPersona {
        MemaryPersona {
            path: PathBuf::from(format!("/fake/{kind}_persona.txt")),
            persona_kind: kind.into(),
            content: body.into(),
            mtime_unix: Some(1_730_000_000),
        }
    }

    #[test]
    fn stream_normalizes_to_reference_project() {
        let r = normalize(raw_from_stream(&stream(), None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Reference);
        assert_eq!(r[0].scope, Scope::Project);
        assert!(r[0].content.contains("Alice"));
        // date parsed.
        assert_eq!(r[0].created_at.timestamp(), 1_777_629_600);
    }

    #[test]
    fn tally_normalizes_with_count_in_content() {
        let r = normalize(raw_from_tally(&tally(), None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Reference);
        assert!(r[0].content.contains("count=3"));
    }

    #[test]
    fn chat_normalizes_to_episode_session() {
        let r = normalize(raw_from_chat(&chat(), None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Episode);
        assert_eq!(r[0].scope, Scope::Session);
        assert_eq!(
            r[0].metadata.get("memary_role").and_then(|v| v.as_str()),
            Some("user")
        );
    }

    #[test]
    fn user_persona_normalizes_to_preference_user() {
        let r = normalize(
            raw_from_persona(&persona("user", "i prefer rust"), None),
            None,
        )
        .unwrap();
        assert_eq!(r[0].kind, Kind::Preference);
        assert_eq!(r[0].scope, Scope::User);
    }

    #[test]
    fn system_persona_normalizes_to_reference_project() {
        let r = normalize(
            raw_from_persona(&persona("system", "I am a helpful agent"), None),
            None,
        )
        .unwrap();
        assert_eq!(r[0].kind, Kind::Reference);
        assert_eq!(r[0].scope, Scope::Project);
    }

    #[test]
    fn unknown_payload_kind_rejected() {
        let raw = RawRecord {
            native_id: "x".into(),
            native_path: None,
            payload: json!({"payload_kind": "memary_NOPE"}),
            captured_at: Utc::now(),
        };
        assert!(normalize(raw, None).is_err());
    }

    #[test]
    fn missing_required_content_rejected() {
        let raw = RawRecord {
            native_id: "x".into(),
            native_path: None,
            payload: json!({"payload_kind": PAYLOAD_KIND_CHAT, "role": "user"}),
            captured_at: Utc::now(),
        };
        assert!(normalize(raw, None).is_err());
    }
}
