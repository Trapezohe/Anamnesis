//! Normalize Memori raw records into `AnamnesisRecord`s.

use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_core::RawRecord;
use chrono::{TimeZone, Utc};
use serde_json::{json, Value};

use crate::scanner::{
    parse_memori_time, MemoriConversationMessage, MemoriConversationSummary, MemoriEntityFact,
    MemoriKgTriple, MemoriProcessAttribute,
};
use crate::ADAPTER_ID;

/// `payload_kind` discriminator: one `memori_entity_fact` row.
pub const PAYLOAD_KIND_ENTITY_FACT: &str = "memori_entity_fact";
/// `payload_kind` discriminator: one `memori_process_attribute` row.
pub const PAYLOAD_KIND_PROCESS_ATTR: &str = "memori_process_attribute";
/// `payload_kind` discriminator: one `memori_conversation_message` row.
pub const PAYLOAD_KIND_MESSAGE: &str = "memori_conversation_message";
/// `payload_kind` discriminator: one `memori_conversation.summary`.
pub const PAYLOAD_KIND_CONV_SUMMARY: &str = "memori_conversation_summary";
/// `payload_kind` discriminator: one `memori_knowledge_graph` triple.
pub const PAYLOAD_KIND_KG_TRIPLE: &str = "memori_kg_triple";

/// Build a `RawRecord` from an entity fact. An Anamnesis round-trip export
/// carries the original provenance in `f.metadata`; when present, restore
/// the original `anamnesis_native_id` so re-import reproduces the same
/// identity (reconcile stays `both`).
pub fn raw_from_entity_fact(f: &MemoriEntityFact, instance: Option<&str>) -> RawRecord {
    let anamnesis_meta = f
        .metadata
        .as_deref()
        .and_then(|m| serde_json::from_str::<Value>(m).ok())
        .filter(|v| v.is_object());
    let restored_native_id = anamnesis_meta
        .as_ref()
        .and_then(|m| m.get("anamnesis_native_id"))
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let native_id =
        restored_native_id.unwrap_or_else(|| synth_id(instance, &format!("entfact|{}", f.uuid)));
    let mut payload = json!({
        "payload_kind": PAYLOAD_KIND_ENTITY_FACT,
        "uuid": f.uuid,
        "entity_external_id": f.entity_external_id,
        "content": f.content,
        "num_times": f.num_times,
        "date_last_time": f.date_last_time,
        "date_created": f.date_created,
    });
    if let Some(meta) = anamnesis_meta {
        payload["anamnesis_meta"] = meta;
    }
    RawRecord {
        native_id,
        native_path: Some(format!("memori_entity_fact::{}", f.uuid)),
        payload,
        captured_at: Utc::now(),
    }
}

/// Build a `RawRecord` from a process attribute.
pub fn raw_from_process_attr(a: &MemoriProcessAttribute, instance: Option<&str>) -> RawRecord {
    let native_id = synth_id(instance, &format!("procattr|{}", a.uuid));
    RawRecord {
        native_id,
        native_path: Some(format!("memori_process_attribute::{}", a.uuid)),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_PROCESS_ATTR,
            "uuid": a.uuid,
            "process_external_id": a.process_external_id,
            "content": a.content,
            "num_times": a.num_times,
            "date_last_time": a.date_last_time,
            "date_created": a.date_created,
        }),
        captured_at: Utc::now(),
    }
}

/// Build a `RawRecord` from a conversation message.
pub fn raw_from_message(m: &MemoriConversationMessage, instance: Option<&str>) -> RawRecord {
    let native_id = synth_id(instance, &format!("msg|{}", m.uuid));
    RawRecord {
        native_id,
        native_path: Some(format!("memori_conversation_message::{}", m.uuid)),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_MESSAGE,
            "uuid": m.uuid,
            "role": m.role,
            "type": m.type_,
            "content": m.content,
            "session_uuid": m.session_uuid,
            "date_created": m.date_created,
        }),
        captured_at: Utc::now(),
    }
}

/// Build a `RawRecord` from a conversation summary.
pub fn raw_from_summary(s: &MemoriConversationSummary, instance: Option<&str>) -> RawRecord {
    let native_id = synth_id(instance, &format!("convsum|{}", s.uuid));
    RawRecord {
        native_id,
        native_path: Some(format!("memori_conversation::{}::summary", s.uuid)),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_CONV_SUMMARY,
            "uuid": s.uuid,
            "session_uuid": s.session_uuid,
            "summary": s.summary,
            "date_created": s.date_created,
        }),
        captured_at: Utc::now(),
    }
}

/// Build a `RawRecord` from a KG triple.
pub fn raw_from_kg_triple(t: &MemoriKgTriple, instance: Option<&str>) -> RawRecord {
    let native_id = synth_id(instance, &format!("kg|{}", t.uuid));
    RawRecord {
        native_id,
        native_path: Some(format!("memori_knowledge_graph::{}", t.uuid)),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_KG_TRIPLE,
            "uuid": t.uuid,
            "entity_external_id": t.entity_external_id,
            "subject": t.subject,
            "predicate": t.predicate,
            "object": t.object,
            "num_times": t.num_times,
            "date_last_time": t.date_last_time,
            "date_created": t.date_created,
        }),
        captured_at: Utc::now(),
    }
}

fn synth_id(instance: Option<&str>, local: &str) -> String {
    let instance = instance.unwrap_or("local");
    let hashed = blake3::hash(local.as_bytes()).to_hex();
    format!("{instance}|{}", &hashed[..32])
}

/// Normalize any Memori `payload_kind` → `AnamnesisRecord`.
pub fn normalize(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let payload_kind = raw
        .payload
        .get("payload_kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("memori: missing payload_kind".into()))?;
    match payload_kind {
        PAYLOAD_KIND_ENTITY_FACT => one(
            raw,
            instance,
            Kind::Fact,
            Scope::User,
            "entity_fact",
            extract_content,
        ),
        PAYLOAD_KIND_PROCESS_ATTR => one(
            raw,
            instance,
            Kind::Reference,
            Scope::Project,
            "process_attribute",
            extract_content,
        ),
        PAYLOAD_KIND_MESSAGE => one(
            raw,
            instance,
            Kind::Episode,
            Scope::Session,
            "conversation_message",
            extract_content,
        ),
        PAYLOAD_KIND_CONV_SUMMARY => one(
            raw,
            instance,
            Kind::Episode,
            Scope::Session,
            "conversation_summary",
            extract_summary,
        ),
        PAYLOAD_KIND_KG_TRIPLE => one(
            raw,
            instance,
            Kind::Fact,
            Scope::User,
            "kg_triple",
            extract_triple_sentence,
        ),
        other => Err(Error::InvalidRecord(format!(
            "memori: unexpected payload_kind {other:?}"
        ))),
    }
}

fn extract_content(payload: &Value) -> Option<String> {
    payload
        .get("content")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
}

fn extract_summary(payload: &Value) -> Option<String> {
    payload
        .get("summary")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
}

fn extract_triple_sentence(payload: &Value) -> Option<String> {
    let s = payload
        .get("subject")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let p = payload
        .get("predicate")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let o = payload.get("object").and_then(|v| v.as_str()).unwrap_or("");
    if s.is_empty() && p.is_empty() && o.is_empty() {
        return None;
    }
    Some(format!("{s} | {p} | {o}"))
}

fn one(
    raw: RawRecord,
    instance: Option<&str>,
    kind: Kind,
    scope: Scope,
    tier: &str,
    extract: fn(&Value) -> Option<String>,
) -> Result<Vec<AnamnesisRecord>> {
    let content = extract(&raw.payload)
        .ok_or_else(|| Error::InvalidRecord(format!("memori: {tier} content missing")))?;

    // Prefer `date_last_time` (the time the fact/attribute/triple was last
    // observed); fall back to `date_created`; last resort `raw.captured_at`.
    let created_at = raw
        .payload
        .get("date_last_time")
        .and_then(|v| v.as_str())
        .and_then(parse_memori_time)
        .or_else(|| {
            raw.payload
                .get("date_created")
                .and_then(|v| v.as_str())
                .and_then(parse_memori_time)
        })
        .and_then(|t| Utc.timestamp_opt(t, 0).single())
        .unwrap_or(raw.captured_at);

    let record_id = RecordId::from_parts(ADAPTER_ID, instance, &raw.native_id);
    // Round-trip exports carry the original provenance under `anamnesis_meta`;
    // restore raw_hash from it so identity matches, else hash the content.
    let anamnesis_meta = raw
        .payload
        .get("anamnesis_meta")
        .and_then(|v| v.as_object());
    let raw_hash = anamnesis_meta
        .and_then(|m| m.get("anamnesis_raw_hash"))
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .unwrap_or_else(|| blake3::hash(content.as_bytes()).to_hex().to_string());

    let mut metadata = serde_json::Map::new();
    metadata.insert("memori_tier".into(), Value::String(tier.into()));
    if let Some(m) = anamnesis_meta {
        for (k, v) in m {
            if k.starts_with("anamnesis_") {
                metadata.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
    }
    for k in [
        "entity_external_id",
        "process_external_id",
        "session_uuid",
        "role",
        "num_times",
        "subject",
        "predicate",
        "object",
    ] {
        if let Some(v) = raw.payload.get(k) {
            if !v.is_null() {
                metadata.insert(format!("memori_{k}"), v.clone());
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
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    }])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fact() -> MemoriEntityFact {
        MemoriEntityFact {
            uuid: "fact-1".into(),
            entity_external_id: Some("user-123".into()),
            content: "user lives in Paris".into(),
            num_times: 3,
            date_last_time: Some("2026-05-01 10:00:00".into()),
            date_created: Some("2026-04-01 10:00:00".into()),
            metadata: None,
        }
    }

    #[test]
    fn entity_fact_normalizes_to_fact_user() {
        let r = normalize(raw_from_entity_fact(&fact(), None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Fact);
        assert_eq!(r[0].scope, Scope::User);
        assert_eq!(
            r[0].metadata.get("memori_tier").and_then(|v| v.as_str()),
            Some("entity_fact")
        );
        // created_at from date_last_time
        assert_eq!(r[0].created_at.timestamp(), 1_777_629_600);
    }

    #[test]
    fn process_attribute_normalizes_to_reference_project() {
        let a = MemoriProcessAttribute {
            uuid: "attr-1".into(),
            process_external_id: Some("my-app".into()),
            content: "prefers JSON".into(),
            num_times: 1,
            date_last_time: None,
            date_created: Some("2026-04-01 10:00:00".into()),
        };
        let r = normalize(raw_from_process_attr(&a, None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Reference);
        assert_eq!(r[0].scope, Scope::Project);
    }

    #[test]
    fn message_normalizes_to_episode_session() {
        let m = MemoriConversationMessage {
            uuid: "msg-1".into(),
            role: "user".into(),
            type_: Some("text".into()),
            content: "hi".into(),
            session_uuid: Some("sess-1".into()),
            date_created: Some("2026-05-01 10:00:00".into()),
        };
        let r = normalize(raw_from_message(&m, None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Episode);
        assert_eq!(r[0].scope, Scope::Session);
        assert_eq!(
            r[0].metadata
                .get("memori_session_uuid")
                .and_then(|v| v.as_str()),
            Some("sess-1")
        );
    }

    #[test]
    fn summary_normalizes_to_episode_session() {
        let s = MemoriConversationSummary {
            uuid: "conv-1".into(),
            session_uuid: Some("sess-1".into()),
            summary: "user asked about colors".into(),
            date_created: Some("2026-05-01 10:00:00".into()),
        };
        let r = normalize(raw_from_summary(&s, None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Episode);
        assert_eq!(r[0].scope, Scope::Session);
        assert_eq!(
            r[0].metadata.get("memori_tier").and_then(|v| v.as_str()),
            Some("conversation_summary")
        );
    }

    #[test]
    fn kg_triple_normalizes_to_fact_user_with_pipe_separated_sentence() {
        let t = MemoriKgTriple {
            uuid: "kg-1".into(),
            entity_external_id: Some("user-123".into()),
            subject: "user :: Person".into(),
            predicate: "lives_in".into(),
            object: "Paris :: City".into(),
            num_times: 2,
            date_last_time: Some("2026-05-01 10:00:00".into()),
            date_created: Some("2026-04-01 10:00:00".into()),
        };
        let r = normalize(raw_from_kg_triple(&t, None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Fact);
        assert_eq!(r[0].scope, Scope::User);
        assert_eq!(r[0].content, "user :: Person | lives_in | Paris :: City");
        assert_eq!(
            r[0].metadata.get("memori_tier").and_then(|v| v.as_str()),
            Some("kg_triple")
        );
    }

    #[test]
    fn unknown_payload_kind_rejected() {
        let raw = RawRecord {
            native_id: "x".into(),
            native_path: None,
            payload: json!({"payload_kind": "memori_NOPE"}),
            captured_at: Utc::now(),
        };
        assert!(normalize(raw, None).is_err());
    }

    #[test]
    fn missing_payload_kind_rejected() {
        let raw = RawRecord {
            native_id: "x".into(),
            native_path: None,
            payload: json!({}),
            captured_at: Utc::now(),
        };
        assert!(normalize(raw, None).is_err());
    }
}
