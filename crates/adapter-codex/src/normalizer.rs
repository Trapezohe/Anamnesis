//! Normalize Codex rollout JSONL files into Episode records.
//!
//! Issue #69 (PR mirroring claude-code PR-H/PR-I): instead of dumping
//! the entire JSONL body as `content`, parse it into structured turns
//! and render as readable markdown. `created_at` / `updated_at` come
//! from the first/last message timestamp (falling back to file mtime,
//! then `captured_at`) so time-window queries carry signal.

use std::path::Path;

use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_core::RawRecord;
use chrono::{DateTime, Utc};
use serde_json::Value;

use crate::session;

/// Payload kind tag.
pub const PAYLOAD_KIND_SESSION: &str = "codex_session";

/// Build a `RawRecord` from a Codex rollout JSONL file. The JSONL is
/// parsed eagerly here so the scanner's per-file IO cost amortizes
/// across the parse work; structured `messages` (rather than the raw
/// text) end up in `payload` so `normalize` can render markdown and
/// hash deterministically without re-parsing.
///
/// `mtime` is the file modification time (PR-I support). The normalizer
/// prefers `first_ts` from parsed messages, falling back to `mtime`,
/// then `captured_at`.
pub fn raw_session(
    path: &Path,
    jsonl_text: &str,
    mtime: Option<DateTime<Utc>>,
    instance: Option<&str>,
) -> RawRecord {
    let parsed = session::parse_jsonl(jsonl_text);
    let native_id = synth_native_id(instance, path);
    let mut payload = serde_json::json!({
        "payload_kind": PAYLOAD_KIND_SESSION,
        "path": path.display().to_string(),
        "messages": parsed.messages,
        "message_count": parsed.messages.len(),
    });
    if let Some(t) = parsed.first_ts {
        payload["first_ts"] = Value::String(t.to_rfc3339());
    }
    if let Some(t) = parsed.last_ts {
        payload["last_ts"] = Value::String(t.to_rfc3339());
    }
    if let Some(t) = mtime {
        payload["mtime"] = Value::String(t.to_rfc3339());
    }
    RawRecord {
        native_id,
        native_path: Some(path.display().to_string()),
        payload,
        captured_at: Utc::now(),
    }
}

fn synth_native_id(instance: Option<&str>, path: &Path) -> String {
    let instance = instance.unwrap_or("default");
    format!("{instance}|{}", path.display())
}

/// Read an RFC3339 timestamp out of a payload field, if present.
fn payload_ts(payload: &Value, key: &str) -> Option<DateTime<Utc>> {
    payload
        .get(key)
        .and_then(|v| v.as_str())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|d| d.with_timezone(&Utc))
}

/// `RawRecord` → `AnamnesisRecord`.
pub fn normalize(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let payload_kind = raw
        .payload
        .get("payload_kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("missing payload_kind".into()))?;
    if payload_kind != PAYLOAD_KIND_SESSION {
        return Err(Error::InvalidRecord(format!(
            "unexpected payload_kind: {payload_kind}"
        )));
    }
    let messages_value = raw
        .payload
        .get("messages")
        .ok_or_else(|| Error::InvalidRecord("codex session payload missing messages".into()))?;
    let messages: Vec<session::SessionMessage> = serde_json::from_value(messages_value.clone())
        .map_err(|e| Error::InvalidRecord(format!("malformed codex session messages: {e}")))?;
    let first_ts = payload_ts(&raw.payload, "first_ts");
    let last_ts = payload_ts(&raw.payload, "last_ts");
    let mtime = payload_ts(&raw.payload, "mtime");

    let parsed = session::ParsedSession {
        messages,
        first_ts,
        last_ts,
    };
    let content = session::render_markdown(&parsed);
    let raw_hash = blake3::hash(content.as_bytes()).to_hex().to_string();
    let id = RecordId::from_parts(crate::ADAPTER_ID, instance, &raw.native_id);

    // PR-I: created_at = first message ts → mtime → captured_at;
    //        updated_at = last message ts → mtime when first_ts present.
    let created_at = first_ts.or(mtime).unwrap_or(raw.captured_at);
    let updated_at = last_ts.or_else(|| if first_ts.is_some() { mtime } else { None });

    Ok(vec![AnamnesisRecord {
        id,
        source: SourceDescriptor {
            adapter: crate::ADAPTER_ID.into(),
            instance: instance.map(str::to_owned),
            version: env!("CARGO_PKG_VERSION").into(),
        },
        content,
        embedding: None,
        scope: Scope::Session,
        kind: Kind::Episode,
        created_at,
        updated_at,
        tags: Vec::new(),
        metadata: serde_json::Map::from_iter([(
            "source_file".into(),
            serde_json::Value::String(raw.native_path.clone().unwrap_or_default()),
        )]),
        provenance: Provenance {
            native_id: raw.native_id,
            native_path: raw.native_path,
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

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(format!("/tmp/{name}.jsonl"))
    }

    fn two_turn_jsonl() -> String {
        [
            r#"{"timestamp":"2026-05-01T00:00:00Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hi codex"}]}}"#,
            r#"{"timestamp":"2026-05-01T00:00:05Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello there"}]}}"#,
        ]
        .join("\n")
    }

    #[test]
    fn raw_session_carries_parsed_messages_into_payload() {
        let p = fixture_path("sess-1");
        let raw = raw_session(&p, &two_turn_jsonl(), None, Some("default"));
        assert_eq!(raw.payload["payload_kind"], PAYLOAD_KIND_SESSION);
        assert_eq!(raw.payload["message_count"], 2);
        assert!(raw.payload.get("messages").is_some());
        assert!(raw.native_id.starts_with("default|"));
    }

    #[test]
    fn normalize_renders_markdown_transcript() {
        let p = fixture_path("sess-2");
        let raw = raw_session(&p, &two_turn_jsonl(), None, Some("default"));
        let recs = normalize(raw, Some("default")).unwrap();
        assert_eq!(recs.len(), 1);
        let r = &recs[0];
        assert_eq!(r.kind, Kind::Episode);
        assert_eq!(r.scope, Scope::Session);
        assert!(r.content.contains("**user**"));
        assert!(r.content.contains("**assistant**"));
        assert!(r.content.contains("hi codex"));
        assert!(r.content.contains("hello there"));
        // raw_hash is deterministic over rendered markdown, not raw JSONL.
        assert_eq!(
            r.provenance.raw_hash,
            blake3::hash(r.content.as_bytes()).to_hex().to_string()
        );
    }

    #[test]
    fn normalize_uses_first_message_timestamp_for_created_at() {
        let p = fixture_path("sess-3");
        let raw = raw_session(&p, &two_turn_jsonl(), None, Some("default"));
        let rec = &normalize(raw, Some("default")).unwrap()[0];
        // first_ts comes from the first message at 2026-05-01T00:00:00Z.
        assert_eq!(rec.created_at.to_rfc3339(), "2026-05-01T00:00:00+00:00");
        // last_ts from the second message.
        assert_eq!(
            rec.updated_at.map(|t| t.to_rfc3339()),
            Some("2026-05-01T00:00:05+00:00".to_string())
        );
    }

    #[test]
    fn normalize_falls_back_to_mtime_when_messages_have_no_timestamps() {
        let p = fixture_path("sess-4");
        // No timestamp fields on the rows — first_ts will be None.
        let no_ts_jsonl = r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"a"}]}}"#;
        let mtime = DateTime::parse_from_rfc3339("2025-01-15T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let raw = raw_session(&p, no_ts_jsonl, Some(mtime), None);
        let rec = &normalize(raw, None).unwrap()[0];
        assert_eq!(rec.created_at, mtime);
        // No first_ts → no fallback for updated_at.
        assert!(rec.updated_at.is_none());
    }

    #[test]
    fn record_id_is_instance_scoped() {
        let p = fixture_path("x");
        let a = &normalize(
            raw_session(&p, &two_turn_jsonl(), None, Some("a")),
            Some("a"),
        )
        .unwrap()[0];
        let b = &normalize(
            raw_session(&p, &two_turn_jsonl(), None, Some("b")),
            Some("b"),
        )
        .unwrap()[0];
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn wrong_payload_kind_errors() {
        let raw = RawRecord {
            native_id: "x".into(),
            native_path: None,
            payload: serde_json::json!({"payload_kind": "wat", "messages": []}),
            captured_at: Utc::now(),
        };
        let err = normalize(raw, None).unwrap_err();
        assert!(format!("{err}").contains("unexpected payload_kind"));
    }

    #[test]
    fn missing_messages_field_errors() {
        let raw = RawRecord {
            native_id: "x".into(),
            native_path: None,
            payload: serde_json::json!({"payload_kind": PAYLOAD_KIND_SESSION}),
            captured_at: Utc::now(),
        };
        let err = normalize(raw, None).unwrap_err();
        assert!(format!("{err}").contains("missing messages"));
    }
}
