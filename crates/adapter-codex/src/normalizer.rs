//! Normalize Codex session files into Episode records.

use std::path::Path;

use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_core::RawRecord;
use chrono::Utc;

/// Payload kind tag.
pub const PAYLOAD_KIND_SESSION: &str = "codex_session";

/// Build a `RawRecord` from a Codex session file.
pub fn raw_session(path: &Path, body: String, instance: Option<&str>) -> RawRecord {
    let native_id = synth_native_id(instance, path);
    let payload = serde_json::json!({
        "payload_kind": PAYLOAD_KIND_SESSION,
        "path": path.display().to_string(),
        "content": body,
    });
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
    let content = raw
        .payload
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("missing content".into()))?
        .to_string();
    let raw_hash = blake3::hash(content.as_bytes()).to_hex().to_string();
    let id = RecordId::from_parts(crate::ADAPTER_ID, instance, &raw.native_id);
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
        created_at: raw.captured_at,
        updated_at: None,
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

    #[test]
    fn raw_session_has_expected_shape() {
        let p = fixture_path("sess-1");
        let raw = raw_session(&p, "body".into(), Some("default"));
        assert_eq!(raw.payload["payload_kind"], PAYLOAD_KIND_SESSION);
        assert_eq!(raw.payload["content"], "body");
        assert!(raw.native_id.starts_with("default|"));
    }

    #[test]
    fn normalize_produces_episode_with_session_scope() {
        let p = fixture_path("sess-1");
        let raw = raw_session(&p, "alpha beta".into(), Some("default"));
        let recs = normalize(raw, Some("default")).unwrap();
        assert_eq!(recs.len(), 1);
        let r = &recs[0];
        assert_eq!(r.kind, Kind::Episode);
        assert_eq!(r.scope, Scope::Session);
        assert_eq!(r.content, "alpha beta");
        assert_eq!(r.source.adapter, "codex");
        assert_eq!(
            r.provenance.raw_hash,
            blake3::hash(b"alpha beta").to_hex().to_string()
        );
    }

    #[test]
    fn record_id_is_instance_scoped() {
        let p = fixture_path("x");
        let a = &normalize(raw_session(&p, "x".into(), Some("a")), Some("a")).unwrap()[0];
        let b = &normalize(raw_session(&p, "x".into(), Some("b")), Some("b")).unwrap()[0];
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn wrong_payload_kind_errors() {
        let raw = RawRecord {
            native_id: "x".into(),
            native_path: None,
            payload: serde_json::json!({"payload_kind": "wat", "content": ""}),
            captured_at: Utc::now(),
        };
        let err = normalize(raw, None).unwrap_err();
        assert!(format!("{err}").contains("unexpected payload_kind"));
    }
}
