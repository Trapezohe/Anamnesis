//! Normalize an MCP `resources/read` response into an `AnamnesisRecord`.
//!
//! We deliberately pick conservative defaults (`Kind::Unknown`,
//! `Scope::Ephemeral`) because the source semantics are opaque — the
//! whole point of this adapter is "ingest something we don't know in
//! advance". Downstream tooling can re-tag based on URI patterns.

use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_core::RawRecord;
use chrono::Utc;

/// Payload kind tag.
pub const PAYLOAD_KIND_RESOURCE: &str = "generic_mcp_resource";

/// Build a `RawRecord` for one MCP resource.
pub fn raw_resource(uri: &str, content: String, instance: Option<&str>) -> RawRecord {
    let native_id = match instance {
        Some(i) => format!("{i}|{uri}"),
        None => format!("upstream|{uri}"),
    };
    let payload = serde_json::json!({
        "payload_kind": PAYLOAD_KIND_RESOURCE,
        "uri": uri,
        "content": content,
    });
    RawRecord {
        native_id,
        native_path: Some(uri.to_string()),
        payload,
        captured_at: Utc::now(),
    }
}

/// `RawRecord` → `AnamnesisRecord`.
pub fn normalize(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let payload_kind = raw
        .payload
        .get("payload_kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("missing payload_kind".into()))?;
    if payload_kind != PAYLOAD_KIND_RESOURCE {
        return Err(Error::InvalidRecord(format!(
            "unexpected payload_kind: {payload_kind}"
        )));
    }
    let uri = raw
        .payload
        .get("uri")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("missing uri".into()))?
        .to_string();
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
        scope: Scope::Ephemeral,
        kind: Kind::Unknown,
        created_at: raw.captured_at,
        updated_at: None,
        tags: vec![uri.clone()],
        metadata: serde_json::Map::from_iter([(
            "source_uri".into(),
            serde_json::Value::String(uri),
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

    #[test]
    fn raw_resource_has_expected_shape() {
        let raw = raw_resource("anamnesis://record/abc", "body".into(), Some("upstream"));
        assert_eq!(raw.payload["payload_kind"], PAYLOAD_KIND_RESOURCE);
        assert_eq!(raw.payload["uri"], "anamnesis://record/abc");
        assert!(raw.native_id.starts_with("upstream|"));
    }

    #[test]
    fn normalize_yields_unknown_kind_ephemeral_scope() {
        let raw = raw_resource("anamnesis://record/x", "body".into(), Some("upstream"));
        let r = &normalize(raw, Some("upstream")).unwrap()[0];
        assert_eq!(r.kind, Kind::Unknown);
        assert_eq!(r.scope, Scope::Ephemeral);
        assert_eq!(r.content, "body");
        assert!(r.tags.iter().any(|t| t == "anamnesis://record/x"));
    }

    #[test]
    fn wrong_payload_kind_errors() {
        let raw = RawRecord {
            native_id: "x".into(),
            native_path: None,
            payload: serde_json::json!({"payload_kind": "nope", "uri": "u", "content": "c"}),
            captured_at: Utc::now(),
        };
        let err = normalize(raw, None).unwrap_err();
        assert!(format!("{err}").contains("unexpected payload_kind"));
    }

    #[test]
    fn record_id_is_instance_scoped() {
        let r1 = raw_resource("u", "x".into(), Some("a"));
        let r2 = raw_resource("u", "x".into(), Some("b"));
        let a = &normalize(r1, Some("a")).unwrap()[0];
        let b = &normalize(r2, Some("b")).unwrap()[0];
        assert_ne!(a.id, b.id);
    }
}
