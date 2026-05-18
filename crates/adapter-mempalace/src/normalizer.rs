//! Normalize MemPalace raw records into `AnamnesisRecord`s.

use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_core::RawRecord;
use chrono::{TimeZone, Utc};
use serde_json::{json, Value};

use crate::scanner::{filed_at_unix, MempalaceDrawer, MempalaceIdentity};
use crate::ADAPTER_ID;

/// `payload_kind` discriminator: the L0 identity file.
pub const PAYLOAD_KIND_IDENTITY: &str = "mempalace_identity";
/// `payload_kind` discriminator: one ChromaDB drawer / closet row.
pub const PAYLOAD_KIND_DRAWER: &str = "mempalace_drawer";

/// Wrap an identity record.
pub fn raw_from_identity(i: &MempalaceIdentity, instance: Option<&str>) -> RawRecord {
    let native_id = synth_id(instance, &format!("identity|{}", i.path.display()));
    RawRecord {
        native_id,
        native_path: Some(i.path.display().to_string()),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_IDENTITY,
            "path": i.path.display().to_string(),
            "content": i.content,
            "mtime_unix": i.mtime_unix,
        }),
        captured_at: Utc::now(),
    }
}

/// Wrap a drawer/closet record.
pub fn raw_from_drawer(d: &MempalaceDrawer, instance: Option<&str>) -> RawRecord {
    // Use the chroma `embedding_id` (e.g. `drawer_default_general_<sha>`) as
    // the local id — it's already content-derived and stable.
    let native_id = synth_id(
        instance,
        &format!("drawer|{}|{}", d.collection_name, d.embedding_id),
    );
    RawRecord {
        native_id,
        native_path: Some(format!("{}::{}", d.collection_name, d.embedding_id)),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_DRAWER,
            "collection_name": d.collection_name,
            "embedding_id": d.embedding_id,
            "content": d.content,
            "metadata": d.metadata,
            "created_unix": d.created_unix,
        }),
        captured_at: Utc::now(),
    }
}

fn synth_id(instance: Option<&str>, local: &str) -> String {
    let instance = instance.unwrap_or("local");
    let hashed = blake3::hash(local.as_bytes()).to_hex();
    format!("{instance}|{}", &hashed[..32])
}

/// Normalize one `RawRecord` (any MemPalace payload_kind) → `AnamnesisRecord`.
pub fn normalize(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let payload_kind = raw
        .payload
        .get("payload_kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("mempalace: missing payload_kind".into()))?;
    match payload_kind {
        PAYLOAD_KIND_IDENTITY => normalize_identity(raw, instance),
        PAYLOAD_KIND_DRAWER => normalize_drawer(raw, instance),
        other => Err(Error::InvalidRecord(format!(
            "mempalace: unexpected payload_kind {other:?}"
        ))),
    }
}

fn normalize_identity(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let content = raw
        .payload
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("mempalace: identity.content missing".into()))?
        .to_string();
    let mtime_unix = raw.payload.get("mtime_unix").and_then(|v| v.as_i64());
    let created_at = mtime_unix
        .and_then(|t| Utc.timestamp_opt(t, 0).single())
        .unwrap_or(raw.captured_at);

    let record_id = RecordId::from_parts(ADAPTER_ID, instance, &raw.native_id);
    let raw_hash = blake3::hash(content.as_bytes()).to_hex().to_string();
    let mut metadata = serde_json::Map::new();
    metadata.insert("mempalace_kind".into(), Value::String("identity".into()));

    Ok(vec![AnamnesisRecord {
        id: record_id,
        source: SourceDescriptor {
            adapter: ADAPTER_ID.into(),
            instance: instance.map(str::to_owned),
            version: env!("CARGO_PKG_VERSION").into(),
        },
        content,
        embedding: None,
        scope: Scope::User,
        kind: Kind::Preference,
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

fn normalize_drawer(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let content = raw
        .payload
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("mempalace: drawer.content missing".into()))?
        .to_string();
    let drawer_metadata = raw.payload.get("metadata").cloned().unwrap_or(Value::Null);
    let collection_name = raw
        .payload
        .get("collection_name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let embedding_id = raw
        .payload
        .get("embedding_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let created_unix = raw.payload.get("created_unix").and_then(|v| v.as_i64());

    // Closets are searchable summaries (Reference / Project); drawers are
    // mined memory chunks (Episode / Project).
    let kind = if collection_name == "mempalace_closets" {
        Kind::Reference
    } else {
        Kind::Episode
    };
    let scope = Scope::Project;

    // Prefer the Chroma row's created_at; fall back to metadata.filed_at;
    // last resort the raw.captured_at.
    let created_at = created_unix
        .and_then(|t| Utc.timestamp_opt(t, 0).single())
        .or_else(|| filed_at_unix(&drawer_metadata).and_then(|t| Utc.timestamp_opt(t, 0).single()))
        .unwrap_or(raw.captured_at);

    let record_id = RecordId::from_parts(ADAPTER_ID, instance, &raw.native_id);
    let raw_hash = blake3::hash(content.as_bytes()).to_hex().to_string();

    let mut metadata = serde_json::Map::new();
    metadata.insert(
        "mempalace_kind".into(),
        Value::String(if collection_name == "mempalace_closets" {
            "closet".into()
        } else {
            "drawer".into()
        }),
    );
    metadata.insert(
        "mempalace_collection".into(),
        Value::String(collection_name),
    );
    metadata.insert("mempalace_embedding_id".into(), Value::String(embedding_id));
    // Surface canonical wing/room/source_file in flat keys for easy filtering.
    for key in ["wing", "room", "source_file", "hall", "added_by"] {
        if let Some(v) = drawer_metadata.get(key) {
            metadata.insert(format!("mempalace_{key}"), v.clone());
        }
    }
    // Keep the full original metadata blob too for callers that want it.
    if !drawer_metadata.is_null() {
        metadata.insert("mempalace_metadata".into(), drawer_metadata);
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

    fn identity() -> MempalaceIdentity {
        MempalaceIdentity {
            path: PathBuf::from("/fake/identity.txt"),
            content: "I am Atlas, an AI for Alice.".into(),
            mtime_unix: Some(1_730_000_000),
        }
    }

    fn drawer(collection: &str, body: &str) -> MempalaceDrawer {
        MempalaceDrawer {
            collection_name: collection.into(),
            embedding_id: format!("{collection}_aaa"),
            content: body.into(),
            metadata: json!({
                "wing": "default",
                "room": "general",
                "source_file": "/repo/CLAUDE.md",
                "filed_at": "2026-05-01T10:00:00Z"
            }),
            created_unix: Some(1_730_000_000),
        }
    }

    #[test]
    fn identity_normalizes_to_preference_user() {
        let r = normalize(raw_from_identity(&identity(), None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Preference);
        assert_eq!(r[0].scope, Scope::User);
        assert_eq!(
            r[0].metadata.get("mempalace_kind").and_then(|v| v.as_str()),
            Some("identity")
        );
    }

    #[test]
    fn drawer_normalizes_to_episode_project() {
        let r = normalize(
            raw_from_drawer(&drawer("mempalace_drawers", "body"), None),
            None,
        )
        .unwrap();
        assert_eq!(r[0].kind, Kind::Episode);
        assert_eq!(r[0].scope, Scope::Project);
        assert_eq!(
            r[0].metadata.get("mempalace_kind").and_then(|v| v.as_str()),
            Some("drawer")
        );
        assert_eq!(
            r[0].metadata.get("mempalace_wing").and_then(|v| v.as_str()),
            Some("default")
        );
    }

    #[test]
    fn closet_normalizes_to_reference_project() {
        let r = normalize(
            raw_from_drawer(&drawer("mempalace_closets", "idx"), None),
            None,
        )
        .unwrap();
        assert_eq!(r[0].kind, Kind::Reference);
        assert_eq!(r[0].scope, Scope::Project);
        assert_eq!(
            r[0].metadata.get("mempalace_kind").and_then(|v| v.as_str()),
            Some("closet")
        );
    }

    #[test]
    fn drawer_created_at_falls_back_to_filed_at_when_chroma_missing() {
        let mut d = drawer("mempalace_drawers", "body");
        d.created_unix = None;
        let r = normalize(raw_from_drawer(&d, None), None).unwrap();
        // metadata.filed_at = "2026-05-01T10:00:00Z" → 1777629600
        assert_eq!(r[0].created_at.timestamp(), 1_777_629_600);
    }

    #[test]
    fn unknown_payload_kind_rejected() {
        let raw = RawRecord {
            native_id: "x".into(),
            native_path: None,
            payload: json!({"payload_kind": "mempalace_NOPE"}),
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
