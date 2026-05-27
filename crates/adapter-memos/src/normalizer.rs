//! Normalize MemOS raw records into `AnamnesisRecord`s.

use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_core::RawRecord;
use chrono::{TimeZone, Utc};
use serde_json::{json, Value};

use crate::scanner::{parse_memos_time, MemosTextItem};
use crate::ADAPTER_ID;

/// `payload_kind` discriminator: one textual MemOS item.
pub const PAYLOAD_KIND_TEXT_ITEM: &str = "memos_text_item";

/// Wrap a textual item.
pub fn raw_from_item(i: &MemosTextItem, instance: Option<&str>) -> RawRecord {
    let native_id = synth_id(
        instance,
        &format!("text|{}|{}", i.cube_dir.display(), i.item_id),
    );
    RawRecord {
        native_id,
        native_path: Some(format!("{}::{}", i.cube_dir.display(), i.item_id)),
        payload: json!({
            "payload_kind": PAYLOAD_KIND_TEXT_ITEM,
            "cube_dir": i.cube_dir.display().to_string(),
            "item_id": i.item_id,
            "content": i.content,
            "memory_type": i.memory_type,
            "user_id": i.user_id,
            "session_id": i.session_id,
            "source": i.source,
            "tags": i.tags,
            "updated_at": i.updated_at,
            "created_at": i.created_at,
            "metadata_raw": i.metadata_raw,
        }),
        captured_at: Utc::now(),
    }
}

fn synth_id(instance: Option<&str>, local: &str) -> String {
    let instance = instance.unwrap_or("local");
    let hashed = blake3::hash(local.as_bytes()).to_hex();
    format!("{instance}|{}", &hashed[..32])
}

/// Normalize one `RawRecord` → `AnamnesisRecord`.
pub fn normalize(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let payload_kind = raw
        .payload
        .get("payload_kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("memos: missing payload_kind".into()))?;
    if payload_kind != PAYLOAD_KIND_TEXT_ITEM {
        return Err(Error::InvalidRecord(format!(
            "memos: unexpected payload_kind {payload_kind:?}"
        )));
    }

    let content = raw
        .payload
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("memos: content missing".into()))?
        .to_string();
    let memory_type = raw
        .payload
        .get("memory_type")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let (kind, scope) = kind_scope_for(memory_type.as_deref());

    // Prefer `updated_at`; fall back to `created_at`; last resort raw.captured_at.
    let created_at = raw
        .payload
        .get("updated_at")
        .and_then(|v| v.as_str())
        .and_then(parse_memos_time)
        .or_else(|| {
            raw.payload
                .get("created_at")
                .and_then(|v| v.as_str())
                .and_then(parse_memos_time)
        })
        .and_then(|t| Utc.timestamp_opt(t, 0).single())
        .unwrap_or(raw.captured_at);

    let record_id = RecordId::from_parts(ADAPTER_ID, instance, &raw.native_id);
    let raw_hash = blake3::hash(content.as_bytes()).to_hex().to_string();

    let mut metadata = serde_json::Map::new();
    metadata.insert("memos_tier".into(), Value::String("text_item".into()));
    for key in [
        "memory_type",
        "user_id",
        "session_id",
        "source",
        "item_id",
        "cube_dir",
    ] {
        if let Some(v) = raw.payload.get(key) {
            if !v.is_null() {
                metadata.insert(format!("memos_{key}"), v.clone());
            }
        }
    }
    let tags = raw
        .payload
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if let Some(meta_raw) = raw.payload.get("metadata_raw") {
        if !meta_raw.is_null() {
            // Surface round-trip provenance (`anamnesis_*`) to the top level so
            // reconcile's identity key sees it; keep the raw blob untouched.
            if let Some(obj) = meta_raw.as_object() {
                for (k, v) in obj {
                    if k.starts_with("anamnesis_") && !matches!(v.as_str(), Some("")) {
                        metadata.entry(k.clone()).or_insert_with(|| v.clone());
                    }
                }
            }
            metadata.insert("memos_metadata_raw".into(), meta_raw.clone());
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
        tags,
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

fn kind_scope_for(memory_type: Option<&str>) -> (Kind, Scope) {
    // MemOS canonical `memory_type` values come from
    // TreeNodeTextualMemoryMetadata. Flat backends sometimes store a
    // free-form `type` instead — we accept those too and treat unknowns
    // conservatively as Reference/User.
    match memory_type.unwrap_or("") {
        "WorkingMemory" => (Kind::Reference, Scope::Ephemeral),
        "LongTermMemory" => (Kind::Fact, Scope::User),
        "UserMemory" | "PreferenceMemory" => (Kind::Preference, Scope::User),
        "OuterMemory" => (Kind::Reference, Scope::User),
        "ToolSchemaMemory" | "SkillMemory" => (Kind::Skill, Scope::Project),
        "ToolTrajectoryMemory" => (Kind::Episode, Scope::Project),
        "RawFileMemory" => (Kind::Reference, Scope::Project),
        // Flat-backend `type` heuristics.
        "fact" => (Kind::Fact, Scope::User),
        "opinion" | "preference" => (Kind::Preference, Scope::User),
        "event" => (Kind::Episode, Scope::User),
        "procedure" | "skill" => (Kind::Skill, Scope::User),
        _ => (Kind::Reference, Scope::User),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn item(memory_type: Option<&str>, body: &str) -> MemosTextItem {
        MemosTextItem {
            cube_dir: PathBuf::from("/fake/cube"),
            item_id: "id-1".into(),
            content: body.into(),
            memory_type: memory_type.map(str::to_owned),
            user_id: Some("u-1".into()),
            session_id: Some("s-1".into()),
            source: Some("conversation".into()),
            tags: vec!["t1".into(), "t2".into()],
            updated_at: Some("2026-05-01T10:00:00".into()),
            created_at: Some("2026-04-01T10:00:00".into()),
            metadata_raw: json!({"memory_type": memory_type, "user_id": "u-1"}),
        }
    }

    #[test]
    fn user_memory_normalizes_to_preference_user() {
        let r = normalize(
            raw_from_item(&item(Some("UserMemory"), "prefer rust"), None),
            None,
        )
        .unwrap();
        assert_eq!(r[0].kind, Kind::Preference);
        assert_eq!(r[0].scope, Scope::User);
        assert_eq!(r[0].tags, vec!["t1", "t2"]);
        // updated_at parsed.
        assert_eq!(r[0].created_at.timestamp(), 1_777_629_600);
    }

    #[test]
    fn long_term_memory_normalizes_to_fact_user() {
        let r = normalize(
            raw_from_item(&item(Some("LongTermMemory"), "Paris is capital"), None),
            None,
        )
        .unwrap();
        assert_eq!(r[0].kind, Kind::Fact);
        assert_eq!(r[0].scope, Scope::User);
    }

    #[test]
    fn working_memory_normalizes_to_reference_ephemeral() {
        let r = normalize(
            raw_from_item(&item(Some("WorkingMemory"), "scratch"), None),
            None,
        )
        .unwrap();
        assert_eq!(r[0].kind, Kind::Reference);
        assert_eq!(r[0].scope, Scope::Ephemeral);
    }

    /// R156: a round-trip export's `anamnesis_*` provenance (carried in the
    /// item `metadata`) is surfaced to top-level metadata so reconcile's
    /// identity key finds `anamnesis_native_id`; the raw blob is kept too.
    #[test]
    fn normalize_surfaces_anamnesis_metadata_top_level() {
        let mut it = item(Some("UserMemory"), "round-trip body");
        it.metadata_raw = json!({
            "anamnesis_native_id": "orig-9",
            "anamnesis_source_adapter": "letta",
            "memory_type": "UserMemory"
        });
        let r = &normalize(raw_from_item(&it, None), None).unwrap()[0];
        assert_eq!(
            r.metadata
                .get("anamnesis_native_id")
                .and_then(|v| v.as_str()),
            Some("orig-9")
        );
        assert_eq!(
            r.metadata
                .get("anamnesis_source_adapter")
                .and_then(|v| v.as_str()),
            Some("letta")
        );
        assert!(r.metadata.contains_key("memos_metadata_raw"));
    }

    /// Native memos data (no anamnesis_*) gains no spurious top-level keys.
    #[test]
    fn normalize_native_metadata_adds_no_anamnesis_keys() {
        let r = &normalize(
            raw_from_item(&item(Some("UserMemory"), "native"), None),
            None,
        )
        .unwrap()[0];
        assert!(!r.metadata.keys().any(|k| k.starts_with("anamnesis_")));
        assert!(r.metadata.contains_key("memos_metadata_raw"));
    }

    #[test]
    fn tool_trajectory_normalizes_to_episode_project() {
        let r = normalize(
            raw_from_item(&item(Some("ToolTrajectoryMemory"), "ran search"), None),
            None,
        )
        .unwrap();
        assert_eq!(r[0].kind, Kind::Episode);
        assert_eq!(r[0].scope, Scope::Project);
    }

    #[test]
    fn skill_memory_normalizes_to_skill_project() {
        let r = normalize(
            raw_from_item(&item(Some("SkillMemory"), "search-web how-to"), None),
            None,
        )
        .unwrap();
        assert_eq!(r[0].kind, Kind::Skill);
        assert_eq!(r[0].scope, Scope::Project);
    }

    #[test]
    fn naive_type_fact_normalizes_to_fact_user() {
        // Flat-backend heuristic: lowercase `fact` in `type` → Fact/User.
        let r = normalize(raw_from_item(&item(Some("fact"), "body"), None), None).unwrap();
        assert_eq!(r[0].kind, Kind::Fact);
        assert_eq!(r[0].scope, Scope::User);
    }

    #[test]
    fn unknown_memory_type_falls_back_to_reference_user() {
        let r = normalize(
            raw_from_item(&item(Some("MysteryType"), "body"), None),
            None,
        )
        .unwrap();
        assert_eq!(r[0].kind, Kind::Reference);
        assert_eq!(r[0].scope, Scope::User);
    }

    #[test]
    fn unknown_payload_kind_rejected() {
        let raw = RawRecord {
            native_id: "x".into(),
            native_path: None,
            payload: json!({"payload_kind": "memos_NOPE"}),
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
