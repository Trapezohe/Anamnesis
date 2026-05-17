//! Normalize raw artifacts produced by the Claude Code scanner into
//! `AnamnesisRecord`s. See `docs/BLUEPRINT.md §6.8` for the mapping rules.

use std::path::Path;

use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_core::RawRecord;
use chrono::Utc;

use crate::frontmatter;

/// Payload `payload_kind` tag for a memory markdown file.
pub const PAYLOAD_KIND_MEMORY: &str = "memory_md";
/// Payload `payload_kind` tag for a conversation JSONL session file.
pub const PAYLOAD_KIND_SESSION: &str = "session_jsonl";

/// Build a `RawRecord` for a memory markdown file.
pub fn raw_memory(path: &Path, body: String, instance: Option<&str>) -> RawRecord {
    let native_id = synth_native_id(instance, "memory", path);
    let payload = serde_json::json!({
        "payload_kind": PAYLOAD_KIND_MEMORY,
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

/// Build a `RawRecord` for a session JSONL file. `body` is the full file
/// content rendered as a single string (the scanner pre-flattens messages).
pub fn raw_session(path: &Path, body: String, instance: Option<&str>) -> RawRecord {
    let native_id = synth_native_id(instance, "session", path);
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

fn synth_native_id(instance: Option<&str>, kind: &str, path: &Path) -> String {
    // Stable across runs; sensitive to path → re-encoded project dirs
    // produce new records (intentional — they really are different sources).
    let instance = instance.unwrap_or("default");
    format!("{instance}|{kind}|{}", path.display())
}

/// Normalize a single `RawRecord` produced by the Claude Code scanner.
pub fn normalize(raw: RawRecord, instance: Option<&str>) -> Result<Vec<AnamnesisRecord>> {
    let payload_kind = raw
        .payload
        .get("payload_kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("missing payload_kind".into()))?
        .to_string();
    let content = raw
        .payload
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::InvalidRecord("missing content".into()))?
        .to_string();
    match payload_kind.as_str() {
        PAYLOAD_KIND_MEMORY => Ok(vec![normalize_memory(&raw, instance, &content)]),
        PAYLOAD_KIND_SESSION => Ok(vec![normalize_session(&raw, instance, content)]),
        other => Err(Error::InvalidRecord(format!(
            "unknown payload_kind: {other}"
        ))),
    }
}

fn normalize_memory(raw: &RawRecord, instance: Option<&str>, raw_text: &str) -> AnamnesisRecord {
    let split = frontmatter::split(raw_text);
    let (kind, scope) = map_memory_type(split.frontmatter.mem_type.as_deref());
    let body = split.body.trim();
    let tags = split
        .frontmatter
        .name
        .iter()
        .cloned()
        .chain(split.frontmatter.description.iter().cloned())
        .collect::<Vec<_>>();
    let raw_hash = blake3::hash(raw_text.as_bytes()).to_hex().to_string();
    let id = RecordId::from_parts(crate::ADAPTER_ID, instance, &raw.native_id);

    // Always carry source_file for provenance; flow every preserved
    // frontmatter extra (originSessionId, node_type, custom annotations…)
    // alongside it so downstream consumers don't lose context.
    let mut metadata = serde_json::Map::new();
    metadata.insert(
        "source_file".into(),
        serde_json::Value::String(raw.native_path.clone().unwrap_or_default()),
    );
    for (k, v) in &split.frontmatter.extras {
        metadata.insert(k.clone(), serde_json::Value::String(v.clone()));
    }

    AnamnesisRecord {
        id,
        source: SourceDescriptor {
            adapter: crate::ADAPTER_ID.into(),
            instance: instance.map(str::to_owned),
            version: env!("CARGO_PKG_VERSION").into(),
        },
        content: body.to_string(),
        embedding: None,
        scope,
        kind,
        created_at: raw.captured_at,
        updated_at: None,
        tags: tags.into_iter().filter(|t| !t.is_empty()).collect(),
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

fn normalize_session(raw: &RawRecord, instance: Option<&str>, content: String) -> AnamnesisRecord {
    let raw_hash = blake3::hash(content.as_bytes()).to_hex().to_string();
    let id = RecordId::from_parts(crate::ADAPTER_ID, instance, &raw.native_id);
    AnamnesisRecord {
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
            native_id: raw.native_id.clone(),
            native_path: raw.native_path.clone(),
            captured_at: raw.captured_at,
            raw_hash,
        },
        schema_version: SCHEMA_VERSION,
    }
}

/// Map Claude Code memory frontmatter `type` to `(Kind, Scope)`.
/// See BLUEPRINT §6.8 rule 1.
pub fn map_memory_type(t: Option<&str>) -> (Kind, Scope) {
    match t {
        Some("user") => (Kind::Fact, Scope::User),
        Some("feedback") => (Kind::Feedback, Scope::User),
        Some("project") => (Kind::Fact, Scope::Project),
        Some("reference") => (Kind::Reference, Scope::User),
        Some("preference") => (Kind::Preference, Scope::User),
        Some("skill") => (Kind::Skill, Scope::User),
        _ => (Kind::Unknown, Scope::Ephemeral),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(format!("/tmp/{name}.md"))
    }

    #[test]
    fn memory_type_mapping() {
        assert_eq!(map_memory_type(Some("user")), (Kind::Fact, Scope::User));
        assert_eq!(
            map_memory_type(Some("feedback")),
            (Kind::Feedback, Scope::User)
        );
        assert_eq!(
            map_memory_type(Some("project")),
            (Kind::Fact, Scope::Project)
        );
        assert_eq!(
            map_memory_type(Some("reference")),
            (Kind::Reference, Scope::User)
        );
        assert_eq!(map_memory_type(None), (Kind::Unknown, Scope::Ephemeral));
        assert_eq!(
            map_memory_type(Some("garbage")),
            (Kind::Unknown, Scope::Ephemeral)
        );
    }

    #[test]
    fn raw_memory_constructor_shape() {
        let p = fixture_path("user_role");
        let raw = raw_memory(&p, "body".into(), Some("default"));
        assert_eq!(
            raw.native_path.as_deref(),
            Some(p.display().to_string()).as_deref()
        );
        assert!(raw.payload["payload_kind"].as_str() == Some(PAYLOAD_KIND_MEMORY));
        assert_eq!(raw.payload["content"].as_str(), Some("body"));
        assert!(raw.native_id.starts_with("default|memory|"));
    }

    #[test]
    fn raw_session_constructor_shape() {
        let p = fixture_path("session-xyz");
        let raw = raw_session(&p, "lots of text".into(), None);
        assert!(raw.native_id.starts_with("default|session|"));
        assert_eq!(raw.payload["payload_kind"], PAYLOAD_KIND_SESSION);
    }

    #[test]
    fn normalize_memory_user_frontmatter() {
        let path = fixture_path("user_role");
        let body = "---\nname: user-prefers-vim\ndescription: vim everywhere\nmetadata:\n  type: user\n---\nUser prefers vim.";
        let raw = raw_memory(&path, body.into(), Some("default"));
        let recs = normalize(raw, Some("default")).unwrap();
        assert_eq!(recs.len(), 1);
        let r = &recs[0];
        assert_eq!(r.kind, Kind::Fact);
        assert_eq!(r.scope, Scope::User);
        assert_eq!(r.content, "User prefers vim.");
        assert!(r.tags.contains(&"user-prefers-vim".to_string()));
        assert!(r.tags.contains(&"vim everywhere".to_string()));
        assert_eq!(r.source.adapter, "claude-code");
        assert_eq!(r.source.instance.as_deref(), Some("default"));
        assert!(!r.provenance.raw_hash.is_empty());
    }

    #[test]
    fn normalize_memory_feedback_keeps_scope_user() {
        let path = fixture_path("feedback_x");
        let body = "---\nname: x\nmetadata:\n  type: feedback\n---\ndon't do Y";
        let raw = raw_memory(&path, body.into(), None);
        let r = &normalize(raw, None).unwrap()[0];
        assert_eq!(r.kind, Kind::Feedback);
        assert_eq!(r.scope, Scope::User);
    }

    #[test]
    fn normalize_memory_project_scope() {
        let path = fixture_path("p");
        let body = "---\nname: p\nmetadata:\n  type: project\n---\nx";
        let r = &normalize(raw_memory(&path, body.into(), None), None).unwrap()[0];
        assert_eq!(r.kind, Kind::Fact);
        assert_eq!(r.scope, Scope::Project);
    }

    #[test]
    fn normalize_memory_without_frontmatter_falls_back_to_unknown() {
        let path = fixture_path("naked");
        let body = "just markdown body, no frontmatter";
        let r = &normalize(raw_memory(&path, body.into(), None), None).unwrap()[0];
        assert_eq!(r.kind, Kind::Unknown);
        assert_eq!(r.scope, Scope::Ephemeral);
        assert_eq!(r.content, body);
    }

    #[test]
    fn normalize_session_is_episode() {
        let path = fixture_path("session-abc");
        let body = r#"{"role":"user","content":"hi"}
{"role":"assistant","content":"hey"}"#;
        let recs = normalize(
            raw_session(&path, body.into(), Some("default")),
            Some("default"),
        )
        .unwrap();
        let r = &recs[0];
        assert_eq!(r.kind, Kind::Episode);
        assert_eq!(r.scope, Scope::Session);
        assert!(r.content.contains("\"role\":\"user\""));
    }

    #[test]
    fn unknown_payload_kind_errors() {
        let raw = RawRecord {
            native_id: "x".into(),
            native_path: None,
            payload: serde_json::json!({"payload_kind": "wat", "content": ""}),
            captured_at: Utc::now(),
        };
        let err = normalize(raw, None).unwrap_err();
        assert!(format!("{err}").contains("unknown payload_kind"));
    }

    #[test]
    fn missing_payload_kind_errors() {
        let raw = RawRecord {
            native_id: "x".into(),
            native_path: None,
            payload: serde_json::json!({"content": ""}),
            captured_at: Utc::now(),
        };
        let err = normalize(raw, None).unwrap_err();
        assert!(format!("{err}").contains("payload_kind"));
    }

    // ─── PR-G end-to-end: top-level type + extras preservation ───

    #[test]
    fn normalize_memory_with_top_level_type_classifies_as_feedback() {
        // Real-world example from BLUEPRINT §18.4 F2: my hand-written
        // ~/.claude/.../feedback_project_location.md was classified as
        // Unknown before PR-G because `type:` was at the top level.
        let path = fixture_path("feedback_project_location");
        let body = "---\nname: 项目目录放桌面\ndescription: 不要默认放 /tmp\ntype: feedback\noriginSessionId: 76a78a2d-e2af-4a15-9be4-f970d9e26e41\n---\n为新项目克隆或搭建工作目录时，默认放在 ~/Desktop。";
        let raw = raw_memory(&path, body.into(), Some("default"));
        let r = &normalize(raw, Some("default")).unwrap()[0];
        assert_eq!(
            r.kind,
            Kind::Feedback,
            "top-level type: feedback must reach the normalizer"
        );
        assert_eq!(r.scope, Scope::User);
        assert_eq!(
            r.metadata.get("originSessionId").and_then(|v| v.as_str()),
            Some("76a78a2d-e2af-4a15-9be4-f970d9e26e41"),
            "originSessionId must be preserved in record.metadata"
        );
    }

    #[test]
    fn normalize_memory_preserves_unknown_keys_in_metadata() {
        let path = fixture_path("reference");
        let body = "---\nname: env-cargo-path\nmetadata:\n  type: reference\n  node_type: memory\nweird_custom_field: keep-me\n---\nbody";
        let raw = raw_memory(&path, body.into(), None);
        let r = &normalize(raw, None).unwrap()[0];
        assert_eq!(r.kind, Kind::Reference);
        // node_type lives inside metadata: block; our minimal parser only
        // grabs `type` from there, so it doesn't reach extras — accepted
        // limitation, documented in BLUEPRINT §18.4 F2 fix proposal.
        // weird_custom_field IS top-level, so it must be preserved.
        assert_eq!(
            r.metadata
                .get("weird_custom_field")
                .and_then(|v| v.as_str()),
            Some("keep-me")
        );
    }

    #[test]
    fn record_id_is_deterministic_and_instance_scoped() {
        let path = fixture_path("a");
        let raw1 = raw_memory(&path, "body".into(), Some("workspace-a"));
        let raw2 = raw_memory(&path, "body".into(), Some("workspace-a"));
        // Same native_id (we synth deterministically), so id collides too.
        let r1 = &normalize(raw1, Some("workspace-a")).unwrap()[0];
        let r2 = &normalize(raw2, Some("workspace-a")).unwrap()[0];
        assert_eq!(r1.id, r2.id);
        // Different instance → different id.
        let raw3 = raw_memory(&path, "body".into(), Some("workspace-b"));
        let r3 = &normalize(raw3, Some("workspace-b")).unwrap()[0];
        assert_ne!(r1.id, r3.id);
    }
}
