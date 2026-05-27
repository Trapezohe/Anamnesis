//! Normalize raw artifacts produced by the Claude Code scanner into
//! `AnamnesisRecord`s. See `docs/BLUEPRINT.md §6.8` for the mapping rules.

use std::path::Path;

use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::{
    AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
};
use anamnesis_core::RawRecord;
use chrono::{DateTime, Utc};

use crate::frontmatter;
use crate::session;

/// Payload `payload_kind` tag for a memory markdown file.
pub const PAYLOAD_KIND_MEMORY: &str = "memory_md";
/// Payload `payload_kind` tag for a conversation JSONL session file.
pub const PAYLOAD_KIND_SESSION: &str = "session_jsonl";

/// Build a `RawRecord` for a memory markdown file.
///
/// `mtime` (when supplied) is preserved in the payload so the normalizer
/// can use it for `created_at` instead of "now" — see PR-I (BLUEPRINT
/// §18.4 F4).
pub fn raw_memory(
    path: &Path,
    body: String,
    mtime: Option<DateTime<Utc>>,
    instance: Option<&str>,
) -> RawRecord {
    let native_id = synth_native_id(instance, "memory", path);
    let mut payload = serde_json::json!({
        "payload_kind": PAYLOAD_KIND_MEMORY,
        "path": path.display().to_string(),
        "content": body,
    });
    if let Some(m) = mtime {
        payload["mtime"] = serde_json::Value::String(m.to_rfc3339());
    }
    RawRecord {
        native_id,
        native_path: Some(path.display().to_string()),
        payload,
        captured_at: Utc::now(),
    }
}

/// Build a `RawRecord` for a session JSONL file.
///
/// `jsonl_text` is parsed into structured turns now (rather than dumped
/// as a raw string) so the normalizer can render readable markdown and
/// the chunker can split on turn boundaries. See PR-H (BLUEPRINT §18.4 F3).
///
/// `mtime` is the file modification time. The normalizer prefers
/// `first_ts` from the parsed session, falling back to `mtime`, then to
/// the import timestamp.
pub fn raw_session(
    path: &Path,
    jsonl_text: &str,
    mtime: Option<DateTime<Utc>>,
    instance: Option<&str>,
) -> RawRecord {
    let native_id = synth_native_id(instance, "session", path);
    let parsed = session::parse_jsonl(jsonl_text);
    let mut payload = serde_json::json!({
        "payload_kind": PAYLOAD_KIND_SESSION,
        "path": path.display().to_string(),
        "messages": parsed.messages,
        "message_count": parsed.messages.len(),
    });
    if let Some(t) = parsed.first_ts {
        payload["first_ts"] = serde_json::Value::String(t.to_rfc3339());
    }
    if let Some(t) = parsed.last_ts {
        payload["last_ts"] = serde_json::Value::String(t.to_rfc3339());
    }
    if let Some(m) = mtime {
        payload["mtime"] = serde_json::Value::String(m.to_rfc3339());
    }
    RawRecord {
        native_id,
        native_path: Some(path.display().to_string()),
        payload,
        captured_at: Utc::now(),
    }
}

/// Parse an RFC3339 timestamp from a payload string field. Returns `None`
/// for absent or malformed values; the caller falls back to `captured_at`.
fn payload_ts(payload: &serde_json::Value, key: &str) -> Option<DateTime<Utc>> {
    payload
        .get(key)
        .and_then(|v| v.as_str())
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|d| d.with_timezone(&Utc))
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
    match payload_kind.as_str() {
        PAYLOAD_KIND_MEMORY => {
            let content = raw
                .payload
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::InvalidRecord("memory payload missing content".into()))?
                .to_string();
            Ok(vec![normalize_memory(&raw, instance, &content)])
        }
        PAYLOAD_KIND_SESSION => Ok(vec![normalize_session(&raw, instance)?]),
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
    // Round-trip exports (`claude-code-dir`) carry the original identity in a
    // frontmatter `anamnesis_native_id`; restore it so re-import reconciles as
    // `both`. Files without it keep the path-synthesized native_id.
    let native_id = split
        .frontmatter
        .extras
        .get("anamnesis_native_id")
        .filter(|s| !s.is_empty())
        .cloned()
        .unwrap_or_else(|| raw.native_id.clone());
    let id = RecordId::from_parts(crate::ADAPTER_ID, instance, &native_id);

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

    // PR-I: prefer file mtime over import wall clock for `created_at` so
    // time-window queries and recency boosts actually carry signal.
    let created_at = payload_ts(&raw.payload, "mtime").unwrap_or(raw.captured_at);

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
        created_at,
        updated_at: None,
        tags: tags.into_iter().filter(|t| !t.is_empty()).collect(),
        metadata,
        provenance: Provenance {
            native_id,
            native_path: raw.native_path.clone(),
            captured_at: raw.captured_at,
            raw_hash,
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    }
}

fn normalize_session(raw: &RawRecord, instance: Option<&str>) -> Result<AnamnesisRecord> {
    // PR-H: reconstruct turns from payload.messages → readable markdown.
    // The scanner already parsed JSONL; here we just render and hash.
    let messages_value = raw
        .payload
        .get("messages")
        .ok_or_else(|| Error::InvalidRecord("session payload missing messages".into()))?;
    let messages: Vec<session::SessionMessage> = serde_json::from_value(messages_value.clone())
        .map_err(|e| Error::InvalidRecord(format!("malformed session messages: {e}")))?;
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

    // Many sessions are empty after filtering (only file-history-snapshot /
    // permission-mode rows). Drop them at the boundary — we don't want
    // empty Episode records polluting the index.
    let content_owned = content;

    // PR-I: created_at = first message ts → mtime → captured_at;
    //        updated_at = last message ts → mtime when first_ts present.
    let created_at = first_ts.or(mtime).unwrap_or(raw.captured_at);
    let updated_at = last_ts.or_else(|| if first_ts.is_some() { mtime } else { None });

    let mut metadata = serde_json::Map::new();
    metadata.insert(
        "source_file".into(),
        serde_json::Value::String(raw.native_path.clone().unwrap_or_default()),
    );
    if let Some(n) = raw.payload.get("message_count").and_then(|v| v.as_u64()) {
        metadata.insert(
            "message_count".into(),
            serde_json::Value::Number(serde_json::Number::from(n)),
        );
    }

    Ok(AnamnesisRecord {
        id,
        source: SourceDescriptor {
            adapter: crate::ADAPTER_ID.into(),
            instance: instance.map(str::to_owned),
            version: env!("CARGO_PKG_VERSION").into(),
        },
        content: content_owned,
        embedding: None,
        scope: Scope::Session,
        kind: Kind::Episode,
        created_at,
        updated_at,
        tags: Vec::new(),
        metadata,
        provenance: Provenance {
            native_id: raw.native_id.clone(),
            native_path: raw.native_path.clone(),
            captured_at: raw.captured_at,
            raw_hash,
            derived_from: None,
        },
        schema_version: SCHEMA_VERSION,
    })
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
        let raw = raw_memory(&p, "body".into(), None, Some("default"));
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
        let raw = raw_session(&p, "lots of text", None, None);
        assert!(raw.native_id.starts_with("default|session|"));
        assert_eq!(raw.payload["payload_kind"], PAYLOAD_KIND_SESSION);
    }

    #[test]
    fn normalize_memory_user_frontmatter() {
        let path = fixture_path("user_role");
        let body = "---\nname: user-prefers-vim\ndescription: vim everywhere\nmetadata:\n  type: user\n---\nUser prefers vim.";
        let raw = raw_memory(&path, body.into(), None, Some("default"));
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
        // Backward compat: no anamnesis sentinel → path-synthesized native_id.
        assert!(r.provenance.native_id.starts_with("default|memory|"));
    }

    /// R155: a `claude-code-dir` round-trip file carries the original identity
    /// in a frontmatter `anamnesis_native_id`; restore it on re-import.
    #[test]
    fn normalize_memory_restores_anamnesis_native_id() {
        let path = fixture_path("anamnesis-export");
        let body = "---\nname: note\nmetadata:\n  type: user\nanamnesis_native_id: note-42\nanamnesis_source_adapter: mem0\n---\nthe body";
        let r = &normalize(raw_memory(&path, body.into(), None, None), None).unwrap()[0];
        assert_eq!(r.provenance.native_id, "note-42");
        assert_eq!(r.content, "the body");
        assert_eq!(
            r.metadata
                .get("anamnesis_source_adapter")
                .and_then(|v| v.as_str()),
            Some("mem0")
        );
    }

    #[test]
    fn normalize_memory_feedback_keeps_scope_user() {
        let path = fixture_path("feedback_x");
        let body = "---\nname: x\nmetadata:\n  type: feedback\n---\ndon't do Y";
        let raw = raw_memory(&path, body.into(), None, None);
        let r = &normalize(raw, None).unwrap()[0];
        assert_eq!(r.kind, Kind::Feedback);
        assert_eq!(r.scope, Scope::User);
    }

    #[test]
    fn normalize_memory_project_scope() {
        let path = fixture_path("p");
        let body = "---\nname: p\nmetadata:\n  type: project\n---\nx";
        let r = &normalize(raw_memory(&path, body.into(), None, None), None).unwrap()[0];
        assert_eq!(r.kind, Kind::Fact);
        assert_eq!(r.scope, Scope::Project);
    }

    #[test]
    fn normalize_memory_without_frontmatter_falls_back_to_unknown() {
        let path = fixture_path("naked");
        let body = "just markdown body, no frontmatter";
        let r = &normalize(raw_memory(&path, body.into(), None, None), None).unwrap()[0];
        assert_eq!(r.kind, Kind::Unknown);
        assert_eq!(r.scope, Scope::Ephemeral);
        assert_eq!(r.content, body);
    }

    #[test]
    fn normalize_session_renders_readable_markdown() {
        // PR-H: JSONL is the real Claude Code shape (typed outer rows with
        // a `message` envelope). After PR-H, content is human-readable
        // markdown, not raw JSON bytes.
        let path = fixture_path("session-abc");
        let body = [
            r#"{"type":"user","message":{"role":"user","content":"hi"},"timestamp":"2026-05-17T03:14:00Z","uuid":"u1"}"#,
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hey"}]},"timestamp":"2026-05-17T03:14:05Z","uuid":"a1"}"#,
        ]
        .join("\n");
        let recs = normalize(
            raw_session(&path, &body, None, Some("default")),
            Some("default"),
        )
        .unwrap();
        let r = &recs[0];
        assert_eq!(r.kind, Kind::Episode);
        assert_eq!(r.scope, Scope::Session);
        assert!(
            r.content.contains("**user**"),
            "rendered markdown must contain `**user**` header"
        );
        assert!(r.content.contains("hi"));
        assert!(r.content.contains("hey"));
        assert!(
            !r.content.contains("\"role\":\"user\""),
            "raw JSON bytes must NOT leak into the rendered content"
        );
        // PR-I: created_at derives from first message timestamp.
        assert_eq!(
            r.created_at.to_rfc3339(),
            "2026-05-17T03:14:00+00:00",
            "created_at must come from first message timestamp"
        );
        assert_eq!(
            r.updated_at.map(|t| t.to_rfc3339()),
            Some("2026-05-17T03:14:05+00:00".to_string()),
            "updated_at must come from last message timestamp"
        );
        // message_count metadata for analytics.
        assert_eq!(
            r.metadata.get("message_count").and_then(|v| v.as_u64()),
            Some(2)
        );
    }

    #[test]
    fn normalize_session_falls_back_to_mtime_when_no_message_timestamps() {
        // Session with messages that have no timestamps — should use mtime.
        let path = fixture_path("session-no-ts");
        let mtime = chrono::DateTime::parse_from_rfc3339("2026-04-01T10:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let body = r#"{"type":"user","message":{"role":"user","content":"hi"}}"#;
        let raw = raw_session(&path, body, Some(mtime), None);
        let r = &normalize(raw, None).unwrap()[0];
        assert_eq!(r.created_at, mtime);
    }

    #[test]
    fn normalize_memory_uses_mtime_for_created_at() {
        let path = fixture_path("mtime_test");
        let mtime = chrono::DateTime::parse_from_rfc3339("2025-12-25T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let body = "---\nname: x\nmetadata:\n  type: user\n---\nbody";
        let r = &normalize(raw_memory(&path, body.into(), Some(mtime), None), None).unwrap()[0];
        assert_eq!(r.created_at, mtime, "memory created_at = file mtime");
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
        let raw = raw_memory(&path, body.into(), None, Some("default"));
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
        let raw = raw_memory(&path, body.into(), None, None);
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
        let raw1 = raw_memory(&path, "body".into(), None, Some("workspace-a"));
        let raw2 = raw_memory(&path, "body".into(), None, Some("workspace-a"));
        // Same native_id (we synth deterministically), so id collides too.
        let r1 = &normalize(raw1, Some("workspace-a")).unwrap()[0];
        let r2 = &normalize(raw2, Some("workspace-a")).unwrap()[0];
        assert_eq!(r1.id, r2.id);
        // Different instance → different id.
        let raw3 = raw_memory(&path, "body".into(), None, Some("workspace-b"));
        let r3 = &normalize(raw3, Some("workspace-b")).unwrap()[0];
        assert_ne!(r1.id, r3.id);
    }
}
