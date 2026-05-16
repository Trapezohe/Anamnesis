//! Core domain types — the cross-adapter memory schema.
//!
//! See `docs/BLUEPRINT.md §4` for the rationale behind each field.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Current schema version. Bumped on breaking changes; minor bumps stay
/// read-compatible with at least one prior version.
pub const SCHEMA_VERSION: u32 = 1;

/// Globally unique record identifier — `blake3(adapter:instance?:native_id)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RecordId(pub String);

impl RecordId {
    /// Compute a deterministic record id from the natural key.
    pub fn from_parts(adapter: &str, instance: Option<&str>, native_id: &str) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(adapter.as_bytes());
        hasher.update(b":");
        hasher.update(instance.unwrap_or("").as_bytes());
        hasher.update(b":");
        hasher.update(native_id.as_bytes());
        Self(hasher.finalize().to_hex().to_string())
    }
}

impl std::fmt::Display for RecordId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Describes where a record came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceDescriptor {
    /// Adapter identifier (`"claude-code"`, `"mem0"`, …).
    pub adapter: String,
    /// Optional instance discriminator (e.g. `"work-vault"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance: Option<String>,
    /// Adapter version reporting itself.
    pub version: String,
}

/// Optional embedding payload. Original vectors are kept verbatim; cross-source
/// search re-embeds lazily when needed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    /// The raw embedding vector.
    pub vector: Vec<f32>,
    /// Embedding model identifier (e.g. `"voyage-3"`).
    pub model: String,
    /// Dimensionality (cached for convenience).
    pub dim: u16,
}

/// How broad is the record's relevance scope?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    /// Stable user-level fact or preference.
    User,
    /// Tied to a specific project/repo.
    Project,
    /// Bounded to a single conversation/session.
    Session,
    /// Short-lived working memory.
    Ephemeral,
}

/// What kind of thing is this record?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// Objective fact ("user prefers zsh").
    Fact,
    /// Stylistic / behavioural preference.
    Preference,
    /// Correction or feedback to an agent.
    Feedback,
    /// External pointer ("bugs tracked in Linear INGEST").
    Reference,
    /// A conversational episode or event log.
    Episode,
    /// A skill or way-of-working the agent should adopt.
    Skill,
    /// Unclassified.
    Unknown,
}

/// Provenance — how to trace this record back to its source-of-truth.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// Native id within the source system.
    pub native_id: String,
    /// File path or DB row reference, if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_path: Option<String>,
    /// When Anamnesis captured this record.
    pub captured_at: DateTime<Utc>,
    /// `blake3` of the raw payload — used for cheap incremental dedup.
    pub raw_hash: String,
}

/// The unified cross-source memory record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnamnesisRecord {
    /// Stable global id.
    pub id: RecordId,
    /// Where this came from.
    pub source: SourceDescriptor,
    /// The actual memory content.
    pub content: String,
    /// Optional embedding (preserved verbatim from the source).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Embedding>,
    /// Relevance scope.
    pub scope: Scope,
    /// Record kind.
    pub kind: Kind,
    /// When the record was originally created in the source.
    pub created_at: DateTime<Utc>,
    /// Last update timestamp, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<DateTime<Utc>>,
    /// Free-form tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Adapter-specific metadata.
    #[serde(default)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
    /// Provenance and audit trail.
    pub provenance: Provenance,
    /// Schema version this record was produced against.
    pub schema_version: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_id_is_deterministic() {
        let a = RecordId::from_parts("claude-code", Some("default"), "abc123");
        let b = RecordId::from_parts("claude-code", Some("default"), "abc123");
        let c = RecordId::from_parts("claude-code", Some("other"), "abc123");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn schema_version_is_one() {
        assert_eq!(SCHEMA_VERSION, 1);
    }

    #[test]
    fn record_roundtrips_through_json() {
        let r = AnamnesisRecord {
            id: RecordId::from_parts("claude-code", None, "x"),
            source: SourceDescriptor {
                adapter: "claude-code".into(),
                instance: None,
                version: "0.0.1".into(),
            },
            content: "user prefers vim".into(),
            embedding: None,
            scope: Scope::User,
            kind: Kind::Preference,
            created_at: Utc::now(),
            updated_at: None,
            tags: vec!["editor".into()],
            metadata: Default::default(),
            provenance: Provenance {
                native_id: "x".into(),
                native_path: Some("memory/editor.md".into()),
                captured_at: Utc::now(),
                raw_hash: "deadbeef".into(),
            },
            schema_version: SCHEMA_VERSION,
        };
        let s = serde_json::to_string(&r).unwrap();
        let back: AnamnesisRecord = serde_json::from_str(&s).unwrap();
        assert_eq!(r, back);
    }
}
