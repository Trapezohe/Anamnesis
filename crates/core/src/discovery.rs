//! Discovery — what `SourceDetector` impls return.
//!
//! Detection is the **metadata-only** phase. It enumerates likely memory
//! sources on the host (paths, DB files, API endpoints), but never reads
//! memory content. The user reviews the list and explicitly opts in before
//! `import` reads any payload.
//!
//! See `docs/BLUEPRINT.md §3.3` for the layering rationale.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// How confident the detector is that the discovered location is a real
/// memory source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    /// Path and structure both match (e.g. `~/.claude/projects/*/conversation.jsonl`).
    /// CLI may suggest a one-shot import.
    High,
    /// Path matches but structure is partial (e.g. dir exists, no files yet).
    /// User confirmation required.
    Medium,
    /// Weak signal — only a suspicious path or old config hit. Shown but not
    /// pre-selected.
    Low,
}

/// What `SourceDetector::detect` returns for each candidate.
///
/// Carries enough for a CLI/UI to render a "found these sources, import? [y/N]"
/// prompt without ever reading user content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetectedSource {
    /// Adapter that produced this detection — must match
    /// `SourceDescriptor::adapter`.
    pub adapter: String,
    /// Optional instance discriminator the detector inferred (e.g.
    /// `"default"`, `"work-vault"`). Adapter decides the convention.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance: Option<String>,
    /// Where the source lives. `path` for filesystem; URL/host string for
    /// network sources. Free-form; only the adapter parses it.
    pub location: String,
    /// Filesystem path if the source is local; convenient for UI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_path: Option<PathBuf>,
    /// Detector's self-rated confidence.
    pub confidence: Confidence,
    /// Cheap record-count estimate (file count, row count, …). `None` if the
    /// adapter cannot estimate without reading content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_records: Option<u64>,
    /// Free-form note rendered to the user (e.g. "writable", "read-only",
    /// "schema v2 detected"). Must not contain memory content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl DetectedSource {
    /// Convenience constructor for filesystem-backed sources.
    pub fn local(
        adapter: impl Into<String>,
        path: impl Into<PathBuf>,
        confidence: Confidence,
    ) -> Self {
        let path = path.into();
        Self {
            adapter: adapter.into(),
            instance: None,
            location: path.display().to_string(),
            local_path: Some(path),
            confidence,
            estimated_records: None,
            note: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_constructor_fills_location() {
        let d = DetectedSource::local("claude-code", "/tmp/x", Confidence::High);
        assert_eq!(d.adapter, "claude-code");
        assert_eq!(d.location, "/tmp/x");
        assert_eq!(
            d.local_path.as_deref(),
            Some(std::path::Path::new("/tmp/x"))
        );
        assert_eq!(d.confidence, Confidence::High);
    }

    #[test]
    fn detected_source_roundtrips_through_json() {
        let d = DetectedSource {
            adapter: "mem0".into(),
            instance: Some("self-hosted".into()),
            location: "/Users/x/.mem0/db.sqlite".into(),
            local_path: Some("/Users/x/.mem0/db.sqlite".into()),
            confidence: Confidence::High,
            estimated_records: Some(1234),
            note: Some("schema v3".into()),
        };
        let s = serde_json::to_string(&d).unwrap();
        let back: DetectedSource = serde_json::from_str(&s).unwrap();
        assert_eq!(d, back);
    }

    #[test]
    fn confidence_serializes_lowercase() {
        let s = serde_json::to_string(&Confidence::Medium).unwrap();
        assert_eq!(s, "\"medium\"");
    }

    #[test]
    fn optional_fields_omitted_when_none() {
        let d = DetectedSource::local("claude-code", "/tmp/x", Confidence::Low);
        let s = serde_json::to_string(&d).unwrap();
        assert!(!s.contains("instance"));
        assert!(!s.contains("estimated_records"));
        assert!(!s.contains("note"));
    }
}
