//! Discovery — `SourceDetector` trait + orchestrator.
//!
//! Detection is the **metadata-only** phase. It enumerates likely memory
//! sources on the host (paths, DB files, API endpoints), but never reads
//! memory content. The user reviews the list and explicitly opts in before
//! `import` reads any payload.
//!
//! See `docs/BLUEPRINT.md §3.3` and §7 (security: discover white-list) for
//! the layering rationale.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

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

/// Options passed to `SourceDetector::detect`.
///
/// Detectors must honour `home_override` so contract tests can run in
/// temp dirs without touching the user's real `$HOME`.
#[derive(Debug, Clone, Default)]
pub struct DetectOpts {
    /// Pretend this is `$HOME` for default-path probes. `None` = use the
    /// real environment.
    pub home_override: Option<PathBuf>,
    /// Extra paths supplied via `--path` that bypass the default whitelist.
    /// Detectors may still ignore paths they don't recognise.
    pub extra_paths: Vec<PathBuf>,
}

/// The detection contract.
///
/// Implementations MUST NOT read user memory content during `detect` —
/// only metadata (path existence, glob counts, DB schema names, API ping
/// status). Reading content is the `import` phase's job and requires
/// explicit user opt-in. See `docs/BLUEPRINT.md §3.3`.
#[async_trait]
pub trait SourceDetector: Send + Sync {
    /// Stable adapter identifier, must match the adapter's
    /// `SourceDescriptor::adapter`.
    fn adapter_id(&self) -> &'static str;

    /// Enumerate likely sources for this adapter. Empty result is valid
    /// and means "I looked and didn't find anything".
    async fn detect(&self, opts: &DetectOpts) -> Result<Vec<DetectedSource>>;
}

/// Orchestrator: runs a collection of detectors and merges their output.
///
/// `Discovery` itself owns no IO state — it is a thin facade that lets the
/// CLI call one method instead of looping over detectors. Per-detector
/// errors are surfaced; one detector failing does not poison results from
/// other detectors.
pub struct Discovery {
    detectors: Vec<Box<dyn SourceDetector>>,
}

impl Discovery {
    /// Empty orchestrator. Register detectors with `register`.
    pub fn new() -> Self {
        Self {
            detectors: Vec::new(),
        }
    }

    /// Register a detector. Order of registration is the order of results.
    pub fn register(mut self, detector: Box<dyn SourceDetector>) -> Self {
        self.detectors.push(detector);
        self
    }

    /// Number of registered detectors — useful for sanity checks in tests
    /// and `anamnesis status`.
    pub fn len(&self) -> usize {
        self.detectors.len()
    }

    /// Returns whether any detector is registered.
    pub fn is_empty(&self) -> bool {
        self.detectors.is_empty()
    }

    /// Run every detector and concatenate their results.
    ///
    /// A detector returning `Err` is logged via `tracing::warn!` and its
    /// results are skipped; remaining detectors still run. Use
    /// `detect_strict` if you want a single error to abort the whole run.
    pub async fn detect_all(&self, opts: &DetectOpts) -> Vec<DetectedSource> {
        let mut out = Vec::new();
        for d in &self.detectors {
            match d.detect(opts).await {
                Ok(found) => out.extend(found),
                Err(e) => {
                    tracing::warn!(
                        adapter = d.adapter_id(),
                        error = %e,
                        "detector failed; skipping its results"
                    );
                }
            }
        }
        out
    }

    /// Strict variant: a single detector error aborts.
    pub async fn detect_strict(&self, opts: &DetectOpts) -> Result<Vec<DetectedSource>> {
        let mut out = Vec::new();
        for d in &self.detectors {
            out.extend(d.detect(opts).await?);
        }
        Ok(out)
    }
}

impl Default for Discovery {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

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

    /// In-memory detector used to exercise the orchestrator.
    struct FakeDetector {
        id: &'static str,
        result: std::sync::Mutex<Vec<DetectedSource>>,
        always_err: bool,
    }

    impl FakeDetector {
        fn ok(id: &'static str, found: Vec<DetectedSource>) -> Self {
            Self {
                id,
                result: std::sync::Mutex::new(found),
                always_err: false,
            }
        }
        fn failing(id: &'static str) -> Self {
            Self {
                id,
                result: std::sync::Mutex::new(Vec::new()),
                always_err: true,
            }
        }
    }

    #[async_trait]
    impl SourceDetector for FakeDetector {
        fn adapter_id(&self) -> &'static str {
            self.id
        }
        async fn detect(&self, _opts: &DetectOpts) -> Result<Vec<DetectedSource>> {
            if self.always_err {
                return Err(Error::Other(format!("{} broken", self.id)));
            }
            Ok(self.result.lock().unwrap().clone())
        }
    }

    #[tokio::test]
    async fn discovery_runs_every_detector_in_registration_order() {
        let a = DetectedSource::local("a-adapter", "/tmp/a", Confidence::High);
        let b = DetectedSource::local("b-adapter", "/tmp/b", Confidence::Medium);
        let d = Discovery::new()
            .register(Box::new(FakeDetector::ok("a-adapter", vec![a.clone()])))
            .register(Box::new(FakeDetector::ok("b-adapter", vec![b.clone()])));
        assert_eq!(d.len(), 2);
        let opts = DetectOpts::default();
        let found = d.detect_all(&opts).await;
        assert_eq!(found, vec![a, b]);
    }

    #[tokio::test]
    async fn detect_all_skips_failing_detectors_but_keeps_others() {
        let a = DetectedSource::local("a-adapter", "/tmp/a", Confidence::High);
        let d = Discovery::new()
            .register(Box::new(FakeDetector::failing("broken")))
            .register(Box::new(FakeDetector::ok("a-adapter", vec![a.clone()])));
        let found = d.detect_all(&DetectOpts::default()).await;
        assert_eq!(found, vec![a]);
    }

    #[tokio::test]
    async fn detect_strict_propagates_first_error() {
        let d = Discovery::new()
            .register(Box::new(FakeDetector::failing("broken")))
            .register(Box::new(FakeDetector::ok("ok", vec![])));
        let err = d.detect_strict(&DetectOpts::default()).await.unwrap_err();
        assert!(format!("{err}").contains("broken"));
    }

    #[tokio::test]
    async fn empty_discovery_returns_empty() {
        let d = Discovery::new();
        assert!(d.is_empty());
        let found = d.detect_all(&DetectOpts::default()).await;
        assert!(found.is_empty());
    }

    #[test]
    fn detect_opts_default_is_clean() {
        let o = DetectOpts::default();
        assert!(o.home_override.is_none());
        assert!(o.extra_paths.is_empty());
    }
}
