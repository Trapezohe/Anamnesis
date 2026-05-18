//! Two-stage session extractor for Anamnesis.
//!
//! Per §-1.5 PR-6 north star item 6:
//!
//! > Session / Episode 类记忆能被显式两阶段抽取为长期 Fact / Preference /
//! > Feedback / Skill，且过程可审计、可重跑、不默默调 LLM。
//!
//! This crate implements **Stage 1** — a fully deterministic gate that
//! selects which `Episode` records are worth handing to an LLM extractor.
//! Stage 1 makes zero LLM calls and zero network requests. It scores each
//! record on observable features (content length, recency, has-user-input,
//! avoids tool-call noise) and emits a ranked list of [`Candidate`]s.
//!
//! Stage 2 — the LLM call that turns a `Candidate` into one or more typed
//! records — is scaffolded as the [`Stage2Plan`] type but not yet wired
//! to a provider. The CLI's `anamnesis extract --dry-run` exists today
//! and shows exactly what *would* be sent; the eventual non-dry-run mode
//! must show the same plan up front, ask for confirmation, and only
//! then call the LLM (§-1.5 #6, §-1.2 #5).
//!
//! ## Anti-goals (per §-1.2)
//!
//! - **No silent LLM calls.** Stage 2 must always be a separate command
//!   (`anamnesis extract`, never inside `anamnesis import`).
//! - **No mutation of source records.** Extracted records get NEW
//!   `RecordId`s; their provenance points back to the source Episode
//!   via [`Candidate::source_record_id`] — the original Episode is left
//!   untouched.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod gate;
pub mod prompt;
pub mod provider;
pub mod stage2;

use anamnesis_core::model::{AnamnesisRecord, Kind, RecordId};

pub use gate::{default_gate, DefaultGate, Stage1Gate, Stage1Score};
pub use prompt::build_prompt;
pub use provider::{cost_preview_line, LlmProvider, MockProvider};
pub use stage2::{
    parse_extracted_items, run_stage2, ExtractedItem, Stage2Report, EXTRACTOR_ADAPTER_ID,
};

/// Target kind for extraction. Maps to `anamnesis-core::Kind` variants
/// users typically want to distill out of raw conversation episodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractKind {
    /// Pull stable facts (long-lived, third-person statements about the
    /// user or the world).
    Fact,
    /// Pull preferences (first-person taste/style/workflow signals).
    Preference,
    /// Pull feedback (corrections the user gave during the session).
    Feedback,
    /// Pull skills (procedure / how-to / repeatable workflow).
    Skill,
}

impl ExtractKind {
    /// The `Kind` value Stage 2 emits onto each extracted record.
    pub fn target_kind(&self) -> Kind {
        match self {
            ExtractKind::Fact => Kind::Fact,
            ExtractKind::Preference => Kind::Preference,
            ExtractKind::Feedback => Kind::Feedback,
            ExtractKind::Skill => Kind::Skill,
        }
    }

    /// Lowercase CLI / docs label.
    pub fn as_str(&self) -> &'static str {
        match self {
            ExtractKind::Fact => "fact",
            ExtractKind::Preference => "preference",
            ExtractKind::Feedback => "feedback",
            ExtractKind::Skill => "skill",
        }
    }

    /// Parse the CLI / config string back to an [`ExtractKind`].
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "fact" => ExtractKind::Fact,
            "preference" | "pref" => ExtractKind::Preference,
            "feedback" => ExtractKind::Feedback,
            "skill" => ExtractKind::Skill,
            _ => return None,
        })
    }
}

/// One Episode record that survived the Stage-1 gate.
///
/// Carries the source `RecordId` so Stage 2 can stamp provenance on each
/// derived record. The Episode `content` is included verbatim — Stage 2
/// is responsible for any further chunking before LLM inference.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Original Episode record.
    pub record: AnamnesisRecord,
    /// Gate-assigned score; higher = more interesting to extract.
    pub score: f32,
    /// Human-readable explanation lines for `--explain` mode.
    pub rationale: Vec<String>,
}

impl Candidate {
    /// Convenience: the source record's id for provenance back-link.
    pub fn source_record_id(&self) -> &RecordId {
        &self.record.id
    }
}

/// Top-level Stage-1 driver: run `gate` over every record in `records`
/// that's already known to be an Episode (callers filter by Kind upstream
/// when reading from the store — we don't re-filter here so this is reusable
/// for other Kinds in future). Returns the surviving candidates sorted by
/// score descending.
///
/// `threshold` is the minimum score (inclusive) for a record to survive.
/// `limit` caps the returned list size (after sorting).
pub fn stage1_select<G: Stage1Gate>(
    records: impl IntoIterator<Item = AnamnesisRecord>,
    gate: &G,
    threshold: f32,
    limit: usize,
) -> Vec<Candidate> {
    let mut scored: Vec<Candidate> = records
        .into_iter()
        .map(|record| {
            let Stage1Score { score, rationale } = gate.score(&record);
            Candidate {
                record,
                score,
                rationale,
            }
        })
        .filter(|c| c.score >= threshold)
        .collect();
    // Sort by score desc (stable so equal scores keep input order).
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(limit);
    scored
}

/// Plan handed to Stage 2 before any LLM call. Calling code is required
/// to display this to the user **before** sending requests (§-1.5 #6).
#[derive(Debug, Clone)]
pub struct Stage2Plan {
    /// What kind the Stage 2 step will emit.
    pub target_kind: ExtractKind,
    /// Candidates that will be sent.
    pub candidates: Vec<Candidate>,
    /// Model id the user has configured (e.g. `"openai:gpt-4o-mini"`).
    pub model_id: String,
    /// Roughly how many LLM calls this plan will incur. Exact policy
    /// (1 call per candidate vs. batched) lives in the Stage 2 impl.
    pub estimated_llm_calls: usize,
}

impl Stage2Plan {
    /// Brief one-line summary suitable for CLI confirmation prompts.
    pub fn summary(&self) -> String {
        format!(
            "Stage 2 plan: {} candidates → {:?} via model {} (~{} LLM call(s))",
            self.candidates.len(),
            self.target_kind,
            self.model_id,
            self.estimated_llm_calls
        )
    }
}

/// Stage-1 → Stage-2 bridge. Builds a [`Stage2Plan`] from the gate's
/// output. The plan is a pure data object; no I/O happens here.
pub fn plan_stage2(
    candidates: Vec<Candidate>,
    target_kind: ExtractKind,
    model_id: impl Into<String>,
) -> Stage2Plan {
    let estimated_llm_calls = candidates.len();
    Stage2Plan {
        target_kind,
        candidates,
        model_id: model_id.into(),
        estimated_llm_calls,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anamnesis_core::model::{
        Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use chrono::Utc;

    fn episode(adapter: &str, content: &str, age_days: i64) -> AnamnesisRecord {
        let id = RecordId::from_parts(adapter, None, content);
        AnamnesisRecord {
            id,
            source: SourceDescriptor {
                adapter: adapter.into(),
                instance: None,
                version: "0".into(),
            },
            content: content.into(),
            embedding: None,
            scope: Scope::Session,
            kind: Kind::Episode,
            created_at: Utc::now() - chrono::Duration::days(age_days),
            updated_at: None,
            tags: vec![],
            metadata: serde_json::Map::new(),
            provenance: Provenance {
                native_id: "n1".into(),
                native_path: None,
                captured_at: Utc::now(),
                raw_hash: "h".into(),
                derived_from: None,
            },
            schema_version: SCHEMA_VERSION,
        }
    }

    #[test]
    fn extract_kind_parse_roundtrip() {
        for k in [
            ExtractKind::Fact,
            ExtractKind::Preference,
            ExtractKind::Feedback,
            ExtractKind::Skill,
        ] {
            assert_eq!(ExtractKind::parse(k.as_str()), Some(k));
        }
        assert_eq!(ExtractKind::parse("PREF"), Some(ExtractKind::Preference));
        assert!(ExtractKind::parse("nope").is_none());
    }

    #[test]
    fn extract_kind_maps_to_anamnesis_kind() {
        assert_eq!(ExtractKind::Fact.target_kind(), Kind::Fact);
        assert_eq!(ExtractKind::Preference.target_kind(), Kind::Preference);
        assert_eq!(ExtractKind::Feedback.target_kind(), Kind::Feedback);
        assert_eq!(ExtractKind::Skill.target_kind(), Kind::Skill);
    }

    #[test]
    fn stage1_select_filters_by_threshold_and_sorts_desc() {
        let r1 = episode("claude-code", "very short", 0);
        let r2 = episode(
            "claude-code",
            "this is a long, content-rich episode with substantial body that the gate should rate highly",
            1,
        );
        let r3 = episode("claude-code", "medium length", 5);
        let gate = default_gate();
        let out = stage1_select([r1, r2, r3], &gate, 0.0, 10);
        assert!(out.len() <= 3);
        // The long one should rank highest.
        assert!(out
            .first()
            .unwrap()
            .record
            .content
            .starts_with("this is a long"));
        // Scores must be monotonically non-increasing.
        for w in out.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn stage1_select_threshold_drops_low_scorers() {
        // `r1` is brackets-only — density sub-score zeros out, length is
        // also below the min_chars floor, so the only sub-score it scrapes
        // is recency. With weight 0.20, max possible score is 0.20.
        let r1 = episode("c", "[]", 0);
        let r2 = episode(
            "c",
            "longer body that the gate likes more than a tiny placeholder noise",
            0,
        );
        let gate = default_gate();
        let out = stage1_select([r1, r2], &gate, 0.5, 10);
        // Only the longer one survives a 0.5 threshold.
        assert!(out.iter().all(|c| c.score >= 0.5));
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn plan_stage2_estimates_one_call_per_candidate() {
        let cs = vec![
            Candidate {
                record: episode("c", "x", 0),
                score: 0.7,
                rationale: vec![],
            },
            Candidate {
                record: episode("c", "y", 0),
                score: 0.8,
                rationale: vec![],
            },
        ];
        let plan = plan_stage2(cs, ExtractKind::Fact, "openai:gpt-4o-mini");
        assert_eq!(plan.estimated_llm_calls, 2);
        assert_eq!(plan.target_kind, ExtractKind::Fact);
        assert_eq!(plan.candidates.len(), 2);
        assert!(plan.summary().contains("Stage 2 plan"));
    }

    #[test]
    fn candidate_source_record_id_is_input_record_id() {
        let r = episode("claude-code", "hello", 0);
        let expected_id = r.id.clone();
        let c = Candidate {
            record: r,
            score: 0.5,
            rationale: vec![],
        };
        assert_eq!(c.source_record_id().0, expected_id.0);
    }
}
