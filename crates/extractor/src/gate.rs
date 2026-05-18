//! Stage 1 — deterministic gate.
//!
//! A `Stage1Gate` impl assigns a `[0.0, 1.0]` score to each candidate
//! Episode record using purely local features (no LLM, no network).
//! Higher score = more interesting to hand to Stage 2.
//!
//! The built-in [`DefaultGate`] uses three signals:
//!
//!   1. **Length** — Episodes shorter than ~40 chars rarely contain
//!      anything worth distilling. Long-but-not-spam episodes get the
//!      highest length sub-score.
//!   2. **Recency** — Newer episodes are weighted higher (decay over
//!      90 days). Avoids re-distilling years-old conversations on
//!      every extract run.
//!   3. **Content density** — Penalize episodes whose body is
//!      dominated by tool-call boilerplate or pure code fences. The
//!      heuristic looks for the ratio of letters-to-total-chars.
//!
//! The gate is intentionally conservative: scores are easy to read by
//! eye in `--explain` mode, and any tuning lives in a single
//! `DefaultGate` struct so it's straightforward to override per-source.

use anamnesis_core::model::AnamnesisRecord;
use chrono::Utc;

/// One gate evaluation result.
#[derive(Debug, Clone)]
pub struct Stage1Score {
    /// Final score in `[0.0, 1.0]`.
    pub score: f32,
    /// One short line per sub-signal explaining how `score` was derived.
    pub rationale: Vec<String>,
}

/// Deterministic Stage-1 gate. Implementations are pure CPU; the contract
/// is that `score(r)` for the same `r` always returns the same score.
pub trait Stage1Gate: Send + Sync {
    /// Stable human-readable identifier (used in audit logs).
    fn name(&self) -> &'static str;
    /// Score one candidate. Must be deterministic.
    fn score(&self, record: &AnamnesisRecord) -> Stage1Score;
}

/// Default heuristic gate. See module docs for the signals it uses.
#[derive(Debug, Clone)]
pub struct DefaultGate {
    /// Min chars for an Episode to be considered "substantive enough"
    /// — anything shorter gets a 0 length sub-score.
    pub min_chars: usize,
    /// Chars where length sub-score plateaus at 1.0.
    pub length_plateau: usize,
    /// Half-life (in days) for the recency exponential decay.
    pub recency_half_life_days: f32,
    /// Pass-mark threshold used by `passes` (purely advisory; the
    /// `stage1_select` driver also accepts a threshold separately).
    pub pass_threshold: f32,
}

impl Default for DefaultGate {
    fn default() -> Self {
        DefaultGate {
            min_chars: 40,
            length_plateau: 600,
            recency_half_life_days: 30.0,
            pass_threshold: 0.4,
        }
    }
}

impl Stage1Gate for DefaultGate {
    fn name(&self) -> &'static str {
        "default-heuristic-v1"
    }

    fn score(&self, record: &AnamnesisRecord) -> Stage1Score {
        let len = record.content.chars().count();
        let length_sub = length_subscore(len, self.min_chars, self.length_plateau);
        let now = Utc::now();
        let age_days = (now - record.created_at).num_seconds() as f32 / 86_400.0;
        let recency_sub = recency_subscore(age_days, self.recency_half_life_days);
        let density_sub = density_subscore(&record.content);

        // Weighted blend. Length and density dominate (signal is local to
        // content); recency is a tiebreaker.
        let score = (0.45 * length_sub) + (0.20 * recency_sub) + (0.35 * density_sub);
        let score = score.clamp(0.0, 1.0);

        let rationale = vec![
            format!(
                "len={} chars → length_sub={:.2} (min={}, plateau={})",
                len, length_sub, self.min_chars, self.length_plateau
            ),
            format!(
                "age={:.1}d → recency_sub={:.2} (half_life={:.0}d)",
                age_days, recency_sub, self.recency_half_life_days
            ),
            format!("letter ratio → density_sub={:.2}", density_sub),
            format!(
                "weighted score = 0.45×len + 0.20×recency + 0.35×density = {:.2}",
                score
            ),
        ];

        Stage1Score { score, rationale }
    }
}

/// Convenience constructor.
pub fn default_gate() -> DefaultGate {
    DefaultGate::default()
}

fn length_subscore(len: usize, min_chars: usize, plateau: usize) -> f32 {
    if len < min_chars {
        return 0.0;
    }
    if len >= plateau {
        return 1.0;
    }
    // Linear ramp from min_chars (0.0) up to plateau (1.0).
    let span = (plateau - min_chars) as f32;
    let over = (len - min_chars) as f32;
    (over / span).clamp(0.0, 1.0)
}

fn recency_subscore(age_days: f32, half_life_days: f32) -> f32 {
    if age_days <= 0.0 {
        return 1.0;
    }
    // Exponential decay: score = 2^(-age / half_life).
    let exp = -age_days / half_life_days;
    2.0_f32.powf(exp).clamp(0.0, 1.0)
}

/// Letter-to-total-char ratio over the content. Anything below ~0.4 is
/// usually a wall of brackets, code fences, or shell output that an LLM
/// would struggle to distill into a clean fact. Above ~0.65 is normal
/// prose and gets a strong density sub-score.
fn density_subscore(content: &str) -> f32 {
    let total = content.chars().count().max(1);
    let letters = content.chars().filter(|c| c.is_alphabetic()).count();
    let ratio = letters as f32 / total as f32;
    // Linear ramp from 0.40 (0.0) to 0.65 (1.0).
    if ratio <= 0.40 {
        return 0.0;
    }
    if ratio >= 0.65 {
        return 1.0;
    }
    let over = ratio - 0.40;
    let span = 0.65 - 0.40;
    (over / span).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use chrono::{Duration, Utc};

    fn record(content: &str, age_days: i64) -> AnamnesisRecord {
        AnamnesisRecord {
            id: RecordId::from_parts("test", None, content),
            source: SourceDescriptor {
                adapter: "test".into(),
                instance: None,
                version: "0".into(),
            },
            content: content.into(),
            embedding: None,
            scope: Scope::Session,
            kind: Kind::Episode,
            created_at: Utc::now() - Duration::days(age_days),
            updated_at: None,
            tags: vec![],
            metadata: serde_json::Map::new(),
            provenance: Provenance {
                native_id: "n1".into(),
                native_path: None,
                captured_at: Utc::now(),
                raw_hash: "h".into(),
            },
            schema_version: SCHEMA_VERSION,
        }
    }

    #[test]
    fn length_subscore_below_min_is_zero() {
        assert_eq!(length_subscore(10, 40, 600), 0.0);
        assert_eq!(length_subscore(39, 40, 600), 0.0);
    }

    #[test]
    fn length_subscore_at_plateau_is_one() {
        assert_eq!(length_subscore(600, 40, 600), 1.0);
        assert_eq!(length_subscore(1200, 40, 600), 1.0);
    }

    #[test]
    fn length_subscore_ramps_linearly() {
        let mid = length_subscore(320, 40, 600);
        // 320 is halfway between 40 and 600.
        assert!((mid - 0.5).abs() < 0.02, "mid={mid}");
    }

    #[test]
    fn recency_subscore_now_is_one() {
        assert!((recency_subscore(0.0, 30.0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn recency_subscore_half_life_is_half() {
        let s = recency_subscore(30.0, 30.0);
        assert!((s - 0.5).abs() < 1e-3, "s={s}");
    }

    #[test]
    fn density_subscore_all_brackets_is_zero() {
        let s = density_subscore("[[[[[[[[[[[[]]]]]]]]]]]]]");
        assert_eq!(s, 0.0);
    }

    #[test]
    fn density_subscore_prose_is_one() {
        let s = density_subscore("The quick brown fox jumps over the lazy dog");
        assert_eq!(s, 1.0);
    }

    #[test]
    fn default_gate_long_recent_prose_scores_high() {
        let long_prose = "This is a substantive episode about how the user prefers Rust over Python and explained why they think strong typing helps them ship faster. ".repeat(10);
        let r = record(&long_prose, 1);
        let Stage1Score { score, .. } = default_gate().score(&r);
        assert!(score > 0.8, "score={score}");
    }

    #[test]
    fn default_gate_short_old_brackets_scores_low() {
        let r = record("[[[[]]]]", 365);
        let Stage1Score { score, .. } = default_gate().score(&r);
        assert!(score < 0.1, "score={score}");
    }

    #[test]
    fn default_gate_is_deterministic() {
        let r = record("hello world ".repeat(20).as_str(), 7);
        let s1 = default_gate().score(&r);
        let s2 = default_gate().score(&r);
        assert!((s1.score - s2.score).abs() < 1e-6);
    }

    #[test]
    fn default_gate_rationale_has_per_signal_lines() {
        let r = record("hello world", 1);
        let Stage1Score { rationale, .. } = default_gate().score(&r);
        // 3 sub-scores + 1 final weighted line = 4 lines.
        assert_eq!(rationale.len(), 4);
        assert!(rationale[0].contains("length_sub"));
        assert!(rationale[1].contains("recency_sub"));
        assert!(rationale[2].contains("density_sub"));
        assert!(rationale[3].contains("weighted score"));
    }

    #[test]
    fn default_gate_name_is_stable_for_audit() {
        // Audit logs use this name; renaming requires a config migration.
        assert_eq!(default_gate().name(), "default-heuristic-v1");
    }
}
