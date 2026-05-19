//! ContextPacker — aggregate `RankedChunk` hits into record-level results
//! ready for MCP / agent consumption.
//!
//! Per BLUEPRINT §6.6.1:
//!   - One PackedRecord per unique parent record.
//!   - All matched chunks for that record kept, ordered by chunk score.
//!   - Provenance preserved (adapter, instance, native_path).
//!   - Diversity cap so one adapter / project can't dominate results.
//!   - Approximate token budget (chunker's heuristic) to bound the
//!     payload returned to LLMs.
//!   - **No LLM summarization** — Phase 1 keeps the data raw.
//!
//! ## Ranking defaults (round-4 consult, BLUEPRINT §17.5)
//!
//! Three signals overlaid on the raw RRF / BM25 / cosine score so the
//! default top-N is useful agent memory rather than noise:
//!
//!   1. **Kind boost** — agents asking "what does the user prefer"
//!      should see `Preference / Feedback / Fact / Skill / Reference`
//!      ahead of `Episode / Unknown` when the underlying recall score
//!      is otherwise tied. We add a tiny per-record bonus, not a re-rank,
//!      so a meaningfully better Episode still wins on raw score.
//!   2. **Recency tiebreaker** — within a *score band* (rounded to
//!      [`SCORE_BAND_PRECISION`] decimal places), newer
//!      `max(updated_at, created_at)` comes first.
//!   3. **Per-project diversity cap** — under
//!      `~/.claude/projects/<proj>/...` and `~/.codex/<proj>/...`,
//!      one project shouldn't claim all 10 slots even when its 1700
//!      episodes all match. Separate from `max_per_source` because
//!      adapter+instance is too coarse.
//!
//! These are *defaults*. The packer remains a pure function over the
//! already-materialized hits + record metadata — no new dependencies,
//! no LLM, no query-path changes.

use anamnesis_core::chunker::estimate_tokens;
use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::{AnamnesisRecord, Kind, RecordId};
use anamnesis_store::Store;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

use crate::hybrid::RankedChunk;

/// Magnitude of the kind boost added to a record's score. Small enough
/// that a non-trivial raw-score difference still wins, large enough to
/// break ties.
pub const KIND_BOOST: f64 = 0.05;

/// Round scores to this many decimal places when grouping for the
/// recency tiebreaker. At RRF's typical `1/(60+rank)` magnitudes (≈ 0.016
/// for the second hit) two decimal places means "scores within 0.01 are
/// considered equivalent", which mirrors how a human reads a ranked list.
pub const SCORE_BAND_PRECISION: u32 = 2;

/// Budget knobs for `pack`.
#[derive(Debug, Clone)]
pub struct ContextBudget {
    /// Hard cap on the number of returned records.
    pub max_records: usize,
    /// Approximate cumulative chunk-token budget. `None` means unbounded.
    /// Counted using the chunker's heuristic, so it's a guide rather than
    /// a hard model-tokenizer limit.
    pub max_total_tokens: Option<u32>,
    /// Per-(adapter, instance) cap. `None` means no diversity restriction.
    pub max_per_source: Option<usize>,
    /// Per-(adapter, instance, project_root) cap. Finer-grained than
    /// `max_per_source` — under `~/.claude/projects/<proj>` one project
    /// shouldn't claim every slot even if it dominates the corpus.
    /// `None` disables.
    pub max_per_project: Option<usize>,
}

impl Default for ContextBudget {
    fn default() -> Self {
        Self {
            max_records: 20,
            max_total_tokens: Some(4_000),
            max_per_source: Some(5),
            max_per_project: Some(3),
        }
    }
}

/// One record returned by the packer, with the chunks that matched.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedRecord {
    /// Full record metadata (provenance, scope, kind, tags, …).
    pub record: AnamnesisRecord,
    /// Chunks that hit the query, score-descending.
    pub matched_chunks: Vec<RankedChunk>,
    /// Best chunk score in this record — the record-level rank key.
    pub score: f64,
}

/// Aggregate hits into records, apply provenance + diversity + budget.
///
/// Hits with a `record_id` that no longer exists in the store (e.g. a
/// race with deletion) are skipped silently.
pub fn pack(
    store: &Store,
    hits: &[RankedChunk],
    budget: &ContextBudget,
) -> Result<Vec<PackedRecord>> {
    // 1. Group by record_id, capture the max score.
    let mut groups: HashMap<String, Vec<RankedChunk>> = HashMap::new();
    let mut order_seen: Vec<String> = Vec::new();
    for h in hits {
        let key = h.record_id.0.clone();
        if !groups.contains_key(&key) {
            order_seen.push(key.clone());
        }
        groups.entry(key).or_default().push(h.clone());
    }
    // 2. Materialize: look up record metadata in one batched query.
    //    Round 63 perf: replace per-id `get_record` loop (one prepare +
    //    query per record) with a single `get_records_by_ids` IN(?) call.
    //    Skip vanished records (missing from the returned map).
    let record_ids: Vec<RecordId> = order_seen.iter().map(|rid| RecordId(rid.clone())).collect();
    let mut record_map = store
        .get_records_by_ids(&record_ids)
        .map_err(|e| Error::Other(format!("store get_records_by_ids: {e}")))?;

    let mut materialized: Vec<PackedRecord> = Vec::new();
    for rid in &order_seen {
        let chunks = groups.remove(rid).expect("group exists by construction");
        let record_id = chunks[0].record_id.clone();
        // `remove` moves the record out of the map; avoids cloning the
        // (potentially large) content field that `get(&id).clone()` would.
        // Each `record_id` in `order_seen` is unique by construction so
        // we never need to read it twice.
        let record = match record_map.remove(&record_id) {
            Some(r) => r,
            None => continue,
        };
        let mut chunks = chunks;
        chunks.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let raw_score = chunks.first().map(|c| c.score).unwrap_or(0.0);
        let score = raw_score + kind_bonus(record.kind);
        materialized.push(PackedRecord {
            record,
            matched_chunks: chunks,
            score,
        });
    }
    // 3. Sort: bucket scores at SCORE_BAND_PRECISION; inside each band
    //    fall back to recency (newer `max(updated_at, created_at)`
    //    first). This is the "small score band" tiebreaker — records
    //    that look equally relevant by raw score get ordered by how
    //    fresh the underlying memory is, which matches user intuition.
    materialized.sort_by(|a, b| {
        let band_b = score_band(b.score);
        let band_a = score_band(a.score);
        band_b
            .cmp(&band_a)
            // Newer first → compare b's timestamp against a's so a larger
            // (more recent) value on `b` returns Greater and pushes `a`
            // after `b`.
            .then_with(|| best_timestamp(&b.record).cmp(&best_timestamp(&a.record)))
            // Stable final tiebreaker so iteration order is deterministic
            // across runs even when band+recency are both equal.
            .then_with(|| a.record.id.0.cmp(&b.record.id.0))
    });

    // 4. Apply per-source diversity cap (preserves overall ordering).
    if let Some(max_per) = budget.max_per_source {
        let mut counts: HashMap<(String, String), usize> = HashMap::new();
        materialized.retain(|p| {
            let key = (
                p.record.source.adapter.clone(),
                p.record.source.instance.clone().unwrap_or_default(),
            );
            let entry = counts.entry(key).or_insert(0);
            if *entry >= max_per {
                false
            } else {
                *entry += 1;
                true
            }
        });
    }

    // 4b. Apply per-project diversity cap. Finer than max_per_source:
    //     a single Claude Code project (or Codex project) shouldn't
    //     claim every slot just because it has more sessions than
    //     others — even if the broader adapter cap allows it.
    if let Some(max_per) = budget.max_per_project {
        let mut counts: HashMap<(String, String, String), usize> = HashMap::new();
        materialized.retain(|p| {
            let key = (
                p.record.source.adapter.clone(),
                p.record.source.instance.clone().unwrap_or_default(),
                extract_project_root(p.record.provenance.native_path.as_deref()),
            );
            let entry = counts.entry(key).or_insert(0);
            if *entry >= max_per {
                false
            } else {
                *entry += 1;
                true
            }
        });
    }

    // 5. Apply max_records cap.
    materialized.truncate(budget.max_records);

    // 6. Apply token budget by trimming late records / late chunks.
    if let Some(max_tokens) = budget.max_total_tokens {
        let mut used = 0u32;
        let mut out: Vec<PackedRecord> = Vec::new();
        for mut p in materialized {
            if used >= max_tokens {
                break;
            }
            // Keep adding chunks until we'd exceed; one chunk minimum so
            // every retained record always has at least one snippet.
            let mut kept: Vec<RankedChunk> = Vec::new();
            for chunk in p.matched_chunks.drain(..) {
                let cost = estimate_tokens(&chunk.content);
                let first_chunk = kept.is_empty();
                let fits = used.saturating_add(cost) <= max_tokens;
                if first_chunk || fits {
                    kept.push(chunk);
                    used = used.saturating_add(cost);
                } else {
                    break;
                }
            }
            p.matched_chunks = kept;
            out.push(p);
        }
        materialized = out;
    }

    Ok(materialized)
}

/// Score boost added to a record based on its `Kind`.
///
/// Kinds the agent ecosystem cares about ("what does the user prefer?",
/// "what fact did we learn?", "what was the user's last correction?")
/// get [`KIND_BOOST`]; pure conversation logs (`Episode`) and unknowns
/// stay at 0. `Reference` is a deliberate inclusion: external pointers
/// are useful agent context.
fn kind_bonus(kind: Kind) -> f64 {
    match kind {
        Kind::Preference | Kind::Feedback | Kind::Fact | Kind::Skill | Kind::Reference => {
            KIND_BOOST
        }
        Kind::Episode | Kind::Unknown => 0.0,
    }
}

/// Round score to [`SCORE_BAND_PRECISION`] decimal places so the
/// tiebreaker treats "very close" scores as equal.
fn score_band(score: f64) -> i64 {
    let mult = 10f64.powi(SCORE_BAND_PRECISION as i32);
    (score * mult).round() as i64
}

/// "Best" timestamp for recency comparisons.
///
/// Uses `max(updated_at, created_at)` so a record that was reaffirmed
/// (a Claude Code memory file rewritten, a mem0 row touched) outranks
/// one that hasn't been seen in a year, even if its original creation
/// is older.
fn best_timestamp(record: &AnamnesisRecord) -> DateTime<Utc> {
    record
        .updated_at
        .unwrap_or(record.created_at)
        .max(record.created_at)
}

/// Extract a "project root" key from a `native_path` so per-project
/// diversity can be enforced.
///
/// Conventions covered (in priority order):
///   - `.../<X>/.claude/projects/<encoded-project>/...` → `<encoded-project>`
///   - `.../<X>/.codex/<encoded-project>/...`         → `<encoded-project>`
///
/// For any other adapter (mem0 row, generic-mcp resource, …), the parent
/// directory of the file is used. When no path is present we fall back
/// to an empty string so all such records share the same diversity
/// bucket (which is the safe under-counting choice — they'll be capped
/// together rather than each treated as its own "project").
fn extract_project_root(native_path: Option<&str>) -> String {
    let Some(path) = native_path else {
        return String::new();
    };
    for marker in &["/.claude/projects/", "/.codex/"] {
        if let Some(idx) = path.find(marker) {
            let after = &path[idx + marker.len()..];
            let end = after.find('/').map(|i| &after[..i]).unwrap_or(after);
            if !end.is_empty() {
                return end.to_string();
            }
        }
    }
    // Fallback: parent directory. Filesystems with no parent (single
    // component) return the path itself so we never collapse to "".
    let p = std::path::Path::new(path);
    if let Some(parent) = p.parent() {
        let s = parent.display().to_string();
        if !s.is_empty() {
            return s;
        }
    }
    path.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use anamnesis_core::chunker::Chunker;
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use chrono::Utc;

    fn rec(
        adapter: &str,
        instance: Option<&str>,
        native_id: &str,
        content: &str,
    ) -> AnamnesisRecord {
        AnamnesisRecord {
            id: RecordId::from_parts(adapter, instance, native_id),
            source: SourceDescriptor {
                adapter: adapter.into(),
                instance: instance.map(str::to_owned),
                version: "0".into(),
            },
            content: content.into(),
            embedding: None,
            scope: Scope::User,
            kind: Kind::Fact,
            created_at: Utc::now(),
            updated_at: None,
            tags: vec![],
            metadata: Default::default(),
            provenance: Provenance {
                native_id: native_id.into(),
                native_path: Some(format!("/p/{native_id}")),
                captured_at: Utc::now(),
                raw_hash: "h".into(),
                derived_from: None,
            },
            schema_version: SCHEMA_VERSION,
        }
    }

    fn hit(record_id: &RecordId, seq: u32, content: &str, score: f64) -> RankedChunk {
        RankedChunk {
            chunk_id: format!("{}:{seq}", record_id.0),
            record_id: record_id.clone(),
            seq,
            content: content.into(),
            score,
            fts_score: Some(score),
            vector_score: None,
            from_fts: true,
            from_vec: false,
        }
    }

    fn seed_store_with(records: &[AnamnesisRecord]) -> Store {
        let store = Store::open_in_memory().unwrap();
        for r in records {
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(r, &chunks, None).unwrap();
        }
        store
    }

    #[test]
    fn empty_hits_yield_empty_output() {
        let store = Store::open_in_memory().unwrap();
        let out = pack(&store, &[], &ContextBudget::default()).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn single_record_aggregates_multiple_chunks() {
        let r = rec("a", None, "x", "alpha alpha alpha");
        let store = seed_store_with(std::slice::from_ref(&r));
        let hits = vec![
            hit(&r.id, 0, "alpha part one", 0.9),
            hit(&r.id, 1, "alpha part two", 0.5),
        ];
        let out = pack(&store, &hits, &ContextBudget::default()).unwrap();
        assert_eq!(out.len(), 1);
        let p = &out[0];
        assert_eq!(p.record.id, r.id);
        assert_eq!(p.matched_chunks.len(), 2);
        // Should be sorted by chunk score, best first.
        assert!(p.matched_chunks[0].score >= p.matched_chunks[1].score);
        // Record-level score = max chunk score + kind_bonus. The `rec`
        // fixture uses Kind::Fact, which gets KIND_BOOST.
        assert!(
            (p.score - (0.9 + KIND_BOOST)).abs() < 1e-9,
            "expected raw 0.9 + KIND_BOOST, got {}",
            p.score,
        );
    }

    #[test]
    fn record_score_is_max_of_chunk_scores() {
        let r = rec("a", None, "x", "x");
        let store = seed_store_with(std::slice::from_ref(&r));
        let hits = vec![hit(&r.id, 0, "x", 0.3), hit(&r.id, 1, "y", 0.8)];
        let out = pack(&store, &hits, &ContextBudget::default()).unwrap();
        // `rec` fixture defaults to Kind::Fact → KIND_BOOST applied.
        assert!(
            (out[0].score - (0.8 + KIND_BOOST)).abs() < 1e-9,
            "expected 0.8 + KIND_BOOST, got {}",
            out[0].score,
        );
    }

    #[test]
    fn output_sorted_by_record_score_descending() {
        let r1 = rec("a", None, "1", "one");
        let r2 = rec("a", None, "2", "two");
        let r3 = rec("a", None, "3", "three");
        let store = seed_store_with(&[r1.clone(), r2.clone(), r3.clone()]);
        let hits = vec![
            hit(&r3.id, 0, "three", 0.1),
            hit(&r1.id, 0, "one", 0.7),
            hit(&r2.id, 0, "two", 0.4),
        ];
        let out = pack(&store, &hits, &ContextBudget::default()).unwrap();
        let ids: Vec<_> = out.iter().map(|p| p.record.id.clone()).collect();
        assert_eq!(ids, vec![r1.id, r2.id, r3.id]);
    }

    #[test]
    fn vanished_records_silently_skipped() {
        let r1 = rec("a", None, "1", "alpha");
        let store = seed_store_with(std::slice::from_ref(&r1));
        let missing_id = RecordId::from_parts("a", None, "gone");
        let hits = vec![
            hit(&r1.id, 0, "alpha", 0.9),
            hit(&missing_id, 0, "ghost", 0.95),
        ];
        let out = pack(&store, &hits, &ContextBudget::default()).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].record.id, r1.id);
    }

    #[test]
    fn max_records_truncates() {
        let r1 = rec("a", None, "1", "one");
        let r2 = rec("a", None, "2", "two");
        let r3 = rec("a", None, "3", "three");
        let store = seed_store_with(&[r1.clone(), r2.clone(), r3.clone()]);
        let hits = vec![
            hit(&r1.id, 0, "one", 0.9),
            hit(&r2.id, 0, "two", 0.8),
            hit(&r3.id, 0, "three", 0.7),
        ];
        let budget = ContextBudget {
            max_records: 2,
            max_total_tokens: None,
            max_per_source: None,
            max_per_project: None,
        };
        let out = pack(&store, &hits, &budget).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].record.id, r1.id);
        assert_eq!(out[1].record.id, r2.id);
    }

    #[test]
    fn diversity_cap_limits_per_source() {
        // 4 hits from adapter "a", 1 from adapter "b". max_per_source=2.
        let a1 = rec("a", None, "1", "one");
        let a2 = rec("a", None, "2", "two");
        let a3 = rec("a", None, "3", "three");
        let a4 = rec("a", None, "4", "four");
        let b1 = rec("b", None, "1", "bb");
        let store = seed_store_with(&[a1.clone(), a2.clone(), a3.clone(), a4.clone(), b1.clone()]);
        let hits = vec![
            hit(&a1.id, 0, "one", 0.9),
            hit(&a2.id, 0, "two", 0.85),
            hit(&a3.id, 0, "three", 0.8),
            hit(&a4.id, 0, "four", 0.75),
            hit(&b1.id, 0, "bb", 0.7),
        ];
        let budget = ContextBudget {
            max_records: 10,
            max_total_tokens: None,
            max_per_source: Some(2),
            max_per_project: None,
        };
        let out = pack(&store, &hits, &budget).unwrap();
        let by_adapter: std::collections::HashMap<String, usize> = {
            let mut m = std::collections::HashMap::new();
            for p in &out {
                *m.entry(p.record.source.adapter.clone()).or_insert(0) += 1;
            }
            m
        };
        assert_eq!(by_adapter.get("a").copied().unwrap_or(0), 2);
        assert_eq!(by_adapter.get("b").copied().unwrap_or(0), 1);
    }

    #[test]
    fn token_budget_keeps_at_least_one_chunk_per_record() {
        // A single very long chunk in r1, plus a small r2. Budget < r1
        // size: pack must still keep one chunk from r1 to surface the hit.
        let big = "word ".repeat(2000); // ~600 tokens at 4 chars/token
        let r1 = rec("a", None, "1", &big);
        let r2 = rec("a", None, "2", "tiny");
        let store = seed_store_with(&[r1.clone(), r2.clone()]);
        let hits = vec![hit(&r1.id, 0, &big, 0.9), hit(&r2.id, 0, "tiny", 0.8)];
        let budget = ContextBudget {
            max_records: 10,
            max_total_tokens: Some(100),
            max_per_source: None,
            max_per_project: None,
        };
        let out = pack(&store, &hits, &budget).unwrap();
        assert!(!out.is_empty());
        assert!(
            !out[0].matched_chunks.is_empty(),
            "first record must keep ≥1 chunk"
        );
    }

    // ─── Round-4 ranking defaults (Codex acceptance) ───

    fn rec_with_kind(adapter: &str, native_id: &str, kind: Kind) -> AnamnesisRecord {
        let mut r = rec(adapter, None, native_id, "shared content");
        r.kind = kind;
        r
    }

    fn rec_with_kind_and_ts(
        adapter: &str,
        native_id: &str,
        kind: Kind,
        created: DateTime<Utc>,
    ) -> AnamnesisRecord {
        let mut r = rec_with_kind(adapter, native_id, kind);
        r.created_at = created;
        r
    }

    fn rec_under_project(
        adapter: &str,
        project: &str,
        native_id: &str,
        kind: Kind,
    ) -> AnamnesisRecord {
        let mut r = rec_with_kind(adapter, native_id, kind);
        // Pretend this record lives under `~/.claude/projects/<project>/...`
        // or `~/.codex/<project>/...` so extract_project_root can pick it up.
        let path = match adapter {
            "claude-code" => format!("/home/u/.claude/projects/{project}/{native_id}.jsonl"),
            "codex" => format!("/home/u/.codex/{project}/{native_id}.json"),
            _ => format!("/p/{project}/{native_id}.md"),
        };
        r.provenance.native_path = Some(path);
        r
    }

    /// Acceptance 1: equal raw BM25, the agent-useful kinds win.
    #[test]
    fn kind_boost_promotes_useful_kinds_over_episode_at_equal_score() {
        let pref = rec_with_kind("a", "pref", Kind::Preference);
        let fact = rec_with_kind("a", "fact", Kind::Fact);
        let feedback = rec_with_kind("a", "feedback", Kind::Feedback);
        let skill = rec_with_kind("a", "skill", Kind::Skill);
        let episode = rec_with_kind("a", "episode", Kind::Episode);
        let unknown = rec_with_kind("a", "unknown", Kind::Unknown);
        let store = seed_store_with(&[
            pref.clone(),
            fact.clone(),
            feedback.clone(),
            skill.clone(),
            episode.clone(),
            unknown.clone(),
        ]);
        // Every hit gets the SAME raw chunk score → only kind decides.
        let hits = vec![
            hit(&episode.id, 0, "shared", 0.5),
            hit(&unknown.id, 0, "shared", 0.5),
            hit(&pref.id, 0, "shared", 0.5),
            hit(&fact.id, 0, "shared", 0.5),
            hit(&feedback.id, 0, "shared", 0.5),
            hit(&skill.id, 0, "shared", 0.5),
        ];
        let budget = ContextBudget {
            max_records: 10,
            max_total_tokens: None,
            max_per_source: None,
            max_per_project: None,
        };
        let out = pack(&store, &hits, &budget).unwrap();
        assert_eq!(out.len(), 6);
        // Top 4 must be the agent-useful kinds (in any order among
        // themselves — the tiebreaker is recency, then id).
        let top_kinds: Vec<Kind> = out.iter().take(4).map(|p| p.record.kind).collect();
        for k in &top_kinds {
            assert!(
                matches!(
                    k,
                    Kind::Preference | Kind::Feedback | Kind::Fact | Kind::Skill
                ),
                "top 4 must be agent-useful kinds, found {k:?}"
            );
        }
        // Episode and Unknown must come last.
        let bottom_kinds: Vec<Kind> = out.iter().skip(4).map(|p| p.record.kind).collect();
        for k in &bottom_kinds {
            assert!(
                matches!(k, Kind::Episode | Kind::Unknown),
                "bottom 2 must be Episode/Unknown, found {k:?}"
            );
        }
    }

    /// Acceptance 2: within a score band, newer wins.
    #[test]
    fn recency_breaks_ties_within_score_band() {
        use chrono::TimeZone;
        let older = rec_with_kind_and_ts(
            "a",
            "older",
            Kind::Episode,
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
        );
        let newer = rec_with_kind_and_ts(
            "a",
            "newer",
            Kind::Episode,
            Utc.with_ymd_and_hms(2026, 5, 1, 0, 0, 0).unwrap(),
        );
        let store = seed_store_with(&[older.clone(), newer.clone()]);
        // Same kind, same chunk score → only recency decides.
        let hits = vec![hit(&older.id, 0, "x", 0.5), hit(&newer.id, 0, "x", 0.5)];
        let budget = ContextBudget {
            max_records: 10,
            max_total_tokens: None,
            max_per_source: None,
            max_per_project: None,
        };
        let out = pack(&store, &hits, &budget).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(
            out[0].record.id, newer.id,
            "newer record must come first within the score band"
        );
        assert_eq!(out[1].record.id, older.id);
    }

    /// Acceptance 3: per-project cap. The 1744:7 spirit — one project
    /// can't claim every slot. We construct 10 hits all from
    /// `~/.claude/projects/dominant/...` plus a few from other places,
    /// and assert no project contributes more than `max_per_project`.
    #[test]
    fn per_project_diversity_cap_limits_slots_per_project() {
        let mut all = Vec::new();
        let mut dominant_records = Vec::new();
        for i in 0..10u32 {
            let r = rec_under_project("claude-code", "dominant", &format!("d-{i}"), Kind::Episode);
            dominant_records.push(r.clone());
            all.push(r);
        }
        let other_a = rec_under_project("claude-code", "other-a", "a", Kind::Episode);
        let other_b = rec_under_project("claude-code", "other-b", "b", Kind::Episode);
        all.push(other_a.clone());
        all.push(other_b.clone());
        let store = seed_store_with(&all);
        let mut hits: Vec<RankedChunk> = dominant_records
            .iter()
            .enumerate()
            .map(|(i, r)| hit(&r.id, 0, "shared", 1.0 - (i as f64) * 0.0001))
            .collect();
        hits.push(hit(&other_a.id, 0, "shared", 0.5));
        hits.push(hit(&other_b.id, 0, "shared", 0.5));
        let budget = ContextBudget {
            max_records: 10,
            max_total_tokens: None,
            max_per_source: None,
            max_per_project: Some(3),
        };
        let out = pack(&store, &hits, &budget).unwrap();
        let mut counts = std::collections::HashMap::<String, usize>::new();
        for p in &out {
            let root = extract_project_root(p.record.provenance.native_path.as_deref());
            *counts.entry(root).or_insert(0) += 1;
        }
        let dominant_count = counts.get("dominant").copied().unwrap_or(0);
        assert_eq!(
            dominant_count, 3,
            "the 'dominant' project must be capped at max_per_project=3, got {dominant_count}"
        );
        // The other projects must each get a slot now that we made room.
        assert!(
            counts.contains_key("other-a"),
            "other-a should not be squeezed out"
        );
        assert!(
            counts.contains_key("other-b"),
            "other-b should not be squeezed out"
        );
    }

    /// Sanity: extract_project_root handles each adapter's path shape.
    #[test]
    fn extract_project_root_parses_known_shapes() {
        assert_eq!(
            extract_project_root(Some(
                "/Users/songsu/.claude/projects/-Users-songsu-Desktop-x/sess.jsonl"
            )),
            "-Users-songsu-Desktop-x"
        );
        assert_eq!(
            extract_project_root(Some(
                "/Users/songsu/.claude/projects/-Users-songsu-Desktop-x/uuid/subagents/a.jsonl"
            )),
            "-Users-songsu-Desktop-x"
        );
        assert_eq!(
            extract_project_root(Some("/Users/u/.codex/repo-name/session.json")),
            "repo-name"
        );
        // Fallback: parent directory.
        assert_eq!(
            extract_project_root(Some("/etc/something/file.md")),
            "/etc/something"
        );
        // None / empty path → empty key, all such records bucket together.
        assert_eq!(extract_project_root(None), "");
    }

    #[test]
    fn provenance_preserved() {
        let r = rec("claude-code", Some("default"), "fbid", "fb");
        let store = seed_store_with(std::slice::from_ref(&r));
        let hits = vec![hit(&r.id, 0, "fb", 0.9)];
        let out = pack(&store, &hits, &ContextBudget::default()).unwrap();
        let p = &out[0];
        assert_eq!(p.record.source.adapter, "claude-code");
        assert_eq!(p.record.source.instance.as_deref(), Some("default"));
        assert_eq!(p.record.provenance.native_id, "fbid");
        assert_eq!(p.record.provenance.native_path.as_deref(), Some("/p/fbid"));
    }
}
