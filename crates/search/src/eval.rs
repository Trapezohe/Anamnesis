//! Retrieval quality measurement — MRR@k and nDCG@k over judged
//! queries.
//!
//! Round 70 motivation: R66–R69 measured search *speed* (criterion
//! benches, MCP request latency surfaced via `doctor`). Nothing
//! measures whether the results are *correct*. Without that, any
//! reranker / RRF tuning / per-stage tracing work is vibes-based and
//! every regression goes unnoticed until a user complains.
//!
//! This module is the measurement *primitive* — given a list of
//! judged queries (with relevant record refs + relevance grades) and
//! a retrieval function, it returns per-query and aggregate
//! MRR@k / nDCG@k. It is intentionally not a reranker, not a
//! corpus, and not a search backend: the CLI `eval-quality`
//! subcommand wires the existing `HybridSearcher` + `pack` pipeline
//! into this module so the *production* retrieval path is what gets
//! scored.
//!
//! Conventions:
//!   - MRR@k: mean reciprocal rank of the first hit with `grade > 0`,
//!     truncated at depth `k`. Queries with no relevant hit in the
//!     top-k contribute `0.0`.
//!   - nDCG@k: graded relevance with gain = `2^grade - 1` and the
//!     standard log2 discount. Ideal DCG computed over the judged
//!     relevant set, sorted by grade descending. Queries with no
//!     judged relevant items at all contribute `0.0` (they don't
//!     count toward the denominator either; the summary excludes
//!     them).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// One row in the judged dataset — what a query is asking for and
/// which records the curator considers relevant (with grades).
///
/// `source` / `kind` / `scope` are optional [`anamnesis_store::SearchFilter`]
/// hints — the CLI honours them when running each query, so a
/// judgment row like "this query should land *only* against mem0
/// records of kind=preference" can be encoded losslessly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgedQuery {
    /// Stable id — used for per-query failure reporting. The curator
    /// picks this; any unique string works.
    pub id: String,
    /// The free-text query string handed to the retrieval pipeline.
    pub query: String,
    /// Adapter filter (mirrors `SearchFilter.source`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Instance filter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance: Option<String>,
    /// Kind filter — `"fact"` / `"preference"` / etc.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Scope filter — `"user"` / `"project"` / etc.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Records the curator considers relevant, with grades.
    pub relevant: Vec<JudgedRecordRef>,
}

/// One curated relevance entry. Either match by `record_id` (the
/// stable natural-key hash from [`anamnesis_core::model::RecordId::from_parts`])
/// or by `(adapter, instance, native_id)` — the latter is friendlier
/// for human-curated fixtures because it avoids needing to compute
/// the hash by hand.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgedRecordRef {
    /// Direct match on `record_id`. Takes precedence when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record_id: Option<String>,
    /// Adapter id (e.g. `"claude-code"`). Used together with
    /// `instance` + `native_id` when `record_id` is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
    /// Instance discriminator. `None` is treated as the default
    /// instance (`""`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance: Option<String>,
    /// Native id as the source adapter sees it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_id: Option<String>,
    /// Relevance grade — `0` = irrelevant (skip), `1` = relevant,
    /// `2` = highly relevant, `3` = canonical answer. Higher
    /// grades earn more nDCG gain (`2^grade - 1`).
    #[serde(default = "default_grade")]
    pub grade: u32,
}

fn default_grade() -> u32 {
    1
}

/// One ranked record the retrieval pipeline returned for a query.
/// Built by the CLI from `HybridSearcher::search_filtered` + `pack`
/// output — `pack` already groups chunks by record, so the eval
/// path operates at record granularity (which is how curators
/// reason: "did the right *memory* come back," not "did one
/// specific chunk come back").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankedRecordRef {
    /// `RecordId.0` — used for direct matching first.
    pub record_id: String,
    /// Adapter / instance / native_id, used for fallback matching
    /// against curator-friendly judgments that don't carry the hash.
    pub adapter: String,
    /// Instance discriminator. Empty string for the default.
    pub instance: String,
    /// Native id as the adapter saw it.
    pub native_id: String,
}

/// Per-query evaluation result. The aggregate ([`QualitySummary`]) is
/// computed from a `&[QueryEval]` so callers can render per-query
/// detail (which query failed?) alongside the summary.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QueryEval {
    /// Mirrors [`JudgedQuery::id`].
    pub id: String,
    /// Reciprocal rank of the first relevant hit (`grade > 0`) at
    /// depth `k`. `0.0` if no relevant hit landed in top-k.
    pub reciprocal_rank: f64,
    /// nDCG@k. `0.0` if the query had no judged relevant items at
    /// all (i.e. nothing to compute against).
    pub ndcg: f64,
    /// Number of judged-relevant records that landed inside top-k.
    pub relevant_in_topk: u32,
    /// Number of judged-relevant records total (across all grades).
    pub judged_relevant: u32,
}

/// Aggregate result over a batch of queries.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QualitySummary {
    /// `k` — the depth used by both metrics. Echoed for the wire.
    pub at: u32,
    /// Number of queries evaluated (the input length).
    pub queries: u32,
    /// Mean reciprocal rank @ k across all queries.
    pub mrr_at_k: f64,
    /// Mean nDCG @ k across queries that had at least one judged
    /// relevant record. Queries with zero judged-relevant entries
    /// are excluded from the denominator (otherwise they always
    /// drag the mean down by definition, which obscures real
    /// regressions).
    pub ndcg_at_k: f64,
    /// Per-query break-out for failure reporting.
    pub per_query: Vec<QueryEval>,
}

/// Evaluate one query's ranked record list against its judgments,
/// at depth `k`. Pure: no I/O, no allocation beyond what `at` needs.
pub fn evaluate_query_at(at: u32, ranked: &[RankedRecordRef], judged: &JudgedQuery) -> QueryEval {
    // Build a lookup from any of the curator's reference forms to
    // the relevance grade. Curators often hand-write
    // `(adapter, native_id)` rather than the hashed `record_id` —
    // we honour both. `record_id` wins when present.
    let mut by_record_id: HashMap<String, u32> = HashMap::new();
    let mut by_natural_key: HashMap<(String, String, String), u32> = HashMap::new();
    let mut judged_relevant: u32 = 0;
    for r in &judged.relevant {
        if r.grade == 0 {
            continue;
        }
        judged_relevant += 1;
        if let Some(rid) = &r.record_id {
            by_record_id.insert(rid.clone(), r.grade);
        }
        // Always also index the natural-key form when supplied so
        // a curator can record both for clarity.
        let (adapter, native_id) = match (&r.adapter, &r.native_id) {
            (Some(a), Some(n)) => (a.clone(), n.clone()),
            _ => continue,
        };
        let instance = r.instance.clone().unwrap_or_default();
        by_natural_key.insert((adapter, instance, native_id), r.grade);
    }

    let depth = (at as usize).min(ranked.len());
    let mut reciprocal_rank = 0.0_f64;
    let mut dcg = 0.0_f64;
    let mut relevant_in_topk: u32 = 0;
    for (i, r) in ranked.iter().take(depth).enumerate() {
        // Resolve grade — prefer record_id, then natural-key.
        let grade = by_record_id
            .get(&r.record_id)
            .copied()
            .or_else(|| {
                by_natural_key
                    .get(&(r.adapter.clone(), r.instance.clone(), r.native_id.clone()))
                    .copied()
            })
            .unwrap_or(0);
        if grade > 0 {
            relevant_in_topk += 1;
            if reciprocal_rank == 0.0 {
                reciprocal_rank = 1.0 / (i as f64 + 1.0);
            }
            // DCG: graded relevance with exponential gain.
            let gain = (2.0_f64).powi(grade as i32) - 1.0;
            let discount = ((i as f64 + 2.0).log2()).max(1.0);
            dcg += gain / discount;
        }
    }

    // Ideal DCG: same gain formula, applied to the top-k of the
    // judged grades sorted descending.
    let mut grades: Vec<u32> = judged
        .relevant
        .iter()
        .filter(|r| r.grade > 0)
        .map(|r| r.grade)
        .collect();
    grades.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = grades
        .iter()
        .take(at as usize)
        .enumerate()
        .map(|(i, g)| {
            let gain = (2.0_f64).powi(*g as i32) - 1.0;
            let discount = ((i as f64 + 2.0).log2()).max(1.0);
            gain / discount
        })
        .sum();
    let ndcg = if idcg > 0.0 { dcg / idcg } else { 0.0 };

    QueryEval {
        id: judged.id.clone(),
        reciprocal_rank,
        ndcg,
        relevant_in_topk,
        judged_relevant,
    }
}

/// Aggregate per-query evaluations into a [`QualitySummary`]. Queries
/// with zero judged-relevant entries are excluded from the nDCG mean
/// so a malformed judgment row can't silently drag the score down
/// (they still count toward the query count and still contribute 0
/// reciprocal-rank to MRR — that's the curator's stated baseline).
pub fn summarize_quality(at: u32, evals: Vec<QueryEval>) -> QualitySummary {
    let queries = evals.len() as u32;
    let mrr_at_k = if queries == 0 {
        0.0
    } else {
        evals.iter().map(|e| e.reciprocal_rank).sum::<f64>() / queries as f64
    };
    let ndcg_denom = evals.iter().filter(|e| e.judged_relevant > 0).count();
    let ndcg_at_k = if ndcg_denom == 0 {
        0.0
    } else {
        evals
            .iter()
            .filter(|e| e.judged_relevant > 0)
            .map(|e| e.ndcg)
            .sum::<f64>()
            / ndcg_denom as f64
    };
    QualitySummary {
        at,
        queries,
        mrr_at_k,
        ndcg_at_k,
        per_query: evals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn judged(id: &str, relevant: Vec<(&str, u32)>) -> JudgedQuery {
        JudgedQuery {
            id: id.into(),
            query: "ignored".into(),
            source: None,
            instance: None,
            kind: None,
            scope: None,
            relevant: relevant
                .into_iter()
                .map(|(rid, g)| JudgedRecordRef {
                    record_id: Some(rid.into()),
                    adapter: None,
                    instance: None,
                    native_id: None,
                    grade: g,
                })
                .collect(),
        }
    }

    fn rnk(rid: &str) -> RankedRecordRef {
        RankedRecordRef {
            record_id: rid.into(),
            adapter: "x".into(),
            instance: String::new(),
            native_id: rid.into(),
        }
    }

    #[test]
    fn mrr_one_when_relevant_is_top_one() {
        let q = judged("q1", vec![("a", 1)]);
        let e = evaluate_query_at(10, &[rnk("a"), rnk("b"), rnk("c")], &q);
        assert!((e.reciprocal_rank - 1.0).abs() < 1e-9);
    }

    #[test]
    fn mrr_one_third_when_relevant_is_position_three() {
        let q = judged("q1", vec![("c", 1)]);
        let e = evaluate_query_at(10, &[rnk("a"), rnk("b"), rnk("c")], &q);
        assert!((e.reciprocal_rank - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn mrr_zero_when_no_relevant_in_topk() {
        let q = judged("q1", vec![("z", 1)]);
        let e = evaluate_query_at(2, &[rnk("a"), rnk("b"), rnk("z")], &q);
        assert_eq!(e.reciprocal_rank, 0.0);
    }

    #[test]
    fn ndcg_is_one_when_perfect_ordering() {
        let q = judged("q1", vec![("a", 3), ("b", 2), ("c", 1)]);
        let e = evaluate_query_at(10, &[rnk("a"), rnk("b"), rnk("c")], &q);
        assert!(
            (e.ndcg - 1.0).abs() < 1e-9,
            "ndcg should be 1.0, got {}",
            e.ndcg
        );
    }

    #[test]
    fn ndcg_strictly_drops_when_top_two_swap() {
        let q = judged("q1", vec![("a", 3), ("b", 2)]);
        let perfect = evaluate_query_at(10, &[rnk("a"), rnk("b")], &q);
        let swapped = evaluate_query_at(10, &[rnk("b"), rnk("a")], &q);
        assert!(
            swapped.ndcg < perfect.ndcg,
            "swap should drop ndcg: perfect={} swapped={}",
            perfect.ndcg,
            swapped.ndcg
        );
    }

    /// Match-by-natural-key path. A curator who hand-writes
    /// `(adapter, native_id)` instead of computing the hashed
    /// `record_id` must still score correctly.
    #[test]
    fn matches_by_natural_key_when_record_id_absent() {
        let q = JudgedQuery {
            id: "q1".into(),
            query: "x".into(),
            source: None,
            instance: None,
            kind: None,
            scope: None,
            relevant: vec![JudgedRecordRef {
                record_id: None,
                adapter: Some("claude-code".into()),
                instance: None,
                native_id: Some("session-42".into()),
                grade: 2,
            }],
        };
        let ranked = vec![RankedRecordRef {
            record_id: "hashed-id-99".into(),
            adapter: "claude-code".into(),
            instance: String::new(),
            native_id: "session-42".into(),
        }];
        let e = evaluate_query_at(10, &ranked, &q);
        assert_eq!(e.reciprocal_rank, 1.0);
        assert!(e.ndcg > 0.0);
    }

    #[test]
    fn summary_excludes_unjudged_queries_from_ndcg_mean() {
        let evals = vec![
            QueryEval {
                id: "judged".into(),
                reciprocal_rank: 1.0,
                ndcg: 1.0,
                relevant_in_topk: 1,
                judged_relevant: 1,
            },
            QueryEval {
                id: "unjudged".into(),
                reciprocal_rank: 0.0,
                ndcg: 0.0,
                relevant_in_topk: 0,
                judged_relevant: 0,
            },
        ];
        let s = summarize_quality(10, evals);
        // MRR averages over both queries (the unjudged one contributes 0).
        assert!((s.mrr_at_k - 0.5).abs() < 1e-9);
        // nDCG averages only over the judged query.
        assert!((s.ndcg_at_k - 1.0).abs() < 1e-9);
    }
}
