//! Hybrid search: FTS5 BM25 + vector kNN, merged via Reciprocal Rank Fusion.
//!
//! Algorithm (BLUEPRINT §6.6 hybrid):
//!
//!   1. Run FTS BM25 over `chunks_fts` → ranked list `A`.
//!   2. If a provider + model_id is configured, embed the query, run
//!      vector kNN over `chunk_embeddings` filtered to that model → ranked list `B`.
//!   3. Reciprocal Rank Fusion: `score(c) = sum( 1 / (K + rank_in_L) )`
//!      where K=60 (the published RRF constant; resilient to outliers).
//!   4. Return top-N ranked chunks.
//!
//! The pure RRF math lives in `rrf` so it's unit-testable without a DB.

use anamnesis_core::embedding::EmbeddingProvider;
#[cfg(test)]
use anamnesis_core::embedding::EmbeddingTask;
use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::RecordId;
use anamnesis_store::{SearchFilter, Store};

/// RRF constant. Published heuristic, robust against rank outliers.
pub const RRF_K: f64 = 60.0;

/// Which retrieval modalities to combine.
///
/// `Serialize` is `rename_all = "lowercase"` so the wire shape stays
/// `"fulltext" / "vector" / "hybrid"` — same casing as
/// `HybridOpts.mode` accepts on input and as the CLI / MCP have
/// always rendered.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// FTS5 BM25 only — no vector lookup, no embedding call.
    Fulltext,
    /// Vector kNN only.
    Vector,
    /// FTS + vector with RRF fusion (default).
    #[default]
    Hybrid,
}

/// Per-query options.
#[derive(Debug, Clone)]
pub struct HybridOpts {
    /// Max chunks to return.
    pub limit: u32,
    /// Candidate pool size per modality before RRF merge. Larger pool =
    /// better recall on rare matches but more work. Computed by
    /// [`candidate_pool_for_limit`] when constructed via
    /// `HybridOpts::for_limit` / `HybridOpts::fulltext_only`.
    pub candidate_pool: u32,
    /// Modalities to combine.
    pub mode: SearchMode,
}

/// Round 136 (PR-78be): the floor on the per-modality candidate pool.
/// Tiny `limit=1` queries previously got pool=4 (`limit * 4`), which
/// starved both FTS and sqlite-vec ANN — a rare-token match could be
/// hiding in candidate 5+ and never reach the RRF merge. 64 is the
/// empirical floor where ANN recall stabilises on the curated
/// benchmark fixtures.
pub const CANDIDATE_POOL_FLOOR: u32 = 64;

/// Round 136 (PR-78be): the ceiling. Without it, a `limit=200` query
/// would push `(limit * 8) = 1600` into vec0 KNN, which is wasted
/// work — the post-RRF top-K rarely benefits from a pool deeper than
/// 512 in profiled traces, and the ANN scan walks the whole
/// candidate set per record.
pub const CANDIDATE_POOL_CEIL: u32 = 512;

/// Round 136 (PR-78be): central candidate-pool sizing policy used by
/// every search entry point (CLI `search`, MCP `search_memories`,
/// MCP `find_related`, …). Returns the per-modality pool size for a
/// given `limit`, applying:
///
/// 1. `limit == 0` short-circuits to 0 (nothing to retrieve).
/// 2. Otherwise compute `limit * 8` (saturating) — wider than the
///    historic `limit * 4` so a `limit=20` query gets pool=160
///    instead of 80, lifting recall on rare-token queries the
///    hybrid RRF merge depends on.
/// 3. Clamp to `[CANDIDATE_POOL_FLOOR, CANDIDATE_POOL_CEIL]`. The
///    floor protects tiny `limit=1/2/5` queries from a starved
///    candidate set; the ceiling protects large-limit queries from
///    paying ANN cost that the post-rerank top-K won't use.
/// 4. Final guard: never below `limit` itself (vanishingly rare with
///    the clamp + 8× scaling, but kept as a correctness invariant
///    so a future tweak of the constants can't ever return a pool
///    smaller than the user asked for).
pub fn candidate_pool_for_limit(limit: u32) -> u32 {
    if limit == 0 {
        return 0;
    }
    limit
        .saturating_mul(8)
        .clamp(CANDIDATE_POOL_FLOOR, CANDIDATE_POOL_CEIL)
        .max(limit)
}

impl Default for HybridOpts {
    fn default() -> Self {
        Self::for_limit(20, SearchMode::Hybrid)
    }
}

impl HybridOpts {
    /// Round 136 (PR-78be): central constructor used by every CLI /
    /// MCP entry point. Routes the per-modality `candidate_pool`
    /// through [`candidate_pool_for_limit`] so a single policy
    /// change covers `anamnesis search`, MCP `search_memories`,
    /// MCP `find_related`, and any future read surface.
    pub fn for_limit(limit: u32, mode: SearchMode) -> Self {
        Self {
            limit,
            candidate_pool: candidate_pool_for_limit(limit),
            mode,
        }
    }

    /// Convenience: only FTS, no embedding call. Now delegates to
    /// [`HybridOpts::for_limit`] so the candidate-pool policy is
    /// applied uniformly.
    pub fn fulltext_only(limit: u32) -> Self {
        Self::for_limit(limit, SearchMode::Fulltext)
    }
}

/// A merged hit returned by the hybrid searcher.
///
/// Carries enough breakdown for an MCP client / agent to understand
/// *why* this chunk surfaced — was it FTS, vector, both? — and to
/// chain into follow-up MCP tools like `trace_provenance` without a
/// second round trip.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedChunk {
    /// Chunk id (`"{record_id}:{seq}"`).
    pub chunk_id: String,
    /// Parent record.
    pub record_id: RecordId,
    /// Per-record chunk index.
    pub seq: u32,
    /// The chunk text (for snippet rendering).
    pub content: String,
    /// Final RRF score (sum of `1/(K+rank)` across hit lists). This is
    /// the value the packer uses for record-level ranking.
    pub score: f64,
    /// The raw FTS bm25 score (already negated so larger = better) when
    /// FTS contributed to this hit. `None` when only vector matched.
    pub fts_score: Option<f64>,
    /// The raw vector cosine when vector kNN contributed. `None` when
    /// only FTS matched.
    pub vector_score: Option<f64>,
    /// `true` if FTS contributed to this hit.
    pub from_fts: bool,
    /// `true` if vector search contributed to this hit.
    pub from_vec: bool,
    /// Round 87 (PR-78i): per-stage score breakdown — the
    /// rank + raw score + RRF contribution from FTS and vector,
    /// plus `rrf_k` so the consumer can verify the arithmetic.
    /// **Always populated**; consumers branch on the inner
    /// `Option`s to know whether each stage contributed. CLI /
    /// MCP only surface this when the caller opts in with
    /// `--explain` / `explain: true` — the field exists on
    /// every chunk so the search layer doesn't need a second
    /// merge variant.
    pub explain: ChunkScoreExplain,
}

/// Round 87 (PR-78i): a single search stage's contribution to a
/// chunk's RRF score. Numeric-only — no record/chunk ids, no
/// query, no snippet — so this fits the same privacy contract
/// as `SearchTrace`.
#[derive(Debug, Clone, PartialEq)]
pub struct StageScore {
    /// 1-based rank in the stage's hit list (1 = top hit).
    pub rank: usize,
    /// Raw score from the underlying index. FTS uses negated
    /// BM25 (larger = better, matches the existing `fts_score`);
    /// vector uses cosine similarity in `[0, 1]`.
    pub raw_score: f64,
    /// `1.0 / (RRF_K + rank)` — what this stage contributed to
    /// the chunk's combined RRF score.
    pub rrf_contribution: f64,
}

/// Round 87 (PR-78i): per-chunk score breakdown attached to
/// every `RankedChunk`. CLI/MCP only render it when the caller
/// opts into `--explain` / `explain: true`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkScoreExplain {
    /// FTS stage contribution. `None` when the chunk surfaced
    /// only via vector kNN.
    pub fts: Option<StageScore>,
    /// Vector stage contribution. `None` when the chunk
    /// surfaced only via FTS.
    pub vector: Option<StageScore>,
    /// The RRF constant used by this merge — pinned so the
    /// consumer can verify `rrf_contribution = 1 / (rrf_k +
    /// rank)`.
    pub rrf_k: f64,
}

impl ChunkScoreExplain {
    /// Empty explain — used as the default in places that don't
    /// participate in RRF (e.g. a fulltext-only path that
    /// rebuilds `RankedChunk`s directly). RRF merge fills the
    /// real values in.
    pub fn empty() -> Self {
        Self {
            fts: None,
            vector: None,
            rrf_k: RRF_K,
        }
    }
}

/// The composer.
///
/// `P: ?Sized` lets callers pass either a concrete provider (`&MyProvider`)
/// or a trait object (`&dyn EmbeddingProvider`) for cases like the MCP
/// server where the provider is chosen at runtime behind a `Box<dyn …>`.
pub struct HybridSearcher<'a, P: EmbeddingProvider + ?Sized> {
    /// Provider used for embedding the query in `Vector`/`Hybrid` modes.
    /// `None` forces `Fulltext` regardless of `HybridOpts::mode`.
    pub provider: Option<&'a P>,
}

impl<'a, P: EmbeddingProvider + ?Sized> HybridSearcher<'a, P> {
    /// Build a searcher that uses `provider` for the vector side.
    pub fn new(provider: &'a P) -> Self {
        Self {
            provider: Some(provider),
        }
    }

    /// Build a fulltext-only searcher.
    pub fn fulltext_only() -> Self {
        Self { provider: None }
    }

    /// Run the search with no filter (`SearchFilter::default`).
    pub async fn search(
        &self,
        store: &Store,
        query: &str,
        opts: &HybridOpts,
    ) -> Result<Vec<RankedChunk>> {
        self.search_filtered(store, query, &SearchFilter::default(), opts)
            .await
    }

    /// Run the search with the given filter pushed into the SQL recall
    /// stage.
    ///
    /// This is the load-bearing entry point for PR-C (BLUEPRINT §17.5).
    /// `filter` shapes the candidate pool *before* `LIMIT` truncates it,
    /// so e.g. `source = "mem0"` returns mem0 chunks even when the
    /// overall corpus is dominated by another adapter.
    ///
    /// Returns only the ranked chunks — the per-stage timing/count
    /// breakdown is computed but discarded. Callers that need the
    /// breakdown (e.g. MCP `search_memories(trace=true)`) should
    /// invoke [`Self::search_filtered_traced`] directly. Round 71's
    /// refactor guarantees the two share a single code path so the
    /// trace can't drift from the live search.
    pub async fn search_filtered(
        &self,
        store: &Store,
        query: &str,
        filter: &SearchFilter,
        opts: &HybridOpts,
    ) -> Result<Vec<RankedChunk>> {
        Ok(self
            .search_filtered_traced(store, query, filter, opts)
            .await?
            .hits)
    }

    /// Same as [`Self::search_filtered`], but also returns a per-stage
    /// breakdown of timings and candidate counts.
    ///
    /// Round 71: built so `search_memories(trace=true)` can report
    /// `embed_query / fts / vec / rrf` wall-time + candidate-pool
    /// shape without us duplicating search logic in a second path.
    /// The MCP / CLI default path goes through
    /// [`Self::search_filtered`], which drops the trace — there's no
    /// observable cost when the trace isn't asked for (timings are
    /// `Instant::elapsed()` on stages we'd run anyway).
    ///
    /// **Privacy**: this primitive only collects sizes and durations.
    /// No query text, snippets, or record/chunk ids cross the
    /// boundary into the trace struct — callers wrapping this for an
    /// external surface (MCP / CLI) can serialise the trace verbatim
    /// without worrying about leaking user content.
    pub async fn search_filtered_traced(
        &self,
        store: &Store,
        query: &str,
        filter: &SearchFilter,
        opts: &HybridOpts,
    ) -> Result<TracedSearchResult> {
        let effective_mode = if self.provider.is_none() {
            SearchMode::Fulltext
        } else {
            opts.mode
        };

        let pool = opts.candidate_pool.max(opts.limit);

        let (fts_hits, fts_ms) =
            if matches!(effective_mode, SearchMode::Fulltext | SearchMode::Hybrid) {
                let t = std::time::Instant::now();
                let hits = store
                    .search_chunks_fts(query, filter, pool)
                    .map_err(|e| Error::Other(format!("store fts: {e}")))?;
                (hits, Some(t.elapsed().as_millis() as u64))
            } else {
                (Vec::new(), None)
            };

        let (vec_hits, embed_query_ms, vec_ms) =
            if matches!(effective_mode, SearchMode::Vector | SearchMode::Hybrid) {
                let provider = self
                    .provider
                    .ok_or_else(|| Error::Other("Vector/Hybrid mode requires a provider".into()))?;
                let t_embed = std::time::Instant::now();
                let qvec = provider.embed_query(query).await?;
                let embed_ms = t_embed.elapsed().as_millis() as u64;
                let t_vec = std::time::Instant::now();
                let hits = store
                    .search_chunks_vec(&qvec, &provider.model_id().0, filter, pool)
                    .map_err(|e| Error::Other(format!("store vec: {e}")))?;
                (
                    hits,
                    Some(embed_ms),
                    Some(t_vec.elapsed().as_millis() as u64),
                )
            } else {
                (Vec::new(), None, None)
            };

        let t_rrf = std::time::Instant::now();
        let mut merged = rrf::merge(&fts_hits, &vec_hits, opts.limit as usize);

        // For each merged hit, look up the actual content from whichever
        // source has it (FTS and vec both carry content; prefer FTS for
        // snippet purposes since BM25 favours the matched terms).
        for m in &mut merged {
            if m.content.is_empty() {
                let pick = fts_hits
                    .iter()
                    .find(|h| h.chunk_id == m.chunk_id)
                    .or_else(|| vec_hits.iter().find(|h| h.chunk_id == m.chunk_id));
                if let Some(p) = pick {
                    m.content = p.content.clone();
                }
            }
        }
        let rrf_ms = t_rrf.elapsed().as_millis() as u64;

        let trace = SearchTrace {
            effective_mode,
            candidate_pool: pool,
            stages_ms: SearchStageTimings {
                embed_query_ms,
                fts_ms,
                vec_ms,
                rrf_ms: Some(rrf_ms),
            },
            counts: SearchStageCounts {
                fts_hits: fts_hits.len() as u32,
                vec_hits: vec_hits.len() as u32,
                ranked_chunks: merged.len() as u32,
            },
        };
        Ok(TracedSearchResult {
            hits: merged,
            trace,
        })
    }
}

/// Output of [`HybridSearcher::search_filtered_traced`] — the ranked
/// chunks plus a per-stage performance breakdown.
#[derive(Debug, Clone)]
pub struct TracedSearchResult {
    /// Top-`limit` chunks, same shape as [`HybridSearcher::search_filtered`].
    pub hits: Vec<RankedChunk>,
    /// Per-stage timing + candidate-count breakdown. Carries no user
    /// content — safe to surface to MCP / CLI clients verbatim.
    pub trace: SearchTrace,
}

/// Run-time breakdown of one search invocation. Strictly numeric +
/// the resolved `effective_mode`; no query text, snippets, or ids.
///
/// Round 71 adds this so `search_memories(trace=true)` can answer
/// "why was that search slow" with the same level of detail an
/// engineer would want from a profiler — without persisting anything,
/// and without exposing what the user typed.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchTrace {
    /// `Fulltext`/`Vector`/`Hybrid` actually used. Differs from
    /// `HybridOpts.mode` when no provider is wired (e.g. the
    /// fastembed model isn't on disk and we fell back to
    /// fulltext-only).
    pub effective_mode: SearchMode,
    /// Per-modality `LIMIT` handed to the recall stage. Mirrors
    /// `opts.candidate_pool.max(opts.limit)`.
    pub candidate_pool: u32,
    /// Per-stage wall time in milliseconds. `None` on stages that
    /// didn't run for this query (e.g. `vec_ms` in fulltext mode).
    pub stages_ms: SearchStageTimings,
    /// Per-stage candidate counts. Useful for "did FTS surface
    /// anything?" diagnostics where `fts_hits = 0` is the answer.
    pub counts: SearchStageCounts,
}

/// Per-stage wall time, milliseconds. `None` for stages skipped
/// under the effective mode.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchStageTimings {
    /// Time spent in `provider.embed_query(...)`. `None` when fulltext.
    pub embed_query_ms: Option<u64>,
    /// Time spent in `store.search_chunks_fts(...)`. `None` when vector-only.
    pub fts_ms: Option<u64>,
    /// Time spent in `store.search_chunks_vec(...)`. `None` when fulltext.
    pub vec_ms: Option<u64>,
    /// Time spent in `rrf::merge(...)` + snippet backfill. Always set.
    pub rrf_ms: Option<u64>,
}

/// Per-stage candidate-count shape.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchStageCounts {
    /// Number of chunks returned by the FTS recall stage.
    pub fts_hits: u32,
    /// Number of chunks returned by the vector recall stage.
    pub vec_hits: u32,
    /// Number of merged chunks after RRF + `limit`.
    pub ranked_chunks: u32,
}

/// Pure RRF logic — separated for unit testing without a DB.
pub mod rrf {
    use super::{ChunkScoreExplain, RankedChunk, StageScore, RRF_K};
    use anamnesis_store::ChunkHit;
    use std::collections::HashMap;

    /// Merge two ranked hit lists via Reciprocal Rank Fusion.
    ///
    /// Returns the top-`limit` items by combined RRF score. Each
    /// returned `RankedChunk` also carries the raw per-modality scores
    /// (`fts_score`, `vector_score`) so downstream consumers — MCP
    /// agents in particular — can explain "why did this surface" without
    /// a round trip back to the index. Round 87 attaches a structured
    /// `explain` block with the 1-based rank and per-stage
    /// `rrf_contribution = 1.0 / (RRF_K + rank)` so callers can
    /// reproduce the arithmetic exactly.
    pub fn merge(fts: &[ChunkHit], vec: &[ChunkHit], limit: usize) -> Vec<RankedChunk> {
        let mut acc: HashMap<String, RankedChunk> = HashMap::new();
        for (rank, hit) in fts.iter().enumerate() {
            let entry = acc
                .entry(hit.chunk_id.clone())
                .or_insert_with(|| RankedChunk {
                    chunk_id: hit.chunk_id.clone(),
                    record_id: hit.record_id.clone(),
                    seq: hit.seq,
                    content: hit.content.clone(),
                    score: 0.0,
                    fts_score: None,
                    vector_score: None,
                    from_fts: false,
                    from_vec: false,
                    explain: ChunkScoreExplain::empty(),
                });
            // 1-based rank — matches operator intuition ("the
            // top FTS hit is rank 1, not rank 0").
            let rank_1 = rank + 1;
            let contribution = 1.0 / (RRF_K + rank_1 as f64);
            entry.score += contribution;
            entry.fts_score = Some(hit.score);
            entry.from_fts = true;
            entry.explain.fts = Some(StageScore {
                rank: rank_1,
                raw_score: hit.score,
                rrf_contribution: contribution,
            });
        }
        for (rank, hit) in vec.iter().enumerate() {
            let entry = acc
                .entry(hit.chunk_id.clone())
                .or_insert_with(|| RankedChunk {
                    chunk_id: hit.chunk_id.clone(),
                    record_id: hit.record_id.clone(),
                    seq: hit.seq,
                    content: hit.content.clone(),
                    score: 0.0,
                    fts_score: None,
                    vector_score: None,
                    from_fts: false,
                    from_vec: false,
                    explain: ChunkScoreExplain::empty(),
                });
            let rank_1 = rank + 1;
            let contribution = 1.0 / (RRF_K + rank_1 as f64);
            entry.score += contribution;
            entry.vector_score = Some(hit.score);
            entry.from_vec = true;
            entry.explain.vector = Some(StageScore {
                rank: rank_1,
                raw_score: hit.score,
                rrf_contribution: contribution,
            });
        }
        let mut out: Vec<RankedChunk> = acc.into_values().collect();
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out.truncate(limit);
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use anamnesis_core::model::RecordId;

        fn hit(id: &str, score: f64) -> ChunkHit {
            ChunkHit {
                chunk_id: id.into(),
                record_id: RecordId(format!("rec-{id}")),
                seq: 0,
                content: format!("content-{id}"),
                score,
            }
        }

        #[test]
        fn empty_inputs_yield_empty_output() {
            let out = merge(&[], &[], 10);
            assert!(out.is_empty());
        }

        #[test]
        fn fts_only_ranks_by_rrf_score() {
            // 3 FTS hits in order a, b, c → rank 1, 2, 3 → scores
            // 1/61, 1/62, 1/63 (descending).
            let out = merge(&[hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)], &[], 3);
            assert_eq!(out.len(), 3);
            assert_eq!(out[0].chunk_id, "a");
            assert_eq!(out[1].chunk_id, "b");
            assert_eq!(out[2].chunk_id, "c");
            assert!(out[0].score > out[1].score);
            assert!(out[1].score > out[2].score);
            assert!(out.iter().all(|r| r.from_fts && !r.from_vec));
        }

        #[test]
        fn vector_only_marks_from_vec() {
            let out = merge(&[], &[hit("x", 0.5)], 10);
            assert_eq!(out.len(), 1);
            assert!(out[0].from_vec);
            assert!(!out[0].from_fts);
        }

        #[test]
        fn hit_in_both_lists_aggregates_score() {
            let fts = vec![hit("a", 0.0), hit("b", 0.0)];
            let vec = vec![hit("a", 0.0), hit("c", 0.0)];
            let out = merge(&fts, &vec, 10);
            // `a` appears in both → score = 1/61 + 1/61, while b and c are
            // each in only one list at rank 1 or 2.
            let a = out.iter().find(|r| r.chunk_id == "a").unwrap();
            let b = out.iter().find(|r| r.chunk_id == "b").unwrap();
            assert!(a.score > b.score, "a should outrank b due to combined RRF");
            assert!(a.from_fts && a.from_vec);
            assert!(b.from_fts && !b.from_vec);
        }

        #[test]
        fn limit_truncates_output() {
            let many: Vec<ChunkHit> = (0..50).map(|i| hit(&format!("h{i}"), 0.0)).collect();
            let out = merge(&many, &[], 5);
            assert_eq!(out.len(), 5);
        }

        #[test]
        fn rrf_constant_lowers_top_rank_advantage() {
            // The k=60 constant means a rank-1 hit has score ~1/61, not 1.0,
            // so a single tail hit isn't crushed by a single head hit.
            let out = merge(&[hit("top", 0.0)], &[], 1);
            assert!((out[0].score - (1.0 / 61.0)).abs() < 1e-9);
        }

        // ─── Round-87 PR-78i: per-stage explain breakdown ──────────

        /// RRF merge populates `explain.fts.rank` (1-based) and
        /// `explain.fts.rrf_contribution = 1/(K+rank)` on every
        /// FTS-only hit. Pinned because the CLI / MCP wire
        /// formats reproduce this arithmetic.
        #[test]
        fn merge_populates_fts_explain_with_rank_and_contribution() {
            let out = merge(&[hit("a", 3.21), hit("b", 1.5)], &[], 10);
            let a = out.iter().find(|r| r.chunk_id == "a").unwrap();
            let a_fts = a
                .explain
                .fts
                .as_ref()
                .expect("FTS-only hit must have fts explain");
            assert_eq!(a_fts.rank, 1);
            assert!((a_fts.raw_score - 3.21).abs() < 1e-9);
            assert!((a_fts.rrf_contribution - 1.0 / 61.0).abs() < 1e-9);
            assert!(
                a.explain.vector.is_none(),
                "FTS-only hit has no vector breakdown"
            );
            assert!((a.explain.rrf_k - RRF_K).abs() < 1e-9);

            let b = out.iter().find(|r| r.chunk_id == "b").unwrap();
            let b_fts = b.explain.fts.as_ref().unwrap();
            assert_eq!(b_fts.rank, 2);
            assert!((b_fts.rrf_contribution - 1.0 / 62.0).abs() < 1e-9);
        }

        /// A chunk that hits in both FTS and vector lists carries
        /// both `explain.fts` and `explain.vector`. Combined
        /// `score` equals the sum of the two `rrf_contribution`s
        /// — operators can verify the arithmetic without trusting
        /// the implementation.
        #[test]
        fn merge_populates_both_stages_when_hit_in_both_lists() {
            let fts = vec![hit("a", 5.0)];
            let vec = vec![hit("a", 0.8)];
            let out = merge(&fts, &vec, 10);
            let a = out.iter().find(|r| r.chunk_id == "a").unwrap();
            let fts_stage = a.explain.fts.as_ref().unwrap();
            let vec_stage = a.explain.vector.as_ref().unwrap();
            assert_eq!(fts_stage.rank, 1);
            assert_eq!(vec_stage.rank, 1);
            let sum = fts_stage.rrf_contribution + vec_stage.rrf_contribution;
            assert!(
                (a.score - sum).abs() < 1e-9,
                "RankedChunk.score must equal Σ rrf_contribution across stages"
            );
        }

        /// `ChunkScoreExplain` carries only numeric data — no
        /// chunk_id, no record_id, no content, no query — so it
        /// fits the same privacy contract as `SearchTrace`. Serde
        /// the struct and grep for any forbidden text.
        #[test]
        fn chunk_explain_struct_carries_only_numeric_fields() {
            let e = ChunkScoreExplain {
                fts: Some(StageScore {
                    rank: 1,
                    raw_score: 3.21,
                    rrf_contribution: 1.0 / 61.0,
                }),
                vector: None,
                rrf_k: RRF_K,
            };
            // Debug-format the struct and assert no string fields
            // exist. (We don't auto-derive Serialize on this type;
            // the wire-format conversion lives at the CLI / MCP
            // layer so the struct itself stays minimal.)
            let dbg = format!("{e:?}");
            for forbidden in ["chunk_id", "record_id", "content", "query", "snippet"] {
                assert!(
                    !dbg.contains(forbidden),
                    "explain struct must NOT carry {forbidden}: {dbg}"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anamnesis_core::chunker::Chunker;
    use anamnesis_core::embedding::ModelId;
    use anamnesis_core::model::{
        AnamnesisRecord, Kind, Provenance, RecordId, Scope, SourceDescriptor, SCHEMA_VERSION,
    };
    use async_trait::async_trait;

    // ─── Round-136 PR-78be: candidate_pool_for_limit policy ─────

    #[test]
    fn candidate_pool_for_limit_zero_short_circuits() {
        // No retrieval requested → no pool to assemble.
        assert_eq!(candidate_pool_for_limit(0), 0);
    }

    #[test]
    fn candidate_pool_for_limit_tiny_limits_hit_the_floor() {
        // The whole point of the floor: a `limit=1` query was
        // previously getting pool=4 (`limit * 4`), which starved
        // both FTS and ANN. 64 is the empirical floor.
        for limit in [1u32, 2, 5, 7] {
            assert_eq!(
                candidate_pool_for_limit(limit),
                CANDIDATE_POOL_FLOOR,
                "limit={limit} must clamp UP to the floor"
            );
        }
    }

    #[test]
    fn candidate_pool_for_limit_midrange_uses_eight_times_scaling() {
        // Inside the floor/ceil window the 8× scaling applies.
        // limit=10 → 80, limit=20 → 160, limit=50 → 400.
        assert_eq!(candidate_pool_for_limit(10), 80);
        assert_eq!(candidate_pool_for_limit(20), 160);
        assert_eq!(candidate_pool_for_limit(50), 400);
    }

    #[test]
    fn candidate_pool_for_limit_large_limits_clamp_to_ceiling() {
        // limit=100 → 800 → clamped down to 512; limit at/below the
        // ceiling never bumps back up because of the `.max(limit)`
        // guard.
        assert_eq!(candidate_pool_for_limit(100), CANDIDATE_POOL_CEIL);
        assert_eq!(candidate_pool_for_limit(500), CANDIDATE_POOL_CEIL);
        // limit ABOVE the ceiling is a corner case: the user asked
        // for more results than the ceiling, so the pool MUST be
        // at least `limit` (otherwise the rerank can't even fill
        // the result set). The clamp protects "ask for 100, scan
        // 1000"; it can't protect "ask for 1000, scan ≥1000".
        assert_eq!(candidate_pool_for_limit(1000), 1000);
        assert_eq!(candidate_pool_for_limit(u32::MAX), u32::MAX);
    }

    #[test]
    fn candidate_pool_for_limit_never_below_limit_itself() {
        // The final `.max(limit)` guard is an invariant: even if a
        // future tweak ever shrank the floor below typical limits,
        // the pool must never return fewer candidates than the
        // caller asked for. We can't break this without the
        // top-level RankedChunk count silently dropping below the
        // user's `limit`.
        for limit in [1u32, 50, 200, 600, 1000] {
            let pool = candidate_pool_for_limit(limit);
            assert!(
                pool >= limit.min(CANDIDATE_POOL_CEIL),
                "pool={pool} must be ≥ min(limit={limit}, ceil={CANDIDATE_POOL_CEIL})"
            );
        }
    }

    #[test]
    fn hybrid_opts_default_routes_through_central_policy() {
        // Smoke: the default constructor honours the policy. If
        // someone reintroduces a hardcoded value here, this test
        // catches it.
        let opts = HybridOpts::default();
        assert_eq!(opts.limit, 20);
        assert_eq!(opts.candidate_pool, candidate_pool_for_limit(20));
        assert!(matches!(opts.mode, SearchMode::Hybrid));
    }

    #[test]
    fn hybrid_opts_for_limit_picks_up_mode() {
        let opts = HybridOpts::for_limit(5, SearchMode::Vector);
        assert_eq!(opts.limit, 5);
        assert_eq!(opts.candidate_pool, CANDIDATE_POOL_FLOOR);
        assert!(matches!(opts.mode, SearchMode::Vector));
    }

    #[test]
    fn hybrid_opts_fulltext_only_delegates_to_policy() {
        let opts = HybridOpts::fulltext_only(7);
        assert_eq!(opts.limit, 7);
        // Below the floor → clamped up.
        assert_eq!(opts.candidate_pool, CANDIDATE_POOL_FLOOR);
        assert!(matches!(opts.mode, SearchMode::Fulltext));
    }
    use chrono::Utc;

    /// Returns a deterministic vector that's 1.0 on the index of one of
    /// {alpha, beta, gamma} and 0.0 elsewhere. Lets us build vectors with
    /// predictable cosine distances without a real model.
    struct ToyProvider {
        id: ModelId,
    }
    impl ToyProvider {
        fn new() -> Self {
            Self {
                id: ModelId::new("test", "toy", 1),
            }
        }
        fn one_hot(label: &str) -> Vec<f32> {
            let mut v = vec![0.0f32; 4];
            match label {
                s if s.contains("alpha") => v[0] = 1.0,
                s if s.contains("beta") => v[1] = 1.0,
                s if s.contains("gamma") => v[2] = 1.0,
                _ => v[3] = 1.0,
            }
            v
        }
    }
    #[async_trait]
    impl EmbeddingProvider for ToyProvider {
        fn model_id(&self) -> ModelId {
            self.id.clone()
        }
        fn dim(&self) -> u16 {
            4
        }
        async fn embed_batch(&self, texts: &[&str], _task: EmbeddingTask) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| Self::one_hot(t)).collect())
        }
    }

    fn rec(id: &str, content: &str) -> AnamnesisRecord {
        AnamnesisRecord {
            id: RecordId::from_parts("a", None, id),
            source: SourceDescriptor {
                adapter: "a".into(),
                instance: None,
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
                native_id: id.into(),
                native_path: None,
                captured_at: Utc::now(),
                raw_hash: "h".into(),
                derived_from: None,
            },
            schema_version: SCHEMA_VERSION,
        }
    }

    async fn seed_with_embeddings(store: &Store, provider: &ToyProvider) {
        store.set_active_model(&provider.model_id().0).unwrap();
        for (id, content) in [
            ("a", "alpha bright morning"),
            ("b", "beta evening tea"),
            ("c", "gamma rays travel quickly"),
        ] {
            let r = rec(id, content);
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(&r, &chunks, None).unwrap();
        }
        // Drain the queue with vectors derived from each chunk's content.
        while let Some(job) = store.claim_next_job(&provider.model_id().0).unwrap() {
            let v = provider
                .embed_batch(&[&job.content], EmbeddingTask::Document)
                .await
                .unwrap()
                .pop()
                .unwrap();
            store.complete_job(&job, &v).unwrap();
        }
    }

    #[tokio::test]
    async fn fulltext_only_returns_fts_hits() {
        let store = Store::open_in_memory().unwrap();
        let provider = ToyProvider::new();
        seed_with_embeddings(&store, &provider).await;
        let searcher = HybridSearcher::<ToyProvider>::fulltext_only();
        let opts = HybridOpts::fulltext_only(10);
        let hits = searcher.search(&store, "alpha", &opts).await.unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0].content.contains("alpha"));
        assert!(hits.iter().all(|h| h.from_fts && !h.from_vec));
    }

    #[tokio::test]
    async fn vector_only_returns_vec_hits() {
        let store = Store::open_in_memory().unwrap();
        let provider = ToyProvider::new();
        seed_with_embeddings(&store, &provider).await;
        let searcher = HybridSearcher::new(&provider);
        let opts = HybridOpts {
            mode: SearchMode::Vector,
            ..HybridOpts::default()
        };
        let hits = searcher
            .search(&store, "alpha is in this query", &opts)
            .await
            .unwrap();
        assert!(!hits.is_empty());
        // ToyProvider gives the query "alpha"-detector vec → matches chunk
        // "alpha bright morning" with cosine 1.0.
        assert!(hits[0].content.contains("alpha"));
        assert!(hits.iter().all(|h| h.from_vec));
    }

    #[tokio::test]
    async fn hybrid_combines_both_modalities() {
        let store = Store::open_in_memory().unwrap();
        let provider = ToyProvider::new();
        seed_with_embeddings(&store, &provider).await;
        let searcher = HybridSearcher::new(&provider);
        let opts = HybridOpts::default();
        let hits = searcher
            .search(&store, "alpha bright", &opts)
            .await
            .unwrap();
        assert!(!hits.is_empty());
        // The "alpha" chunk should be in the top result and be flagged
        // by both modalities (FTS hits 'alpha', vec hits 1-hot[0]).
        let top = &hits[0];
        assert!(top.from_fts && top.from_vec);
        assert!(top.content.contains("alpha"));
    }

    #[tokio::test]
    async fn empty_query_returns_no_hits_in_fulltext() {
        let store = Store::open_in_memory().unwrap();
        let provider = ToyProvider::new();
        seed_with_embeddings(&store, &provider).await;
        let searcher = HybridSearcher::new(&provider);
        let opts = HybridOpts::fulltext_only(10);
        // PR-Jieba (round-5): the FTS layer now short-circuits an empty
        // / punctuation-only query into an empty result set rather than
        // letting SQLite raise on `MATCH ''`. An empty user query truly
        // has zero matches; surfacing that as an error was an artefact
        // of FTS5's strictness, not a useful signal.
        let hits = searcher.search(&store, "", &opts).await.unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn missing_provider_forces_fulltext_mode() {
        let store = Store::open_in_memory().unwrap();
        let provider = ToyProvider::new();
        seed_with_embeddings(&store, &provider).await;
        let searcher = HybridSearcher::<ToyProvider>::fulltext_only();
        // Caller asked for Hybrid but no provider → effective Fulltext.
        let opts = HybridOpts {
            mode: SearchMode::Hybrid,
            ..HybridOpts::default()
        };
        let hits = searcher.search(&store, "alpha", &opts).await.unwrap();
        assert!(hits.iter().all(|h| h.from_fts && !h.from_vec));
    }

    #[tokio::test]
    async fn limit_caps_results() {
        let store = Store::open_in_memory().unwrap();
        let provider = ToyProvider::new();
        seed_with_embeddings(&store, &provider).await;
        let searcher = HybridSearcher::new(&provider);
        let opts = HybridOpts {
            limit: 1,
            ..HybridOpts::default()
        };
        let hits = searcher.search(&store, "alpha", &opts).await.unwrap();
        assert_eq!(hits.len(), 1);
    }

    // ─── Round-71: per-stage search tracing ─────────────────────────

    #[tokio::test]
    async fn search_trace_reports_fulltext_stage_counts_only() {
        let store = Store::open_in_memory().unwrap();
        let provider = ToyProvider::new();
        seed_with_embeddings(&store, &provider).await;
        let searcher = HybridSearcher::<ToyProvider>::fulltext_only();
        let opts = HybridOpts::fulltext_only(10);
        let traced = searcher
            .search_filtered_traced(&store, "alpha", &SearchFilter::default(), &opts)
            .await
            .unwrap();

        assert_eq!(traced.trace.effective_mode, SearchMode::Fulltext);
        assert_eq!(
            traced.trace.candidate_pool,
            opts.candidate_pool.max(opts.limit)
        );
        // Fulltext mode: FTS ran, embed/vec did not.
        assert!(
            traced.trace.stages_ms.fts_ms.is_some(),
            "fts stage should be timed"
        );
        assert!(traced.trace.stages_ms.embed_query_ms.is_none());
        assert!(traced.trace.stages_ms.vec_ms.is_none());
        assert!(traced.trace.stages_ms.rrf_ms.is_some());
        // Counts: FTS saw something; vec untouched.
        assert!(traced.trace.counts.fts_hits >= 1);
        assert_eq!(traced.trace.counts.vec_hits, 0);
        assert_eq!(
            traced.trace.counts.ranked_chunks as usize,
            traced.hits.len(),
            "ranked_chunks should match returned hits len",
        );
    }

    #[tokio::test]
    async fn search_trace_reports_all_stages_in_hybrid_mode() {
        let store = Store::open_in_memory().unwrap();
        let provider = ToyProvider::new();
        seed_with_embeddings(&store, &provider).await;
        let searcher = HybridSearcher::new(&provider);
        let opts = HybridOpts {
            mode: SearchMode::Hybrid,
            ..HybridOpts::default()
        };
        let traced = searcher
            .search_filtered_traced(&store, "alpha", &SearchFilter::default(), &opts)
            .await
            .unwrap();

        assert_eq!(traced.trace.effective_mode, SearchMode::Hybrid);
        // Every stage must have a timing in hybrid mode.
        assert!(traced.trace.stages_ms.fts_ms.is_some());
        assert!(traced.trace.stages_ms.embed_query_ms.is_some());
        assert!(traced.trace.stages_ms.vec_ms.is_some());
        assert!(traced.trace.stages_ms.rrf_ms.is_some());
        // Both modalities contributed.
        assert!(traced.trace.counts.fts_hits >= 1);
        assert!(traced.trace.counts.vec_hits >= 1);
    }

    /// `search_filtered` is now defined as
    /// `search_filtered_traced(...).hits` — this guards against a
    /// future refactor accidentally re-introducing a second search
    /// code path with subtly different ranking.
    #[tokio::test]
    async fn search_filtered_returns_same_hits_as_traced_primitive() {
        let store = Store::open_in_memory().unwrap();
        let provider = ToyProvider::new();
        seed_with_embeddings(&store, &provider).await;
        let searcher = HybridSearcher::new(&provider);
        let opts = HybridOpts {
            mode: SearchMode::Hybrid,
            ..HybridOpts::default()
        };
        let plain = searcher
            .search_filtered(&store, "alpha", &SearchFilter::default(), &opts)
            .await
            .unwrap();
        let traced = searcher
            .search_filtered_traced(&store, "alpha", &SearchFilter::default(), &opts)
            .await
            .unwrap();
        assert_eq!(plain.len(), traced.hits.len());
        for (a, b) in plain.iter().zip(traced.hits.iter()) {
            assert_eq!(a.chunk_id, b.chunk_id);
            assert_eq!(a.record_id, b.record_id);
            assert!((a.score - b.score).abs() < 1e-9);
        }
    }

    /// Privacy guard: the serialised trace payload must contain only
    /// numeric stage shape + mode. Specifically: no query text, no
    /// chunk content, no record/chunk ids, no path strings. Any
    /// future field that smuggles user content trips this test.
    #[tokio::test]
    async fn search_trace_serialised_payload_excludes_user_content() {
        let store = Store::open_in_memory().unwrap();
        let provider = ToyProvider::new();
        seed_with_embeddings(&store, &provider).await;
        let searcher = HybridSearcher::new(&provider);
        let opts = HybridOpts::default();
        let traced = searcher
            .search_filtered_traced(
                &store,
                "alpha distinct phrase",
                &SearchFilter::default(),
                &opts,
            )
            .await
            .unwrap();
        let json = serde_json::to_string(&traced.trace).unwrap();
        for forbidden in [
            "alpha distinct phrase",
            "bright morning",
            "evening tea",
            // Chunk ids embed the record id, which is the sha hash.
            // We don't try to enumerate hashes; the content checks
            // above + the field-list whitelist below cover it.
        ] {
            assert!(
                !json.contains(forbidden),
                "trace payload must not contain {forbidden:?}: {json}",
            );
        }
        // Whitelist top-level shape — any new field that isn't one of
        // these needs an explicit privacy review.
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let obj = v.as_object().expect("trace serialises to an object");
        let allowed = ["effective_mode", "candidate_pool", "stages_ms", "counts"];
        for k in obj.keys() {
            assert!(
                allowed.contains(&k.as_str()),
                "unexpected top-level trace field {k:?}: would need a privacy review",
            );
        }
    }
}
