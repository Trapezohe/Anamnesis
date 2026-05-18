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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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
    /// better recall on rare matches but more work. Default: limit * 4.
    pub candidate_pool: u32,
    /// Modalities to combine.
    pub mode: SearchMode,
}

impl Default for HybridOpts {
    fn default() -> Self {
        Self {
            limit: 20,
            candidate_pool: 80,
            mode: SearchMode::Hybrid,
        }
    }
}

impl HybridOpts {
    /// Convenience: only FTS, no embedding call.
    pub fn fulltext_only(limit: u32) -> Self {
        Self {
            limit,
            candidate_pool: limit.saturating_mul(4).max(limit),
            mode: SearchMode::Fulltext,
        }
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
    pub async fn search_filtered(
        &self,
        store: &Store,
        query: &str,
        filter: &SearchFilter,
        opts: &HybridOpts,
    ) -> Result<Vec<RankedChunk>> {
        let effective_mode = if self.provider.is_none() {
            SearchMode::Fulltext
        } else {
            opts.mode
        };

        let pool = opts.candidate_pool.max(opts.limit);

        let fts_hits = if matches!(effective_mode, SearchMode::Fulltext | SearchMode::Hybrid) {
            store
                .search_chunks_fts(query, filter, pool)
                .map_err(|e| Error::Other(format!("store fts: {e}")))?
        } else {
            Vec::new()
        };

        let vec_hits = if matches!(effective_mode, SearchMode::Vector | SearchMode::Hybrid) {
            let provider = self
                .provider
                .ok_or_else(|| Error::Other("Vector/Hybrid mode requires a provider".into()))?;
            let qvec = provider.embed_query(query).await?;
            store
                .search_chunks_vec(&qvec, &provider.model_id().0, filter, pool)
                .map_err(|e| Error::Other(format!("store vec: {e}")))?
        } else {
            Vec::new()
        };

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
        Ok(merged)
    }
}

/// Pure RRF logic — separated for unit testing without a DB.
pub mod rrf {
    use super::{RankedChunk, RRF_K};
    use anamnesis_store::ChunkHit;
    use std::collections::HashMap;

    /// Merge two ranked hit lists via Reciprocal Rank Fusion.
    ///
    /// Returns the top-`limit` items by combined RRF score. Each
    /// returned `RankedChunk` also carries the raw per-modality scores
    /// (`fts_score`, `vector_score`) so downstream consumers — MCP
    /// agents in particular — can explain "why did this surface" without
    /// a round trip back to the index.
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
                });
            entry.score += 1.0 / (RRF_K + rank as f64 + 1.0);
            entry.fts_score = Some(hit.score);
            entry.from_fts = true;
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
                });
            entry.score += 1.0 / (RRF_K + rank as f64 + 1.0);
            entry.vector_score = Some(hit.score);
            entry.from_vec = true;
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
}
