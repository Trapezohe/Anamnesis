//! Embedding worker — drains `embedding_jobs` against an
//! `EmbeddingProvider` and writes vectors back to the store.
//!
//! The worker is intentionally simple: one job at a time, one provider,
//! one store. Concurrency comes from running multiple workers (each on
//! its own Store handle / connection) if you really need it. For Phase 1
//! a single worker keeps semantics obvious and avoids SQLite contention.

use anamnesis_core::embedding::{EmbeddingProvider, EmbeddingTask};
use anamnesis_core::error::{Error, Result};
use anamnesis_store::Store;
use serde::{Deserialize, Serialize};

/// Wrap a store-layer error into a core error (core can't depend on store,
/// so we do the conversion at the call site).
fn s2c(e: anamnesis_store::StoreError) -> Error {
    Error::Other(format!("store: {e}"))
}

/// Per-drain summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DrainSummary {
    /// Model that produced the embeddings.
    pub model_id: String,
    /// Jobs that completed successfully.
    pub processed: u64,
    /// Jobs that failed (now in status='failed').
    pub failed: u64,
}

/// The worker. Stateless aside from the provider it wraps.
pub struct EmbeddingWorker<'a, P: EmbeddingProvider> {
    provider: &'a P,
}

impl<'a, P: EmbeddingProvider> EmbeddingWorker<'a, P> {
    /// Build a worker that drains jobs targeted at `provider.model_id()`.
    pub fn new(provider: &'a P) -> Self {
        Self { provider }
    }

    /// Process one job, if any. Returns:
    ///   - `Some(true)`  job claimed and completed
    ///   - `Some(false)` job claimed but embedding failed (marked failed)
    ///   - `None`        queue was empty for this model
    pub async fn run_once(&self, store: &Store) -> Result<Option<bool>> {
        let model_id = self.provider.model_id().0;
        let job = match store.claim_next_job(&model_id).map_err(s2c)? {
            Some(j) => j,
            None => return Ok(None),
        };
        match self
            .provider
            .embed_batch(&[&job.content], EmbeddingTask::Document)
            .await
        {
            Ok(mut vectors) => match vectors.pop() {
                Some(v) if v.len() as u16 == self.provider.dim() => {
                    store.complete_job(&job, &v).map_err(s2c)?;
                    Ok(Some(true))
                }
                Some(v) => {
                    let msg = format!(
                        "provider returned vec of dim {} but trait says dim {}",
                        v.len(),
                        self.provider.dim()
                    );
                    store.fail_job(job.job_id, &msg).map_err(s2c)?;
                    Ok(Some(false))
                }
                None => {
                    store
                        .fail_job(job.job_id, "provider returned no vectors")
                        .map_err(s2c)?;
                    Ok(Some(false))
                }
            },
            Err(e) => {
                store.fail_job(job.job_id, &format!("{e}")).map_err(s2c)?;
                Ok(Some(false))
            }
        }
    }

    /// Run until the queue for this model is empty. Returns aggregate counts.
    ///
    /// Round 63 perf: claims up to `BATCH = 64` jobs per `claim_next_jobs`
    /// and hands them to `embed_batch` in one call. For 1000 jobs that's
    /// `~16 provider requests + ~32 SQLite transactions` instead of
    /// `~1000 + ~2000`. If `embed_batch` fails for the whole batch (network
    /// error, single bad text poisoning a remote-batch call), each job is
    /// retried via the per-job path so a single bad chunk doesn't kill 63
    /// healthy siblings.
    pub async fn drain(&self, store: &Store) -> Result<DrainSummary> {
        const BATCH: usize = 64;
        let model_id = self.provider.model_id().0;
        let mut summary = DrainSummary {
            model_id: model_id.clone(),
            processed: 0,
            failed: 0,
        };
        loop {
            let jobs = store.claim_next_jobs(&model_id, BATCH).map_err(s2c)?;
            if jobs.is_empty() {
                break;
            }
            self.process_batch(store, jobs, &mut summary).await?;
        }
        Ok(summary)
    }

    /// Embed a single claimed batch. On batch-level success the store
    /// commits all vectors in one transaction; on batch-level failure we
    /// fall back to per-job processing so one bad text doesn't poison the
    /// whole drain step.
    async fn process_batch(
        &self,
        store: &Store,
        jobs: Vec<anamnesis_store::PendingEmbeddingJob>,
        summary: &mut DrainSummary,
    ) -> Result<()> {
        let texts: Vec<&str> = jobs.iter().map(|j| j.content.as_str()).collect();
        let expected_dim = self.provider.dim();
        match self
            .provider
            .embed_batch(&texts, EmbeddingTask::Document)
            .await
        {
            Ok(vectors)
                if vectors.len() == jobs.len()
                    && vectors.iter().all(|v| v.len() as u16 == expected_dim) =>
            {
                store.complete_jobs_batch(&jobs, &vectors).map_err(s2c)?;
                summary.processed += jobs.len() as u64;
                Ok(())
            }
            Ok(_vectors_mismatched) => {
                // Provider returned the wrong number of vectors, or one was
                // the wrong dim. Don't trust any of them — re-route each
                // through the per-job path so the working ones get retried
                // and the genuinely bad ones get marked failed individually.
                tracing::warn!(
                    expected = jobs.len(),
                    "embed_batch returned mismatched vector count/dim; falling back per-job"
                );
                self.fallback_per_job(store, jobs, summary).await
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    batch_size = jobs.len(),
                    "embed_batch failed for whole batch; falling back per-job",
                );
                self.fallback_per_job(store, jobs, summary).await
            }
        }
    }

    async fn fallback_per_job(
        &self,
        store: &Store,
        jobs: Vec<anamnesis_store::PendingEmbeddingJob>,
        summary: &mut DrainSummary,
    ) -> Result<()> {
        let expected_dim = self.provider.dim();
        for job in jobs {
            match self
                .provider
                .embed_batch(&[&job.content], EmbeddingTask::Document)
                .await
            {
                Ok(mut vectors) => match vectors.pop() {
                    Some(v) if v.len() as u16 == expected_dim => {
                        store.complete_job(&job, &v).map_err(s2c)?;
                        summary.processed += 1;
                    }
                    Some(v) => {
                        let msg = format!(
                            "provider returned vec of dim {} but trait says dim {}",
                            v.len(),
                            expected_dim
                        );
                        store.fail_job(job.job_id, &msg).map_err(s2c)?;
                        summary.failed += 1;
                    }
                    None => {
                        store
                            .fail_job(job.job_id, "provider returned no vectors")
                            .map_err(s2c)?;
                        summary.failed += 1;
                    }
                },
                Err(e) => {
                    store.fail_job(job.job_id, &format!("{e}")).map_err(s2c)?;
                    summary.failed += 1;
                }
            }
        }
        Ok(())
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
    use anamnesis_store::SearchFilter;
    use async_trait::async_trait;
    use chrono::Utc;

    /// Deterministic in-memory provider: vector is `[hash(text) mod 13 / 13.0; dim]`.
    /// Failing mode triggers an Err on any call.
    struct FakeProvider {
        id: ModelId,
        dim: u16,
        fail: bool,
        dim_mismatch: bool,
    }

    impl FakeProvider {
        fn new(model: &str, dim: u16) -> Self {
            Self {
                id: ModelId::new("test", model, 1),
                dim,
                fail: false,
                dim_mismatch: false,
            }
        }
        fn failing(model: &str, dim: u16) -> Self {
            Self {
                fail: true,
                ..Self::new(model, dim)
            }
        }
        fn wrong_dim(model: &str, dim: u16) -> Self {
            Self {
                dim_mismatch: true,
                ..Self::new(model, dim)
            }
        }
    }

    #[async_trait]
    impl EmbeddingProvider for FakeProvider {
        fn model_id(&self) -> ModelId {
            self.id.clone()
        }
        fn dim(&self) -> u16 {
            self.dim
        }
        async fn embed_batch(&self, texts: &[&str], _task: EmbeddingTask) -> Result<Vec<Vec<f32>>> {
            if self.fail {
                return Err(anamnesis_core::error::Error::Other("boom".into()));
            }
            let real_dim = if self.dim_mismatch {
                self.dim + 1
            } else {
                self.dim
            };
            Ok(texts
                .iter()
                .map(|t| {
                    let mut h = blake3::Hasher::new();
                    h.update(t.as_bytes());
                    let bytes = h.finalize();
                    let n = (u32::from_le_bytes([
                        bytes.as_bytes()[0],
                        bytes.as_bytes()[1],
                        bytes.as_bytes()[2],
                        bytes.as_bytes()[3],
                    ]) % 13) as f32;
                    vec![n / 13.0; real_dim as usize]
                })
                .collect())
        }
    }

    fn record(adapter: &str, id: &str, content: &str) -> AnamnesisRecord {
        AnamnesisRecord {
            id: RecordId::from_parts(adapter, None, id),
            source: SourceDescriptor {
                adapter: adapter.into(),
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

    fn seed(store: &Store, model_id: &str, records: &[(&str, &str)]) {
        store.set_active_model(model_id).unwrap();
        for (id, content) in records {
            let r = record("a", id, content);
            let chunks = Chunker::default().chunk(&r.id, &r.content);
            store.upsert_record(&r, &chunks, None).unwrap();
        }
    }

    #[tokio::test]
    async fn drain_processes_all_pending_jobs() {
        let store = Store::open_in_memory().unwrap();
        let provider = FakeProvider::new("fake", 4);
        seed(
            &store,
            &provider.model_id().0,
            &[("a", "alpha"), ("b", "beta")],
        );

        let worker = EmbeddingWorker::new(&provider);
        let summary = worker.drain(&store).await.unwrap();

        assert_eq!(summary.model_id, provider.model_id().0);
        assert_eq!(summary.processed, 2);
        assert_eq!(summary.failed, 0);

        let n: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(1) FROM embedding_jobs WHERE status = 'done'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 2);

        let emb_count: i64 = store
            .conn()
            .query_row("SELECT COUNT(1) FROM chunk_embeddings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(emb_count, 2);
    }

    #[tokio::test]
    async fn drain_only_touches_matching_model() {
        let store = Store::open_in_memory().unwrap();
        let provider_a = FakeProvider::new("model-a", 4);
        let provider_b = FakeProvider::new("model-b", 4);
        seed(&store, &provider_a.model_id().0, &[("x", "x")]);
        store
            .rebuild_embedding_jobs(&provider_b.model_id().0)
            .unwrap();

        let summary = EmbeddingWorker::new(&provider_a)
            .drain(&store)
            .await
            .unwrap();
        assert_eq!(summary.processed, 1);
        let pending_b: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(1) FROM embedding_jobs WHERE model_id = ?1 AND status = 'pending'",
                [&provider_b.model_id().0],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(pending_b, 1, "drainer must not touch other models' jobs");
    }

    #[tokio::test]
    async fn provider_errors_mark_jobs_failed() {
        let store = Store::open_in_memory().unwrap();
        let provider = FakeProvider::failing("fake", 4);
        seed(
            &store,
            &provider.model_id().0,
            &[("a", "alpha"), ("b", "beta")],
        );

        let summary = EmbeddingWorker::new(&provider).drain(&store).await.unwrap();
        assert_eq!(summary.processed, 0);
        assert_eq!(summary.failed, 2);

        let failed: i64 = store
            .conn()
            .query_row(
                "SELECT COUNT(1) FROM embedding_jobs WHERE status = 'failed'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(failed, 2);
    }

    #[tokio::test]
    async fn dim_mismatch_marks_failed_not_completed() {
        let store = Store::open_in_memory().unwrap();
        let provider = FakeProvider::wrong_dim("fake", 4);
        seed(&store, &provider.model_id().0, &[("a", "alpha")]);
        let summary = EmbeddingWorker::new(&provider).drain(&store).await.unwrap();
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.processed, 0);
    }

    #[tokio::test]
    async fn empty_queue_is_no_op() {
        let store = Store::open_in_memory().unwrap();
        let provider = FakeProvider::new("fake", 4);
        let summary = EmbeddingWorker::new(&provider).drain(&store).await.unwrap();
        assert_eq!(summary.processed, 0);
        assert_eq!(summary.failed, 0);
    }

    #[tokio::test]
    async fn run_once_returns_none_when_empty() {
        let store = Store::open_in_memory().unwrap();
        let provider = FakeProvider::new("fake", 4);
        assert!(EmbeddingWorker::new(&provider)
            .run_once(&store)
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn completed_embeddings_are_searchable_via_vec() {
        let store = Store::open_in_memory().unwrap();
        let provider = FakeProvider::new("fake", 4);
        seed(
            &store,
            &provider.model_id().0,
            &[("a", "alpha"), ("b", "beta")],
        );
        EmbeddingWorker::new(&provider).drain(&store).await.unwrap();
        // Query with same vector that "alpha" produced — it must be the
        // top-1 hit.
        let alpha_vec = provider
            .embed_batch(&["alpha"], EmbeddingTask::Document)
            .await
            .unwrap()
            .pop()
            .unwrap();
        let hits = store
            .search_chunks_vec(
                &alpha_vec,
                &provider.model_id().0,
                &SearchFilter::default(),
                2,
            )
            .unwrap();
        assert!(!hits.is_empty());
        assert!((hits[0].score - 1.0).abs() < 1e-6);
    }
}
