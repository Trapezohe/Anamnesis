//! `EmbeddingProvider` trait — the only seam between Anamnesis's RAG layer
//! and the underlying model runtime.
//!
//! ## Why a trait
//!
//! Anamnesis runs its **own** RAG stack (see `docs/BLUEPRINT.md §6.6.1`).
//! Source-system vectors (mem0, Hermes, …) are kept only as `provenance` and
//! never enter the retrieval path. This trait is therefore the *only* path by
//! which any vector ever reaches the index.
//!
//! ## Invariants every implementor must hold
//!
//! 1. **Stable `model_id`** — must be deterministic for the lifetime of a
//!    given (provider, model, version) tuple. The store uses
//!    `(content_hash, model_id)` as the embedding cache key; a drifting id
//!    silently invalidates the cache or, worse, mixes incompatible vectors.
//! 2. **Stable `dim`** — must match the size of every vector returned.
//! 3. **Deterministic normalization** — vectors returned by `embed_query` and
//!    `embed_batch` must be in the same numeric regime (e.g. both L2-
//!    normalised) so cosine similarity is meaningful.
//! 4. **Pure** — the trait itself does no IO scheduling; callers (the
//!    embedding worker) own batching, retry, and concurrency.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Stable identifier for an embedding model.
///
/// Format convention (not enforced, but recommended):
/// `"<provider>:<model>:<version>"`, e.g. `"local:multilingual-e5-small:1"`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(pub String);

impl ModelId {
    /// Build a model id from provider + model name + version.
    pub fn new(provider: &str, model: &str, version: u32) -> Self {
        Self(format!("{provider}:{model}:{version}"))
    }

    /// Borrow as `&str`.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Hint to the provider for asymmetric models (e.g. e5 / bge use different
/// prefixes for queries vs documents).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingTask {
    /// The text is a user search query.
    Query,
    /// The text is an indexed document/chunk.
    Document,
}

/// The only seam by which vectors enter the Anamnesis index.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Stable id — see invariants in the module docs.
    fn model_id(&self) -> ModelId;

    /// Vector dimensionality. Must match every vector returned.
    fn dim(&self) -> u16;

    /// Embed a single query string. Default impl forwards to `embed_batch`.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let mut out = self.embed_batch(&[text], EmbeddingTask::Query).await?;
        out.pop()
            .ok_or_else(|| crate::error::Error::Other("provider returned no vector".into()))
    }

    /// Embed a batch of texts. The provider is responsible for chunking the
    /// batch into model-friendly sizes; the worker calling this owns the
    /// outer batching loop.
    async fn embed_batch(&self, texts: &[&str], task: EmbeddingTask) -> Result<Vec<Vec<f32>>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_id_format_is_stable() {
        let id = ModelId::new("local", "multilingual-e5-small", 1);
        assert_eq!(id.as_str(), "local:multilingual-e5-small:1");
    }

    #[test]
    fn model_id_roundtrips_through_json() {
        let id = ModelId::new("local", "bge-m3", 1);
        let s = serde_json::to_string(&id).unwrap();
        // serde_transparent → just the string, not an object
        assert_eq!(s, "\"local:bge-m3:1\"");
        let back: ModelId = serde_json::from_str(&s).unwrap();
        assert_eq!(back, id);
    }

    /// Minimal in-memory provider used to lock the trait shape and prove the
    /// default `embed_query` forwarding works.
    struct FakeProvider {
        id: ModelId,
        dim: u16,
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
            // Deterministic dummy vector: length 4, filled with text length / 100.
            Ok(texts
                .iter()
                .map(|t| {
                    let v = (t.len() as f32) / 100.0;
                    vec![v; self.dim as usize]
                })
                .collect())
        }
    }

    #[tokio::test]
    async fn default_embed_query_forwards_to_batch() {
        let p = FakeProvider {
            id: ModelId::new("test", "fake", 1),
            dim: 4,
        };
        let v = p.embed_query("hello world").await.unwrap();
        assert_eq!(v.len(), 4);
        assert!((v[0] - 0.11).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn batch_returns_one_vector_per_input() {
        let p = FakeProvider {
            id: ModelId::new("test", "fake", 1),
            dim: 4,
        };
        let v = p
            .embed_batch(&["a", "bb", "ccc"], EmbeddingTask::Document)
            .await
            .unwrap();
        assert_eq!(v.len(), 3);
        assert!(v.iter().all(|row| row.len() == 4));
    }

    #[tokio::test]
    async fn embed_query_propagates_empty_provider_result() {
        struct Empty;
        #[async_trait]
        impl EmbeddingProvider for Empty {
            fn model_id(&self) -> ModelId {
                ModelId::new("test", "empty", 1)
            }
            fn dim(&self) -> u16 {
                4
            }
            async fn embed_batch(
                &self,
                _texts: &[&str],
                _task: EmbeddingTask,
            ) -> Result<Vec<Vec<f32>>> {
                Ok(vec![])
            }
        }
        let err = Empty.embed_query("x").await.unwrap_err();
        assert!(format!("{err}").contains("no vector"));
    }
}
