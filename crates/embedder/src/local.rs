//! Local `EmbeddingProvider` implemented on top of `fastembed-rs`.
//!
//! Gated behind the `local-fastembed` cargo feature so dev iterations
//! that don't need the ONNX runtime can compile fast.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anamnesis_core::embedding::{EmbeddingProvider, EmbeddingTask, ModelId};
use anamnesis_core::error::{Error, Result};
use async_trait::async_trait;

use crate::registry::CuratedModel;

/// `EmbeddingProvider` backed by a fastembed-managed ONNX model.
pub struct LocalFastembedProvider {
    model_info: &'static CuratedModel,
    model_id: ModelId,
    cache_dir: PathBuf,
    // `TextEmbedding::embed` is `&mut`, so wrap in Mutex for `&self` access
    // from the async trait. The queue worker calls one batch at a time;
    // contention is a non-issue.
    inner: Mutex<fastembed::TextEmbedding>,
}

impl std::fmt::Debug for LocalFastembedProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalFastembedProvider")
            .field("model_id", &self.model_id)
            .field("dim", &self.model_info.dim)
            .field("cache_dir", &self.cache_dir)
            .finish()
    }
}

impl LocalFastembedProvider {
    /// Build a provider for the curated `key` (see `registry::REGISTRY`).
    /// Downloads the model on first use; subsequent runs read from cache.
    pub fn new(key: &str, cache_dir: impl AsRef<Path>) -> Result<Self> {
        let info = crate::registry::by_key(key).ok_or_else(|| {
            Error::Other(format!(
                "unknown curated model: {key} (try one of: {})",
                crate::registry::available().join(", ")
            ))
        })?;
        if !info.is_local {
            return Err(Error::Other(format!(
                "model {key} is a cloud provider; use the cloud provider instead"
            )));
        }
        let cache_dir = cache_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&cache_dir).map_err(Error::Io)?;
        let fast_model = map_to_fastembed(info)?;
        let opts = fastembed::InitOptions::new(fast_model).with_cache_dir(cache_dir.clone());
        let inner = fastembed::TextEmbedding::try_new(opts)
            .map_err(|e| Error::Other(format!("fastembed init {key}: {e}")))?;
        Ok(Self {
            model_info: info,
            model_id: ModelId::new("local", info.key, 1),
            cache_dir,
            inner: Mutex::new(inner),
        })
    }

    /// Where the model files are cached.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// The curated model entry this provider serves.
    pub fn model_info(&self) -> &'static CuratedModel {
        self.model_info
    }

    fn prefixed(&self, texts: &[&str], task: EmbeddingTask) -> Vec<String> {
        let prefix = match task {
            EmbeddingTask::Query => self.model_info.query_prefix,
            EmbeddingTask::Document => self.model_info.doc_prefix,
        };
        match prefix {
            Some(p) => texts.iter().map(|t| format!("{p}{t}")).collect(),
            None => texts.iter().map(|t| (*t).to_owned()).collect(),
        }
    }
}

#[async_trait]
impl EmbeddingProvider for LocalFastembedProvider {
    fn model_id(&self) -> ModelId {
        self.model_id.clone()
    }

    fn dim(&self) -> u16 {
        self.model_info.dim
    }

    async fn embed_batch(&self, texts: &[&str], task: EmbeddingTask) -> Result<Vec<Vec<f32>>> {
        let inputs = self.prefixed(texts, task);
        // Synchronous CPU-bound call. The embedding worker is single-batch
        // at a time so blocking the runtime is acceptable; users who run
        // many parallel embedders should drive each on its own runtime.
        let guard = self.inner.lock().expect("provider inner mutex poisoned");
        guard
            .embed(inputs, None)
            .map_err(|e| Error::Other(format!("fastembed embed: {e}")))
    }
}

fn map_to_fastembed(info: &CuratedModel) -> Result<fastembed::EmbeddingModel> {
    use fastembed::EmbeddingModel as FE;
    Ok(match info.key {
        "default" => FE::MultilingualE5Small,
        "tiny" => FE::AllMiniLML6V2Q,
        "en" => FE::BGESmallENV15,
        "multi-strong" => FE::MultilingualE5Base,
        other => {
            return Err(Error::Other(format!(
                "no fastembed mapping for curated model: {other}"
            )))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static FE_CACHE_TMP_NONCE: AtomicU64 = AtomicU64::new(0);

    fn tmp_cache() -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = FE_CACHE_TMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "anamnesis-fe-cache-{nonce}-{pid}-{seq}",
            pid = std::process::id()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn unknown_key_errors() {
        let r = LocalFastembedProvider::new("nope-not-a-model", tmp_cache());
        assert!(r.is_err());
        let msg = format!("{}", r.unwrap_err());
        assert!(msg.contains("unknown curated model"));
        assert!(msg.contains("default")); // suggestion list rendered
    }

    #[test]
    fn cloud_voyage_rejected_by_local_provider() {
        let r = LocalFastembedProvider::new("cloud-voyage", tmp_cache());
        let err = r.unwrap_err();
        assert!(format!("{err}").contains("cloud provider"));
    }

    #[test]
    fn every_local_key_has_a_fastembed_mapping() {
        for m in crate::registry::local_only() {
            assert!(
                map_to_fastembed(m).is_ok(),
                "missing fastembed mapping for {}",
                m.key
            );
        }
    }

    // The instantiation + embed tests actually download the model
    // (~120 MB for `default`). They're gated behind FASTEMBED_DOWNLOAD=1
    // so plain `cargo test` stays fast and CI can opt in.
    fn allow_download() -> bool {
        std::env::var("FASTEMBED_DOWNLOAD").ok().as_deref() == Some("1")
    }

    #[tokio::test]
    async fn end_to_end_embed_with_real_model() {
        if !allow_download() {
            eprintln!("skipping: FASTEMBED_DOWNLOAD != 1");
            return;
        }
        let provider = LocalFastembedProvider::new("default", tmp_cache()).unwrap();
        assert_eq!(provider.dim(), 384);
        assert_eq!(provider.model_id().as_str(), "local:default:1");
        let v = provider
            .embed_batch(&["hello", "用户偏好"], EmbeddingTask::Document)
            .await
            .unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].len(), 384);
        assert_eq!(v[1].len(), 384);
        // E5 returns L2-normalized vectors → magnitude ~1.0
        let mag = (v[0].iter().map(|x| x * x).sum::<f32>()).sqrt();
        assert!(
            (mag - 1.0).abs() < 0.1,
            "expected ~L2-normalized vector, got mag {mag}"
        );
    }
}
