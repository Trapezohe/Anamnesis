//! Voyage AI cloud embedding provider.
//!
//! Per BLUEPRINT §16.8 the curated registry advertises `cloud-voyage` as
//! an opt-in option. Reads `VOYAGE_API_KEY` (or a custom env var). All
//! calls are explicit — never invoked unless the user picks the
//! `cloud-voyage` model AND the binary was built with the
//! `cloud-voyage` feature.

use anamnesis_core::embedding::{EmbeddingProvider, EmbeddingTask, ModelId};
use anamnesis_core::error::{Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Default Voyage API endpoint.
pub const VOYAGE_API_BASE: &str = "https://api.voyageai.com/v1/embeddings";

/// Voyage model id (the only one we curate today).
pub const VOYAGE_MODEL: &str = "voyage-3";

/// Voyage cloud `EmbeddingProvider`.
pub struct VoyageProvider {
    api_key: String,
    api_base: String,
    model: String,
    client: reqwest::Client,
    model_id: ModelId,
}

impl std::fmt::Debug for VoyageProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoyageProvider")
            .field("api_base", &self.api_base)
            .field("model", &self.model)
            .field("api_key", &"<redacted>")
            .finish()
    }
}

impl VoyageProvider {
    /// Build with the default model (`voyage-3`) reading
    /// `VOYAGE_API_KEY` from env.
    pub fn from_env() -> Result<Self> {
        let key = std::env::var("VOYAGE_API_KEY")
            .map_err(|_| Error::Other("VOYAGE_API_KEY not set in environment".into()))?;
        Self::with_key(&key)
    }

    /// Build with an explicit API key.
    pub fn with_key(api_key: &str) -> Result<Self> {
        Self::new(api_key, VOYAGE_API_BASE, VOYAGE_MODEL)
    }

    /// Build with explicit base URL + model — for tests against a mock
    /// server.
    pub fn new(api_key: &str, api_base: &str, model: &str) -> Result<Self> {
        if api_key.is_empty() {
            return Err(Error::Other("VOYAGE_API_KEY is empty".into()));
        }
        let client = reqwest::Client::builder()
            .build()
            .map_err(|e| Error::Other(format!("reqwest client: {e}")))?;
        Ok(Self {
            api_key: api_key.to_string(),
            api_base: api_base.to_string(),
            model: model.to_string(),
            client,
            model_id: ModelId::new("voyage", model, 1),
        })
    }
}

#[derive(Debug, Serialize)]
struct VoyageRequest<'a> {
    input: &'a [&'a str],
    model: &'a str,
    input_type: &'static str,
}

#[derive(Debug, Deserialize)]
struct VoyageResponse {
    data: Vec<VoyageData>,
}

#[derive(Debug, Deserialize)]
struct VoyageData {
    embedding: Vec<f32>,
}

#[async_trait]
impl EmbeddingProvider for VoyageProvider {
    fn model_id(&self) -> ModelId {
        self.model_id.clone()
    }

    fn dim(&self) -> u16 {
        // voyage-3 returns 1024-dim vectors.
        1024
    }

    async fn embed_batch(&self, texts: &[&str], task: EmbeddingTask) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let input_type = match task {
            EmbeddingTask::Query => "query",
            EmbeddingTask::Document => "document",
        };
        let body = VoyageRequest {
            input: texts,
            model: &self.model,
            input_type,
        };
        let response = self
            .client
            .post(&self.api_base)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Other(format!("voyage request: {e}")))?;
        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<no body>".to_string());
            return Err(Error::Other(format!("voyage HTTP {status}: {body}")));
        }
        let parsed: VoyageResponse = response
            .json()
            .await
            .map_err(|e| Error::Other(format!("voyage parse: {e}")))?;
        if parsed.data.len() != texts.len() {
            return Err(Error::Other(format!(
                "voyage returned {} vectors for {} inputs",
                parsed.data.len(),
                texts.len()
            )));
        }
        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_api_key_errors() {
        let r = VoyageProvider::with_key("");
        assert!(r.is_err());
    }

    #[test]
    fn from_env_without_key_errors() {
        // Remove and restore.
        let prev = std::env::var_os("VOYAGE_API_KEY");
        std::env::remove_var("VOYAGE_API_KEY");
        let r = VoyageProvider::from_env();
        if let Some(v) = prev {
            std::env::set_var("VOYAGE_API_KEY", v);
        }
        assert!(r.is_err());
    }

    #[test]
    fn constructor_sets_model_id() {
        let p = VoyageProvider::with_key("sk-test").unwrap();
        assert_eq!(p.model_id().as_str(), "voyage:voyage-3:1");
        assert_eq!(p.dim(), 1024);
    }

    // Real-network test is opt-in via VOYAGE_API_KEY + VOYAGE_TEST=1.
    #[tokio::test]
    async fn end_to_end_against_real_voyage_api() {
        if std::env::var("VOYAGE_TEST").ok().as_deref() != Some("1") {
            eprintln!("skipping: VOYAGE_TEST != 1");
            return;
        }
        let provider = VoyageProvider::from_env().expect("VOYAGE_API_KEY required");
        let vectors = provider
            .embed_batch(&["hello world", "用户偏好"], EmbeddingTask::Document)
            .await
            .expect("voyage call");
        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0].len(), 1024);
    }
}
