//! Detector for the generic MCP adapter — pings `/healthz` on the
//! configured upstream and reports reachability.

use anamnesis_core::discovery::{Confidence, DetectOpts, DetectedSource, SourceDetector};
use anamnesis_core::error::Result;
use async_trait::async_trait;

/// `GenericMcpDetector` requires explicit configuration — there's no
/// well-known default location to probe.
pub struct GenericMcpDetector {
    /// Base URL of the upstream MCP HTTP server (`http://127.0.0.1:7878`).
    pub url: String,
    /// Optional bearer token. The detector only hits `/healthz`, which
    /// doesn't require auth in the anamnesis-mcp server, but other
    /// implementations may.
    pub token: Option<String>,
    client: reqwest::Client,
}

impl GenericMcpDetector {
    /// Build a detector pointing at `url`.
    pub fn new(url: impl Into<String>, token: Option<String>) -> Self {
        Self {
            url: url.into(),
            token,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl SourceDetector for GenericMcpDetector {
    fn adapter_id(&self) -> &'static str {
        crate::ADAPTER_ID
    }

    async fn detect(&self, _opts: &DetectOpts) -> Result<Vec<DetectedSource>> {
        let healthz = format!("{}/healthz", self.url.trim_end_matches('/'));
        let mut req = self.client.get(&healthz);
        if let Some(t) = self.token.as_deref() {
            req = req.bearer_auth(t);
        }
        match req.send().await {
            Ok(r) if r.status().is_success() => Ok(vec![DetectedSource {
                adapter: crate::ADAPTER_ID.into(),
                instance: Some("upstream".into()),
                location: self.url.clone(),
                local_path: None,
                confidence: Confidence::High,
                estimated_records: None,
                note: Some("upstream MCP reachable".into()),
            }]),
            Ok(r) => Ok(vec![DetectedSource {
                adapter: crate::ADAPTER_ID.into(),
                instance: Some("upstream".into()),
                location: self.url.clone(),
                local_path: None,
                confidence: Confidence::Low,
                estimated_records: None,
                note: Some(format!("upstream returned HTTP {}", r.status())),
            }]),
            Err(_) => Ok(Vec::new()),
        }
    }
}
