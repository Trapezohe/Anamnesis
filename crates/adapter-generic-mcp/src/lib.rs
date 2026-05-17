//! Generic MCP adapter — turns any MCP-aware HTTP server into a
//! memory source for Anamnesis. Closes the loop on BLUEPRINT §11
//! Phase 4 "generic MCP adapter / reverse mode".
//!
//! The adapter speaks the same minimal HTTP JSON-RPC profile that
//! `anamnesis-mcp --sse` serves:
//!
//!   POST {url}/mcp                   — JSON-RPC body
//!   Authorization: Bearer <token>   — required when the server demands it
//!
//! Behaviour:
//!   - `detect` hits `{url}/healthz` (no auth) to verify reachability.
//!   - `scan` calls `resources/list`, then `resources/read` for each
//!     URI. Each resource payload becomes one `RawRecord`.
//!   - `normalize` maps payload → `AnamnesisRecord` with
//!     `Kind::Unknown` + `Scope::Ephemeral` as conservative defaults.
//!     Downstream packing can re-tag.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod detector;
pub mod normalizer;

use std::sync::Arc;

use anamnesis_core::adapter::{HealthStatus, MemoryAdapter, RawRecord, ScanOpts};
use anamnesis_core::error::{Error, Result};
use anamnesis_core::model::{AnamnesisRecord, SourceDescriptor};
use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use serde_json::{json, Value};

pub use detector::GenericMcpDetector;

/// Stable adapter id.
pub const ADAPTER_ID: &str = "generic-mcp";

/// Adapter configuration.
#[derive(Debug, Clone)]
pub struct GenericMcpConfig {
    /// Base URL of the upstream MCP HTTP server (e.g.
    /// `http://127.0.0.1:7878`). The adapter appends `/mcp` for JSON-RPC
    /// calls and `/healthz` for the detector ping.
    pub url: String,
    /// Pre-shared bearer token; `None` skips the Authorization header.
    pub token: Option<String>,
    /// Optional instance discriminator.
    pub instance: Option<String>,
}

/// The adapter.
pub struct GenericMcpAdapter {
    config: Arc<GenericMcpConfig>,
    client: reqwest::Client,
}

impl GenericMcpAdapter {
    /// Build with explicit config.
    pub fn new(config: GenericMcpConfig) -> Self {
        let client = reqwest::Client::builder()
            .build()
            .expect("reqwest client must build");
        Self {
            config: Arc::new(config),
            client,
        }
    }

    /// HTTP endpoint for JSON-RPC.
    pub fn endpoint(&self) -> String {
        format!("{}/mcp", self.config.url.trim_end_matches('/'))
    }

    /// Generic JSON-RPC call (public so library users can reach
    /// methods we don't wrap explicitly).
    pub async fn rpc(&self, method: &str, params: Value) -> Result<Value> {
        let payload = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        });
        let mut req = self.client.post(self.endpoint()).json(&payload);
        if let Some(t) = self.config.token.as_deref() {
            req = req.bearer_auth(t);
        }
        let resp = req.send().await.map_err(|e| Error::Adapter {
            adapter: ADAPTER_ID.into(),
            message: format!("send: {e}"),
        })?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp
                .text()
                .await
                .unwrap_or_else(|_| "<no body>".to_string());
            return Err(Error::Adapter {
                adapter: ADAPTER_ID.into(),
                message: format!("HTTP {status}: {body}"),
            });
        }
        let parsed: Value = resp.json().await.map_err(|e| Error::Adapter {
            adapter: ADAPTER_ID.into(),
            message: format!("parse: {e}"),
        })?;
        if let Some(err) = parsed.get("error") {
            return Err(Error::Adapter {
                adapter: ADAPTER_ID.into(),
                message: format!("rpc error: {err}"),
            });
        }
        Ok(parsed.get("result").cloned().unwrap_or(Value::Null))
    }
}

#[async_trait]
impl MemoryAdapter for GenericMcpAdapter {
    fn descriptor(&self) -> SourceDescriptor {
        SourceDescriptor {
            adapter: ADAPTER_ID.into(),
            instance: self.config.instance.clone(),
            version: env!("CARGO_PKG_VERSION").into(),
        }
    }

    fn scan<'a>(&'a self, opts: ScanOpts) -> BoxStream<'a, Result<RawRecord>> {
        // Round-20 (§-1.5 PR-4b): generic-mcp can't filter by
        // `opts.since` today because upstream `resources/list` exposes
        // neither per-resource `updated_at` nor a cursor-by-time
        // protocol. That gap belongs to §-1.5 PR-2 (resources/list
        // pagination). For now: emit a single one-shot warning per scan
        // when `opts.since` is set + `opts.full` is false, then return
        // every available resource. This is honest about the limitation
        // and preserves the existing migration loop's correctness
        // (the importer's raw_hash fast-path still skips no-op upserts).
        if !opts.full && opts.since.is_some() {
            tracing::warn!(
                adapter = ADAPTER_ID,
                instance = ?self.config.instance,
                since = ?opts.since,
                "generic-mcp adapter does not support `--since` filtering yet \
                 (waiting on §-1.5 PR-2: resources/list pagination + timestamps); \
                 returning all available upstream resources"
            );
        }
        let cfg = self.config.clone();
        let client = self.client.clone();
        // We fetch lazily inside the stream so the importer's async
        // runtime drives the HTTP calls. No block_in_place needed.
        let fut = async move { async_stream::fetch_all(cfg, client).await };
        let once = stream::once(fut).flat_map(|result| match result {
            Ok(raws) => stream::iter(raws.into_iter().map(Ok)).boxed(),
            Err(e) => stream::once(async move { Err(e) }).boxed(),
        });
        Box::pin(once)
    }

    fn normalize(&self, raw: RawRecord) -> Result<Vec<AnamnesisRecord>> {
        normalizer::normalize(raw, self.config.instance.as_deref())
    }

    async fn health(&self) -> HealthStatus {
        let healthz = format!("{}/healthz", self.config.url.trim_end_matches('/'));
        match self.client.get(&healthz).send().await {
            Ok(r) if r.status().is_success() => HealthStatus {
                ok: true,
                detail: format!("upstream MCP reachable at {}", self.config.url),
            },
            Ok(r) => HealthStatus {
                ok: false,
                detail: format!("upstream MCP returned {}", r.status()),
            },
            Err(e) => HealthStatus {
                ok: false,
                detail: format!("upstream MCP unreachable: {e}"),
            },
        }
    }
}

mod async_stream {
    //! Async fetcher for resources/list + resources/read. Called lazily
    //! by the stream returned from `MemoryAdapter::scan`.

    use super::*;

    /// Hard cap so a misbehaving / hostile upstream that returns
    /// `nextCursor` forever can't make the adapter spin indefinitely.
    /// 1000 pages × 1000 records = 1M record ceiling per scan, which
    /// is far above any realistic single migration.
    const MAX_LIST_PAGES: u32 = 1000;

    pub async fn fetch_all(
        cfg: Arc<GenericMcpConfig>,
        client: reqwest::Client,
    ) -> Result<Vec<RawRecord>> {
        let url = cfg.url.clone();
        let token = cfg.token.clone();
        let endpoint = format!("{}/mcp", url.trim_end_matches('/'));

        // Round-21 (§-1.5 PR-2): page through `resources/list` via
        // `cursor` / `nextCursor` so full migrations are no longer
        // truncated at the upstream's first-page limit. The previous
        // call sent `params: {}` (always page 1) and trusted whatever
        // the upstream chose to put in `resources` — typically capped
        // at 100. Now we follow `nextCursor` until the upstream stops
        // returning one OR we hit `MAX_LIST_PAGES`.
        let mut uris: Vec<String> = Vec::new();
        let mut cursor: Option<String> = None;
        let mut pages_seen: u32 = 0;
        loop {
            pages_seen += 1;
            if pages_seen > MAX_LIST_PAGES {
                tracing::warn!(
                    adapter = ADAPTER_ID,
                    "resources/list reached MAX_LIST_PAGES={MAX_LIST_PAGES}; \
                     refusing to follow another cursor (possible upstream loop)"
                );
                break;
            }
            let params = match cursor.as_deref() {
                Some(c) => json!({ "cursor": c }),
                None => json!({}),
            };
            let list_payload = json!({
                "jsonrpc": "2.0",
                "id": pages_seen,
                "method": "resources/list",
                "params": params,
            });
            let mut req = client.post(&endpoint).json(&list_payload);
            if let Some(t) = token.as_deref() {
                req = req.bearer_auth(t);
            }
            let body: Value = req
                .send()
                .await
                .map_err(|e| Error::Adapter {
                    adapter: ADAPTER_ID.into(),
                    message: format!("list send (page {pages_seen}): {e}"),
                })?
                .json()
                .await
                .map_err(|e| Error::Adapter {
                    adapter: ADAPTER_ID.into(),
                    message: format!("list parse (page {pages_seen}): {e}"),
                })?;
            if let Some(arr) = body["result"]["resources"].as_array() {
                for r in arr {
                    if let Some(u) = r["uri"].as_str() {
                        // Skip template URIs (contain placeholders).
                        if !u.contains('{') {
                            uris.push(u.to_owned());
                        }
                    }
                }
            }
            // Follow `nextCursor` if present and non-empty.
            cursor = body["result"]["nextCursor"]
                .as_str()
                .filter(|s| !s.is_empty())
                .map(str::to_owned);
            if cursor.is_none() {
                break;
            }
        }

        // 2. resources/read for each concrete URI.
        let mut raws = Vec::new();
        for uri in uris {
            let read_payload = json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "resources/read",
                "params": {"uri": uri}
            });
            let mut req2 = client.post(&endpoint).json(&read_payload);
            if let Some(t) = token.as_deref() {
                req2 = req2.bearer_auth(t);
            }
            let resp = match req2.send().await {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(uri = %uri, error = %e, "resources/read failed");
                    continue;
                }
            };
            let body: Value = match resp.json().await {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!(uri = %uri, error = %e, "resources/read parse failed");
                    continue;
                }
            };
            if let Some(arr) = body["result"]["contents"].as_array() {
                if let Some(first) = arr.first() {
                    let text = first["text"].as_str().unwrap_or("");
                    raws.push(normalizer::raw_resource(
                        &uri,
                        text.to_owned(),
                        cfg.instance.as_deref(),
                    ));
                }
            }
        }
        Ok(raws)
    }
}

/// Convenience constructor.
pub fn generic_mcp_adapter(
    url: impl Into<String>,
    token: Option<&str>,
    instance: Option<&str>,
) -> GenericMcpAdapter {
    GenericMcpAdapter::new(GenericMcpConfig {
        url: url.into(),
        token: token.map(str::to_owned),
        instance: instance.map(str::to_owned),
    })
}
