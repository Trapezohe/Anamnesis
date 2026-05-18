//! Anthropic Messages API `LlmProvider` (feature-gated).
//!
//! Talks to Anthropic's official Messages API at
//! `POST {api_base}/v1/messages`. Wire format differs from OpenAI:
//!
//! - **Auth**: `x-api-key` header (not bearer)
//! - **Versioning**: `anthropic-version` header (currently `2023-06-01`)
//! - **Body**: `{model, max_tokens, messages: [{role, content}]}` — note
//!   `max_tokens` is **required** for Anthropic (OpenAI defaults it).
//! - **Response**: `{content: [{type: "text", text: "…"}]}` — we
//!   concatenate every `text` block (Anthropic sometimes splits the
//!   response across multiple blocks for reasoning models).
//!
//! ## Setup
//!
//! ```ignore
//! use anamnesis_extractor::{AnthropicProvider, LlmProvider};
//!
//! let provider = AnthropicProvider::new("claude-3-5-sonnet-20241022")
//!     .with_api_key(std::env::var("ANTHROPIC_API_KEY").unwrap());
//! let raw = provider.complete("…prompt…").await?;
//! ```
//!
//! ## Why a separate provider
//!
//! OpenAI-compatible servers (Ollama, vLLM, OpenRouter, …) speak
//! Chat Completions. Anthropic itself doesn't — its `/v1/messages`
//! shape is meaningfully different. Building two thin providers
//! beats trying to shoehorn one into the other and paying for the
//! abstraction at every request site.

use anamnesis_core::error::{Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::provider::LlmProvider;
use crate::retry::{retry_with_backoff, RetryPolicy, RetryStep};

/// Default base URL — points at Anthropic's hosted API.
pub const ANTHROPIC_DEFAULT_API_BASE: &str = "https://api.anthropic.com";

/// `anthropic-version` header value the provider sends with every
/// request. Pinned so a future API version change is an explicit code
/// edit, not a silent behavior shift.
pub const ANTHROPIC_VERSION_HEADER: &str = "2023-06-01";

/// Default request timeout (90s). Anthropic's `claude-3-5-sonnet` is
/// fast; the larger budget covers `claude-3-opus` thinking time.
pub const ANTHROPIC_DEFAULT_TIMEOUT_SECS: u64 = 90;

/// Default sampling temperature — same conservative 0.1 the OpenAI
/// provider uses. Stage 2 wants stable extraction.
pub const ANTHROPIC_DEFAULT_TEMPERATURE: f32 = 0.1;

/// Default `max_tokens` for the response. Anthropic requires the field
/// in every request; 1024 is generous for one extraction batch.
pub const ANTHROPIC_DEFAULT_MAX_TOKENS: u32 = 1024;

/// Anthropic Messages API provider.
#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    model: String,
    api_base: String,
    api_key: Option<String>,
    temperature: f32,
    max_tokens: u32,
    timeout_secs: u64,
    retry: RetryPolicy,
}

impl AnthropicProvider {
    /// Build a provider for `model` (e.g. `"claude-3-5-sonnet-20241022"`,
    /// `"claude-3-5-haiku-20241022"`, `"claude-3-opus-20240229"`).
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_base: ANTHROPIC_DEFAULT_API_BASE.into(),
            api_key: None,
            temperature: ANTHROPIC_DEFAULT_TEMPERATURE,
            max_tokens: ANTHROPIC_DEFAULT_MAX_TOKENS,
            timeout_secs: ANTHROPIC_DEFAULT_TIMEOUT_SECS,
            retry: RetryPolicy::default(),
        }
    }

    /// Override the API base URL (useful for proxies).
    pub fn with_api_base(mut self, base: impl Into<String>) -> Self {
        self.api_base = base.into();
        self
    }

    /// Set the `x-api-key`.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Override sampling temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Override `max_tokens` — useful for longer extraction batches.
    pub fn with_max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = max;
        self
    }

    /// Override the per-request HTTP timeout.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Override the retry policy. Defaults to 3 attempts with
    /// exponential backoff (1s → 2s, capped at 16s).
    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry = policy;
        self
    }

    /// Convenience: cap retries at `n` total attempts (keeps default backoff).
    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.retry.max_attempts = n.max(1);
        self
    }

    /// Bare model name (no `"anthropic:"` prefix).
    pub fn model_name(&self) -> &str {
        &self.model
    }

    /// Resolved base URL.
    pub fn api_base(&self) -> &str {
        &self.api_base
    }

    /// Resolved retry policy (for audit-log visibility).
    pub fn retry_policy(&self) -> RetryPolicy {
        self.retry
    }
}

/// Request body sent to `POST {api_base}/v1/messages`.
#[derive(Debug, Serialize)]
struct MessagesRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    temperature: f32,
    messages: Vec<MessagesItem<'a>>,
}

#[derive(Debug, Serialize)]
struct MessagesItem<'a> {
    role: &'a str,
    content: &'a str,
}

/// Slice of the response we care about.
#[derive(Debug, Deserialize)]
struct MessagesResponse {
    content: Vec<MessagesContentBlock>,
}

#[derive(Debug, Deserialize)]
struct MessagesContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn model_id(&self) -> &str {
        // Same convention as OpenAiProvider — return the bare model name.
        // Vendor-disambiguating prefix is the CLI banner's job.
        &self.model
    }

    fn estimate_tokens(&self, prompt: &str) -> usize {
        // Anthropic's tokenizer is closer to Claude's BPE than to OpenAI's
        // cl100k_base, but ~4 chars/token still holds in expectation for
        // English-heavy prompts. The cost preview only needs the right
        // order of magnitude.
        prompt.chars().count().div_ceil(4)
    }

    async fn complete(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/v1/messages", self.api_base.trim_end_matches('/'));
        let body = MessagesRequest {
            model: &self.model,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            messages: vec![MessagesItem {
                role: "user",
                content: prompt,
            }],
        };
        let key = self.api_key.as_deref().ok_or_else(|| {
            Error::Other("anthropic provider: x-api-key missing; call with_api_key() first".into())
        })?;
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .build()
            .map_err(|e| Error::Other(format!("anthropic client build: {e}")))?;

        retry_with_backoff(self.retry, |_attempt| async {
            let resp = match client
                .post(&url)
                .header("x-api-key", key)
                .header("anthropic-version", ANTHROPIC_VERSION_HEADER)
                .header("content-type", "application/json")
                .json(&body)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    return RetryStep::Retry {
                        message: format!("anthropic request: {e}"),
                        retry_after: None,
                    };
                }
            };
            let status = resp.status();
            let retry_after_hint = retry_after_from_headers(resp.headers());
            let raw = match resp.text().await {
                Ok(s) => s,
                Err(e) => {
                    return RetryStep::Retry {
                        message: format!("anthropic read body: {e}"),
                        retry_after: retry_after_hint,
                    };
                }
            };
            if status.as_u16() == 429 || status.is_server_error() {
                return RetryStep::Retry {
                    message: format!("anthropic HTTP {status}: {}", truncate(&raw, 400)),
                    retry_after: retry_after_hint,
                };
            }
            if !status.is_success() {
                return RetryStep::Fatal(format!(
                    "anthropic HTTP {status}: {}",
                    truncate(&raw, 400)
                ));
            }
            let parsed: MessagesResponse = match serde_json::from_str(&raw) {
                Ok(v) => v,
                Err(e) => {
                    return RetryStep::Fatal(format!(
                        "anthropic response parse: {e}; body={}",
                        truncate(&raw, 200)
                    ));
                }
            };
            let combined = combine_text_blocks(&parsed.content);
            if combined.is_empty() {
                return RetryStep::Fatal("anthropic response had no text content blocks".into());
            }
            RetryStep::Done(combined)
        })
        .await
    }
}

fn retry_after_from_headers(headers: &reqwest::header::HeaderMap) -> Option<std::time::Duration> {
    let raw = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    let secs: u64 = raw.trim().parse().ok()?;
    Some(std::time::Duration::from_secs(secs))
}

fn combine_text_blocks(blocks: &[MessagesContentBlock]) -> String {
    let mut out = String::new();
    for blk in blocks {
        if blk.block_type != "text" {
            continue;
        }
        if let Some(text) = &blk.text {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(text);
        }
    }
    out
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let head: String = s.chars().take(max).collect();
    format!("{head}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_chain_records_config() {
        let p = AnthropicProvider::new("claude-3-5-sonnet-20241022")
            .with_api_base("https://api.anthropic.com")
            .with_api_key("sk-ant-test")
            .with_temperature(0.3)
            .with_max_tokens(2048)
            .with_timeout(45);
        assert_eq!(p.model_name(), "claude-3-5-sonnet-20241022");
        assert_eq!(p.api_base(), "https://api.anthropic.com");
        assert_eq!(p.temperature, 0.3);
        assert_eq!(p.max_tokens, 2048);
        assert_eq!(p.timeout_secs, 45);
        assert_eq!(p.api_key.as_deref(), Some("sk-ant-test"));
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn default_constants_are_sensible() {
        assert_eq!(ANTHROPIC_DEFAULT_API_BASE, "https://api.anthropic.com");
        assert_eq!(ANTHROPIC_VERSION_HEADER, "2023-06-01");
        assert!((30..=600).contains(&ANTHROPIC_DEFAULT_TIMEOUT_SECS));
        assert!((0.0..=1.0).contains(&ANTHROPIC_DEFAULT_TEMPERATURE));
        assert!(ANTHROPIC_DEFAULT_MAX_TOKENS >= 256);
    }

    #[test]
    fn model_id_is_model_name_no_prefix() {
        let p = AnthropicProvider::new("claude-3-5-haiku-20241022");
        assert_eq!(p.model_id(), "claude-3-5-haiku-20241022");
    }

    #[test]
    fn token_estimate_is_proportional() {
        let p = AnthropicProvider::new("claude-3-5-sonnet-20241022");
        let short = p.estimate_tokens("hi");
        let long = p.estimate_tokens(&"x".repeat(400));
        assert!(long > short);
        assert!(long >= 100);
    }

    #[test]
    fn request_body_has_max_tokens_required_field() {
        // Anthropic rejects requests without max_tokens. Confirm we
        // always include it (the OpenAI provider doesn't need to).
        let body = MessagesRequest {
            model: "claude-3-5-sonnet-20241022",
            max_tokens: 1024,
            temperature: 0.1,
            messages: vec![MessagesItem {
                role: "user",
                content: "hello",
            }],
        };
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["model"], "claude-3-5-sonnet-20241022");
        assert_eq!(json["max_tokens"], 1024);
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["messages"][0]["content"], "hello");
    }

    #[test]
    fn response_parses_single_text_block() {
        // Real Anthropic response body.
        let body = serde_json::json!({
            "id": "msg_xxx",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "[{\"content\":\"fact\",\"confidence\":0.9}]"}
            ],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let parsed: MessagesResponse = serde_json::from_value(body).unwrap();
        let combined = combine_text_blocks(&parsed.content);
        assert!(combined.contains("fact"));
    }

    #[test]
    fn response_concatenates_multi_text_blocks() {
        // Reasoning-style models split output across several text
        // blocks (sometimes with thinking blocks interleaved we skip).
        let body = serde_json::json!({
            "id": "msg_yyy",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "ignore me"},
                {"type": "text", "text": "first chunk"},
                {"type": "text", "text": "second chunk"}
            ],
            "model": "claude-3-5-sonnet-20241022"
        });
        let parsed: MessagesResponse = serde_json::from_value(body).unwrap();
        let combined = combine_text_blocks(&parsed.content);
        assert_eq!(combined, "first chunk\nsecond chunk");
    }

    #[test]
    fn truncate_caps_long_strings() {
        let s = "x".repeat(1000);
        let out = truncate(&s, 100);
        assert!(out.chars().count() <= 101);
        assert!(out.ends_with('…'));
    }

    #[tokio::test]
    async fn complete_without_api_key_errors_loudly() {
        let p = AnthropicProvider::new("claude-3-5-sonnet-20241022");
        let err = p.complete("hello").await.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("x-api-key missing"));
    }
}
