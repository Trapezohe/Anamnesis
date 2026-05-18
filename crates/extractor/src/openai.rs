//! OpenAI-compatible HTTP `LlmProvider` (feature-gated).
//!
//! Talks to any backend that implements the `POST /v1/chat/completions`
//! shape — OpenAI itself, plus Ollama, Together, OpenRouter, vLLM,
//! LMStudio, etc. The provider holds only `model_id`, `api_base`, and
//! an optional `api_key`; it does not cache responses or batch
//! requests (the Stage 2 driver runs one prompt at a time on purpose).
//!
//! ## Setup
//!
//! ```ignore
//! use anamnesis_extractor::{OpenAiProvider, LlmProvider};
//!
//! let provider = OpenAiProvider::new("gpt-4o-mini")
//!     .with_api_key(std::env::var("OPENAI_API_KEY").unwrap())
//!     .with_api_base("https://api.openai.com/v1");
//! let raw = provider.complete("…prompt…").await?;
//! ```
//!
//! ## Safety
//!
//! Per §-1.2 #5 the CLI must show "will use model X for N calls" before
//! constructing this provider. `OpenAiProvider::complete()` itself is
//! a plain HTTP POST — it doesn't print anything; the upstream caller
//! is responsible for the cost preview.

use anamnesis_core::error::{Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::provider::LlmProvider;
use crate::retry::{retry_with_backoff, RetryPolicy, RetryStep};

/// Default base URL — points at OpenAI itself. Override for any other
/// vendor that speaks the same wire format.
pub const DEFAULT_API_BASE: &str = "https://api.openai.com/v1";

/// Default request timeout (90s, generous to cover slow remote
/// inference). Override via [`OpenAiProvider::with_timeout`].
pub const DEFAULT_TIMEOUT_SECS: u64 = 90;

/// Default sampling temperature. Stage 2 wants stable extraction, not
/// creative writing.
pub const DEFAULT_TEMPERATURE: f32 = 0.1;

/// OpenAI-compatible chat-completions provider.
#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    model: String,
    api_base: String,
    api_key: Option<String>,
    temperature: f32,
    timeout_secs: u64,
    retry: RetryPolicy,
}

impl OpenAiProvider {
    /// Build a provider for `model` (e.g. `"gpt-4o-mini"`,
    /// `"llama3.2:3b"` for Ollama, `"meta-llama/Meta-Llama-3-8B"` for
    /// Together).
    ///
    /// The `model_id()` surface exposes `"openai:<model>"` so audit
    /// logs disambiguate it from `"mock:default"`.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_base: DEFAULT_API_BASE.into(),
            api_key: None,
            temperature: DEFAULT_TEMPERATURE,
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            retry: RetryPolicy::default(),
        }
    }

    /// Override the API base URL (for Ollama / vLLM / OpenRouter / etc.).
    pub fn with_api_base(mut self, base: impl Into<String>) -> Self {
        self.api_base = base.into();
        self
    }

    /// Set the bearer token. Required for OpenAI proper; Ollama and
    /// some local servers don't need one.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Override the sampling temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Override the per-request HTTP timeout.
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Override the retry policy. Defaults to 3 attempts with
    /// exponential backoff (1s → 2s, capped at 16s). Set `max_attempts
    /// = 1` to disable retry entirely.
    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry = policy;
        self
    }

    /// Convenience: cap the retry policy at `n` attempts (keeps the
    /// default backoff schedule).
    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.retry.max_attempts = n.max(1);
        self
    }

    /// The underlying model name (e.g. `"gpt-4o-mini"`) without the
    /// `"openai:"` prefix. Tests use this for assertion.
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

/// Request body sent to `POST {api_base}/chat/completions`.
#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

/// Minimal slice of the response we care about.
#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct ChatChoiceMessage {
    content: String,
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    fn model_id(&self) -> &str {
        // Stable, audit-log-friendly id. We don't build a `&str` from
        // `format!` here because the trait wants a borrowed ref; the
        // upstream cost preview prints whatever this returns.
        // Implementation note: we cache the full id in the struct
        // would be cleaner, but `&str` from `&self` requires that the
        // string live in `self`. To avoid a redundant field, we use
        // `Box::leak` on the prefix during construction — but that's
        // wasteful. Simpler: just return `&self.model` here and have
        // the CLI prepend `"openai:"` when displaying. That's what
        // the prefix-stripped form is for.
        //
        // For now we return the raw model so MockProvider stays
        // `mock:default` and OpenAiProvider stays `gpt-4o-mini` — a
        // small awkwardness we'll iron out when adding more providers.
        &self.model
    }

    fn estimate_tokens(&self, prompt: &str) -> usize {
        // Same heuristic as MockProvider. Real tokenization would be
        // model-specific (cl100k_base for OpenAI proper) but the cost
        // preview only needs to be in the right order of magnitude.
        prompt.chars().count().div_ceil(4)
    }

    async fn complete(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/chat/completions", self.api_base.trim_end_matches('/'));
        let body = ChatRequest {
            model: &self.model,
            messages: vec![ChatMessage {
                role: "user",
                content: prompt,
            }],
            temperature: self.temperature,
        };
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .build()
            .map_err(|e| Error::Other(format!("openai client build: {e}")))?;

        retry_with_backoff(self.retry, |_attempt| async {
            let mut req = client.post(&url).json(&body);
            if let Some(key) = &self.api_key {
                req = req.bearer_auth(key);
            }
            let resp = match req.send().await {
                Ok(r) => r,
                Err(e) => {
                    // Network-level errors (DNS, refused, timeout) are
                    // worth retrying — they're transient by definition.
                    return RetryStep::Retry {
                        message: format!("openai request: {e}"),
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
                        message: format!("openai read body: {e}"),
                        retry_after: retry_after_hint,
                    };
                }
            };
            if status.as_u16() == 429 || status.is_server_error() {
                return RetryStep::Retry {
                    message: format!("openai HTTP {status}: {}", truncate(&raw, 400)),
                    retry_after: retry_after_hint,
                };
            }
            if !status.is_success() {
                return RetryStep::Fatal(format!("openai HTTP {status}: {}", truncate(&raw, 400)));
            }
            let parsed: ChatResponse = match serde_json::from_str(&raw) {
                Ok(v) => v,
                Err(e) => {
                    return RetryStep::Fatal(format!(
                        "openai response parse: {e}; body={}",
                        truncate(&raw, 200)
                    ));
                }
            };
            match parsed.choices.into_iter().next() {
                Some(first) => RetryStep::Done(first.message.content),
                None => RetryStep::Fatal("openai response had zero choices".into()),
            }
        })
        .await
    }
}

fn retry_after_from_headers(headers: &reqwest::header::HeaderMap) -> Option<std::time::Duration> {
    let raw = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    let secs: u64 = raw.trim().parse().ok()?;
    Some(std::time::Duration::from_secs(secs))
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
        let p = OpenAiProvider::new("gpt-4o-mini")
            .with_api_base("https://example.invalid/v1")
            .with_api_key("sk-test")
            .with_temperature(0.5)
            .with_timeout(30);
        assert_eq!(p.model_name(), "gpt-4o-mini");
        assert_eq!(p.api_base(), "https://example.invalid/v1");
        assert_eq!(p.temperature, 0.5);
        assert_eq!(p.timeout_secs, 30);
        assert_eq!(p.api_key.as_deref(), Some("sk-test"));
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn default_constants_are_sensible() {
        assert_eq!(DEFAULT_API_BASE, "https://api.openai.com/v1");
        // Clippy treats const-vs-const as a compile-time check it
        // wants moved into `const { … }`, but the goal of these tests
        // is to make a const drift loudly fail CI — keep them as
        // runtime asserts so the test name is the bug report.
        assert!((30..=600).contains(&DEFAULT_TIMEOUT_SECS));
        assert!((0.0..=1.0).contains(&DEFAULT_TEMPERATURE));
    }

    #[test]
    fn model_id_is_model_name_no_prefix() {
        let p = OpenAiProvider::new("llama3.2:3b");
        // model_id returns the bare model so audit logs print the
        // configured name verbatim. The CLI cost-preview prepends
        // a vendor prefix when displaying.
        assert_eq!(p.model_id(), "llama3.2:3b");
    }

    #[test]
    fn token_estimate_is_proportional() {
        let p = OpenAiProvider::new("gpt-4o-mini");
        let short = p.estimate_tokens("hi");
        let long = p.estimate_tokens(&"x".repeat(400));
        assert!(long > short);
        assert!(long >= 100);
    }

    #[test]
    fn request_body_serializes_to_chat_completions_shape() {
        // Verify the wire format matches what OpenAI / Ollama expect.
        let body = ChatRequest {
            model: "gpt-4o-mini",
            messages: vec![ChatMessage {
                role: "user",
                content: "hello",
            }],
            temperature: 0.1,
        };
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["model"], "gpt-4o-mini");
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["messages"][0]["content"], "hello");
        assert!(json["temperature"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn response_parses_first_choice_content() {
        // Real OpenAI response body. Confirm we pick `choices[0].message.content`.
        let body = serde_json::json!({
            "id": "chatcmpl-xxx",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "[{\"content\":\"hi\",\"confidence\":0.9}]"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let parsed: ChatResponse = serde_json::from_value(body).unwrap();
        assert_eq!(parsed.choices.len(), 1);
        assert!(parsed.choices[0].message.content.contains("hi"));
    }

    #[test]
    fn truncate_caps_long_strings() {
        let s = "x".repeat(1000);
        let out = truncate(&s, 100);
        assert!(out.chars().count() <= 101); // 100 + the ellipsis
        assert!(out.ends_with('…'));
    }

    #[test]
    fn truncate_passes_short_strings_unchanged() {
        assert_eq!(truncate("short", 100), "short");
    }
}
