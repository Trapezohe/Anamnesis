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

    /// The underlying model name (e.g. `"gpt-4o-mini"`) without the
    /// `"openai:"` prefix. Tests use this for assertion.
    pub fn model_name(&self) -> &str {
        &self.model
    }

    /// Resolved base URL.
    pub fn api_base(&self) -> &str {
        &self.api_base
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
        let mut req = client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| Error::Other(format!("openai request: {e}")))?;
        let status = resp.status();
        let raw = resp
            .text()
            .await
            .map_err(|e| Error::Other(format!("openai read body: {e}")))?;
        if !status.is_success() {
            return Err(Error::Other(format!(
                "openai HTTP {status}: {}",
                truncate(&raw, 400)
            )));
        }
        let parsed: ChatResponse = serde_json::from_str(&raw).map_err(|e| {
            Error::Other(format!(
                "openai response parse: {e}; body={}",
                truncate(&raw, 200)
            ))
        })?;
        let first = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| Error::Other("openai response had zero choices".into()))?;
        Ok(first.message.content)
    }
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
