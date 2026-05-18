//! LLM provider trait + a deterministic mock impl.
//!
//! Per §-1.5 #6 / §-1.2 #5, the extractor must never silently call a
//! cloud LLM. That contract is enforced at the [`LlmProvider`] trait
//! boundary: every concrete impl is constructed explicitly with a model
//! id and any required credentials. The CLI's `anamnesis extract`
//! command prints which provider+model it's about to use **before**
//! sending any request.
//!
//! This crate currently ships one concrete provider:
//!
//! - [`MockProvider`] — fully deterministic, makes zero network calls.
//!   Used by the extractor's own test suite and as the default when the
//!   CLI runs `extract --no-dry-run` without a configured provider.
//!   The real OpenAI-compatible HTTP provider lands in a follow-up PR
//!   (it pulls in `reqwest` and needs `tokio`, so it gets its own
//!   feature flag).

use async_trait::async_trait;

use anamnesis_core::error::Result;

/// Pluggable LLM completion provider.
///
/// All methods are intentionally narrow — callers should be able to swap
/// implementations without touching the prompt templates or the Stage 2
/// driver. Implementations MUST be deterministic in `model_id()` and
/// `estimate_tokens()` so the cost preview is stable across runs.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Stable model identifier (e.g. `"mock:default"`,
    /// `"openai:gpt-4o-mini"`). Surfaced in audit logs and cost
    /// previews — renaming a model id is a breaking change.
    fn model_id(&self) -> &str;

    /// Heuristic token estimate for a prompt. Implementations are free
    /// to be inexact (~N chars / 4 is the standard cheap estimate for
    /// English); the value is used only for the up-front cost preview,
    /// not for billing.
    fn estimate_tokens(&self, prompt: &str) -> usize;

    /// Run one completion. Returns the raw model output as a string —
    /// the caller is responsible for parsing structured (JSON) output.
    ///
    /// Implementations should propagate transport / API errors as
    /// `Error::Other` so the Stage 2 driver can decide whether to skip
    /// the candidate or abort the run.
    async fn complete(&self, prompt: &str) -> Result<String>;
}

/// Deterministic mock — no network. Returns a fixed-shape JSON
/// response so the rest of the Stage 2 pipeline (prompt template →
/// LLM call → JSON parse → record build → store write) can be
/// exercised end-to-end in tests and on machines where no real
/// provider is configured.
///
/// The output is intentionally trivial (one item, content = "[mock]
/// {first 80 chars of input}"). Calling code that wants more
/// interesting fixtures should write a custom `LlmProvider` impl —
/// `MockProvider` is for the no-LLM-available default.
#[derive(Debug, Clone)]
pub struct MockProvider {
    model_id: String,
}

impl MockProvider {
    /// Build a mock provider with the given model id (defaults to
    /// `"mock:default"`).
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
        }
    }

    /// Default mock — model id `"mock:default"`.
    pub fn default_instance() -> Self {
        Self::new("mock:default")
    }
}

#[async_trait]
impl LlmProvider for MockProvider {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn estimate_tokens(&self, prompt: &str) -> usize {
        // ~4 chars / token (English-leaning rule of thumb). For CJK the
        // ratio is closer to 1:1 in tokens but most callers feed mixed
        // content; the cost preview is permissive either way.
        prompt.chars().count().div_ceil(4)
    }

    async fn complete(&self, prompt: &str) -> Result<String> {
        // First 80 chars of the prompt is the most likely place the
        // candidate's content snippet lives. Use it as a deterministic
        // mock output so test assertions can be specific.
        let snippet: String = prompt.chars().take(80).collect();
        let response = serde_json::json!([
            {
                "content": format!("[mock] {snippet}"),
                "confidence": 0.5,
            }
        ]);
        Ok(response.to_string())
    }
}

/// Helper: build a tracing-safe display string for the provider line
/// printed before any LLM call. Used by the CLI for `extract`'s
/// up-front cost preview.
pub fn cost_preview_line(
    provider_model: &str,
    candidates_count: usize,
    total_tokens: usize,
) -> String {
    format!(
        "Stage 2 plan: {candidates_count} candidate(s) → model {provider_model} \
         (~{total_tokens} input token(s) total)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_provider_model_id_is_stable() {
        let p = MockProvider::default_instance();
        assert_eq!(p.model_id(), "mock:default");
        let p2 = MockProvider::new("mock:special");
        assert_eq!(p2.model_id(), "mock:special");
    }

    #[test]
    fn mock_provider_token_estimate_is_proportional_to_length() {
        let p = MockProvider::default_instance();
        let short = p.estimate_tokens("hello");
        let long = p.estimate_tokens(&"x".repeat(400));
        assert!(long > short);
        assert!(long >= 100, "400 chars / 4 ≈ 100 tokens, got {long}");
    }

    #[tokio::test]
    async fn mock_provider_returns_deterministic_json() {
        let p = MockProvider::default_instance();
        let r1 = p.complete("test prompt for fact extraction").await.unwrap();
        let r2 = p.complete("test prompt for fact extraction").await.unwrap();
        assert_eq!(r1, r2, "mock must be deterministic");
        // Must parse as a JSON array of objects with content/confidence.
        let parsed: serde_json::Value = serde_json::from_str(&r1).unwrap();
        let arr = parsed.as_array().expect("array");
        assert!(!arr.is_empty());
        assert!(arr[0].get("content").is_some());
    }

    #[test]
    fn cost_preview_line_includes_all_three_inputs() {
        let line = cost_preview_line("openai:gpt-4o-mini", 7, 1234);
        assert!(line.contains("7 candidate"));
        assert!(line.contains("openai:gpt-4o-mini"));
        assert!(line.contains("1234 input token"));
    }
}
