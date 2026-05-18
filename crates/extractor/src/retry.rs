//! Retry-with-backoff helper used by the real LLM providers.
//!
//! HTTP-only providers (`OpenAiProvider`, `AnthropicProvider`) wrap
//! their actual request in [`retry_with_backoff`]. When the provider
//! sees a 429 or 5xx, it returns [`RetryStep::Retry`]; the helper
//! sleeps and tries again until `max_attempts` is exhausted.
//!
//! ## Why a separate module
//!
//! The retry logic is pure async control flow — no HTTP, no JSON.
//! Lives outside the feature-gated provider modules so the helper
//! itself stays tested even when `openai-provider` / `anthropic-provider`
//! features are off (which is the case on the `--no-default-features`
//! CI leg).
//!
//! ## Backoff schedule
//!
//! Exponential: `delay = base * 2^(attempt - 1)`, capped at `max`.
//! With the defaults (base=1000ms, max=16000ms, 3 attempts) that's:
//!
//! - attempt 1 fails → sleep 1s
//! - attempt 2 fails → sleep 2s
//! - attempt 3 fails → return error
//!
//! If the server returned a `Retry-After: <seconds>` header, we honor
//! whichever is **longer** between that hint and our computed backoff
//! — never shorter, so a buggy `Retry-After: 0` can't bypass the
//! schedule.

use std::time::Duration;

use anamnesis_core::error::{Error, Result};

/// Retry behavior. Both LLM providers expose
/// `with_max_retries(n)` / `with_retry_base_delay(d)` builder methods
/// that thread into one of these.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// Total attempts (including the first). `1` disables retry.
    pub max_attempts: u32,
    /// Delay before the first retry. Doubles every subsequent retry.
    pub base_delay_ms: u64,
    /// Upper bound on any single sleep.
    pub max_delay_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay_ms: 1_000,
            max_delay_ms: 16_000,
        }
    }
}

/// Outcome of one attempt that the wrapped closure returns.
pub enum RetryStep<T> {
    /// Closure succeeded — return the value, no more retries.
    Done(T),
    /// Closure failed but the failure is retryable (e.g. 429 / 5xx).
    /// `retry_after` is honored when set (e.g. parsed from a
    /// `Retry-After` header) — never shortens the computed backoff.
    Retry {
        /// Diagnostic message for the eventual failure path.
        message: String,
        /// Server-supplied hint, if any.
        retry_after: Option<Duration>,
    },
    /// Closure failed in a way retrying won't help (auth, malformed
    /// request, parsed-response-was-garbage). Bail immediately.
    Fatal(String),
}

/// Wrap an async closure with retry-with-backoff semantics. The
/// closure receives the 1-based attempt number so it can log
/// "attempt 2 of 3" or similar.
pub async fn retry_with_backoff<F, Fut, T>(policy: RetryPolicy, mut op: F) -> Result<T>
where
    F: FnMut(u32) -> Fut,
    Fut: std::future::Future<Output = RetryStep<T>>,
{
    let max = policy.max_attempts.max(1);
    let mut last_error = "no attempts run".to_string();
    for attempt in 1..=max {
        match op(attempt).await {
            RetryStep::Done(value) => return Ok(value),
            RetryStep::Fatal(msg) => return Err(Error::Other(msg)),
            RetryStep::Retry {
                message,
                retry_after,
            } => {
                last_error = message;
                if attempt == max {
                    break; // out of attempts
                }
                let delay = compute_delay(&policy, attempt, retry_after);
                tracing::debug!(
                    attempt,
                    max,
                    delay_ms = delay.as_millis() as u64,
                    "extractor retry: sleeping before next attempt"
                );
                tokio::time::sleep(delay).await;
            }
        }
    }
    Err(Error::Other(format!(
        "exhausted {max} retry attempt(s): {last_error}"
    )))
}

/// Compute the sleep duration for the next attempt. Exponential
/// backoff, capped at `policy.max_delay_ms`, and at least as long as
/// any server-supplied `Retry-After`. Pure function — tested directly.
pub fn compute_delay(
    policy: &RetryPolicy,
    failed_attempt: u32,
    retry_after: Option<Duration>,
) -> Duration {
    let exponent = failed_attempt.saturating_sub(1);
    // Cap the exponent so 2^exp doesn't overflow u64. With base 1000ms,
    // 2^63 ms ≈ 292 million years, so any exp >= 50 is silly.
    let exponent = exponent.min(50);
    let multiplier = 1u64.checked_shl(exponent).unwrap_or(u64::MAX);
    let raw_ms = policy
        .base_delay_ms
        .saturating_mul(multiplier)
        .min(policy.max_delay_ms);
    let computed = Duration::from_millis(raw_ms);
    match retry_after {
        Some(hint) if hint > computed => hint,
        _ => computed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn compute_delay_doubles_then_caps() {
        let policy = RetryPolicy {
            max_attempts: 5,
            base_delay_ms: 1000,
            max_delay_ms: 4000,
        };
        // attempt 1 just failed → sleep base
        assert_eq!(compute_delay(&policy, 1, None), Duration::from_millis(1000));
        // attempt 2 just failed → 2 * base
        assert_eq!(compute_delay(&policy, 2, None), Duration::from_millis(2000));
        // attempt 3 just failed → 4 * base
        assert_eq!(compute_delay(&policy, 3, None), Duration::from_millis(4000));
        // attempt 4 → would be 8 * base but capped at 4000
        assert_eq!(compute_delay(&policy, 4, None), Duration::from_millis(4000));
    }

    #[test]
    fn compute_delay_honors_retry_after_when_longer() {
        let policy = RetryPolicy::default();
        let hint = Duration::from_secs(30);
        let d = compute_delay(&policy, 1, Some(hint));
        assert_eq!(d, hint);
    }

    #[test]
    fn compute_delay_ignores_retry_after_when_shorter_than_backoff() {
        // A buggy `Retry-After: 0` must NOT bypass our schedule.
        let policy = RetryPolicy {
            max_attempts: 3,
            base_delay_ms: 2000,
            max_delay_ms: 16000,
        };
        let d = compute_delay(&policy, 1, Some(Duration::from_secs(0)));
        assert_eq!(d, Duration::from_millis(2000));
    }

    #[test]
    fn compute_delay_huge_attempt_does_not_overflow() {
        let policy = RetryPolicy::default();
        // Way past the saturating_shl edge.
        let d = compute_delay(&policy, 1000, None);
        // Should cap at max_delay_ms, not panic / overflow.
        assert_eq!(d, Duration::from_millis(policy.max_delay_ms));
    }

    #[tokio::test]
    async fn retry_returns_first_success_no_sleeps() {
        let calls = Arc::new(AtomicU32::new(0));
        let calls_clone = calls.clone();
        let result: Result<i32> = retry_with_backoff(RetryPolicy::default(), move |_| {
            let calls = calls_clone.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                RetryStep::Done(42)
            }
        })
        .await;
        assert_eq!(result.unwrap(), 42);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retry_fatal_short_circuits_with_no_retries() {
        let calls = Arc::new(AtomicU32::new(0));
        let calls_clone = calls.clone();
        let result: Result<i32> = retry_with_backoff(
            RetryPolicy {
                max_attempts: 5,
                base_delay_ms: 1,
                max_delay_ms: 1,
            },
            move |_| {
                let calls = calls_clone.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    RetryStep::Fatal("auth failed".into())
                }
            },
        )
        .await;
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("auth failed"));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "Fatal must not trigger any retry"
        );
    }

    #[tokio::test]
    async fn retry_succeeds_after_n_retries() {
        let calls = Arc::new(AtomicU32::new(0));
        let calls_clone = calls.clone();
        let result: Result<&'static str> = retry_with_backoff(
            RetryPolicy {
                max_attempts: 4,
                base_delay_ms: 1, // tiny so the test runs fast
                max_delay_ms: 1,
            },
            move |attempt| {
                let calls = calls_clone.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    if attempt < 3 {
                        RetryStep::Retry {
                            message: format!("flaky at attempt {attempt}"),
                            retry_after: None,
                        }
                    } else {
                        RetryStep::Done("ok")
                    }
                }
            },
        )
        .await;
        assert_eq!(result.unwrap(), "ok");
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn retry_exhausted_returns_last_error() {
        let result: Result<i32> = retry_with_backoff(
            RetryPolicy {
                max_attempts: 2,
                base_delay_ms: 1,
                max_delay_ms: 1,
            },
            |attempt| async move {
                RetryStep::Retry {
                    message: format!("attempt {attempt} failed"),
                    retry_after: None,
                }
            },
        )
        .await;
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("exhausted 2 retry attempt"));
        assert!(err.contains("attempt 2 failed"));
    }

    #[tokio::test]
    async fn retry_max_attempts_zero_clamps_to_one() {
        // 0 would underflow; the helper must clamp to 1.
        let calls = Arc::new(AtomicU32::new(0));
        let calls_clone = calls.clone();
        let result: Result<&'static str> = retry_with_backoff(
            RetryPolicy {
                max_attempts: 0,
                base_delay_ms: 1,
                max_delay_ms: 1,
            },
            move |_| {
                let calls = calls_clone.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    RetryStep::Done("only call")
                }
            },
        )
        .await;
        assert_eq!(result.unwrap(), "only call");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
