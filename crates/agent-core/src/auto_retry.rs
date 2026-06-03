//! Application-level retry logic.
//!
//! Distinct from the per-provider HTTP retry (`provider.rs`) which only
//! retries the same HTTP call. This module's `is_retryable_error` /
//! `compute_retry_delay` operate on **finalised assistant messages** that
//! came back with `stop_reason = Error` — i.e. the streaming completed but
//! the model / proxy returned an error. The harness uses these to decide
//! whether to drop the error message and re-run the same agent turn.

use crate::overflow::is_context_overflow;
use crate::types::{AgentMessage, StopReason};
use once_cell::sync::Lazy;
use regex::Regex;

/// Patterns that indicate transient failures worth retrying (overloaded
/// providers, rate limits, 5xx, network / connection / websocket drops, ...).
static RETRYABLE_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)overloaded|provider.?returned.?error|rate.?limit|too many requests|429|500|502|503|504|service.?unavailable|server.?error|internal.?error|network.?error|connection.?error|connection.?refused|connection.?lost|websocket.?closed|websocket.?error|other side closed|fetch failed|upstream.?connect|reset before headers|socket hang up|ended without|stream ended before message_stop|http2 request did not get a response|timed? out|timeout|terminated|retry delay",
    )
    .expect("retryable pattern must compile")
});

/// Settings for the retry state machine. Serde-friendly (camelCase, missing
/// fields fall back to [`Default`]) so it can be embedded directly in
/// persisted config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct RetrySettings {
    pub enabled: bool,
    pub max_retries: u32,
    pub base_delay_ms: u64,
}

impl Default for RetrySettings {
    fn default() -> Self {
        Self {
            enabled: true,
            max_retries: 3,
            base_delay_ms: 1_000,
        }
    }
}

/// Whether `msg` represents a transient error the harness should retry.
///
/// Returns `false` for context-overflow errors (those are handled by
/// compaction, not retry) and for any non-Error-terminated message.
pub fn is_retryable_error(msg: &AgentMessage, context_window: Option<u64>) -> bool {
    let AgentMessage::Assistant { stop_reason, error_message, .. } = msg else {
        return false;
    };
    if *stop_reason != StopReason::Error {
        return false;
    }
    let Some(err) = error_message.as_deref() else {
        return false;
    };
    // Context overflow -> compaction handles it.
    if is_context_overflow(msg, context_window) {
        return false;
    }
    RETRYABLE_PATTERN.is_match(err)
}

/// Compute the next retry delay (exponential backoff: `base * 2^(attempt-1)`).
/// `attempt` is 1-indexed (the first retry uses `base_delay_ms`).
///
/// Returns `None` once `attempt > max_retries` to signal "give up".
pub fn compute_retry_delay(attempt: u32, settings: &RetrySettings) -> Option<u64> {
    if !settings.enabled || attempt == 0 || attempt > settings.max_retries {
        return None;
    }
    Some(settings.base_delay_ms.saturating_mul(1u64 << (attempt - 1)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AgentMessage, Api, Usage};

    fn errored(msg: &str) -> AgentMessage {
        AgentMessage::Assistant {
            content: vec![],
            api: Api::Anthropic,
            provider: "p".into(),
            model: "m".into(),
            usage: Usage::default(),
            stop_reason: StopReason::Error,
            error_message: Some(msg.into()),
            timestamp: 0,
        }
    }

    #[test]
    fn detects_overloaded() {
        assert!(is_retryable_error(&errored("Anthropic API: overloaded_error"), None));
    }

    #[test]
    fn detects_5xx() {
        for s in &[
            "HTTP 500: server_error",
            "502 Bad Gateway",
            "503 service unavailable",
            "504 timeout",
        ] {
            assert!(is_retryable_error(&errored(s), None), "should retry: {}", s);
        }
    }

    #[test]
    fn detects_rate_limit() {
        assert!(is_retryable_error(&errored("rate limit exceeded"), None));
        assert!(is_retryable_error(&errored("429 too many requests"), None));
    }

    #[test]
    fn detects_network_errors() {
        for s in &[
            "fetch failed",
            "socket hang up",
            "connection refused",
            "connection lost",
            "websocket closed",
            "stream ended before message_stop",
            "http2 request did not get a response",
            "request timed out",
        ] {
            assert!(is_retryable_error(&errored(s), None), "should retry: {}", s);
        }
    }

    #[test]
    fn skips_non_error_messages() {
        let m = AgentMessage::Assistant {
            content: vec![],
            api: Api::Anthropic,
            provider: "p".into(),
            model: "m".into(),
            usage: Usage::default(),
            stop_reason: StopReason::EndTurn,
            error_message: None,
            timestamp: 0,
        };
        assert!(!is_retryable_error(&m, None));
    }

    #[test]
    fn skips_context_overflow() {
        // Overflow errors must NOT be retried — compaction handles them.
        let m = errored("prompt is too long: 213462 tokens > 200000 maximum");
        assert!(!is_retryable_error(&m, Some(200_000)));
    }

    #[test]
    fn skips_user_messages() {
        let m = AgentMessage::user_text("hi");
        assert!(!is_retryable_error(&m, None));
    }

    #[test]
    fn skips_unrelated_errors() {
        // Auth, validation, etc. should not retry.
        assert!(!is_retryable_error(&errored("invalid api key"), None));
        assert!(!is_retryable_error(&errored("malformed request body"), None));
    }

    #[test]
    fn backoff_doubles_each_attempt() {
        let s = RetrySettings {
            enabled: true,
            max_retries: 3,
            base_delay_ms: 1000,
        };
        assert_eq!(compute_retry_delay(1, &s), Some(1000));
        assert_eq!(compute_retry_delay(2, &s), Some(2000));
        assert_eq!(compute_retry_delay(3, &s), Some(4000));
        assert_eq!(compute_retry_delay(4, &s), None); // exceeded
    }

    #[test]
    fn backoff_disabled_returns_none() {
        let s = RetrySettings {
            enabled: false,
            max_retries: 3,
            base_delay_ms: 1000,
        };
        assert_eq!(compute_retry_delay(1, &s), None);
    }

    #[test]
    fn backoff_zero_attempt_returns_none() {
        let s = RetrySettings::default();
        assert_eq!(compute_retry_delay(0, &s), None);
    }
}
