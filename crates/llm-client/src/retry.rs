//! Retry helpers with HTTP-status-prefix matching.
//!
//! Trigger retry when the error message starts with `HTTP 5` or `HTTP 429`. Honour
//! `Retry-After` if the server supplied it.

use std::time::Duration;
use crate::provider::LlmError;

/// Should this error trigger a retry?
pub fn should_retry(err: &LlmError) -> bool {
    match err {
        LlmError::RateLimit { .. } => true,
        LlmError::Http { status, .. } => *status >= 500 || *status == 429,
        LlmError::Reqwest(e) => {
            // Network errors (timeout, connection refused, etc.) are retryable.
            e.is_timeout() || e.is_connect() || e.is_request()
        }
        _ => false,
    }
}

/// Compute the delay for the Nth retry attempt with optional override + cap.
pub fn retry_delay(
    attempt: u32,
    initial_ms: u64,
    max_delay_ms: Option<u64>,
    retry_after_secs: Option<u64>,
) -> Duration {
    let mut ms = if let Some(secs) = retry_after_secs {
        secs.saturating_mul(1000)
    } else {
        initial_ms.saturating_mul(2u64.saturating_pow(attempt))
    };
    if let Some(cap) = max_delay_ms {
        if ms > cap { ms = cap; }
    }
    Duration::from_millis(ms)
}

/// Format an HTTP error message with the status code prefix so the retry layer
/// can pattern-match on it (`HTTP <status>: <body>` shape).
pub fn format_http_error(status: u16, body: &str) -> String {
    format!("HTTP {}: {}", status, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_retry_5xx() {
        let err = LlmError::Http { status: 503, message: "Service Unavailable".into() };
        assert!(should_retry(&err));
    }

    #[test]
    fn test_should_retry_429() {
        let err = LlmError::Http { status: 429, message: "rate limited".into() };
        assert!(should_retry(&err));
        let err2 = LlmError::RateLimit { retry_after_secs: 5 };
        assert!(should_retry(&err2));
    }

    #[test]
    fn test_no_retry_4xx() {
        let err = LlmError::Http { status: 400, message: "bad request".into() };
        assert!(!should_retry(&err));
        let err2 = LlmError::AuthError;
        assert!(!should_retry(&err2));
    }

    #[test]
    fn test_retry_delay_exp() {
        assert_eq!(retry_delay(0, 500, None, None), Duration::from_millis(500));
        assert_eq!(retry_delay(1, 500, None, None), Duration::from_millis(1000));
        assert_eq!(retry_delay(2, 500, None, None), Duration::from_millis(2000));
    }

    #[test]
    fn test_retry_delay_cap() {
        assert_eq!(retry_delay(10, 500, Some(5000), None), Duration::from_millis(5000));
    }

    #[test]
    fn test_retry_delay_after() {
        assert_eq!(retry_delay(0, 500, None, Some(7)), Duration::from_millis(7000));
    }

    #[test]
    fn test_format_http_error() {
        assert_eq!(format_http_error(503, "down"), "HTTP 503: down");
    }
}
