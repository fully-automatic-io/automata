use llm_client::retry::{format_http_error, retry_delay, should_retry};
use llm_client::provider::LlmError;
use std::time::Duration;

#[test]
fn test_http_503_triggers_retry() {
    let err = LlmError::Http { status: 503, message: format_http_error(503, "Service Unavailable") };
    assert!(should_retry(&err));
    assert!(err.to_string().starts_with("HTTP error 503: HTTP 503:"));
}

#[test]
fn test_http_429_triggers_retry() {
    let err = LlmError::Http { status: 429, message: format_http_error(429, "Too Many Requests") };
    assert!(should_retry(&err));
}

#[test]
fn test_http_400_does_not_retry() {
    let err = LlmError::Http { status: 400, message: format_http_error(400, "Bad Request") };
    assert!(!should_retry(&err));
}

#[test]
fn test_http_401_does_not_retry() {
    let err = LlmError::AuthError;
    assert!(!should_retry(&err));
}

#[test]
fn test_rate_limit_triggers_retry() {
    let err = LlmError::RateLimit { retry_after_secs: 5 };
    assert!(should_retry(&err));
}

#[test]
fn test_retry_delay_exponential_backoff() {
    assert_eq!(retry_delay(0, 500, None, None), Duration::from_millis(500));
    assert_eq!(retry_delay(1, 500, None, None), Duration::from_millis(1000));
    assert_eq!(retry_delay(2, 500, None, None), Duration::from_millis(2000));
}

#[test]
fn test_retry_delay_capped() {
    let capped = retry_delay(10, 500, Some(5000), None);
    assert_eq!(capped, Duration::from_millis(5000));
}

#[test]
fn test_retry_delay_retry_after_header() {
    let delay = retry_delay(0, 500, None, Some(7));
    assert_eq!(delay, Duration::from_millis(7000));
}

#[test]
fn test_format_http_error_prefix() {
    let msg = format_http_error(503, "down");
    assert_eq!(msg, "HTTP 503: down");
}

#[test]
fn test_http_error_message_starts_with_status() {
    // Verify the error message format that providers emit (52e13870 requirement).
    let msg = format_http_error(503, "Service Unavailable");
    assert!(msg.starts_with("HTTP 503:"));
}
