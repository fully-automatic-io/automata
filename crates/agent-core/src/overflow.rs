//! Context-overflow error detection.
//!
//! Different LLM providers signal "input exceeds context window" in different
//! ways. This module centralises pattern matching against provider error
//! messages plus silent-overflow detection (provider accepts the request but
//! `usage.input` exceeds the model's context window) so callers can trigger
//! auto-compaction or surface a friendly error.

use crate::types::{AgentMessage, StopReason};
use once_cell::sync::Lazy;
use regex::RegexSet;

// ============================================================================
// Pattern sets
// ============================================================================

/// Error-message patterns indicating a context-overflow condition. Order
/// doesn't matter — `RegexSet::is_match` returns true if any pattern hits.
///
/// Sample messages per provider:
///
/// - Anthropic: `prompt is too long: 213462 tokens > 200000 maximum`
/// - Anthropic 413: `request_too_large`
/// - OpenAI: `Your input exceeds the context window of this model`
/// - OpenAI/LiteLLM: `Requested token count exceeds the model's maximum context length of 131072 tokens`
/// - Google: `The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)`
/// - xAI: `This model's maximum prompt length is 131072 but the request contains 537812 tokens`
/// - Groq: `Please reduce the length of the messages or completion`
/// - OpenRouter: `This endpoint's maximum context length is X tokens`
/// - OpenRouter/Poolside: `Input length X exceeds the maximum allowed input length of Y tokens`
/// - Together AI: `The input (X tokens) is longer than the model's context length (Y tokens)`
/// - llama.cpp: `the request exceeds the available context size, try increasing it`
/// - LM Studio: `tokens to keep from the initial prompt is greater than the context length`
/// - GitHub Copilot: `prompt token count of X exceeds the limit of Y`
/// - MiniMax: `invalid params, context window exceeds limit`
/// - Kimi For Coding: `Your request exceeded model token limit: X (requested: Y)`
/// - Mistral: `Prompt contains X tokens ... too large for model with Y maximum context length`
/// - Cerebras: `400/413 status code (no body)`
/// - Ollama: `prompt too long; exceeded max context length by X tokens`
const OVERFLOW_PATTERNS: &[&str] = &[
    r"(?i)prompt is too long",
    r"(?i)request_too_large",
    r"(?i)input is too long for requested model",
    r"(?i)exceeds the context window",
    r"(?i)exceeds (?:the )?(?:model'?s )?maximum context length of [\d,]+ tokens?",
    r"(?i)input token count.*exceeds the maximum",
    r"(?i)maximum prompt length is \d+",
    r"(?i)reduce the length of the messages",
    r"(?i)maximum context length is \d+ tokens",
    r"(?i)exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?",
    r"(?i)input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
    r"(?i)exceeds the limit of \d+",
    r"(?i)exceeds the available context size",
    r"(?i)greater than the context length",
    r"(?i)context window exceeds limit",
    r"(?i)exceeded model token limit",
    r"(?i)too large for model with \d+ maximum context length",
    r"(?i)model_context_window_exceeded",
    r"(?i)prompt too long; exceeded (?:max )?context length",
    r"(?i)context[_ ]length[_ ]exceeded",
    r"(?i)too many tokens",
    r"(?i)token limit exceeded",
    r"(?i)^4(?:00|13)\s*(?:status code)?\s*\(no body\)",
];

/// Patterns that should override an overflow match — e.g. throttling errors
/// that happen to mention "too many tokens". Checked before `OVERFLOW_PATTERNS`.
const NON_OVERFLOW_PATTERNS: &[&str] = &[
    r"(?i)^(Throttling error|Service unavailable):",
    r"(?i)rate limit",
    r"(?i)too many requests",
];

static OVERFLOW_SET: Lazy<RegexSet> =
    Lazy::new(|| RegexSet::new(OVERFLOW_PATTERNS).expect("overflow patterns must compile"));
static NON_OVERFLOW_SET: Lazy<RegexSet> =
    Lazy::new(|| RegexSet::new(NON_OVERFLOW_PATTERNS).expect("non-overflow patterns must compile"));

// ============================================================================
// Public API
// ============================================================================

/// Check whether an assistant message represents a context-overflow condition.
///
/// Three cases are recognised:
///
/// 1. **Error-based overflow** — `stop_reason == Error` and `error_message`
///    matches one of [`OVERFLOW_PATTERNS`] (and *not* a non-overflow pattern).
/// 2. **Silent overflow** (z.ai-style) — `stop_reason == EndTurn` but
///    `usage.input + usage.cache_read > context_window`. Requires
///    `context_window` to be supplied.
/// 3. **Length-stop overflow** (Xiaomi MiMo-style) — `stop_reason == MaxTokens`
///    with `output == 0` and input filling ≥99% of `context_window`. The
///    server truncated the oversized input and left no room to generate.
///
/// Returns `false` for non-assistant messages.
pub fn is_context_overflow(message: &AgentMessage, context_window: Option<u64>) -> bool {
    let AgentMessage::Assistant {
        stop_reason,
        error_message,
        usage,
        ..
    } = message
    else {
        return false;
    };

    // Case 1: error-message pattern.
    if *stop_reason == StopReason::Error {
        if let Some(msg) = error_message.as_deref() {
            if !NON_OVERFLOW_SET.is_match(msg) && OVERFLOW_SET.is_match(msg) {
                return true;
            }
        }
    }

    let Some(window) = context_window else { return false };

    // Case 2: silent overflow.
    if *stop_reason == StopReason::EndTurn {
        let input_tokens = usage.input + usage.cache_read;
        if input_tokens > window {
            return true;
        }
    }

    // Case 3: length-stop overflow (truncated input + zero output + filled context).
    if *stop_reason == StopReason::MaxTokens && usage.output == 0 {
        let input_tokens = usage.input + usage.cache_read;
        // Match pi-mono's 99% threshold so a small token-counting drift doesn't miss it.
        if (input_tokens as f64) >= (window as f64) * 0.99 {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AgentMessage, Api, StopReason, Usage};

    fn err_assistant(msg: &str) -> AgentMessage {
        AgentMessage::Assistant {
            content: vec![],
            api: Api::Anthropic,
            provider: "test".into(),
            model: "test".into(),
            usage: Usage::default(),
            stop_reason: StopReason::Error,
            error_message: Some(msg.into()),
            timestamp: 0,
        }
    }

    fn ok_with_usage(input: u64, output: u64, cache_read: u64) -> AgentMessage {
        AgentMessage::Assistant {
            content: vec![],
            api: Api::Anthropic,
            provider: "test".into(),
            model: "test".into(),
            usage: Usage { input, output, cache_read, ..Default::default() },
            stop_reason: StopReason::EndTurn,
            error_message: None,
            timestamp: 0,
        }
    }

    fn length_with_usage(input: u64, output: u64, cache_read: u64) -> AgentMessage {
        AgentMessage::Assistant {
            content: vec![],
            api: Api::Anthropic,
            provider: "test".into(),
            model: "test".into(),
            usage: Usage { input, output, cache_read, ..Default::default() },
            stop_reason: StopReason::MaxTokens,
            error_message: None,
            timestamp: 0,
        }
    }

    // ─── error-message detection ───

    #[test]
    fn detects_anthropic_overflow() {
        assert!(is_context_overflow(
            &err_assistant("prompt is too long: 213462 tokens > 200000 maximum"),
            None,
        ));
    }

    #[test]
    fn detects_anthropic_request_too_large() {
        assert!(is_context_overflow(
            &err_assistant("413 {\"error\":{\"type\":\"request_too_large\",\"message\":\"Request exceeds the maximum size\"}}"),
            None,
        ));
    }

    #[test]
    fn detects_openai_context_window() {
        assert!(is_context_overflow(
            &err_assistant("Your input exceeds the context window of this model"),
            None,
        ));
    }

    #[test]
    fn detects_openai_compat_max_context_length() {
        assert!(is_context_overflow(
            &err_assistant("Requested token count exceeds the model's maximum context length of 131072 tokens"),
            None,
        ));
    }

    #[test]
    fn detects_google_overflow() {
        assert!(is_context_overflow(
            &err_assistant("The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)"),
            None,
        ));
    }

    #[test]
    fn detects_xai_overflow() {
        assert!(is_context_overflow(
            &err_assistant("This model's maximum prompt length is 131072 but the request contains 537812 tokens"),
            None,
        ));
    }

    #[test]
    fn detects_openrouter_poolside() {
        assert!(is_context_overflow(
            &err_assistant("Input length 213462 exceeds the maximum allowed input length of 200000 tokens"),
            None,
        ));
    }

    #[test]
    fn detects_together_ai() {
        assert!(is_context_overflow(
            &err_assistant("The input (213462 tokens) is longer than the model's context length (200000 tokens)"),
            None,
        ));
    }

    #[test]
    fn detects_cerebras_no_body() {
        assert!(is_context_overflow(&err_assistant("413 status code (no body)"), None));
        assert!(is_context_overflow(&err_assistant("400 (no body)"), None));
    }

    // ─── non-overflow exclusions ───

    #[test]
    fn skips_throttling_too_many_tokens() {
        // Bedrock throttling error contains "too many tokens" but is not overflow.
        assert!(!is_context_overflow(
            &err_assistant("Throttling error: Too many tokens, please wait before trying again."),
            None,
        ));
    }

    #[test]
    fn skips_rate_limit() {
        assert!(!is_context_overflow(
            &err_assistant("rate limit exceeded, please retry"),
            None,
        ));
    }

    // ─── silent overflow (Case 2) ───

    #[test]
    fn detects_silent_overflow_zai_style() {
        // Successful response but usage.input > context_window.
        assert!(is_context_overflow(&ok_with_usage(200_001, 100, 0), Some(200_000)));
    }

    #[test]
    fn detects_silent_overflow_with_cache_read() {
        // Cache reads count toward input.
        assert!(is_context_overflow(&ok_with_usage(100_000, 100, 100_001), Some(200_000)));
    }

    #[test]
    fn under_window_is_not_overflow() {
        assert!(!is_context_overflow(&ok_with_usage(100, 100, 0), Some(200_000)));
    }

    #[test]
    fn silent_overflow_needs_context_window() {
        assert!(!is_context_overflow(&ok_with_usage(200_001, 100, 0), None));
    }

    // ─── length-stop overflow (Case 3) ───

    #[test]
    fn detects_length_stop_overflow_xiaomi_style() {
        // Server truncated input to fit, output=0, input filled the window.
        assert!(is_context_overflow(&length_with_usage(199_500, 0, 0), Some(200_000)));
    }

    #[test]
    fn length_stop_with_output_is_normal_max_tokens() {
        // Real max-tokens generation (output > 0) is not overflow.
        assert!(!is_context_overflow(&length_with_usage(199_500, 200, 0), Some(200_000)));
    }

    #[test]
    fn length_stop_below_99_percent_is_not_overflow() {
        // Input only at 90% of window — caller hit max_tokens normally.
        assert!(!is_context_overflow(&length_with_usage(180_000, 0, 0), Some(200_000)));
    }

    // ─── non-assistant messages ───

    #[test]
    fn user_message_is_not_overflow() {
        let m = AgentMessage::user_text("hi");
        assert!(!is_context_overflow(&m, Some(200_000)));
    }
}
