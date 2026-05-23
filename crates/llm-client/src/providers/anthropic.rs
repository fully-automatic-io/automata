
use crate::provider::{AuthMethod, LlmError, LlmProvider, LlmStream, ProviderConfig};
use crate::streaming::parse_sse_event;
use crate::types::{
    ContentPart, LlmMessage, LlmRequest, LlmResponse, MessageContent, StopReason,
    ThinkingBudgets, ThinkingLevel, Usage,
};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::time::Duration;

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";

/// Resolve adaptive vs budget thinking. Explicit `force` overrides; otherwise
/// fall back to substring matching on the model id.
fn resolve_adaptive_thinking(model_id: &str, force: Option<bool>) -> bool {
    if let Some(b) = force { return b; }
    let m = model_id;
    m.contains("opus-4-6") || m.contains("opus-4.6")
        || m.contains("opus-4-7") || m.contains("opus-4.7")
        || m.contains("sonnet-4-6") || m.contains("sonnet-4.6")
}

fn map_thinking_level_to_anthropic_effort(level: ThinkingLevel) -> &'static str {
    match level {
        ThinkingLevel::Off => "low",      // never reached (filtered by caller)
        ThinkingLevel::Minimal => "low",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::XHigh => "xhigh",
    }
}

fn pick_thinking_budget(level: ThinkingLevel, budgets: Option<&ThinkingBudgets>) -> Option<u32> {
    let b = budgets?;
    match level {
        ThinkingLevel::Minimal => b.minimal,
        ThinkingLevel::Low => b.low,
        ThinkingLevel::Medium => b.medium,
        ThinkingLevel::High => b.high,
        ThinkingLevel::XHigh => b.high,  // no separate xhigh slot; high suffices
        ThinkingLevel::Off => None,
    }
}

fn needs_interleaved_thinking_beta(body: &serde_json::Value) -> bool {
    body.get("thinking")
        .and_then(|t| t.get("type"))
        .and_then(|t| t.as_str())
        == Some("enabled")
}

pub struct AnthropicProvider {
    config: ProviderConfig,
    client: Client,
}

impl AnthropicProvider {
    pub fn new(config: ProviderConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to build HTTP client");
        Self { config, client }
    }

    fn base_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or(ANTHROPIC_API_URL)
    }

    fn build_body(&self, request: &LlmRequest, stream: bool) -> serde_json::Value {
        let messages: Vec<serde_json::Value> = request.messages.iter()
            .filter_map(convert_message)
            .collect();

        let tools: Vec<serde_json::Value> = request.tools.iter()
            .map(|t| json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }))
            .collect();

        let mut body = json!({
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_tokens.unwrap_or(8192),
            "stream": stream,
        });

        if let Some(system) = &request.system {
            body["system"] = json!(system);
        }
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        if let Some(temp) = request.temperature {
            body["temperature"] = json!(temp);
        }
        if !request.stop_sequences.is_empty() {
            body["stop_sequences"] = json!(request.stop_sequences);
        }

        // Resolve thinking config:
        //   - Adaptive (Opus 4.6+, Sonnet 4.6, or forced via provider_options) →
        //     { type: "adaptive", display: "summarized" } + output_config.effort
        //   - Budget (older Claude 4) → { type: "enabled", budget_tokens: N }
        //   - None → no thinking field
        if let Some(level) = request.reasoning_effort {
            if level != ThinkingLevel::Off {
                let force = request.anthropic_options()
                    .and_then(|o| o.force_adaptive_thinking);
                if resolve_adaptive_thinking(&request.model, force) {
                    body["thinking"] = json!({ "type": "adaptive", "display": "summarized" });
                    body["output_config"] = json!({
                        "effort": map_thinking_level_to_anthropic_effort(level),
                    });
                } else if let Some(budget) = pick_thinking_budget(level, request.thinking_budgets.as_ref()) {
                    body["thinking"] = json!({ "type": "enabled", "budget_tokens": budget });
                }
            }
        }

        body
    }

    async fn send_with_retry(&self, body: serde_json::Value, _stream: bool) -> Result<reqwest::Response, LlmError> {
        use crate::retry::{format_http_error, retry_delay, should_retry};
        let mut last_err: Option<LlmError> = None;
        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let after = match &last_err {
                    Some(LlmError::RateLimit { retry_after_secs }) => Some(*retry_after_secs),
                    _ => None,
                };
                tokio::time::sleep(retry_delay(attempt - 1, 500, Some(30_000), after)).await;
            }
            let req = self.client
                .post(self.base_url())
                .header("anthropic-version", ANTHROPIC_VERSION)
                .header("content-type", "application/json");
            let req = match self.config.auth_method {
                AuthMethod::ApiKeyHeader => req.header("x-api-key", &self.config.api_key),
                AuthMethod::Bearer => req.bearer_auth(&self.config.api_key),
            };
            // Older Claude 4 models (budget thinking) need the interleaved-thinking
            // beta header to interleave thinking blocks with tool calls; adaptive
            // models have it built in.
            let req = if needs_interleaved_thinking_beta(&body) {
                req.header("anthropic-beta", INTERLEAVED_THINKING_BETA)
            } else {
                req
            };
            let resp = req.json(&body).send().await;

            match resp {
                Ok(r) if r.status().is_success() => return Ok(r),
                Ok(r) => {
                    let status = r.status().as_u16();
                    if status == 429 {
                        let secs = r.headers().get("retry-after")
                            .and_then(|v| v.to_str().ok())
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(5);
                        last_err = Some(LlmError::RateLimit { retry_after_secs: secs });
                        continue;
                    }
                    if status == 401 { return Err(LlmError::AuthError); }
                    let body = r.text().await.unwrap_or_default();
                    // Prefix status onto message body so the retry layer (and callers) can match (52e13870).
                    let err = LlmError::Http { status, message: format_http_error(status, &body) };
                    if should_retry(&err) {
                        last_err = Some(err);
                        continue;
                    }
                    return Err(err);
                }
                Err(e) => {
                    let err = LlmError::Reqwest(e);
                    if should_retry(&err) {
                        last_err = Some(err);
                        continue;
                    }
                    return Err(err);
                }
            }
        }
        Err(last_err.unwrap_or(LlmError::Other("retry exhausted".into())))
    }
}

fn convert_message(msg: &LlmMessage) -> Option<serde_json::Value> {
    match msg {
        LlmMessage::User { content, .. } => Some(json!({
            "role": "user",
            "content": content_blocks_to_json(&to_blocks(content)),
        })),
        LlmMessage::Assistant { content, .. } => Some(json!({
            "role": "assistant",
            "content": content.iter().map(convert_content_part).collect::<Vec<_>>(),
        })),
        LlmMessage::ToolResult { tool_call_id, content, is_error, .. } => Some(json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": content.iter()
                    .filter_map(|p| if let ContentPart::Text { text } = p { Some(text.as_str()) } else { None })
                    .collect::<Vec<_>>()
                    .join(""),
                "is_error": is_error,
            }],
        })),
        // Custom roles (Custom / BashExecution / BranchSummary / CompactionSummary)
        // are collapsed to user content via `convert_to_llm` before reaching
        // the provider; treat any unexpected role as a no-op for safety.
        _ => None,
    }
}

fn to_blocks(content: &MessageContent) -> Vec<ContentPart> {
    match content {
        MessageContent::String(s) => vec![ContentPart::Text { text: s.clone() }],
        MessageContent::Blocks(b) => b.clone(),
    }
}

fn content_blocks_to_json(blocks: &[ContentPart]) -> Vec<serde_json::Value> {
    blocks.iter().map(convert_content_part).collect()
}

fn convert_content_part(part: &ContentPart) -> serde_json::Value {
    match part {
        ContentPart::Text { text } => json!({"type": "text", "text": text}),
        ContentPart::Image { data, mime_type } => json!({
            "type": "image",
            "source": {"type": "base64", "media_type": mime_type, "data": data},
        }),
        ContentPart::ToolCall { id, name, arguments } => json!({
            "type": "tool_use", "id": id, "name": name, "input": arguments,
        }),
        ContentPart::ToolResult { tool_call_id, content, is_error, .. } => json!({
            "type": "tool_result", "tool_use_id": tool_call_id, "content": content, "is_error": is_error,
        }),
        ContentPart::Thinking { thinking } => json!({"type": "thinking", "thinking": thinking}),
    }
}

fn parse_stop_reason(s: &str) -> StopReason {
    match s {
        "end_turn" => StopReason::EndTurn,
        "max_tokens" => StopReason::MaxTokens,
        "stop_sequence" => StopReason::StopSequence,
        "tool_use" => StopReason::ToolUse,
        _ => StopReason::EndTurn,
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
        let body = self.build_body(&request, false);
        let resp = self.send_with_retry(body, false).await?;

        #[derive(Deserialize)]
        struct ApiResponse {
            id: String,
            model: String,
            content: Vec<serde_json::Value>,
            stop_reason: String,
            usage: ApiUsage,
        }
        #[derive(Deserialize)]
        struct ApiUsage {
            input_tokens: u64,
            output_tokens: u64,
            #[serde(default)]
            cache_creation_input_tokens: u64,
            #[serde(default)]
            cache_read_input_tokens: u64,
        }

        let api_resp: ApiResponse = resp.json().await
            .map_err(|e| LlmError::SerializationError(e.to_string()))?;

        let content: Vec<ContentPart> = api_resp.content.into_iter()
            .filter_map(|v| {
                let t = v.get("type")?.as_str()?;
                match t {
                    "text" => Some(ContentPart::Text {
                        text: v.get("text")?.as_str()?.to_string(),
                    }),
                    "tool_use" => Some(ContentPart::ToolCall {
                        id: v.get("id")?.as_str()?.to_string(),
                        name: v.get("name")?.as_str()?.to_string(),
                        arguments: v.get("input")?.clone(),
                    }),
                    _ => None,
                }
            })
            .collect();

        Ok(LlmResponse {
            id: api_resp.id,
            model: api_resp.model,
            content,
            stop_reason: parse_stop_reason(&api_resp.stop_reason),
            usage: Usage {
                input: api_resp.usage.input_tokens,
                output: api_resp.usage.output_tokens,
                cache_read: api_resp.usage.cache_read_input_tokens,
                cache_write: api_resp.usage.cache_creation_input_tokens,
                total_tokens: api_resp.usage.input_tokens + api_resp.usage.output_tokens,
                cost: Default::default(),
            },
        })
    }

    async fn stream(&self, request: LlmRequest) -> Result<LlmStream, LlmError> {
        let body = self.build_body(&request, true);
        let resp = self.send_with_retry(body, true).await?;

        let byte_stream = resp.bytes_stream();
        let stream = byte_stream.eventsource().filter_map(|result| async move {
            match result {
                Ok(event) => {
                    let parsed = parse_sse_event(&event.event, &event.data);
                    parsed.map(Ok)
                }
                Err(e) => Some(Err(LlmError::StreamError(e.to_string()))),
            }
        });

        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AnthropicOptions, ProviderOptions};

    fn provider() -> AnthropicProvider {
        AnthropicProvider::new(ProviderConfig::new("test"))
    }

    fn req(model: &str) -> LlmRequest {
        LlmRequest { model: model.into(), ..Default::default() }
    }

    #[test]
    fn adaptive_thinking_emitted_for_opus_4_7() {
        let mut r = req("claude-opus-4-7");
        r.reasoning_effort = Some(ThinkingLevel::High);
        let body = provider().build_body(&r, false);
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["thinking"]["display"], "summarized");
        assert_eq!(body["output_config"]["effort"], "high");
        assert!(!needs_interleaved_thinking_beta(&body));
    }

    #[test]
    fn budget_thinking_for_older_claude() {
        let mut r = req("claude-sonnet-4-5");
        r.reasoning_effort = Some(ThinkingLevel::High);
        r.thinking_budgets = Some(ThinkingBudgets {
            high: Some(8192), ..Default::default()
        });
        let body = provider().build_body(&r, false);
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 8192);
        assert!(body.get("output_config").is_none());
        assert!(needs_interleaved_thinking_beta(&body));
    }

    #[test]
    fn thinking_off_emits_no_field() {
        let body = provider().build_body(&req("claude-opus-4-7"), false);
        assert!(body.get("thinking").is_none());

        let mut r = req("claude-opus-4-7");
        r.reasoning_effort = Some(ThinkingLevel::Off);
        let body = provider().build_body(&r, false);
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn force_adaptive_thinking_overrides_substring() {
        // Unknown alias gets adaptive when explicitly forced.
        let mut r = req("bedrock-claude-something");
        r.reasoning_effort = Some(ThinkingLevel::Medium);
        r.provider_options = Some(ProviderOptions::Anthropic(AnthropicOptions {
            force_adaptive_thinking: Some(true),
        }));
        let body = provider().build_body(&r, false);
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["output_config"]["effort"], "medium");
    }

    #[test]
    fn force_adaptive_thinking_false_overrides_substring() {
        // Adaptive-id model forced into budget mode.
        let mut r = req("claude-opus-4-7");
        r.reasoning_effort = Some(ThinkingLevel::High);
        r.thinking_budgets = Some(ThinkingBudgets {
            high: Some(4096), ..Default::default()
        });
        r.provider_options = Some(ProviderOptions::Anthropic(AnthropicOptions {
            force_adaptive_thinking: Some(false),
        }));
        let body = provider().build_body(&r, false);
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 4096);
    }
}
