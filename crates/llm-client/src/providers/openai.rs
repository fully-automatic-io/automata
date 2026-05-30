
use crate::provider::{LlmError, LlmProvider, LlmStream, ProviderConfig};
use crate::streaming::{Delta, LlmEvent};
use crate::types::{
    CacheRetention, ContentPart, LlmMessage, LlmRequest, LlmResponse, MessageContent, StopReason,
    Usage,
};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::time::Duration;

const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

pub struct OpenAIProvider {
    config: ProviderConfig,
    client: Client,
}

impl OpenAIProvider {
    pub fn new(config: ProviderConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to build HTTP client");
        Self { config, client }
    }

    fn base_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or(OPENAI_API_URL)
    }

    fn build_body(&self, request: &LlmRequest, stream: bool) -> serde_json::Value {
        let messages: Vec<serde_json::Value> = request.messages.iter()
            .filter_map(convert_message)
            .collect();

        let tools: Vec<serde_json::Value> = request.tools.iter()
            .map(|t| json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                }
            }))
            .collect();

        let mut body = json!({
            "model": request.model,
            "messages": messages,
            "stream": stream,
        });

        if let Some(max_tokens) = request.max_tokens {
            // GPT-5 / o-series use `max_completion_tokens`; legacy chat uses `max_tokens`.
            // We forward both to keep older endpoints happy.
            body["max_tokens"] = json!(max_tokens);
            body["max_completion_tokens"] = json!(max_tokens);
        }
        if let Some(temp) = request.temperature {
            body["temperature"] = json!(temp);
        }
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        if !request.stop_sequences.is_empty() {
            body["stop"] = json!(request.stop_sequences);
        }
        // Stream usage chunks in SSE responses.
        if stream {
            body["stream_options"] = json!({"include_usage": true});
        }

        // `reasoning_effort` is forwarded as a typed field on LlmRequest (set by callers
        // based on the user's thinking level). The retry/option layer ensures it's a valid level.
        if let Some(eff) = &request.reasoning_effort {
            body["reasoning_effort"] = json!(eff);
        }

        // Prompt-cache control:
        //   - native OpenAI: top-level `prompt_cache_retention: "24h"` for long cache
        //   - Anthropic-compat aliases (set via OpenaiOptions): inject inline
        //     `cache_control: {type: "ephemeral", ttl?: "1h"}` on system / last
        //     tool / last user message
        if let Some(retention) = request.cache_retention
            && retention != CacheRetention::None {
                let use_anthropic_style = request
                    .openai_options()
                    .and_then(|o| o.anthropic_cache_control)
                    .unwrap_or(false);
                if use_anthropic_style {
                    let cache_control = match retention {
                        CacheRetention::Long => json!({"type": "ephemeral", "ttl": "1h"}),
                        CacheRetention::Short => json!({"type": "ephemeral"}),
                        CacheRetention::None => json!(null),
                    };
                    apply_anthropic_cache_control(&mut body, &cache_control);
                } else if retention == CacheRetention::Long {
                    body["prompt_cache_retention"] = json!("24h");
                }
            }

        body
    }

    async fn send_with_retry(&self, body: serde_json::Value) -> Result<reqwest::Response, LlmError> {
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
            let resp = self.client
                .post(self.base_url())
                .bearer_auth(&self.config.api_key)
                .json(&body)
                .send().await;

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

/// Apply Anthropic-style `cache_control` hints to an OpenAI Chat Completions
/// request body. Inserts on the system message (text content), the last tool
/// definition, and the last user / assistant turn.
fn apply_anthropic_cache_control(body: &mut serde_json::Value, cache_control: &serde_json::Value) {
    // 1. System (or developer) message — promote string content to an array
    //    of `{type: "text", text, cache_control}` parts.
    if let Some(messages) = body.get_mut("messages").and_then(|m| m.as_array_mut()) {
        for msg in messages.iter_mut() {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
            if role == "system" || role == "developer" {
                attach_cache_to_text_content(msg, cache_control);
                break;
            }
        }
        // 2. Last user/assistant turn.
        for msg in messages.iter_mut().rev() {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
            if role == "user" || role == "assistant" {
                attach_cache_to_text_content(msg, cache_control);
                break;
            }
        }
    }
    // 3. Last tool definition.
    if let Some(tools) = body.get_mut("tools").and_then(|t| t.as_array_mut())
        && let Some(last) = tools.last_mut()
            && let Some(obj) = last.as_object_mut() {
                obj.insert("cache_control".into(), cache_control.clone());
            }
}

/// Attach `cache_control` to the last text part of a message, promoting flat
/// string content to a single-element array if needed.
fn attach_cache_to_text_content(msg: &mut serde_json::Value, cache_control: &serde_json::Value) {
    let content = match msg.get_mut("content") {
        Some(c) => c,
        None => return,
    };
    if let Some(s) = content.as_str() {
        let s = s.to_string();
        *content = json!([{ "type": "text", "text": s, "cache_control": cache_control }]);
        return;
    }
    if let Some(arr) = content.as_array_mut() {
        // Find last text-like part; fallback to last part.
        let last_text_idx = arr.iter().rposition(|p| {
            matches!(p.get("type").and_then(|t| t.as_str()), Some("text") | Some("input_text"))
        });
        let idx = last_text_idx.or_else(|| arr.len().checked_sub(1));
        if let Some(i) = idx
            && let Some(obj) = arr[i].as_object_mut() {
                obj.insert("cache_control".into(), cache_control.clone());
            }
    }
}

fn convert_message(msg: &LlmMessage) -> Option<serde_json::Value> {
    match msg {
        LlmMessage::User { content, .. } => {
            let text = extract_text(content);
            Some(json!({"role": "user", "content": text}))
        }
        LlmMessage::Assistant { content, .. } => {
            // Reasoning replay normalization: Chat Completions has no first-class
            // place for prior `thinking` blocks. We drop them on replay. If the
            // remaining content is empty (model produced thinking-only), skip the
            // message entirely so the endpoint doesn't reject it.
            let text: String = content.iter()
                .filter_map(|p| if let ContentPart::Text { text } = p { Some(text.as_str()) } else { None })
                .collect::<Vec<_>>()
                .join("");
            let tool_calls: Vec<serde_json::Value> = content.iter()
                .filter_map(|p| {
                    if let ContentPart::ToolCall { id, name, arguments } = p {
                        Some(json!({
                            "id": id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments.to_string()}
                        }))
                    } else { None }
                })
                .collect();
            if text.is_empty() && tool_calls.is_empty() {
                return None;
            }
            if tool_calls.is_empty() {
                Some(json!({"role": "assistant", "content": text}))
            } else {
                Some(json!({"role": "assistant", "content": text, "tool_calls": tool_calls}))
            }
        }
        LlmMessage::ToolResult { tool_call_id, content, is_error: _, .. } => {
            let text = content.iter()
                .filter_map(|p| if let ContentPart::Text { text } = p { Some(text.as_str()) } else { None })
                .collect::<Vec<_>>()
                .join("");
            Some(json!({"role": "tool", "tool_call_id": tool_call_id, "content": text}))
        }
        // Custom roles are collapsed before reaching the provider.
        _ => None,
    }
}

fn extract_text(content: &MessageContent) -> String {
    match content {
        MessageContent::String(s) => s.clone(),
        MessageContent::Blocks(blocks) => blocks.iter()
            .filter_map(|p| if let ContentPart::Text { text } = p { Some(text.as_str()) } else { None })
            .collect::<Vec<_>>()
            .join(""),
    }
}

#[async_trait]
impl LlmProvider for OpenAIProvider {
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
        let body = self.build_body(&request, false);
        let resp = self.send_with_retry(body).await?;

        #[derive(Deserialize)]
        struct ApiResponse {
            id: String,
            model: String,
            choices: Vec<Choice>,
            usage: ApiUsage,
        }
        #[derive(Deserialize)]
        struct Choice {
            message: Message,
            finish_reason: String,
        }
        #[derive(Deserialize)]
        struct Message {
            content: Option<String>,
            tool_calls: Option<Vec<ToolCall>>,
        }
        #[derive(Deserialize)]
        struct ToolCall {
            id: String,
            function: FunctionCall,
        }
        #[derive(Deserialize)]
        struct FunctionCall {
            name: String,
            arguments: String,
        }
        #[derive(Deserialize)]
        struct ApiUsage {
            prompt_tokens: u64,
            completion_tokens: u64,
        }

        let api_resp: ApiResponse = resp.json().await
            .map_err(|e| LlmError::SerializationError(e.to_string()))?;

        let choice = api_resp.choices.into_iter().next()
            .ok_or_else(|| LlmError::InvalidRequest("No choices".to_string()))?;

        let mut content = vec![];
        if let Some(text) = choice.message.content
            && !text.is_empty() {
                content.push(ContentPart::Text { text });
            }
        if let Some(tool_calls) = choice.message.tool_calls {
            for tc in tool_calls {
                let input = serde_json::from_str(&tc.function.arguments)
                    .unwrap_or(serde_json::Value::Null);
                content.push(ContentPart::ToolCall {
                    id: tc.id,
                    name: tc.function.name,
                    arguments: input,
                });
            }
        }

        let stop_reason = match choice.finish_reason.as_str() {
            "stop" => StopReason::EndTurn,
            "length" => StopReason::MaxTokens,
            "tool_calls" => StopReason::ToolUse,
            "content_filter" => StopReason::ContentFilter,
            _ => StopReason::EndTurn,
        };

        Ok(LlmResponse {
            id: api_resp.id,
            model: api_resp.model,
            content,
            stop_reason,
            usage: Usage {
                input: api_resp.usage.prompt_tokens,
                output: api_resp.usage.completion_tokens,
                total_tokens: api_resp.usage.prompt_tokens + api_resp.usage.completion_tokens,
                ..Default::default()
            },
        })
    }

    async fn stream(&self, request: LlmRequest) -> Result<LlmStream, LlmError> {
        let body = self.build_body(&request, true);
        let resp = self.send_with_retry(body).await?;

        let byte_stream = resp.bytes_stream();
        let stream = byte_stream.eventsource().filter_map(|result| async move {
            match result {
                Ok(event) => {
                    if event.data == "[DONE]" {
                        return Some(Ok(LlmEvent::MessageStop));
                    }
                    let v: serde_json::Value = serde_json::from_str(&event.data).ok()?;
                    let delta = v.get("choices")?.get(0)?.get("delta")?;
                    delta.get("content").and_then(|c| c.as_str()).map(|text| {
                        Ok(LlmEvent::ContentBlockDelta {
                            index: 0,
                            delta: Delta::TextDelta { text: text.to_string() },
                        })
                    })
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
    use crate::types::{AnthropicOptions, OpenaiOptions, ProviderOptions};

    fn provider() -> OpenAIProvider {
        OpenAIProvider::new(ProviderConfig::new("test"))
    }

    fn req(model: &str) -> LlmRequest {
        LlmRequest {
            model: model.into(),
            messages: vec![LlmMessage::user_text("hi")],
            ..Default::default()
        }
    }

    #[test]
    fn cache_long_native_emits_prompt_cache_retention() {
        let mut r = req("gpt-5");
        r.cache_retention = Some(CacheRetention::Long);
        let body = provider().build_body(&r, false);
        assert_eq!(body["prompt_cache_retention"], "24h");
        assert!(body.get("messages").and_then(|m| m.as_array())
            .and_then(|a| a.first())
            .and_then(|m| m.get("content"))
            .map(|c| c.is_string())
            .unwrap_or(false), "native cache must not promote content shape");
    }

    #[test]
    fn cache_short_native_no_emit() {
        // Short-cache is OpenAI's default behavior; no top-level field needed.
        let mut r = req("gpt-5");
        r.cache_retention = Some(CacheRetention::Short);
        let body = provider().build_body(&r, false);
        assert!(body.get("prompt_cache_retention").is_none());
    }

    #[test]
    fn cache_none_emits_nothing() {
        let mut r = req("gpt-5");
        r.cache_retention = Some(CacheRetention::None);
        let body = provider().build_body(&r, false);
        assert!(body.get("prompt_cache_retention").is_none());
    }

    #[test]
    fn cache_long_anthropic_style_inline_control() {
        let mut r = req("deepseek-chat");
        r.cache_retention = Some(CacheRetention::Long);
        r.provider_options = Some(ProviderOptions::Openai(OpenaiOptions {
            anthropic_cache_control: Some(true),
        }));
        let body = provider().build_body(&r, false);
        assert!(body.get("prompt_cache_retention").is_none());
        // Last user message should now be an array with a `cache_control` part.
        let messages = body["messages"].as_array().unwrap();
        let last = messages.last().unwrap();
        let parts = last["content"].as_array().expect("content promoted to array");
        assert_eq!(parts[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(parts[0]["cache_control"]["ttl"], "1h");
    }

    #[test]
    fn cache_short_anthropic_style_no_ttl() {
        let mut r = req("deepseek-chat");
        r.cache_retention = Some(CacheRetention::Short);
        r.provider_options = Some(ProviderOptions::Openai(OpenaiOptions {
            anthropic_cache_control: Some(true),
        }));
        let body = provider().build_body(&r, false);
        let messages = body["messages"].as_array().unwrap();
        let last = messages.last().unwrap();
        let parts = last["content"].as_array().unwrap();
        assert_eq!(parts[0]["cache_control"]["type"], "ephemeral");
        assert!(parts[0]["cache_control"].get("ttl").is_none());
    }

    #[test]
    fn cache_anthropic_style_attaches_to_last_tool() {
        let mut r = req("deepseek-chat");
        r.cache_retention = Some(CacheRetention::Long);
        r.tools = vec![crate::types::ToolDefinition {
            name: "first".into(),
            description: "f".into(),
            input_schema: serde_json::json!({}),
        }, crate::types::ToolDefinition {
            name: "second".into(),
            description: "s".into(),
            input_schema: serde_json::json!({}),
        }];
        r.provider_options = Some(ProviderOptions::Openai(OpenaiOptions {
            anthropic_cache_control: Some(true),
        }));
        let body = provider().build_body(&r, false);
        let tools = body["tools"].as_array().unwrap();
        assert!(tools[0].get("cache_control").is_none(), "first tool unchanged");
        assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn anthropic_options_on_openai_request_ignored() {
        // Type system can't prevent caller mismatching; runtime simply ignores.
        let mut r = req("gpt-5");
        r.cache_retention = Some(CacheRetention::Long);
        r.provider_options = Some(ProviderOptions::Anthropic(AnthropicOptions {
            force_adaptive_thinking: Some(true),
        }));
        let body = provider().build_body(&r, false);
        // Falls back to native cache (anthropic_cache_control is None).
        assert_eq!(body["prompt_cache_retention"], "24h");
    }
}
