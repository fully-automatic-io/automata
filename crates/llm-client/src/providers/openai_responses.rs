//! OpenAI Responses API provider.
//!
//! The Responses API is OpenAI's newer endpoint (`/v1/responses`) used by
//! GPT-5 and Codex/Computer-Use models. Unlike Chat Completions it accepts a
//! flat `input` array of typed items, supports first-class `reasoning` items
//! with `summary`/`content`/`encrypted_content`, and uses `function_call` /
//! `function_call_output` instead of `tool_calls` / `tool` messages.

use crate::provider::{LlmError, LlmProvider, LlmStream, ProviderConfig};
use crate::streaming::LlmEvent;
use crate::types::{
    ContentPart, LlmMessage, LlmRequest, LlmResponse, MessageContent, StopReason, Usage,
};
use async_trait::async_trait;
use futures::stream;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::time::Duration;

const OPENAI_RESPONSES_URL: &str = "https://api.openai.com/v1/responses";

pub struct OpenAIResponsesProvider {
    config: ProviderConfig,
    client: Client,
}

impl OpenAIResponsesProvider {
    pub fn new(config: ProviderConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to build HTTP client");
        Self { config, client }
    }

    fn base_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or(OPENAI_RESPONSES_URL)
    }

    fn build_body(&self, request: &LlmRequest, stream: bool) -> serde_json::Value {
        let input: Vec<serde_json::Value> = request.messages.iter()
            .flat_map(convert_message)
            .collect();

        let tools: Vec<serde_json::Value> = request.tools.iter()
            .map(|t| json!({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            }))
            .collect();

        let mut body = json!({
            "model": request.model,
            "input": input,
            "stream": stream,
        });

        if let Some(sys) = &request.system {
            body["instructions"] = json!(sys);
        }
        if let Some(max) = request.max_tokens {
            body["max_output_tokens"] = json!(max);
        }
        if let Some(temp) = request.temperature {
            body["temperature"] = json!(temp);
        }
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        // Reasoning effort forwarded as `{reasoning: {effort: ...}}`.
        if let Some(eff) = &request.reasoning_effort {
            body["reasoning"] = json!({"effort": eff.as_str()});
        }
        if !request.openai_response_includes.is_empty() {
            body["include"] = json!(request.openai_response_includes);
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
                    let body_txt = r.text().await.unwrap_or_default();
                    let err = LlmError::Http { status, message: format_http_error(status, &body_txt) };
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

#[async_trait]
impl LlmProvider for OpenAIResponsesProvider {
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
        let body = self.build_body(&request, false);
        let resp = self.send_with_retry(body).await?;

        #[derive(Deserialize)]
        struct ApiResponse {
            id: String,
            model: String,
            output: Vec<serde_json::Value>,
            #[serde(default)]
            usage: Option<ApiUsage>,
            #[serde(default)]
            status: Option<String>,
        }
        #[derive(Deserialize)]
        struct ApiUsage {
            input_tokens: u64,
            output_tokens: u64,
            #[serde(default)]
            input_tokens_details: Option<TokensDetails>,
        }
        #[derive(Deserialize)]
        struct TokensDetails {
            #[serde(default)]
            cached_tokens: u64,
        }

        let api_resp: ApiResponse = resp.json().await
            .map_err(|e| LlmError::SerializationError(e.to_string()))?;

        let content: Vec<ContentPart> = api_resp.output.iter()
            .flat_map(parse_output_item)
            .collect();

        let stop_reason = match api_resp.status.as_deref() {
            Some("incomplete") => StopReason::MaxTokens,
            _ => {
                // If any output item is a function_call, signal ToolUse; else EndTurn.
                let has_tool = api_resp.output.iter().any(|v| {
                    v.get("type").and_then(|t| t.as_str()) == Some("function_call")
                });
                if has_tool { StopReason::ToolUse } else { StopReason::EndTurn }
            }
        };

        let usage = api_resp.usage.map(|u| {
            let cache_read = u.input_tokens_details.as_ref().map(|d| d.cached_tokens).unwrap_or(0);
            Usage {
                input: u.input_tokens.saturating_sub(cache_read),
                output: u.output_tokens,
                cache_read,
                cache_write: 0,
                total_tokens: u.input_tokens + u.output_tokens,
                cost: Default::default(),
            }
        }).unwrap_or_default();

        Ok(LlmResponse {
            id: api_resp.id,
            model: api_resp.model,
            content,
            stop_reason,
            usage,
        })
    }

    async fn stream(&self, request: LlmRequest) -> Result<LlmStream, LlmError> {
        // Streaming will be added when the agent loop migrates to consume the
        // Responses event format. For now we expose a non-streaming wrapper so
        // callers can use complete() and still receive an LlmStream-compatible
        // single-shot terminal event.
        let resp = self.complete(request).await?;
        let events = vec![
            Ok(LlmEvent::MessageStart { id: resp.id.clone(), model: resp.model.clone() }),
            Ok(LlmEvent::MessageStop),
        ];
        Ok(Box::pin(stream::iter(events)))
    }
}

/// Convert one of our `LlmMessage`s into zero or more Responses-API input items.
fn convert_message(msg: &LlmMessage) -> Vec<serde_json::Value> {
    match msg {
        LlmMessage::User { content, .. } => {
            let content_arr: Vec<serde_json::Value> = to_blocks(content).iter()
                .map(convert_user_content_part)
                .collect();
            vec![json!({ "type": "message", "role": "user", "content": content_arr })]
        }
        LlmMessage::Assistant { content, .. } => {
            // Each ContentPart maps to its own input item.
            let mut out = vec![];
            let mut text_buf = String::new();
            for part in content {
                match part {
                    ContentPart::Text { text } => {
                        text_buf.push_str(text);
                    }
                    ContentPart::Thinking { thinking } => {
                        // Reasoning items are first-class in the Responses API.
                        out.push(json!({
                            "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": thinking}],
                        }));
                    }
                    ContentPart::ToolCall { id, name, arguments } => {
                        // Flush any pending text first.
                        if !text_buf.is_empty() {
                            out.push(json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": std::mem::take(&mut text_buf)}],
                            }));
                        }
                        out.push(json!({
                            "type": "function_call",
                            "call_id": id,
                            "name": name,
                            "arguments": arguments.to_string(),
                        }));
                    }
                    _ => {}
                }
            }
            if !text_buf.is_empty() {
                out.push(json!({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text_buf}],
                }));
            }
            out
        }
        LlmMessage::ToolResult { tool_call_id, content, .. } => {
            let text: String = content.iter()
                .filter_map(|p| if let ContentPart::Text { text } = p { Some(text.as_str()) } else { None })
                .collect::<Vec<_>>()
                .join("");
            vec![json!({
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": text,
            })]
        }
        // Custom roles are collapsed before reaching the provider.
        _ => vec![],
    }
}

fn convert_user_content_part(p: &ContentPart) -> serde_json::Value {
    match p {
        ContentPart::Text { text } => json!({"type": "input_text", "text": text}),
        ContentPart::Image { data, mime_type } => json!({
            "type": "input_image",
            "image_url": format!("data:{};base64,{}", mime_type, data),
        }),
        _ => json!({"type": "input_text", "text": ""}),
    }
}

fn to_blocks(content: &MessageContent) -> Vec<ContentPart> {
    match content {
        MessageContent::String(s) => vec![ContentPart::Text { text: s.clone() }],
        MessageContent::Blocks(b) => b.clone(),
    }
}

/// Parse a single output item from a Responses-API response.
fn parse_output_item(item: &serde_json::Value) -> Vec<ContentPart> {
    let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("");
    match item_type {
        "message" => {
            let mut out = vec![];
            if let Some(arr) = item.get("content").and_then(|c| c.as_array()) {
                for c in arr {
                    let ctype = c.get("type").and_then(|t| t.as_str()).unwrap_or("");
                    if ctype == "output_text" {
                        if let Some(text) = c.get("text").and_then(|t| t.as_str()) {
                            out.push(ContentPart::Text { text: text.to_string() });
                        }
                    }
                }
            }
            out
        }
        "reasoning" => {
            let mut out = vec![];
            if let Some(summary) = item.get("summary").and_then(|s| s.as_array()) {
                for s in summary {
                    if let Some(text) = s.get("text").and_then(|t| t.as_str()) {
                        out.push(ContentPart::Thinking { thinking: text.to_string() });
                    }
                }
            }
            out
        }
        "function_call" => {
            let id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or_default().to_string();
            let name = item.get("name").and_then(|v| v.as_str()).unwrap_or_default().to_string();
            let args_raw = item.get("arguments").and_then(|v| v.as_str()).unwrap_or("{}");
            let arguments: serde_json::Value = serde_json::from_str(args_raw).unwrap_or(serde_json::json!({}));
            vec![ContentPart::ToolCall { id, name, arguments }]
        }
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_message_conversion() {
        let msg = LlmMessage::user_text("hello");
        let items = convert_message(&msg);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "message");
        assert_eq!(items[0]["role"], "user");
        assert_eq!(items[0]["content"][0]["type"], "input_text");
    }

    #[test]
    fn test_assistant_with_thinking_and_text() {
        let msg = LlmMessage::Assistant {
            content: vec![
                ContentPart::Thinking { thinking: "ponder".into() },
                ContentPart::Text { text: "hi".into() },
            ],
            api: crate::types::Api::Openai,
            provider: "openai".into(),
            model: "gpt-5".into(),
            usage: Usage::default(),
            stop_reason: StopReason::EndTurn,
            error_message: None,
            timestamp: 0,
        };
        let items = convert_message(&msg);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["type"], "reasoning");
        assert_eq!(items[1]["type"], "message");
        assert_eq!(items[1]["content"][0]["type"], "output_text");
    }

    #[test]
    fn test_assistant_tool_use_split() {
        let msg = LlmMessage::Assistant {
            content: vec![
                ContentPart::Text { text: "use a tool".into() },
                ContentPart::ToolCall {
                    id: "call_1".into(),
                    name: "bash".into(),
                    arguments: serde_json::json!({"cmd": "ls"}),
                },
            ],
            api: crate::types::Api::Openai,
            provider: "openai".into(),
            model: "gpt-5".into(),
            usage: Usage::default(),
            stop_reason: StopReason::ToolUse,
            error_message: None,
            timestamp: 0,
        };
        let items = convert_message(&msg);
        // Should produce: text message, then function_call (in that order).
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["type"], "message");
        assert_eq!(items[1]["type"], "function_call");
        assert_eq!(items[1]["call_id"], "call_1");
    }

    #[test]
    fn test_tool_result_conversion() {
        let msg = LlmMessage::ToolResult {
            tool_call_id: "call_1".into(),
            tool_name: "bash".into(),
            content: vec![ContentPart::Text { text: "output".into() }],
            details: None,
            is_error: false,
            timestamp: 0,
        };
        let items = convert_message(&msg);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "function_call_output");
        assert_eq!(items[0]["call_id"], "call_1");
        assert_eq!(items[0]["output"], "output");
    }

    #[test]
    fn test_parse_output_function_call() {
        let item = serde_json::json!({
            "type": "function_call",
            "call_id": "c1",
            "name": "bash",
            "arguments": "{\"cmd\":\"ls\"}"
        });
        let parts = parse_output_item(&item);
        assert_eq!(parts.len(), 1);
        if let ContentPart::ToolCall { id, name, arguments } = &parts[0] {
            assert_eq!(id, "c1");
            assert_eq!(name, "bash");
            assert_eq!(arguments["cmd"], "ls");
        } else {
            panic!("expected ToolCall");
        }
    }

    #[test]
    fn test_parse_output_reasoning() {
        let item = serde_json::json!({
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "thinking..."}]
        });
        let parts = parse_output_item(&item);
        assert_eq!(parts.len(), 1);
        if let ContentPart::Thinking { thinking } = &parts[0] {
            assert_eq!(thinking, "thinking...");
        } else {
            panic!("expected Thinking");
        }
    }
}
