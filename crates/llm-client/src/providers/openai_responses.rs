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
        use eventsource_stream::Eventsource;
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let body = self.build_body(&request, true);
        let resp = self.send_with_retry(body).await?;

        let state = Arc::new(Mutex::new(TranslateState {
            current_index: 0,
            current_kind: BlockKind::None,
            tool_buf: String::new(),
        }));

        let byte_stream = resp.bytes_stream();
        let sse = byte_stream.eventsource();
        let stream = futures::StreamExt::flat_map(sse, move |result| {
            let state = state.clone();
            futures::stream::once(async move { translate_event(result, state).await })
        });
        let stream = futures::StreamExt::flat_map(stream, futures::stream::iter);

        Ok(Box::pin(stream))
    }
}

#[derive(Copy, Clone, PartialEq)]
enum BlockKind { None, Text, Thinking, ToolCall }

struct TranslateState {
    current_index: usize,
    current_kind: BlockKind,
    tool_buf: String,
}

async fn translate_event(
    result: Result<eventsource_stream::Event, eventsource_stream::EventStreamError<reqwest::Error>>,
    state: std::sync::Arc<tokio::sync::Mutex<TranslateState>>,
) -> Vec<Result<LlmEvent, LlmError>> {
    use crate::streaming::Delta;
    use crate::types::ContentBlock;

    let event = match result {
        Ok(e) => e,
        Err(e) => return vec![Err(LlmError::StreamError(e.to_string()))],
    };
    if event.data == "[DONE]" {
        return vec![Ok(LlmEvent::MessageStop)];
    }
    let payload: serde_json::Value = match serde_json::from_str(&event.data) {
        Ok(v) => v,
        Err(_) => return vec![],
    };

    let mut out: Vec<Result<LlmEvent, LlmError>> = Vec::new();
    let mut s = state.lock().await;

    match event.event.as_str() {
        "response.created" => {
            if let Some(resp) = payload.get("response") {
                let id = resp.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let model = resp.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string();
                out.push(Ok(LlmEvent::MessageStart { id, model }));
            }
        }
        "response.output_item.added" => {
            let item = match payload.get("item") {
                Some(v) => v,
                None => return out,
            };
            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
            s.current_index = payload.get("output_index")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize)
                .unwrap_or(s.current_index);
            match item_type {
                "reasoning" => {
                    s.current_kind = BlockKind::Thinking;
                    out.push(Ok(LlmEvent::ContentBlockStart {
                        index: s.current_index,
                        content_block: ContentBlock::Thinking { thinking: String::new() },
                    }));
                }
                "message" => {
                    s.current_kind = BlockKind::Text;
                    out.push(Ok(LlmEvent::ContentBlockStart {
                        index: s.current_index,
                        content_block: ContentBlock::Text { text: String::new() },
                    }));
                }
                "function_call" => {
                    s.current_kind = BlockKind::ToolCall;
                    s.tool_buf.clear();
                    let id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    out.push(Ok(LlmEvent::ContentBlockStart {
                        index: s.current_index,
                        content_block: ContentBlock::ToolCall {
                            id,
                            name,
                            arguments: serde_json::json!({}),
                        },
                    }));
                }
                _ => {}
            }
        }
        "response.output_text.delta"
        | "response.reasoning_text.delta"
        | "response.reasoning_summary_text.delta" => {
            let delta_text = payload.get("delta").and_then(|v| v.as_str()).unwrap_or("").to_string();
            if delta_text.is_empty() {
                return out;
            }
            let delta = if matches!(s.current_kind, BlockKind::Thinking) {
                Delta::ThinkingDelta { thinking: delta_text }
            } else {
                Delta::TextDelta { text: delta_text }
            };
            out.push(Ok(LlmEvent::ContentBlockDelta { index: s.current_index, delta }));
        }
        "response.function_call_arguments.delta" => {
            let delta_text = payload.get("delta").and_then(|v| v.as_str()).unwrap_or("").to_string();
            if !delta_text.is_empty() {
                s.tool_buf.push_str(&delta_text);
                out.push(Ok(LlmEvent::ContentBlockDelta {
                    index: s.current_index,
                    delta: Delta::InputJsonDelta { partial_json: delta_text },
                }));
            }
        }
        "response.function_call_arguments.done" => {
            if let Some(full) = payload.get("arguments").and_then(|v| v.as_str())
                && full.starts_with(&s.tool_buf) && full.len() > s.tool_buf.len() {
                    let tail = full[s.tool_buf.len()..].to_string();
                    out.push(Ok(LlmEvent::ContentBlockDelta {
                        index: s.current_index,
                        delta: Delta::InputJsonDelta { partial_json: tail },
                    }));
                }
        }
        "response.output_item.done" => {
            out.push(Ok(LlmEvent::ContentBlockStop { index: s.current_index }));
            s.current_kind = BlockKind::None;
        }
        "response.completed" => {
            let usage = payload.get("response").and_then(|r| r.get("usage")).map(|u| {
                let cache_read = u.get("input_tokens_details")
                    .and_then(|d| d.get("cached_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let input_total = u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                Usage {
                    input: input_total.saturating_sub(cache_read),
                    output: u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                    cache_read,
                    cache_write: 0,
                    total_tokens: u.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
                    cost: Default::default(),
                }
            });
            let resp = payload.get("response");
            let status = resp.and_then(|r| r.get("status")).and_then(|v| v.as_str()).unwrap_or("");
            let stop_reason = if status == "incomplete" {
                Some(StopReason::MaxTokens)
            } else {
                let has_tool = resp
                    .and_then(|r| r.get("output"))
                    .and_then(|o| o.as_array())
                    .map(|arr| arr.iter().any(|v| v.get("type").and_then(|t| t.as_str()) == Some("function_call")))
                    .unwrap_or(false);
                Some(if has_tool { StopReason::ToolUse } else { StopReason::EndTurn })
            };
            out.push(Ok(LlmEvent::MessageDelta {
                delta: crate::streaming::MessageDelta { stop_reason, stop_sequence: None },
                usage,
            }));
            out.push(Ok(LlmEvent::MessageStop));
        }
        "response.failed" | "error" => {
            let msg = payload
                .get("response")
                .and_then(|r| r.get("error"))
                .and_then(|e| e.get("message"))
                .and_then(|v| v.as_str())
                .or_else(|| payload.get("error").and_then(|e| e.get("message")).and_then(|v| v.as_str()))
                .unwrap_or("Responses API error")
                .to_string();
            out.push(Ok(LlmEvent::Error {
                error: crate::streaming::StreamError {
                    error_type: "responses_error".into(),
                    message: msg,
                },
            }));
        }
        _ => {}
    }
    out
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
                    if ctype == "output_text"
                        && let Some(text) = c.get("text").and_then(|t| t.as_str()) {
                            out.push(ContentPart::Text { text: text.to_string() });
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

    // ── streaming-translation tests ──
    use eventsource_stream::Event;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    fn ev(name: &str, json: serde_json::Value) -> Result<Event, eventsource_stream::EventStreamError<reqwest::Error>> {
        Ok(Event {
            event: name.into(),
            data: serde_json::to_string(&json).unwrap(),
            id: String::new(),
            retry: None,
        })
    }

    fn fresh_state() -> Arc<Mutex<TranslateState>> {
        Arc::new(Mutex::new(TranslateState {
            current_index: 0,
            current_kind: BlockKind::None,
            tool_buf: String::new(),
        }))
    }

    #[tokio::test]
    async fn translate_response_created_emits_message_start() {
        let s = fresh_state();
        let evs = translate_event(
            ev("response.created", serde_json::json!({"response": {"id": "r1", "model": "gpt-5"}})),
            s,
        ).await;
        match &evs[0] {
            Ok(LlmEvent::MessageStart { id, model }) => {
                assert_eq!(id, "r1");
                assert_eq!(model, "gpt-5");
            }
            _ => panic!("expected MessageStart, got {:?}", evs[0]),
        }
    }

    #[tokio::test]
    async fn translate_text_block_lifecycle() {
        let s = fresh_state();
        // item.added (message)
        let _ = translate_event(
            ev("response.output_item.added", serde_json::json!({
                "output_index": 0, "item": {"type": "message"}
            })),
            s.clone(),
        ).await;
        // delta
        let evs = translate_event(
            ev("response.output_text.delta", serde_json::json!({"delta": "Hi "})),
            s.clone(),
        ).await;
        match &evs[0] {
            Ok(LlmEvent::ContentBlockDelta { index: 0, delta: crate::streaming::Delta::TextDelta { text } }) => {
                assert_eq!(text, "Hi ");
            }
            _ => panic!("expected TextDelta, got {:?}", evs[0]),
        }
        // item.done
        let evs = translate_event(
            ev("response.output_item.done", serde_json::json!({"item": {"type": "message"}})),
            s.clone(),
        ).await;
        match &evs[0] {
            Ok(LlmEvent::ContentBlockStop { index: 0 }) => {}
            _ => panic!("expected ContentBlockStop, got {:?}", evs[0]),
        }
    }

    #[tokio::test]
    async fn translate_tool_call_args_streaming() {
        let s = fresh_state();
        let _ = translate_event(
            ev("response.output_item.added", serde_json::json!({
                "output_index": 0,
                "item": {"type": "function_call", "call_id": "tc1", "name": "bash"}
            })),
            s.clone(),
        ).await;
        let evs = translate_event(
            ev("response.function_call_arguments.delta", serde_json::json!({"delta": "{\"cmd"})),
            s.clone(),
        ).await;
        match &evs[0] {
            Ok(LlmEvent::ContentBlockDelta { delta: crate::streaming::Delta::InputJsonDelta { partial_json }, .. }) => {
                assert_eq!(partial_json, "{\"cmd");
            }
            _ => panic!("expected InputJsonDelta, got {:?}", evs[0]),
        }
        // .done with the canonical full string emits any tail delta only.
        let evs = translate_event(
            ev("response.function_call_arguments.done", serde_json::json!({"arguments": "{\"cmd\":\"ls\"}"})),
            s.clone(),
        ).await;
        match &evs[0] {
            Ok(LlmEvent::ContentBlockDelta { delta: crate::streaming::Delta::InputJsonDelta { partial_json }, .. }) => {
                assert_eq!(partial_json, "\":\"ls\"}");
            }
            _ => panic!("expected tail InputJsonDelta, got {:?}", evs[0]),
        }
    }

    #[tokio::test]
    async fn translate_response_completed_emits_usage_and_stop() {
        let s = fresh_state();
        let evs = translate_event(
            ev("response.completed", serde_json::json!({
                "response": {
                    "status": "completed",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "total_tokens": 150,
                        "input_tokens_details": {"cached_tokens": 30}
                    },
                    "output": [{"type": "message"}]
                }
            })),
            s,
        ).await;
        // First event: MessageDelta with usage.
        match &evs[0] {
            Ok(LlmEvent::MessageDelta { delta, usage }) => {
                assert_eq!(delta.stop_reason, Some(StopReason::EndTurn));
                let u = usage.as_ref().unwrap();
                assert_eq!(u.input, 70);  // 100 - 30 cached
                assert_eq!(u.output, 50);
                assert_eq!(u.cache_read, 30);
                assert_eq!(u.total_tokens, 150);
            }
            _ => panic!("expected MessageDelta, got {:?}", evs[0]),
        }
        // Second event: MessageStop.
        match &evs[1] {
            Ok(LlmEvent::MessageStop) => {}
            _ => panic!("expected MessageStop, got {:?}", evs[1]),
        }
    }

    #[tokio::test]
    async fn translate_tool_use_stop_reason() {
        let s = fresh_state();
        let evs = translate_event(
            ev("response.completed", serde_json::json!({
                "response": {
                    "status": "completed",
                    "output": [{"type": "function_call"}]
                }
            })),
            s,
        ).await;
        match &evs[0] {
            Ok(LlmEvent::MessageDelta { delta, .. }) => {
                assert_eq!(delta.stop_reason, Some(StopReason::ToolUse));
            }
            _ => panic!("expected MessageDelta with ToolUse"),
        }
    }

    #[tokio::test]
    async fn translate_response_incomplete_maps_to_max_tokens() {
        let s = fresh_state();
        let evs = translate_event(
            ev("response.completed", serde_json::json!({
                "response": {"status": "incomplete", "output": []}
            })),
            s,
        ).await;
        match &evs[0] {
            Ok(LlmEvent::MessageDelta { delta, .. }) => {
                assert_eq!(delta.stop_reason, Some(StopReason::MaxTokens));
            }
            _ => panic!("expected MessageDelta with MaxTokens"),
        }
    }

    #[tokio::test]
    async fn translate_response_failed_emits_error() {
        let s = fresh_state();
        let evs = translate_event(
            ev("response.failed", serde_json::json!({
                "response": {"error": {"message": "boom"}}
            })),
            s,
        ).await;
        match &evs[0] {
            Ok(LlmEvent::Error { error }) => assert_eq!(error.message, "boom"),
            _ => panic!("expected Error, got {:?}", evs[0]),
        }
    }
}
