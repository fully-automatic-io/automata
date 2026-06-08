use crate::provider::{LlmError, LlmProvider, LlmStream, ProviderConfig};
use crate::streaming::{Delta, LlmEvent, MessageDelta, StreamError};
use crate::types::{
    CacheRetention, ContentPart, Cost, LlmMessage, LlmRequest, LlmResponse, MessageContent,
    ModelCost, StopReason, Usage,
};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::collections::{HashMap, HashSet};
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
        let messages: Vec<serde_json::Value> =
            request.messages.iter().filter_map(convert_message).collect();

        let tools: Vec<serde_json::Value> = request
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    }
                })
            })
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
            && retention != CacheRetention::None
        {
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

    async fn send_with_retry(
        &self,
        body: serde_json::Value,
    ) -> Result<reqwest::Response, LlmError> {
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
            let resp = self
                .client
                .post(self.base_url())
                .bearer_auth(&self.config.api_key)
                .json(&body)
                .send()
                .await;

            match resp {
                Ok(r) if r.status().is_success() => return Ok(r),
                Ok(r) => {
                    let status = r.status().as_u16();
                    if status == 429 {
                        let secs = r
                            .headers()
                            .get("retry-after")
                            .and_then(|v| v.to_str().ok())
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(5);
                        last_err = Some(LlmError::RateLimit { retry_after_secs: secs });
                        continue;
                    }
                    if status == 401 {
                        return Err(LlmError::AuthError);
                    }
                    let body = r.text().await.unwrap_or_default();
                    // Prefix status onto message body so the retry layer (and callers) can match (52e13870).
                    let err = LlmError::Http {
                        status,
                        message: format_http_error(status, &body),
                    };
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
        && let Some(obj) = last.as_object_mut()
    {
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
            && let Some(obj) = arr[i].as_object_mut()
        {
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
            // Chat Completions has no first-class replay slot for prior
            // thinking blocks. OpenAI-compatible providers that require replay
            // still accept those blocks as plain assistant text parts.
            let text: String = content
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } | ContentPart::Thinking { thinking: text } => {
                        Some(text.as_str())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
            let tool_calls: Vec<serde_json::Value> = content
                .iter()
                .filter_map(|p| {
                    if let ContentPart::ToolCall { id, name, arguments } = p {
                        Some(json!({
                            "id": id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments.to_string()}
                        }))
                    } else {
                        None
                    }
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
            let text = content
                .iter()
                .filter_map(|p| {
                    if let ContentPart::Text { text } = p {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
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
        MessageContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|p| {
                if let ContentPart::Text { text } = p {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

#[derive(Default)]
struct OpenAiStreamState {
    next_content_index: usize,
    text_index: Option<usize>,
    thinking_index: Option<usize>,
    tool_indices_by_stream_index: HashMap<usize, usize>,
    tool_indices_by_id: HashMap<String, usize>,
    active_content_indices: Vec<usize>,
    stopped_content_indices: HashSet<usize>,
    has_finish_reason: bool,
    is_complete: bool,
    model_cost: Option<ModelCost>,
}

impl OpenAiStreamState {
    fn new(model_cost: Option<ModelCost>) -> Self {
        Self { model_cost, ..Default::default() }
    }

    fn parse_sse_data(&mut self, data: &str) -> Vec<Result<LlmEvent, LlmError>> {
        if self.is_complete {
            return vec![];
        }
        if data == "[DONE]" {
            self.is_complete = true;
            if self.has_finish_reason {
                return vec![Ok(LlmEvent::MessageStop)];
            }
            let mut events = self.stop_active_blocks();
            events.push(Ok(LlmEvent::Error {
                error: StreamError {
                    error_type: "stream_error".into(),
                    message: "Stream ended without finish_reason".into(),
                },
            }));
            return events;
        }

        let value: serde_json::Value = match serde_json::from_str(data) {
            Ok(value) => value,
            Err(err) => {
                return vec![Err(LlmError::SerializationError(err.to_string()))];
            }
        };

        let mut events = Vec::new();
        let mut usage = value
            .get("usage")
            .filter(|usage| !usage.is_null())
            .map(|usage| parse_openai_usage(usage, self.model_cost.as_ref()));

        let choice = value
            .get("choices")
            .and_then(|choices| choices.as_array())
            .and_then(|choices| choices.first());

        if let Some(choice) = choice {
            if usage.is_none() {
                usage = choice
                    .get("usage")
                    .filter(|usage| !usage.is_null())
                    .map(|usage| parse_openai_usage(usage, self.model_cost.as_ref()));
            }

            if let Some(delta) = choice.get("delta") {
                self.push_delta_events(delta, &mut events);
            }

            if let Some(reason) = choice.get("finish_reason").and_then(|value| value.as_str()) {
                events.extend(self.finish_reason_events(reason, usage.take()));
            }
        }

        if let Some(usage) = usage {
            events.push(Ok(LlmEvent::MessageDelta {
                delta: MessageDelta { stop_reason: None, stop_sequence: None },
                usage: Some(usage),
            }));
        }

        events
    }

    fn push_delta_events(
        &mut self,
        delta: &serde_json::Value,
        events: &mut Vec<Result<LlmEvent, LlmError>>,
    ) {
        if let Some(text) = delta.get("content").and_then(|value| value.as_str())
            && !text.is_empty()
        {
            let index = self.ensure_text_block(events);
            events.push(Ok(LlmEvent::ContentBlockDelta {
                index,
                delta: Delta::TextDelta { text: text.to_string() },
            }));
        }

        if let Some(thinking) = first_reasoning_delta(delta) {
            let index = self.ensure_thinking_block(events);
            events.push(Ok(LlmEvent::ContentBlockDelta {
                index,
                delta: Delta::ThinkingDelta { thinking },
            }));
        }

        if let Some(tool_calls) = delta.get("tool_calls").and_then(|value| value.as_array()) {
            for tool_call in tool_calls {
                let index = self.ensure_tool_call_block(tool_call, events);
                if let Some(arguments) = tool_call
                    .get("function")
                    .and_then(|function| function.get("arguments"))
                    .and_then(|value| value.as_str())
                    && !arguments.is_empty()
                {
                    events.push(Ok(LlmEvent::ContentBlockDelta {
                        index,
                        delta: Delta::InputJsonDelta { partial_json: arguments.to_string() },
                    }));
                }
            }
        }
    }

    fn ensure_text_block(&mut self, events: &mut Vec<Result<LlmEvent, LlmError>>) -> usize {
        if let Some(index) = self.text_index {
            return index;
        }
        let index = self.allocate_content_index();
        self.text_index = Some(index);
        self.mark_active(index);
        events.push(Ok(LlmEvent::ContentBlockStart {
            index,
            content_block: ContentPart::Text { text: String::new() },
        }));
        index
    }

    fn ensure_thinking_block(&mut self, events: &mut Vec<Result<LlmEvent, LlmError>>) -> usize {
        if let Some(index) = self.thinking_index {
            return index;
        }
        let index = self.allocate_content_index();
        self.thinking_index = Some(index);
        self.mark_active(index);
        events.push(Ok(LlmEvent::ContentBlockStart {
            index,
            content_block: ContentPart::Thinking { thinking: String::new() },
        }));
        index
    }

    fn ensure_tool_call_block(
        &mut self,
        tool_call: &serde_json::Value,
        events: &mut Vec<Result<LlmEvent, LlmError>>,
    ) -> usize {
        let stream_index = tool_call
            .get("index")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize);
        let id = tool_call.get("id").and_then(|value| value.as_str()).unwrap_or("");
        let name = tool_call
            .get("function")
            .and_then(|function| function.get("name"))
            .and_then(|value| value.as_str())
            .unwrap_or("");

        if let Some(stream_index) = stream_index
            && let Some(index) = self.tool_indices_by_stream_index.get(&stream_index).copied()
        {
            if !id.is_empty() {
                self.tool_indices_by_id.insert(id.to_string(), index);
            }
            return index;
        }
        if !id.is_empty()
            && let Some(index) = self.tool_indices_by_id.get(id).copied()
        {
            if let Some(stream_index) = stream_index {
                self.tool_indices_by_stream_index.insert(stream_index, index);
            }
            return index;
        }

        let index = self.allocate_content_index();
        if let Some(stream_index) = stream_index {
            self.tool_indices_by_stream_index.insert(stream_index, index);
        }
        if !id.is_empty() {
            self.tool_indices_by_id.insert(id.to_string(), index);
        }
        self.mark_active(index);
        events.push(Ok(LlmEvent::ContentBlockStart {
            index,
            content_block: ContentPart::ToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments: serde_json::json!({}),
            },
        }));
        index
    }

    fn finish_reason_events(
        &mut self,
        reason: &str,
        usage: Option<Usage>,
    ) -> Vec<Result<LlmEvent, LlmError>> {
        self.has_finish_reason = true;
        let mut events = self.stop_active_blocks();
        match map_openai_finish_reason(reason) {
            Ok(stop_reason) => {
                events.push(Ok(LlmEvent::MessageDelta {
                    delta: MessageDelta {
                        stop_reason: Some(stop_reason),
                        stop_sequence: None,
                    },
                    usage,
                }));
            }
            Err(message) => {
                self.is_complete = true;
                if let Some(usage) = usage {
                    events.push(Ok(LlmEvent::MessageDelta {
                        delta: MessageDelta { stop_reason: None, stop_sequence: None },
                        usage: Some(usage),
                    }));
                }
                events.push(Ok(LlmEvent::Error {
                    error: StreamError {
                        error_type: "finish_reason".into(),
                        message,
                    },
                }));
            }
        }
        events
    }

    fn stop_active_blocks(&mut self) -> Vec<Result<LlmEvent, LlmError>> {
        let mut events = Vec::new();
        for index in self.active_content_indices.clone() {
            if self.stopped_content_indices.insert(index) {
                events.push(Ok(LlmEvent::ContentBlockStop { index }));
            }
        }
        events
    }

    fn allocate_content_index(&mut self) -> usize {
        let index = self.next_content_index;
        self.next_content_index += 1;
        index
    }

    fn mark_active(&mut self, index: usize) {
        if !self.active_content_indices.contains(&index) {
            self.active_content_indices.push(index);
        }
    }
}

fn first_reasoning_delta(delta: &serde_json::Value) -> Option<String> {
    ["reasoning_content", "reasoning", "reasoning_text"]
        .into_iter()
        .find_map(|field| {
            delta
                .get(field)
                .and_then(|value| value.as_str())
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
        })
}

fn map_openai_finish_reason(reason: &str) -> Result<StopReason, String> {
    match reason {
        "stop" | "end" => Ok(StopReason::EndTurn),
        "length" => Ok(StopReason::MaxTokens),
        "function_call" | "tool_calls" => Ok(StopReason::ToolUse),
        "content_filter" => Err("Provider finish_reason: content_filter".into()),
        "network_error" => Err("Provider finish_reason: network_error".into()),
        other => Err(format!("Provider finish_reason: {}", other)),
    }
}

fn parse_openai_usage(raw: &serde_json::Value, model_cost: Option<&ModelCost>) -> Usage {
    let prompt_tokens = raw.get("prompt_tokens").and_then(|value| value.as_u64()).unwrap_or(0);
    let completion_tokens =
        raw.get("completion_tokens").and_then(|value| value.as_u64()).unwrap_or(0);
    let prompt_details = raw.get("prompt_tokens_details");
    let cache_read = prompt_details
        .and_then(|details| details.get("cached_tokens"))
        .and_then(|value| value.as_u64())
        .or_else(|| raw.get("prompt_cache_hit_tokens").and_then(|value| value.as_u64()))
        .unwrap_or(0);
    let cache_write = prompt_details
        .and_then(|details| details.get("cache_write_tokens"))
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let input = prompt_tokens.saturating_sub(cache_read.saturating_add(cache_write));
    let mut usage = Usage {
        input,
        output: completion_tokens,
        cache_read,
        cache_write,
        total_tokens: input + completion_tokens + cache_read + cache_write,
        cost: Cost::default(),
    };
    if let Some(cost) = model_cost {
        usage.cost.input = cost.input / 1_000_000.0 * usage.input as f64;
        usage.cost.output = cost.output / 1_000_000.0 * usage.output as f64;
        usage.cost.cache_read = cost.cache_read / 1_000_000.0 * usage.cache_read as f64;
        usage.cost.cache_write = cost.cache_write / 1_000_000.0 * usage.cache_write as f64;
        usage.cost.total =
            usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
    }
    usage
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
            usage: serde_json::Value,
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
        let api_resp: ApiResponse =
            resp.json().await.map_err(|e| LlmError::SerializationError(e.to_string()))?;

        let choice = api_resp
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::InvalidRequest("No choices".to_string()))?;

        let mut content = vec![];
        if let Some(text) = choice.message.content
            && !text.is_empty()
        {
            content.push(ContentPart::Text { text });
        }
        if let Some(tool_calls) = choice.message.tool_calls {
            for tc in tool_calls {
                let input =
                    serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::Value::Null);
                content.push(ContentPart::ToolCall {
                    id: tc.id,
                    name: tc.function.name,
                    arguments: input,
                });
            }
        }

        let stop_reason = match choice.finish_reason.as_str() {
            "stop" | "end" => StopReason::EndTurn,
            "length" => StopReason::MaxTokens,
            "tool_calls" | "function_call" => StopReason::ToolUse,
            "content_filter" => StopReason::ContentFilter,
            _ => StopReason::Error,
        };
        let usage = parse_openai_usage(&api_resp.usage, request.model_cost.as_ref());

        Ok(LlmResponse {
            id: api_resp.id,
            model: api_resp.model,
            content,
            stop_reason,
            usage,
        })
    }

    async fn stream(&self, request: LlmRequest) -> Result<LlmStream, LlmError> {
        let body = self.build_body(&request, true);
        let resp = self.send_with_retry(body).await?;

        let byte_stream = resp.bytes_stream();
        let mut state = OpenAiStreamState::new(request.model_cost.clone());
        let stream = byte_stream.eventsource().flat_map(move |result| {
            let events = match result {
                Ok(event) => state.parse_sse_data(&event.data),
                Err(e) => vec![Err(LlmError::StreamError(e.to_string()))],
            };
            futures::stream::iter(events)
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
        assert!(
            body.get("messages")
                .and_then(|m| m.as_array())
                .and_then(|a| a.first())
                .and_then(|m| m.get("content"))
                .map(|c| c.is_string())
                .unwrap_or(false),
            "native cache must not promote content shape"
        );
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
        r.provider_options =
            Some(ProviderOptions::Openai(OpenaiOptions { anthropic_cache_control: Some(true) }));
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
        r.provider_options =
            Some(ProviderOptions::Openai(OpenaiOptions { anthropic_cache_control: Some(true) }));
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
        r.tools = vec![
            crate::types::ToolDefinition {
                name: "first".into(),
                description: "f".into(),
                input_schema: serde_json::json!({}),
            },
            crate::types::ToolDefinition {
                name: "second".into(),
                description: "s".into(),
                input_schema: serde_json::json!({}),
            },
        ];
        r.provider_options =
            Some(ProviderOptions::Openai(OpenaiOptions { anthropic_cache_control: Some(true) }));
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
            ..Default::default()
        }));
        let body = provider().build_body(&r, false);
        // Falls back to native cache (anthropic_cache_control is None).
        assert_eq!(body["prompt_cache_retention"], "24h");
    }

    #[test]
    fn empty_tools_are_omitted_from_payload() {
        let mut r = req("gpt-4o-mini");
        r.tools = Vec::new();

        let body = provider().build_body(&r, true);

        assert!(body.get("tools").is_none());
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn non_empty_tools_are_forwarded() {
        let mut r = req("gpt-4o-mini");
        r.tools = vec![crate::types::ToolDefinition {
            name: "ping".into(),
            description: "Ping tool".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"]
            }),
        }];

        let body = provider().build_body(&r, true);
        let tool = &body["tools"][0]["function"];

        assert_eq!(tool["name"], "ping");
        assert_eq!(tool["description"], "Ping tool");
        assert_eq!(tool["parameters"]["properties"]["ok"]["type"], "boolean");
    }

    #[test]
    fn forwards_reasoning_effort() {
        let mut r = req("deepseek-reasoner");
        r.reasoning_effort = Some(crate::types::ThinkingLevel::Medium);

        let body = provider().build_body(&r, true);

        assert_eq!(body["reasoning_effort"], "medium");
    }

    #[test]
    fn assistant_thinking_replays_as_text_for_chat_completions() {
        let mut r = req("deepseek-reasoner");
        r.messages = vec![
            LlmMessage::user_text("hello"),
            LlmMessage::Assistant {
                content: vec![
                    ContentPart::Thinking { thinking: "internal reasoning".into() },
                    ContentPart::Text { text: "visible answer".into() },
                ],
                api: crate::types::Api::Openai,
                provider: "deepseek".into(),
                model: "deepseek-reasoner".into(),
                usage: Usage::default(),
                stop_reason: StopReason::EndTurn,
                error_message: None,
                timestamp: 2,
            },
            LlmMessage::user_text("continue"),
        ];

        let body = provider().build_body(&r, false);

        assert_eq!(body["messages"][1]["role"], "assistant");
        assert_eq!(body["messages"][1]["content"], "internal reasoningvisible answer");
    }

    #[test]
    fn assistant_thinking_only_replay_is_not_dropped() {
        let mut r = req("deepseek-reasoner");
        r.messages = vec![
            LlmMessage::user_text("hello"),
            LlmMessage::Assistant {
                content: vec![ContentPart::Thinking { thinking: "internal reasoning".into() }],
                api: crate::types::Api::Openai,
                provider: "deepseek".into(),
                model: "deepseek-reasoner".into(),
                usage: Usage::default(),
                stop_reason: StopReason::EndTurn,
                error_message: None,
                timestamp: 2,
            },
            LlmMessage::user_text("continue"),
        ];

        let body = provider().build_body(&r, false);

        assert_eq!(body["messages"].as_array().unwrap().len(), 3);
        assert_eq!(body["messages"][1]["content"], "internal reasoning");
    }

    fn unwrap_events(events: Vec<Result<LlmEvent, LlmError>>) -> Vec<LlmEvent> {
        events.into_iter().map(Result::unwrap).collect()
    }

    #[test]
    fn stream_parser_emits_text_reasoning_tool_usage_and_finish_reason() {
        let mut state = OpenAiStreamState::new(Some(ModelCost {
            input: 2.0,
            output: 4.0,
            cache_read: 1.0,
            cache_write: 3.0,
        }));

        let first = unwrap_events(state.parse_sse_data(
            r#"{"id":"chatcmpl-test","choices":[{"index":0,"delta":{"content":"hello","reasoning_content":"think","tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"bash","arguments":"{\"cmd\""}}]},"finish_reason":null}]}"#,
        ));
        assert!(matches!(
            first[0],
            LlmEvent::ContentBlockStart {
                index: 0,
                content_block: ContentPart::Text { .. }
            }
        ));
        assert!(matches!(
            first[1],
            LlmEvent::ContentBlockDelta {
                index: 0,
                delta: Delta::TextDelta { ref text }
            } if text == "hello"
        ));
        assert!(matches!(
            first[2],
            LlmEvent::ContentBlockStart {
                index: 1,
                content_block: ContentPart::Thinking { .. }
            }
        ));
        assert!(matches!(
            first[3],
            LlmEvent::ContentBlockDelta {
                index: 1,
                delta: Delta::ThinkingDelta { ref thinking }
            } if thinking == "think"
        ));
        assert!(matches!(
            first[4],
            LlmEvent::ContentBlockStart {
                index: 2,
                content_block: ContentPart::ToolCall { ref id, ref name, .. }
            } if id == "call-1" && name == "bash"
        ));
        assert!(matches!(
            first[5],
            LlmEvent::ContentBlockDelta {
                index: 2,
                delta: Delta::InputJsonDelta { ref partial_json }
            } if partial_json == "{\"cmd\""
        ));

        let second = unwrap_events(state.parse_sse_data(
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"ls\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":30,"cache_write_tokens":10}}}"#,
        ));
        assert!(matches!(
            second[0],
            LlmEvent::ContentBlockDelta {
                index: 2,
                delta: Delta::InputJsonDelta { ref partial_json }
            } if partial_json == ":\"ls\"}"
        ));
        assert!(matches!(second[1], LlmEvent::ContentBlockStop { index: 0 }));
        assert!(matches!(second[2], LlmEvent::ContentBlockStop { index: 1 }));
        assert!(matches!(second[3], LlmEvent::ContentBlockStop { index: 2 }));
        match &second[4] {
            LlmEvent::MessageDelta {
                delta:
                    MessageDelta {
                        stop_reason: Some(StopReason::ToolUse), ..
                    },
                usage: Some(usage),
            } => {
                assert_eq!(usage.input, 60);
                assert_eq!(usage.output, 20);
                assert_eq!(usage.cache_read, 30);
                assert_eq!(usage.cache_write, 10);
                assert_eq!(usage.total_tokens, 120);
                assert!((usage.cost.input - 0.00012).abs() < f64::EPSILON);
                assert!((usage.cost.output - 0.00008).abs() < f64::EPSILON);
                assert!((usage.cost.cache_read - 0.00003).abs() < f64::EPSILON);
                assert!((usage.cost.cache_write - 0.00003).abs() < f64::EPSILON);
            }
            other => panic!("unexpected event: {other:?}"),
        }

        let done = unwrap_events(state.parse_sse_data("[DONE]"));
        assert!(matches!(done.as_slice(), [LlmEvent::MessageStop]));
    }

    #[test]
    fn stream_parser_errors_when_done_arrives_without_finish_reason() {
        let mut state = OpenAiStreamState::new(None);
        let events = unwrap_events(state.parse_sse_data(
            r#"{"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}"#,
        ));
        assert!(events.iter().any(|event| matches!(
            event,
            LlmEvent::ContentBlockDelta {
                delta: Delta::TextDelta { text },
                ..
            } if text == "partial"
        )));

        let done = unwrap_events(state.parse_sse_data("[DONE]"));
        assert!(matches!(done[0], LlmEvent::ContentBlockStop { index: 0 }));
        assert!(matches!(
            done[1],
            LlmEvent::Error {
                error: StreamError { ref message, .. }
            } if message == "Stream ended without finish_reason"
        ));
    }
}
