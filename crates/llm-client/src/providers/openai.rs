
use crate::provider::{LlmError, LlmProvider, LlmStream, ProviderConfig};
use crate::streaming::{Delta, LlmEvent};
use crate::types::{ContentPart, LlmMessage, LlmRequest, LlmResponse, StopReason, Usage, MessageContent};
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
        if let Some(text) = choice.message.content {
            if !text.is_empty() {
                content.push(ContentPart::Text { text });
            }
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
