
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
            .map(convert_message)
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
            body["max_tokens"] = json!(max_tokens);
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

        body
    }

    async fn send_with_retry(&self, body: serde_json::Value) -> Result<reqwest::Response, LlmError> {
        let mut last_err = None;
        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(500 * 2u64.pow(attempt - 1))).await;
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
                        tokio::time::sleep(Duration::from_secs(secs)).await;
                        continue;
                    }
                    if status == 401 { return Err(LlmError::AuthError); }
                    let msg = r.text().await.unwrap_or_default();
                    last_err = Some(LlmError::Http { status, message: msg });
                    if status >= 500 { continue; }
                    break;
                }
                Err(e) => { last_err = Some(LlmError::Reqwest(e)); }
            }
        }
        Err(last_err.unwrap())
    }
}

fn convert_message(msg: &LlmMessage) -> serde_json::Value {
    match msg {
        LlmMessage::User { content, .. } => {
            let text = extract_text(content);
            json!({"role": "user", "content": text})
        }
        LlmMessage::Assistant { content, .. } => {
            let text: String = content.iter()
                .filter_map(|p| if let ContentPart::Text { text } = p { Some(text.as_str()) } else { None })
                .collect::<Vec<_>>()
                .join("");
            let tool_calls: Vec<serde_json::Value> = content.iter()
                .filter_map(|p| {
                    if let ContentPart::ToolUse { id, name, arguments } = p {
                        Some(json!({
                            "id": id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments.to_string()}
                        }))
                    } else { None }
                })
                .collect();
            if tool_calls.is_empty() {
                json!({"role": "assistant", "content": text})
            } else {
                json!({"role": "assistant", "content": text, "tool_calls": tool_calls})
            }
        }
        LlmMessage::ToolResult { tool_call_id, content, is_error: _, .. } => {
            let text = content.iter()
                .filter_map(|p| if let ContentPart::Text { text } = p { Some(text.as_str()) } else { None })
                .collect::<Vec<_>>()
                .join("");
            json!({"role": "tool", "tool_call_id": tool_call_id, "content": text})
        }
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
                content.push(ContentPart::ToolUse {
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
