
use crate::provider::{AuthMethod, LlmError, LlmProvider, LlmStream, ProviderConfig};
use crate::streaming::parse_sse_event;
use crate::types::{ContentPart, LlmMessage, LlmRequest, LlmResponse, StopReason, Usage, MessageContent};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::time::Duration;

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

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

        body
    }

    async fn send_with_retry(&self, body: serde_json::Value, _stream: bool) -> Result<reqwest::Response, LlmError> {
        let mut last_err = None;
        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(500 * 2u64.pow(attempt - 1))).await;
            }
            let req = self.client
                .post(self.base_url())
                .header("anthropic-version", ANTHROPIC_VERSION)
                .header("content-type", "application/json");
            let req = match self.config.auth_method {
                AuthMethod::ApiKeyHeader => req.header("x-api-key", &self.config.api_key),
                AuthMethod::Bearer => req.bearer_auth(&self.config.api_key),
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
        ContentPart::ToolUse { id, name, arguments } => json!({
            "type": "tool_use", "id": id, "name": name, "input": arguments,
        }),
        ContentPart::ToolResult { tool_use_id, content, is_error } => json!({
            "type": "tool_result", "tool_use_id": tool_use_id, "content": content, "is_error": is_error,
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
                    "tool_use" => Some(ContentPart::ToolUse {
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
