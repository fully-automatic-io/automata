
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// API / Provider
// ============================================================================

pub type Api = String;
pub type Provider = String;

// ============================================================================
// Transport
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Transport {
    #[default]
    Sse,
    Websocket,
    Auto,
}

// ============================================================================
// Cache Retention
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CacheRetention {
    #[default]
    Short,
    Long,
    None,
}

// ============================================================================
// Thinking Budgets
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThinkingBudgets {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimal: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub low: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub medium: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub high: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xhigh: Option<u32>,
}

// ============================================================================
// Model
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCost {
    pub input: f64,
    pub output: f64,
    #[serde(rename = "cacheRead")]
    pub cache_read: f64,
    #[serde(rename = "cacheWrite")]
    pub cache_write: f64,
}

impl Default for ModelCost {
    fn default() -> Self {
        Self { input: 0.0, output: 0.0, cache_read: 0.0, cache_write: 0.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: Api,
    pub provider: Provider,
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    pub reasoning: bool,
    pub input: Vec<String>,
    pub cost: ModelCost,
    #[serde(rename = "contextWindow")]
    pub context_window: u64,
    #[serde(rename = "maxTokens")]
    pub max_tokens: u64,
}

// ============================================================================
// Content Parts — matches pi-ai content blocks
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    #[serde(rename = "toolCall")]
    ToolUse {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    #[serde(rename = "toolResult")]
    ToolResult {
        #[serde(rename = "toolCallId")]
        tool_use_id: String,
        content: String,
        #[serde(rename = "isError", default)]
        is_error: bool,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
}

// ============================================================================
// Messages — matches pi-ai Message union
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum LlmMessage {
    #[serde(rename = "user")]
    User {
        content: MessageContent,
        #[serde(default)]
        timestamp: u64,
    },
    #[serde(rename = "assistant")]
    Assistant {
        content: Vec<ContentPart>,
        api: String,
        provider: String,
        model: String,
        usage: Usage,
        #[serde(rename = "stopReason")]
        stop_reason: String,
        #[serde(rename = "errorMessage", skip_serializing_if = "Option::is_none")]
        error_message: Option<String>,
        timestamp: u64,
    },
    #[serde(rename = "toolResult")]
    ToolResult {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName", skip_serializing_if = "Option::is_none")]
        tool_name: Option<String>,
        content: Vec<ContentPart>,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<serde_json::Value>,
        #[serde(rename = "isError", default)]
        is_error: bool,
        timestamp: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    String(String),
    Blocks(Vec<ContentPart>),
}

impl Default for MessageContent {
    fn default() -> Self { Self::Blocks(vec![]) }
}

impl LlmMessage {
    pub fn system(content: impl Into<String>) -> serde_json::Value {
        serde_json::json!({"role": "system", "content": content.into()})
    }

    pub fn user_text(text: impl Into<String>) -> Self {
        Self::User {
            content: MessageContent::Blocks(vec![ContentPart::Text { text: text.into() }]),
            timestamp: 0,
        }
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::Assistant {
            content: vec![ContentPart::Text { text: text.into() }],
            api: "unknown".into(),
            provider: "unknown".into(),
            model: "unknown".into(),
            usage: Usage::default(),
            stop_reason: "stop".into(),
            error_message: None,
            timestamp: 0,
        }
    }

    pub fn role(&self) -> &str {
        match self {
            Self::User { .. } => "user",
            Self::Assistant { .. } => "assistant",
            Self::ToolResult { .. } => "toolResult",
        }
    }
}

// ============================================================================
// Tool Definition
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

// ============================================================================
// Request / Response
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    pub model: String,
    pub messages: Vec<LlmMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, serde_json::Value>,
}

impl LlmRequest {
    pub fn new(model: impl Into<String>, messages: Vec<LlmMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            tools: vec![],
            system: None,
            max_tokens: None,
            temperature: None,
            stop_sequences: vec![],
            extra: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub id: String,
    pub model: String,
    pub content: Vec<ContentPart>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

// ============================================================================
// Stop Reason
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    #[serde(rename = "stop")]
    EndTurn,
    #[serde(rename = "length")]
    MaxTokens,
    #[serde(rename = "stop_sequence")]
    StopSequence,
    #[serde(rename = "toolUse")]
    ToolUse,
    #[serde(rename = "content_filter")]
    ContentFilter,
    #[serde(rename = "error")]
    Error,
    #[serde(rename = "aborted")]
    Aborted,
}

// ============================================================================
// Usage — matches pi-ai Usage exactly
// ============================================================================

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    #[serde(rename = "cacheRead")]
    pub cache_read: u64,
    #[serde(rename = "cacheWrite")]
    pub cache_write: u64,
    #[serde(rename = "totalTokens")]
    pub total_tokens: u64,
    pub cost: UsageCost,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct UsageCost {
    pub input: f64,
    pub output: f64,
    #[serde(rename = "cacheRead")]
    pub cache_read: f64,
    #[serde(rename = "cacheWrite")]
    pub cache_write: f64,
    pub total: f64,
}

// ============================================================================
// Context — matches pi-ai Context
// ============================================================================

#[derive(Debug, Clone)]
pub struct Context {
    pub system_prompt: String,
    pub messages: Vec<LlmMessage>,
    pub tools: Vec<ToolDefinition>,
}

// ============================================================================
// SimpleStreamOptions — matches pi-ai SimpleStreamOptions
// ============================================================================

#[derive(Debug, Clone)]
pub struct SimpleStreamOptions {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub reasoning: Option<String>,
    pub api_key: Option<String>,
    pub transport: Option<Transport>,
    pub cache_retention: Option<CacheRetention>,
    pub session_id: Option<String>,
    pub headers: Option<HashMap<String, String>>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub max_retry_delay_ms: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_message() {
        let msg = LlmMessage::user_text("hello");
        assert_eq!(msg.role(), "user");
    }

    #[test]
    fn test_assistant_message() {
        let msg = LlmMessage::assistant_text("hi");
        assert_eq!(msg.role(), "assistant");
    }

    #[test]
    fn test_usage_default() {
        let usage = Usage::default();
        assert_eq!(usage.input, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn test_model_serde() {
        let model = Model {
            id: "claude-sonnet-4-6".into(),
            name: "Claude Sonnet 4.6".into(),
            api: "anthropic-messages".into(),
            provider: "anthropic".into(),
            base_url: "https://api.anthropic.com".into(),
            reasoning: false,
            input: vec!["text".into(), "image".into()],
            cost: ModelCost::default(),
            context_window: 200000,
            max_tokens: 16384,
        };
        let json = serde_json::to_string(&model).unwrap();
        assert!(json.contains("claude-sonnet-4-6"));
        let deser: Model = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.provider, "anthropic");
    }
}
