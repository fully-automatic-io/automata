
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ============================================================================
// Thinking Level
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    #[default]
    #[serde(rename = "off")]
    Off,
    #[serde(rename = "minimal")]
    Minimal,
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "high")]
    High,
    #[serde(rename = "xhigh")]
    XHigh,
}

// ============================================================================
// Tool Execution Mode
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ToolExecutionMode {
    #[default]
    Sequential,
    Parallel,
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
// Content Blocks
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    #[serde(rename = "toolCall")]
    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    #[serde(rename = "toolResult")]
    ToolResult {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName", skip_serializing_if = "Option::is_none")]
        tool_name: Option<String>,
        content: Vec<ContentBlock>,
        #[serde(rename = "isError", default)]
        is_error: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<serde_json::Value>,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
}

// ============================================================================
// Usage & Cost
// ============================================================================

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Cost {
    pub input: f64,
    pub output: f64,
    #[serde(rename = "cacheRead")]
    pub cache_read: f64,
    #[serde(rename = "cacheWrite")]
    pub cache_write: f64,
    pub total: f64,
}

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
    pub cost: Cost,
}

impl Usage {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn add(&mut self, other: &Usage) {
        self.input += other.input;
        self.output += other.output;
        self.cache_read += other.cache_read;
        self.cache_write += other.cache_write;
        self.total_tokens += other.total_tokens;
        self.cost.input += other.cost.input;
        self.cost.output += other.cost.output;
        self.cost.cache_read += other.cost.cache_read;
        self.cost.cache_write += other.cost.cache_write;
        self.cost.total += other.cost.total;
    }
}

// ============================================================================
// Transport
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Transport {
    #[default]
    Sse,
    Json,
}

// ============================================================================
// Thinking Budgets
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThinkingBudgets {
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
// Model (minimal — provided by llm-client)
// ============================================================================

/// Model info used within agent-core.
/// Full model type is in llm-client; agent-core uses a trait/trait object.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub api: String,
    pub provider: String,
    pub base_url: String,
    pub reasoning: bool,
    pub input: Vec<String>,
    pub context_window: u64,
    pub max_tokens: u64,
}

impl Default for ModelInfo {
    fn default() -> Self {
        Self {
            id: "unknown".to_string(),
            name: "unknown".to_string(),
            api: "unknown".to_string(),
            provider: "unknown".to_string(),
            base_url: String::new(),
            reasoning: false,
            input: vec![],
            context_window: 0,
            max_tokens: 0,
        }
    }
}

// ============================================================================
// Messages
// ============================================================================

/// A message in the agent conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    #[serde(rename = "user")]
    User {
        content: MessageContent,
        #[serde(default)]
        timestamp: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<serde_json::Value>,
    },
    #[serde(rename = "assistant")]
    Assistant {
        content: Vec<ContentBlock>,
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
        #[serde(rename = "toolName")]
        tool_name: String,
        content: Vec<ContentBlock>,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<serde_json::Value>,
        #[serde(rename = "isError", default)]
        is_error: bool,
        timestamp: u64,
    },
}

/// Content can be a string (simple) or an array of content blocks (structured).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    String(String),
    Blocks(Vec<ContentBlock>),
}

impl Default for MessageContent {
    fn default() -> Self {
        Self::Blocks(vec![])
    }
}

/// CustomAgentMessages extension point.
/// Apps extend this via declaration merging in TS; in Rust we use a type-erased
/// approach with serde_json::Value for custom fields.
pub type AgentMessage = serde_json::Value;

/// Convert an AgentMessage to a typed Message if possible.
pub fn try_into_message(msg: &AgentMessage) -> Option<Message> {
    serde_json::from_value(msg.clone()).ok()
}

// ============================================================================
// Tool Result
// ============================================================================

/// Final or partial result produced by a tool.
#[derive(Debug, Clone, Serialize)]
pub struct AgentToolResult<T = serde_json::Value> {
    pub content: Vec<ContentBlock>,
    pub details: T,
    pub terminate: bool,
}

impl<T: Default> AgentToolResult<T> {
    pub fn error_text(message: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text {
                text: message.into(),
            }],
            details: T::default(),
            terminate: false,
        }
    }
}

/// Callback used by tools to stream partial execution updates.
pub type AgentToolUpdateCallback<T = serde_json::Value> =
    Box<dyn Fn(AgentToolResult<T>) + Send + Sync>;

// ============================================================================
// Agent Tool Call
// ============================================================================

/// A tool call content block extracted from an assistant message.
#[derive(Debug, Clone)]
pub struct AgentToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

// ============================================================================
// Tool Execution Hooks Contexts
// ============================================================================

#[derive(Debug, Clone)]
pub struct BeforeToolCallContext {
    pub assistant_message: AgentMessage,
    pub tool_call: AgentToolCall,
    pub args: serde_json::Value,
    pub context: AgentContext,
}

#[derive(Debug, Clone)]
pub struct BeforeToolCallResult {
    pub block: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AfterToolCallContext {
    pub assistant_message: AgentMessage,
    pub tool_call: AgentToolCall,
    pub args: serde_json::Value,
    pub result: AgentToolResult,
    pub is_error: bool,
    pub context: AgentContext,
}

#[derive(Debug, Clone)]
pub struct AfterToolCallResult {
    pub content: Option<Vec<ContentBlock>>,
    pub details: Option<serde_json::Value>,
    pub is_error: Option<bool>,
    pub terminate: Option<bool>,
}

// ============================================================================
// Agent Context & Config
// ============================================================================

/// Context snapshot passed into the low-level agent loop.
#[derive(Debug, Clone)]
pub struct AgentContext {
    pub system_prompt: String,
    pub messages: Vec<AgentMessage>,
    pub tools: Vec<String>,
}

/// Configuration for the agent loop.
/// Uses async callbacks — implementors use async_trait or Box<dyn Fn> + Pin.
#[derive(Clone)]
pub struct AgentLoopConfig {
    pub model: ModelInfo,

    pub convert_to_llm: std::sync::Arc<
        dyn Fn(Vec<AgentMessage>) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Vec<Message>> + Send>,
            > + Send
            + Sync,
    >,

    pub transform_context: Option<
        std::sync::Arc<
            dyn Fn(
                    Vec<AgentMessage>,
                    Option<tokio_util::sync::CancellationToken>,
                ) -> std::pin::Pin<
                    Box<dyn std::future::Future<Output = Vec<AgentMessage>> + Send>,
                > + Send
                + Sync,
        >,
    >,

    pub get_api_key:
        Option<std::sync::Arc<dyn Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<String>> + Send>> + Send + Sync>>,

    pub api_key: Option<String>,

    pub get_steering_messages: Option<
        std::sync::Arc<
            dyn Fn() -> std::pin::Pin<
                    Box<dyn std::future::Future<Output = Vec<AgentMessage>> + Send>,
                > + Send
                + Sync,
        >,
    >,

    pub get_follow_up_messages: Option<
        std::sync::Arc<
            dyn Fn() -> std::pin::Pin<
                    Box<dyn std::future::Future<Output = Vec<AgentMessage>> + Send>,
                > + Send
                + Sync,
        >,
    >,

    pub tool_execution: ToolExecutionMode,

    pub before_tool_call: Option<
        std::sync::Arc<
            dyn Fn(
                    BeforeToolCallContext,
                    Option<tokio_util::sync::CancellationToken>,
                ) -> std::pin::Pin<
                    Box<
                        dyn std::future::Future<Output = Option<BeforeToolCallResult>>
                            + Send,
                    >,
                > + Send
                + Sync,
        >,
    >,

    pub after_tool_call: Option<
        std::sync::Arc<
            dyn Fn(
                    AfterToolCallContext,
                    Option<tokio_util::sync::CancellationToken>,
                ) -> std::pin::Pin<
                    Box<
                        dyn std::future::Future<Output = Option<AfterToolCallResult>>
                            + Send,
                    >,
                > + Send
                + Sync,
        >,
    >,

    pub session_id: Option<String>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub transport: Transport,
    pub max_retry_delay_ms: Option<u64>,
    pub reasoning: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

impl AgentLoopConfig {
    pub fn new(
        model: ModelInfo,
        convert_to_llm: std::sync::Arc<
            dyn Fn(Vec<AgentMessage>) -> std::pin::Pin<
                    Box<dyn std::future::Future<Output = Vec<Message>> + Send>,
                > + Send
                + Sync,
        >,
    ) -> Self {
        Self {
            model,
            convert_to_llm,
            transform_context: None,
            get_api_key: None,
            api_key: None,
            get_steering_messages: None,
            get_follow_up_messages: None,
            tool_execution: ToolExecutionMode::Parallel,
            before_tool_call: None,
            after_tool_call: None,
            session_id: None,
            thinking_budgets: None,
            transport: Transport::Sse,
            max_retry_delay_ms: None,
            reasoning: None,
            temperature: None,
            max_tokens: None,
        }
    }
}

impl std::fmt::Debug for AgentLoopConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentLoopConfig")
            .field("model", &self.model)
            .field("tool_execution", &self.tool_execution)
            .field("session_id", &self.session_id)
            .field("transport", &self.transport)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Agent State
// ============================================================================

/// Mutable agent state with copy-on-write semantics for tools and messages.
pub struct AgentState {
    system_prompt: String,
    model: ModelInfo,
    thinking_level: ThinkingLevel,
    tools: Vec<String>,
    messages: Vec<AgentMessage>,
    pub is_streaming: bool,
    pub streaming_message: Option<AgentMessage>,
    pub pending_tool_calls: HashSet<String>,
    pub error_message: Option<String>,
}

impl AgentState {
    pub fn new(
        system_prompt: Option<String>,
        model: Option<ModelInfo>,
        thinking_level: Option<ThinkingLevel>,
        tools: Option<Vec<String>>,
        messages: Option<Vec<AgentMessage>>,
    ) -> Self {
        Self {
            system_prompt: system_prompt.unwrap_or_default(),
            model: model.unwrap_or_default(),
            thinking_level: thinking_level.unwrap_or_default(),
            tools: tools.unwrap_or_default(),
            messages: messages.unwrap_or_default(),
            is_streaming: false,
            streaming_message: None,
            pending_tool_calls: HashSet::new(),
            error_message: None,
        }
    }

    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    pub fn set_system_prompt(&mut self, prompt: String) {
        self.system_prompt = prompt;
    }

    pub fn model(&self) -> &ModelInfo {
        &self.model
    }

    pub fn set_model(&mut self, model: ModelInfo) {
        self.model = model;
    }

    pub fn thinking_level(&self) -> ThinkingLevel {
        self.thinking_level
    }

    pub fn set_thinking_level(&mut self, level: ThinkingLevel) {
        self.thinking_level = level;
    }

    /// Get tools — returns a copy (copy-on-read semantics matching TS getter).
    pub fn tools(&self) -> Vec<String> {
        self.tools.clone()
    }

    /// Set tools — copies the provided slice (copy-on-write matching TS setter).
    pub fn set_tools(&mut self, tools: &[String]) {
        self.tools = tools.to_vec();
    }

    /// Get messages — returns a copy.
    pub fn messages(&self) -> Vec<AgentMessage> {
        self.messages.clone()
    }

    /// Set messages — copies the provided slice.
    pub fn set_messages(&mut self, messages: &[AgentMessage]) {
        self.messages = messages.to_vec();
    }

    /// Push a single message to the transcript.
    pub fn push_message(&mut self, message: AgentMessage) {
        self.messages.push(message);
    }
}

impl std::fmt::Debug for AgentState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentState")
            .field("system_prompt", &self.system_prompt)
            .field("model", &self.model)
            .field("thinking_level", &self.thinking_level)
            .field("tools_count", &self.tools.len())
            .field("messages_count", &self.messages.len())
            .field("is_streaming", &self.is_streaming)
            .field("error_message", &self.error_message)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thinking_level_serde() {
        let level = ThinkingLevel::XHigh;
        let json = serde_json::to_string(&level).unwrap();
        assert_eq!(json, r#""xhigh""#);

        let deser: ThinkingLevel = serde_json::from_str(r#""off""#).unwrap();
        assert_eq!(deser, ThinkingLevel::Off);

        let deser: ThinkingLevel = serde_json::from_str(r#""minimal""#).unwrap();
        assert_eq!(deser, ThinkingLevel::Minimal);
    }

    #[test]
    fn test_agent_state_copy_semantics() {
        let mut state = AgentState::new(None, None, None, None, None);
        state.set_tools(&["tool_a".to_string(), "tool_b".to_string()]);

        let tools = state.tools();
        assert_eq!(tools.len(), 2);

        // Verify we got a copy, not a reference
        state.set_tools(&["tool_c".to_string()]);
        assert_eq!(tools.len(), 2); // original copy unchanged
        assert_eq!(state.tools().len(), 1);
    }

    #[test]
    fn test_usage_add() {
        let mut a = Usage {
            input: 100,
            output: 50,
            total_tokens: 150,
            ..Default::default()
        };
        let b = Usage {
            input: 200,
            output: 100,
            total_tokens: 300,
            ..Default::default()
        };
        a.add(&b);
        assert_eq!(a.input, 300);
        assert_eq!(a.output, 150);
        assert_eq!(a.total_tokens, 450);
    }

    #[test]
    fn test_agent_loop_config_debug() {
        let config = AgentLoopConfig::new(
            ModelInfo::default(),
            std::sync::Arc::new(|msgs| {
                Box::pin(async move {
                    msgs.into_iter()
                        .filter_map(|m| try_into_message(&m))
                        .collect()
                })
            }),
        );
        let debug = format!("{:?}", config);
        assert!(debug.contains("AgentLoopConfig"));
    }

    #[test]
    fn test_content_block_serde() {
        let text = ContentBlock::Text {
            text: "hello".to_string(),
        };
        let json = serde_json::to_string(&text).unwrap();
        assert_eq!(json, r#"{"type":"text","text":"hello"}"#);

        let tool_call = ContentBlock::ToolCall {
            id: "tc1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({"command": "ls"}),
        };
        let json = serde_json::to_string(&tool_call).unwrap();
        assert!(json.contains("toolCall"));
        assert!(json.contains("bash"));
    }

    #[test]
    fn test_tool_execution_mode_serde() {
        assert_eq!(
            serde_json::to_string(&ToolExecutionMode::Sequential).unwrap(),
            r#""sequential""#
        );
        assert_eq!(
            serde_json::to_string(&ToolExecutionMode::Parallel).unwrap(),
            r#""parallel""#
        );
    }
}
