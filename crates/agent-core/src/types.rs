// Canonical type definitions for agent-core.
//
// This module is the single source of truth for protocol types (LlmMessage,
// ContentBlock, Usage, Model, ...) and agent types (AgentMessage, AgentContext,
// AgentLoopConfig, AgentState, ...). The llm-client crate depends on agent-core
// and re-exports these types; coding-agent uses them via agent-core.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use crate::tool::AgentTool;

// ============================================================================
// Transport / CacheRetention / ThinkingBudgets / ThinkingLevel / ToolExec
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Transport {
    #[default]
    Sse,
    Websocket,
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CacheRetention {
    #[default]
    Short,
    Long,
    None,
}

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    #[default]
    #[serde(rename = "off")]
    Off,
    Minimal,
    Low,
    Medium,
    High,
    #[serde(rename = "xhigh")]
    XHigh,
}

impl ThinkingLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
        }
    }

    pub fn from_wire_str(s: &str) -> Option<Self> {
        match s {
            "off" => Some(Self::Off),
            "minimal" => Some(Self::Minimal),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" => Some(Self::XHigh),
            _ => None,
        }
    }
}

// ============================================================================
// Wire-protocol family — closed set; one variant per supported request shape.
// `Provider` (vendor name) stays a String because it's open: anthropic /
// openai / deepseek / bedrock / cloudflare / fireworks / etc.
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Api {
    /// Anthropic Messages-style (`/v1/messages`, content blocks, thinking blocks).
    #[default]
    Anthropic,
    /// OpenAI Chat Completions (`/v1/chat/completions`).
    Openai,
    /// OpenAI Responses API (`/v1/responses`).
    #[serde(rename = "openai-responses")]
    OpenaiResponses,
    /// Mock/test sink — no real wire shape.
    Mock,
}

impl Api {
    pub fn as_wire_str(&self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::Openai => "openai",
            Self::OpenaiResponses => "openai-responses",
            Self::Mock => "mock",
        }
    }

    pub fn from_wire_str(s: &str) -> Option<Self> {
        match s {
            "anthropic" => Some(Self::Anthropic),
            "openai" => Some(Self::Openai),
            "openai-responses" => Some(Self::OpenaiResponses),
            "mock" => Some(Self::Mock),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ToolExecutionMode {
    Sequential,
    #[default]
    Parallel,
}

// ============================================================================
// Stop Reason
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum StopReason {
    #[default]
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

impl StopReason {
    pub fn as_wire_str(&self) -> &'static str {
        match self {
            Self::EndTurn => "stop",
            Self::MaxTokens => "length",
            Self::StopSequence => "stop_sequence",
            Self::ToolUse => "toolUse",
            Self::ContentFilter => "content_filter",
            Self::Error => "error",
            Self::Aborted => "aborted",
        }
    }

    pub fn from_wire_str(s: &str) -> Option<Self> {
        match s {
            "stop" | "end_turn" => Some(Self::EndTurn),
            "length" | "max_tokens" => Some(Self::MaxTokens),
            "stop_sequence" => Some(Self::StopSequence),
            "toolUse" | "tool_use" => Some(Self::ToolUse),
            "content_filter" => Some(Self::ContentFilter),
            "error" => Some(Self::Error),
            "aborted" => Some(Self::Aborted),
            _ => None,
        }
    }
}

// ============================================================================
// Usage / Cost
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

// `UsageCost` is an alias for `Cost` used by some llm-client provider crates.
pub type UsageCost = Cost;

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
        Self {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
        }
    }
}

/// API-specific compatibility flags that override the provider's default
/// behaviour for a given model. Mirrors pi-mono's `compat` object; only the
/// flags the Rust providers actually consume are modelled. Unset flags fall
/// back to substring matching on the model id (see the Anthropic provider).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCompat {
    /// Force adaptive thinking (`thinking.type = "adaptive"`) regardless of id.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "forceAdaptiveThinking"
    )]
    pub force_adaptive_thinking: Option<bool>,
    /// Whether the model accepts the Anthropic `temperature` field. Claude
    /// Opus 4.7+ reject non-default temperature values.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "supportsTemperature"
    )]
    pub supports_temperature: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: Api,
    pub provider: String,
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    pub reasoning: bool,
    pub input: Vec<String>,
    pub cost: ModelCost,
    #[serde(rename = "contextWindow")]
    pub context_window: u64,
    #[serde(rename = "maxTokens")]
    pub max_tokens: u64,
    #[serde(default)]
    pub compat: ModelCompat,
}

impl Default for Model {
    fn default() -> Self {
        Self {
            id: "unknown".into(),
            name: "unknown".into(),
            api: Api::Anthropic,
            provider: "unknown".into(),
            base_url: String::new(),
            reasoning: false,
            input: vec![],
            cost: ModelCost::default(),
            context_window: 0,
            max_tokens: 0,
            compat: ModelCompat::default(),
        }
    }
}

impl Model {
    /// Cap `requested` so it never exceeds either the model's `max_tokens` or
    /// `context_window - input_tokens_estimate`.
    pub fn cap_output_budget(&self, requested: Option<u32>, input_tokens_estimate: u64) -> u32 {
        let model_cap = if self.max_tokens > 0 {
            self.max_tokens
        } else {
            u64::MAX
        };
        let context_cap = self.context_window.saturating_sub(input_tokens_estimate);
        let asked = requested.map(u64::from).unwrap_or(model_cap);
        let capped = asked.min(model_cap).min(context_cap.max(1));
        capped.try_into().unwrap_or(u32::MAX)
    }

    /// Clamp `reasoning` to a level this model supports. Models without
    /// reasoning return None. `Some(ThinkingLevel::Off)` also collapses to None.
    pub fn clamp_reasoning(&self, reasoning: Option<ThinkingLevel>) -> Option<ThinkingLevel> {
        if !self.reasoning {
            return None;
        }
        let level = reasoning?;
        if level == ThinkingLevel::Off {
            return None;
        }
        Some(level)
    }
}

/// Lightweight model identifier — used inside AgentState / AgentLoopConfig
/// when only id+provider+api+context-window are needed (e.g. message attribution).
/// Full `Model` carries pricing and metadata.
#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub api: Api,
    pub provider: String,
    pub base_url: String,
    pub reasoning: bool,
    pub input: Vec<String>,
    pub context_window: u64,
    pub max_tokens: u64,
}

impl From<&Model> for ModelInfo {
    fn from(m: &Model) -> Self {
        Self {
            id: m.id.clone(),
            name: m.name.clone(),
            api: m.api,
            provider: m.provider.clone(),
            base_url: m.base_url.clone(),
            reasoning: m.reasoning,
            input: m.input.clone(),
            context_window: m.context_window,
            max_tokens: m.max_tokens,
        }
    }
}

impl From<Model> for ModelInfo {
    fn from(m: Model) -> Self {
        Self::from(&m)
    }
}

// ============================================================================
// Content Blocks — single canonical representation
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
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
        /// Tool args are open by design — each tool defines its own JSON Schema.
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
        /// Tool-specific extra details (each tool defines its own struct).
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<serde_json::Value>,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
}

// `ContentPart` is the alias used inside llm-client and provider crates.
pub type ContentPart = ContentBlock;

// ============================================================================
// MessageContent — string OR structured blocks
// ============================================================================

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

impl MessageContent {
    pub fn into_blocks(self) -> Vec<ContentBlock> {
        match self {
            Self::String(s) => vec![ContentBlock::Text { text: s }],
            Self::Blocks(b) => b,
        }
    }
    pub fn as_blocks(&self) -> Vec<ContentBlock> {
        self.clone().into_blocks()
    }
}

// ============================================================================
// AgentMessage — typed enum covering both protocol and custom variants.
// Replaces the previous `serde_json::Value` alias and the parallel `LlmMessage`.
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum AgentMessage {
    /// User-supplied input.
    #[serde(rename = "user")]
    User {
        content: MessageContent,
        #[serde(default)]
        timestamp: u64,
        /// Caller-attached metadata (e.g. trace ids); shape is application-defined.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<serde_json::Value>,
    },

    /// Streamed assistant response.
    #[serde(rename = "assistant")]
    Assistant {
        content: Vec<ContentBlock>,
        api: Api,
        provider: String,
        model: String,
        usage: Usage,
        #[serde(rename = "stopReason")]
        stop_reason: StopReason,
        #[serde(
            rename = "errorMessage",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        error_message: Option<String>,
        #[serde(default)]
        timestamp: u64,
    },

    /// Result of a tool call (separate role).
    #[serde(rename = "toolResult")]
    ToolResult {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        content: Vec<ContentBlock>,
        /// Tool-specific extra details (each tool defines its own struct).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<serde_json::Value>,
        #[serde(rename = "isError", default)]
        is_error: bool,
        #[serde(default)]
        timestamp: u64,
    },

    /// Custom: arbitrary user-tagged message that converts to user content.
    #[serde(rename = "custom")]
    Custom {
        /// Plugin-defined message kind (extensible — open string by design).
        #[serde(rename = "customType")]
        custom_type: String,
        /// Plugin-defined payload — shape is set by `custom_type`.
        content: serde_json::Value,
        #[serde(default)]
        display: bool,
        /// Plugin-defined extra metadata.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<serde_json::Value>,
        #[serde(default)]
        timestamp: u64,
    },

    /// Bash execution record (used by the coding-agent's bash extension).
    #[serde(rename = "bashExecution")]
    BashExecution {
        command: String,
        output: String,
        #[serde(rename = "exitCode")]
        exit_code: Option<i32>,
        #[serde(default)]
        cancelled: bool,
        #[serde(default)]
        truncated: bool,
        #[serde(
            rename = "fullOutputPath",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        full_output_path: Option<String>,
        #[serde(default)]
        timestamp: u64,
        #[serde(
            rename = "excludeFromContext",
            default,
            skip_serializing_if = "std::ops::Not::not"
        )]
        exclude_from_context: bool,
    },

    /// Branch summary recorded when navigating away from a branch.
    #[serde(rename = "branchSummary")]
    BranchSummary {
        summary: String,
        #[serde(rename = "fromId")]
        from_id: String,
        #[serde(default)]
        timestamp: u64,
    },

    /// Compaction summary inserted at the front of compacted context.
    #[serde(rename = "compactionSummary")]
    CompactionSummary {
        summary: String,
        #[serde(rename = "tokensBefore", default)]
        tokens_before: u64,
        #[serde(default)]
        timestamp: u64,
    },
}

impl AgentMessage {
    pub fn role(&self) -> &'static str {
        match self {
            Self::User { .. } => "user",
            Self::Assistant { .. } => "assistant",
            Self::ToolResult { .. } => "toolResult",
            Self::Custom { .. } => "custom",
            Self::BashExecution { .. } => "bashExecution",
            Self::BranchSummary { .. } => "branchSummary",
            Self::CompactionSummary { .. } => "compactionSummary",
        }
    }

    pub fn user_text(text: impl Into<String>) -> Self {
        Self::User {
            content: MessageContent::Blocks(vec![ContentBlock::Text { text: text.into() }]),
            timestamp: now_ts(),
            metadata: None,
        }
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::Assistant {
            content: vec![ContentBlock::Text { text: text.into() }],
            api: Api::Anthropic,
            provider: "unknown".into(),
            model: "unknown".into(),
            usage: Usage::default(),
            stop_reason: StopReason::EndTurn,
            error_message: None,
            timestamp: now_ts(),
        }
    }

    /// Returns the assistant content array if this is an assistant message.
    pub fn assistant_content(&self) -> Option<&[ContentBlock]> {
        if let Self::Assistant { content, .. } = self {
            Some(content)
        } else {
            None
        }
    }

    /// Returns the assistant `errorMessage` field if set.
    pub fn assistant_error_message(&self) -> Option<&str> {
        if let Self::Assistant { error_message, .. } = self {
            error_message.as_deref()
        } else {
            None
        }
    }

    /// Returns the assistant `stopReason` if this is an assistant message.
    pub fn stop_reason(&self) -> Option<StopReason> {
        if let Self::Assistant { stop_reason, .. } = self {
            Some(*stop_reason)
        } else {
            None
        }
    }

    /// Wire-format stop reason (`"stop"`, `"error"`, ...) for legacy callers.
    pub fn stop_reason_str(&self) -> Option<&'static str> {
        self.stop_reason().map(|r| r.as_wire_str())
    }

    /// Convert to a JSON value; matches the previous wire format used by
    /// session storage and the streaming bridge.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    /// Parse from JSON. Returns `None` on unknown roles or malformed shape.
    pub fn from_json(value: serde_json::Value) -> Option<Self> {
        serde_json::from_value(value).ok()
    }
}

fn now_ts() -> u64 {
    chrono::Utc::now().timestamp_millis().max(0) as u64
}

/// Alias for `AgentMessage` used by llm-client and provider crates.
pub type LlmMessage = AgentMessage;
pub type Message = AgentMessage;

// ============================================================================
// Tool Definition
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema describing the tool's parameters — must accept any shape.
    pub input_schema: serde_json::Value,
}

// ============================================================================
// LlmRequest / LlmResponse — used by llm-client providers.
// ============================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LlmRequest {
    pub model: String,
    pub messages: Vec<AgentMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ThinkingLevel>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub openai_response_includes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_retention: Option<CacheRetention>,
    /// Anthropic adaptive-thinking budgets keyed by `ThinkingLevel`. Only
    /// consumed when the resolved thinking mode is "budget" (older Claude 4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_budgets: Option<ThinkingBudgets>,
    /// Provider-specific overrides (e.g. Anthropic's `forceAdaptiveThinking`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_options: Option<ProviderOptions>,
}

impl LlmRequest {
    /// Return the Anthropic-specific options block, if the request carries one.
    pub fn anthropic_options(&self) -> Option<&AnthropicOptions> {
        match &self.provider_options {
            Some(ProviderOptions::Anthropic(o)) => Some(o),
            _ => None,
        }
    }

    /// Return the OpenAI-specific options block, if the request carries one.
    pub fn openai_options(&self) -> Option<&OpenaiOptions> {
        match &self.provider_options {
            Some(ProviderOptions::Openai(o)) => Some(o),
            _ => None,
        }
    }
}

/// Provider-specific overrides carried alongside `LlmRequest`. The `provider`
/// tag discriminates which subset of knobs is in effect; mismatched
/// combinations (e.g. setting Anthropic options on an OpenAI request) are
/// silently ignored by the provider that doesn't recognize them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "lowercase")]
pub enum ProviderOptions {
    Anthropic(AnthropicOptions),
    Openai(OpenaiOptions),
}

/// Provider-specific knobs Anthropic providers respect. Add slots as new
/// override needs emerge.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnthropicOptions {
    /// Force adaptive thinking (`thinking.type = "adaptive"`) regardless of
    /// model id. `None` falls back to substring matching on the model id.
    /// Set explicitly for custom Anthropic-compatible aliases (Bedrock,
    /// Cloudflare AI Gateway, Fireworks, etc.).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "forceAdaptiveThinking"
    )]
    pub force_adaptive_thinking: Option<bool>,
    /// Whether the model accepts the `temperature` field. `None` falls back to
    /// substring matching on the model id. Claude Opus 4.7+ set this `false`.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "supportsTemperature"
    )]
    pub supports_temperature: Option<bool>,
}

/// Provider-specific knobs OpenAI providers respect. Reserved for future
/// use; empty for now.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpenaiOptions {
    /// Use Anthropic-style `cache_control: {type: "ephemeral"}` inline cache
    /// hints instead of the native `prompt_cache_retention` parameter. Set
    /// for OpenAI-compatible endpoints that proxy to Anthropic (DeepSeek,
    /// Cloudflare Workers AI, Together AI, etc.).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "anthropicCacheControl"
    )]
    pub anthropic_cache_control: Option<bool>,
}

impl LlmRequest {
    pub fn new(model: impl Into<String>, messages: Vec<AgentMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub id: String,
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

// ============================================================================
// SimpleStreamOptions / Context
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct SimpleStreamOptions {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub reasoning: Option<ThinkingLevel>,
    pub api_key: Option<String>,
    pub transport: Option<Transport>,
    pub cache_retention: Option<CacheRetention>,
    pub session_id: Option<String>,
    pub headers: Option<HashMap<String, String>>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub max_retry_delay_ms: Option<u64>,
}

/// Context passed into the low-level agent loop. Carries the system prompt,
/// the message transcript, and the tool set in scope.
#[derive(Clone)]
pub struct AgentContext {
    pub system_prompt: String,
    pub messages: Vec<AgentMessage>,
    pub tools: Vec<Arc<dyn AgentTool>>,
}

impl std::fmt::Debug for AgentContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentContext")
            .field("system_prompt_len", &self.system_prompt.len())
            .field("messages_count", &self.messages.len())
            .field("tool_count", &self.tools.len())
            .finish()
    }
}

// ============================================================================
// Tool Result / Tool Call structures (used by the agent loop)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolResult<T = serde_json::Value> {
    pub content: Vec<ContentBlock>,
    /// Tool-specific metadata (each tool serializes its own typed `details`
    /// struct — `EditToolDetails`, `BashToolDetails`, etc.).
    pub details: T,
    pub terminate: bool,
}

impl<T: Default> AgentToolResult<T> {
    pub fn error_text(message: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text { text: message.into() }],
            details: T::default(),
            terminate: false,
        }
    }
}

pub type AgentToolUpdateCallback<T = serde_json::Value> =
    Box<dyn Fn(AgentToolResult<T>) + Send + Sync>;

#[derive(Debug, Clone)]
pub struct AgentToolCall {
    pub id: String,
    pub name: String,
    /// Tool-defined argument shape — each tool validates its own JSON Schema.
    pub arguments: serde_json::Value,
}

// ============================================================================
// Hook contexts — passed into before/after tool-call hooks and the
// shouldStopAfterTurn / prepareNextTurn hooks.
// ============================================================================

#[derive(Clone)]
pub struct BeforeToolCallContext {
    pub assistant_message: AgentMessage,
    pub tool_call: AgentToolCall,
    /// Validated tool arguments (shape defined by the tool).
    pub args: serde_json::Value,
    pub context: AgentContext,
}

#[derive(Debug, Clone, Default)]
pub struct BeforeToolCallResult {
    pub block: bool,
    pub reason: Option<String>,
}

#[derive(Clone)]
pub struct AfterToolCallContext {
    pub assistant_message: AgentMessage,
    pub tool_call: AgentToolCall,
    /// Validated tool arguments (shape defined by the tool).
    pub args: serde_json::Value,
    pub result: AgentToolResult,
    pub is_error: bool,
    pub context: AgentContext,
}

#[derive(Debug, Clone, Default)]
pub struct AfterToolCallResult {
    pub content: Option<Vec<ContentBlock>>,
    /// Tool-specific override for the result's `details` field.
    pub details: Option<serde_json::Value>,
    pub is_error: Option<bool>,
    pub terminate: Option<bool>,
}

#[derive(Clone)]
pub struct ShouldStopAfterTurnContext {
    pub assistant_message: AgentMessage,
    pub tool_results: Vec<AgentMessage>,
    pub context: AgentContext,
    pub has_more_tool_calls: bool,
}

#[derive(Clone)]
pub struct PrepareNextTurnContext {
    pub last_assistant_message: AgentMessage,
    pub tool_results: Vec<AgentMessage>,
    pub context: AgentContext,
}

/// Config snapshot returned by [`PrepareNextTurnFn`]. Any `Some` field
/// overrides the loop's state for the next turn; `None` keeps the current
/// value. `messages` are injected (as pending) before the next assistant
/// response. Mirrors pi-mono's `AgentLoopTurnUpdate`.
#[derive(Clone, Default)]
pub struct TurnUpdate {
    /// Replace the working context (messages + tools + system prompt).
    pub context: Option<AgentContext>,
    /// Switch the model for subsequent turns.
    pub model: Option<ModelInfo>,
    /// Switch the reasoning level. `Some(ThinkingLevel::Off)` disables it.
    pub thinking_level: Option<ThinkingLevel>,
    /// Extra messages to inject before the next assistant response.
    pub messages: Vec<AgentMessage>,
}

// ============================================================================
// Hook type aliases — small named closure types for AgentLoopConfig.
// ============================================================================

pub type ConvertToLlmFn = Arc<
    dyn Fn(Vec<AgentMessage>) -> Pin<Box<dyn Future<Output = Vec<AgentMessage>> + Send>>
        + Send
        + Sync,
>;
pub type TransformContextFn = Arc<
    dyn Fn(
            Vec<AgentMessage>,
            Option<CancellationToken>,
        ) -> Pin<Box<dyn Future<Output = Vec<AgentMessage>> + Send>>
        + Send
        + Sync,
>;
pub type GetApiKeyFn =
    Arc<dyn Fn(String) -> Pin<Box<dyn Future<Output = Option<String>> + Send>> + Send + Sync>;
pub type GetMessagesFn =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Vec<AgentMessage>> + Send>> + Send + Sync>;
pub type BeforeToolCallFn = Arc<
    dyn Fn(
            BeforeToolCallContext,
            Option<CancellationToken>,
        ) -> Pin<Box<dyn Future<Output = Option<BeforeToolCallResult>> + Send>>
        + Send
        + Sync,
>;
pub type AfterToolCallFn = Arc<
    dyn Fn(
            AfterToolCallContext,
            Option<CancellationToken>,
        ) -> Pin<Box<dyn Future<Output = Option<AfterToolCallResult>> + Send>>
        + Send
        + Sync,
>;
pub type ShouldStopAfterTurnFn = Arc<
    dyn Fn(
            ShouldStopAfterTurnContext,
            Option<CancellationToken>,
        ) -> Pin<Box<dyn Future<Output = bool> + Send>>
        + Send
        + Sync,
>;
pub type PrepareNextTurnFn = Arc<
    dyn Fn(
            PrepareNextTurnContext,
            Option<CancellationToken>,
        ) -> Pin<Box<dyn Future<Output = Option<TurnUpdate>> + Send>>
        + Send
        + Sync,
>;
pub type OnPayloadFn =
    Arc<dyn Fn(serde_json::Value) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;
pub type OnResponseFn =
    Arc<dyn Fn(serde_json::Value) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

// ============================================================================
// AgentLoopConfig
// ============================================================================

#[derive(Clone)]
pub struct AgentLoopConfig {
    pub model: ModelInfo,
    pub convert_to_llm: ConvertToLlmFn,
    pub transform_context: Option<TransformContextFn>,
    pub get_api_key: Option<GetApiKeyFn>,
    pub api_key: Option<String>,

    pub get_steering_messages: Option<GetMessagesFn>,
    pub get_follow_up_messages: Option<GetMessagesFn>,
    pub tool_execution: ToolExecutionMode,

    pub before_tool_call: Option<BeforeToolCallFn>,
    pub after_tool_call: Option<AfterToolCallFn>,
    pub should_stop_after_turn: Option<ShouldStopAfterTurnFn>,
    pub prepare_next_turn: Option<PrepareNextTurnFn>,
    pub on_payload: Option<OnPayloadFn>,
    pub on_response: Option<OnResponseFn>,

    pub session_id: Option<String>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub transport: Transport,
    pub cache_retention: Option<CacheRetention>,
    pub max_retry_delay_ms: Option<u64>,
    pub timeout_ms: Option<u64>,
    pub max_retries: Option<u32>,
    pub headers: Option<HashMap<String, String>>,
    /// Caller-defined extra metadata forwarded to the provider.
    pub metadata: Option<serde_json::Value>,
    pub reasoning: Option<ThinkingLevel>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    /// Provider-specific overrides forwarded to `LlmRequest.provider_options`.
    pub provider_options: Option<ProviderOptions>,
}

impl AgentLoopConfig {
    pub fn new(model: ModelInfo, convert_to_llm: ConvertToLlmFn) -> Self {
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
            should_stop_after_turn: None,
            prepare_next_turn: None,
            on_payload: None,
            on_response: None,
            session_id: None,
            thinking_budgets: None,
            transport: Transport::Sse,
            cache_retention: None,
            max_retry_delay_ms: None,
            timeout_ms: None,
            max_retries: None,
            headers: None,
            metadata: None,
            reasoning: None,
            temperature: None,
            max_tokens: None,
            provider_options: None,
        }
    }
}

impl std::fmt::Debug for AgentLoopConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentLoopConfig")
            .field("model", &self.model.id)
            .field("tool_execution", &self.tool_execution)
            .field("transport", &self.transport)
            .field("session_id", &self.session_id)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// AgentState
// ============================================================================

pub struct AgentState {
    system_prompt: String,
    model: ModelInfo,
    thinking_level: ThinkingLevel,
    tools: Vec<Arc<dyn AgentTool>>,
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
        tools: Option<Vec<Arc<dyn AgentTool>>>,
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

    pub fn tools(&self) -> Vec<Arc<dyn AgentTool>> {
        self.tools.clone()
    }
    pub fn set_tools(&mut self, tools: Vec<Arc<dyn AgentTool>>) {
        self.tools = tools;
    }

    pub fn messages(&self) -> Vec<AgentMessage> {
        self.messages.clone()
    }
    pub fn set_messages(&mut self, messages: Vec<AgentMessage>) {
        self.messages = messages;
    }
    pub fn push_message(&mut self, message: AgentMessage) {
        self.messages.push(message);
    }

    /// Fold an [`crate::event::AgentEvent`] into the running state. Mirrors
    /// pi-mono's `Agent.processEvents`: tracks the streaming message, appends
    /// finalized messages, maintains the pending-tool-call set, and captures
    /// the last turn's error message.
    pub fn apply_event(&mut self, event: &crate::event::AgentEvent) {
        use crate::event::AgentEvent;
        match event {
            AgentEvent::MessageStart { message } => {
                self.is_streaming = true;
                self.streaming_message = Some(message.clone());
            }
            AgentEvent::MessageUpdate { partial, .. } => {
                self.is_streaming = true;
                self.streaming_message = Some(partial.clone().into_finalized());
            }
            AgentEvent::MessageEnd { message } => {
                self.is_streaming = false;
                self.streaming_message = None;
                self.messages.push(message.clone());
            }
            AgentEvent::ToolExecutionStart { tool_call_id, .. } => {
                self.pending_tool_calls.insert(tool_call_id.clone());
            }
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
                self.pending_tool_calls.remove(tool_call_id);
            }
            AgentEvent::TurnEnd { message, .. } => {
                if let Some(err) = message.assistant_error_message() {
                    self.error_message = Some(err.to_string());
                }
            }
            AgentEvent::AgentEnd { .. } => {
                self.is_streaming = false;
                self.streaming_message = None;
            }
            _ => {}
        }
    }

    /// Clear transcript, streaming/runtime state, and error. Queue clearing is
    /// the `Agent`'s responsibility (it owns the queues). Mirrors pi-mono
    /// `Agent.reset`.
    pub fn reset_runtime(&mut self) {
        self.messages.clear();
        self.is_streaming = false;
        self.streaming_message = None;
        self.pending_tool_calls.clear();
        self.error_message = None;
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
        assert_eq!(serde_json::to_string(&ThinkingLevel::XHigh).unwrap(), r#""xhigh""#);
        let deser: ThinkingLevel = serde_json::from_str(r#""off""#).unwrap();
        assert_eq!(deser, ThinkingLevel::Off);
    }

    #[test]
    fn test_tool_execution_mode_serde() {
        assert_eq!(
            serde_json::to_string(&ToolExecutionMode::Sequential).unwrap(),
            r#""sequential""#
        );
        assert_eq!(serde_json::to_string(&ToolExecutionMode::Parallel).unwrap(), r#""parallel""#);
    }

    #[test]
    fn test_agent_message_user_text_roundtrip() {
        let msg = AgentMessage::user_text("hello");
        let json = msg.to_json();
        assert_eq!(json["role"], "user");
        let back = AgentMessage::from_json(json).unwrap();
        assert_eq!(back.role(), "user");
    }

    #[test]
    fn test_agent_message_tool_result_serde() {
        let msg = AgentMessage::ToolResult {
            tool_call_id: "tc1".into(),
            tool_name: "bash".into(),
            content: vec![ContentBlock::Text { text: "out".into() }],
            details: None,
            is_error: false,
            timestamp: 0,
        };
        let json = msg.to_json();
        assert_eq!(json["role"], "toolResult");
        assert_eq!(json["toolCallId"], "tc1");
    }

    #[test]
    fn test_agent_message_custom_serde() {
        let msg = AgentMessage::Custom {
            custom_type: "artifact".into(),
            content: serde_json::json!("hi"),
            display: true,
            details: None,
            timestamp: 1,
        };
        let json = msg.to_json();
        assert_eq!(json["role"], "custom");
        assert_eq!(json["customType"], "artifact");
        let back = AgentMessage::from_json(json).unwrap();
        assert_eq!(back.role(), "custom");
    }

    #[test]
    fn test_cap_output_budget_within_limits() {
        let m = Model {
            reasoning: true,
            max_tokens: 16384,
            context_window: 200000,
            ..Default::default()
        };
        assert_eq!(m.cap_output_budget(Some(4000), 1000), 4000);
    }

    #[test]
    fn test_clamp_reasoning_no_support() {
        let m = Model::default();
        assert_eq!(m.clamp_reasoning(Some(ThinkingLevel::High)), None);
    }

    #[test]
    fn test_stop_reason_round_trip() {
        for r in [
            StopReason::EndTurn,
            StopReason::MaxTokens,
            StopReason::StopSequence,
            StopReason::ToolUse,
            StopReason::ContentFilter,
            StopReason::Error,
            StopReason::Aborted,
        ] {
            assert_eq!(StopReason::from_wire_str(r.as_wire_str()), Some(r));
        }
        // accept legacy aliases too
        assert_eq!(StopReason::from_wire_str("end_turn"), Some(StopReason::EndTurn));
        assert_eq!(StopReason::from_wire_str("tool_use"), Some(StopReason::ToolUse));
        assert_eq!(StopReason::from_wire_str("max_tokens"), Some(StopReason::MaxTokens));
        assert_eq!(StopReason::from_wire_str("garbage"), None);
    }

    #[test]
    fn test_thinking_level_round_trip() {
        for l in [
            ThinkingLevel::Off,
            ThinkingLevel::Minimal,
            ThinkingLevel::Low,
            ThinkingLevel::Medium,
            ThinkingLevel::High,
            ThinkingLevel::XHigh,
        ] {
            assert_eq!(ThinkingLevel::from_wire_str(l.as_str()), Some(l));
        }
        assert_eq!(ThinkingLevel::from_wire_str("garbage"), None);
    }

    #[test]
    fn test_assistant_message_stop_reason_jsonl_compat() {
        // Wire format unchanged: stopReason: "stop" round-trips to EndTurn.
        let json = r#"{"role":"assistant","content":[{"type":"text","text":"hi"}],"api":"anthropic","provider":"p","model":"m","usage":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},"stopReason":"stop","timestamp":0}"#;
        let m: AgentMessage = serde_json::from_str(json).unwrap();
        match m {
            AgentMessage::Assistant { stop_reason, .. } => {
                assert_eq!(stop_reason, StopReason::EndTurn)
            }
            _ => panic!("expected Assistant"),
        }
    }

    #[test]
    fn test_api_round_trip() {
        for api in [Api::Anthropic, Api::Openai, Api::OpenaiResponses, Api::Mock] {
            assert_eq!(Api::from_wire_str(api.as_wire_str()), Some(api));
            let v = serde_json::to_value(api).unwrap();
            assert_eq!(v.as_str().unwrap(), api.as_wire_str());
        }
        assert_eq!(Api::from_wire_str("garbage"), None);
    }

    #[test]
    fn test_provider_options_anthropic_serde() {
        let opts = ProviderOptions::Anthropic(AnthropicOptions {
            force_adaptive_thinking: Some(true),
            ..Default::default()
        });
        let v = serde_json::to_value(&opts).unwrap();
        assert_eq!(v["provider"], "anthropic");
        assert_eq!(v["forceAdaptiveThinking"], true);
        let round: ProviderOptions = serde_json::from_value(v).unwrap();
        match round {
            ProviderOptions::Anthropic(o) => assert_eq!(o.force_adaptive_thinking, Some(true)),
            _ => panic!("expected anthropic"),
        }
    }

    #[test]
    fn test_provider_options_openai_serde() {
        let opts = ProviderOptions::Openai(OpenaiOptions::default());
        let v = serde_json::to_value(&opts).unwrap();
        assert_eq!(v["provider"], "openai");
    }

    #[test]
    fn test_llm_request_anthropic_options_helper() {
        let r = LlmRequest {
            provider_options: Some(ProviderOptions::Anthropic(AnthropicOptions {
                force_adaptive_thinking: Some(true),
                ..Default::default()
            })),
            ..Default::default()
        };
        assert!(r.anthropic_options().is_some());
        assert_eq!(r.anthropic_options().unwrap().force_adaptive_thinking, Some(true));
        assert!(r.openai_options().is_none());
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
        assert_eq!(a.total_tokens, 450);
    }
}
