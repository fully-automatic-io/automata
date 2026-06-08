pub mod provider;
pub mod retry;
pub mod streaming;
pub mod types;

pub mod providers {
    pub mod anthropic;
    pub mod openai;
    pub mod openai_responses;
}

// Auto-retry / overflow detection live in agent-core (they need AgentMessage
// types and the harness uses them). Re-export so existing call sites work.
pub use agent_core::auto_retry::{RetrySettings, compute_retry_delay, is_retryable_error};
pub use agent_core::overflow::is_context_overflow;
pub use provider::{AuthMethod, LlmError, LlmProvider, LlmStream, ProviderConfig};
pub use providers::anthropic::AnthropicProvider;
pub use providers::openai::OpenAIProvider;
pub use retry::{format_http_error, retry_delay, should_retry};
pub use streaming::{AssistantMessageEvent, LlmEvent};
pub use types::{
    CacheRetention, ContentPart, Context, Cost, LlmMessage, LlmRequest, LlmResponse,
    MessageContent, Model, ModelCost, SimpleStreamOptions, StopReason, ThinkingBudgets,
    ToolDefinition, Transport, Usage, UsageCost,
};
