// Type re-exports — canonical definitions live in agent-core.
//
// llm-client depends on agent-core and re-exports the protocol types so
// existing call sites continue to compile unchanged.

pub use agent_core::types::{
    AnthropicOptions, Api, CacheRetention, ContentBlock, ContentPart, LlmMessage, LlmRequest,
    LlmResponse, MessageContent, Model, ModelCost, OpenaiOptions, ProviderOptions,
    SimpleStreamOptions, StopReason, ThinkingBudgets, ThinkingLevel, ToolDefinition, Transport,
    Usage, UsageCost,
};

/// `Context` mirrors `AgentContext` minus the concrete `Vec<Arc<dyn AgentTool>>`
/// shape — providers want a flat tool-definition list, not the trait object.
/// Kept as a thin local struct so provider code doesn't pull in `AgentTool`.
#[derive(Debug, Clone)]
pub struct Context {
    pub system_prompt: String,
    pub messages: Vec<LlmMessage>,
    pub tools: Vec<ToolDefinition>,
}
