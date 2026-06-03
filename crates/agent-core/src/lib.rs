// Agent Core — foundational types and the low-level agent loop.

pub mod agent;
pub mod agent_loop;
pub mod auto_retry;
pub mod event;
pub mod harness;
pub mod overflow;
pub mod proxy;
pub mod queue;
pub mod tool;
pub mod types;

// ============================================================================
// Re-exports
// ============================================================================

pub use agent::{Agent, AgentOptions, AgentSnapshot, PromptInput};
pub use agent_loop::{
    AgentEventSink, AgentLoop, AssistantMessageEventStream, StreamFn, StreamFnInput,
    tools_to_definitions,
};
pub use auto_retry::{RetrySettings, compute_retry_delay, is_retryable_error};
pub use event::{
    AgentEvent, AgentEventChannel, AgentEventListener, AgentEventReceiver, AssistantMessageEvent,
    EventStream, FnEventListener,
};
pub use harness::messages::{convert_to_llm, default_convert_to_llm};
pub use overflow::is_context_overflow;
pub use queue::{MessageQueue, QueueMode};
pub use tool::{
    AgentTool, ToolDefinitionWrapper, ToolRegistry, create_error_tool_result,
    create_success_tool_result, downcast_details, validate_tool_arguments,
};
pub use types::{
    AfterToolCallContext, AfterToolCallFn, AfterToolCallResult, AgentContext, AgentLoopConfig,
    AgentMessage, AgentState, AgentToolCall, AgentToolResult, AgentToolUpdateCallback,
    AnthropicOptions, Api, BeforeToolCallContext, BeforeToolCallFn, BeforeToolCallResult,
    CacheRetention, ContentBlock, ContentPart, ConvertToLlmFn, Cost, GetApiKeyFn, GetMessagesFn,
    LlmMessage, LlmRequest, LlmResponse, Message, MessageContent, Model, ModelCompat, ModelCost,
    ModelInfo, OnPayloadFn, OnResponseFn, OpenaiOptions, PrepareNextTurnContext, PrepareNextTurnFn,
    ProviderOptions, ShouldStopAfterTurnContext, ShouldStopAfterTurnFn, SimpleStreamOptions,
    StopReason, ThinkingBudgets, ThinkingLevel, ToolDefinition, ToolExecutionMode,
    TransformContextFn, Transport, TurnUpdate, Usage, UsageCost,
};
