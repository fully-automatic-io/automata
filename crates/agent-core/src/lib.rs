// Agent Core - Core agent framework
//
// This crate provides the foundational types and traits for building AI agents.

pub mod agent_loop;
pub mod event;
pub mod hooks;
pub mod message;
pub mod proxy;
pub mod queue;
pub mod tool;
pub mod types;

// Re-export agent module (legacy compatibility wrapper)
pub mod agent;
pub use agent::Agent;

pub use agent_loop::{run_agent_loop, run_agent_loop_continue, AgentEventSink};
pub use event::{
    AgentEvent, AgentEventListener, AgentEventReceiver, AgentEventChannel,
    AssistantMessageEvent, EventStream, FnEventListener,
};
pub use hooks::{
    AfterToolCallHook, BeforeToolCallHook, ConvertToLlmHook, DefaultConvertToLlmHook,
    HookRegistry, TransformContextHook, default_convert_to_llm,
};
pub use message::{AgentMessage as AgentMessageTrait, MessageRole, StandardMessage};
pub use queue::{MessageQueue, QueueMode};
pub use tool::{AgentTool, ToolRegistry, create_error_tool_result, validate_tool_arguments};
pub use types::{
    AgentContext, AgentLoopConfig, AgentMessage, AgentState, AgentToolCall, AgentToolResult,
    AgentToolUpdateCallback, AfterToolCallContext, AfterToolCallResult,
    BeforeToolCallContext, BeforeToolCallResult, ContentBlock, Cost, Message,
    MessageContent, ModelInfo, StopReason, ThinkingBudgets, ThinkingLevel,
    ToolExecutionMode, Transport, Usage,
};
