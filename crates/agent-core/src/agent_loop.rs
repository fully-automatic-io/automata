
use crate::event::{AgentEvent, AssistantMessageEvent, EventStream};
use crate::tool::{AgentTool, validate_tool_arguments};
use crate::types::{
    AfterToolCallContext, AgentContext, AgentLoopConfig, AgentMessage, AgentToolCall,
    AgentToolResult, AgentToolUpdateCallback, Message,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

// ============================================================================
// Types
// ============================================================================

pub type AgentEventSink =
    Arc<dyn Fn(AgentEvent) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

pub type StreamFn = Arc<
    dyn Fn(
            StreamFnInput,
        )
            -> Pin<Box<dyn Future<Output = Result<AssistantMessageEventStream, String>> + Send>>
        + Send
        + Sync,
>;

pub type AssistantMessageEventStream = EventStream<AssistantMessageEvent, AgentMessage>;

#[derive(Clone)]
pub struct StreamFnInput {
    pub model: crate::types::ModelInfo,
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub tools: Vec<Arc<dyn AgentTool>>,
    pub api_key: Option<String>,
    pub signal: Option<CancellationToken>,
    pub session_id: Option<String>,
    pub thinking_budgets: Option<crate::types::ThinkingBudgets>,
    pub transport: crate::types::Transport,
    pub max_retry_delay_ms: Option<u64>,
    pub reasoning: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

// ============================================================================
// Preparation types
// ============================================================================

enum PreparedToolCall {
    Ready {
        tool_call: AgentToolCall,
        tool: Arc<dyn AgentTool>,
        args: serde_json::Value,
    },
    Immediate {
        tool_call: AgentToolCall,
        result: AgentToolResult,
        is_error: bool,
    },
}

#[derive(Clone)]
struct FinalizedToolCall {
    tool_call: AgentToolCall,
    result: AgentToolResult,
    is_error: bool,
}

struct ExecutedToolCall {
    result: AgentToolResult,
    is_error: bool,
}

// ============================================================================
// Public API
// ============================================================================

pub async fn run_agent_loop(
    prompts: Vec<AgentMessage>,
    context: AgentContext,
    config: &AgentLoopConfig,
    tools: &[Arc<dyn AgentTool>],
    emit: &AgentEventSink,
    signal: Option<CancellationToken>,
    stream_fn: &StreamFn,
) -> Vec<AgentMessage> {
    let mut new_messages: Vec<AgentMessage> = prompts.clone();
    let mut current_context = AgentContext {
        system_prompt: context.system_prompt.clone(),
        messages: {
            let mut msgs = context.messages;
            msgs.extend(prompts.clone());
            msgs
        },
        tools: context.tools.clone(),
    };

    emit(AgentEvent::AgentStart).await;
    emit(AgentEvent::TurnStart).await;
    for prompt in &prompts {
        emit(AgentEvent::MessageStart { message: prompt.clone() }).await;
        emit(AgentEvent::MessageEnd { message: prompt.clone() }).await;
    }

    run_loop(&mut current_context, &mut new_messages, config, tools, emit, signal, stream_fn).await;
    new_messages
}

pub async fn run_agent_loop_continue(
    context: AgentContext,
    config: &AgentLoopConfig,
    tools: &[Arc<dyn AgentTool>],
    emit: &AgentEventSink,
    signal: Option<CancellationToken>,
    stream_fn: &StreamFn,
) -> Vec<AgentMessage> {
    if context.messages.is_empty() {
        panic!("Cannot continue: no messages in context");
    }
    let last_role = context
        .messages
        .last()
        .and_then(|m| m.get("role"))
        .and_then(|r| r.as_str())
        .unwrap_or("");
    if last_role == "assistant" {
        panic!("Cannot continue from message role: assistant");
    }

    let mut new_messages: Vec<AgentMessage> = vec![];
    let mut current_context = context.clone();

    emit(AgentEvent::AgentStart).await;
    emit(AgentEvent::TurnStart).await;
    run_loop(&mut current_context, &mut new_messages, config, tools, emit, signal, stream_fn).await;
    new_messages
}

// ============================================================================
// Main loop
// ============================================================================

async fn run_loop(
    current_context: &mut AgentContext,
    new_messages: &mut Vec<AgentMessage>,
    config: &AgentLoopConfig,
    tools: &[Arc<dyn AgentTool>],
    emit: &AgentEventSink,
    signal: Option<CancellationToken>,
    stream_fn: &StreamFn,
) {
    let mut first_turn = true;
    let mut pending_messages: Vec<AgentMessage> = get_steering_messages(config, true).await;

    loop {
        let mut has_more_tool_calls = true;

        while has_more_tool_calls || !pending_messages.is_empty() {
            if let Some(ref token) = signal {
                if token.is_cancelled() {
                    emit(AgentEvent::AgentEnd { messages: new_messages.clone() }).await;
                    return;
                }
            }

            if !first_turn {
                emit(AgentEvent::TurnStart).await;
            } else {
                first_turn = false;
            }

            if !pending_messages.is_empty() {
                for message in &pending_messages {
                    emit(AgentEvent::MessageStart { message: message.clone() }).await;
                    emit(AgentEvent::MessageEnd { message: message.clone() }).await;
                    current_context.messages.push(message.clone());
                    new_messages.push(message.clone());
                }
                pending_messages.clear();
            }

            let message = stream_assistant_response(
                current_context,
                config,
                tools,
                emit,
                signal.clone(),
                stream_fn,
            )
            .await;
            new_messages.push(message.clone());

            let stop_reason = message.get("stopReason").and_then(|s| s.as_str()).unwrap_or("stop");

            if stop_reason == "error" || stop_reason == "aborted" {
                emit(AgentEvent::TurnEnd { message, tool_results: vec![] }).await;
                emit(AgentEvent::AgentEnd { messages: new_messages.clone() }).await;
                return;
            }

            let tool_calls = extract_tool_calls(&message);
            let mut tool_results: Vec<AgentMessage> = vec![];
            has_more_tool_calls = false;

            if !tool_calls.is_empty() {
                let batch = execute_tool_calls(
                    current_context,
                    &tool_calls,
                    config,
                    tools,
                    emit,
                    signal.clone(),
                )
                .await;
                for result in &batch.messages {
                    current_context.messages.push(result.clone());
                    new_messages.push(result.clone());
                }
                tool_results = batch.messages;
                has_more_tool_calls = !batch.terminate;
            }

            emit(AgentEvent::TurnEnd { message, tool_results }).await;
            pending_messages = get_steering_messages(config, false).await;
        }

        let follow_up = get_follow_up_messages(config).await;
        if !follow_up.is_empty() {
            pending_messages = follow_up;
            continue;
        }
        break;
    }

    emit(AgentEvent::AgentEnd { messages: new_messages.clone() }).await;
}

// ============================================================================
// Stream assistant response
// ============================================================================

async fn stream_assistant_response(
    context: &mut AgentContext,
    config: &AgentLoopConfig,
    tools: &[Arc<dyn AgentTool>],
    emit: &AgentEventSink,
    signal: Option<CancellationToken>,
    stream_fn: &StreamFn,
) -> AgentMessage {
    let mut messages = context.messages.clone();
    if let Some(ref transform_fn) = config.transform_context {
        messages = transform_fn(messages.clone(), signal.clone()).await;
    }

    let llm_messages = (config.convert_to_llm)(messages.clone()).await;

    let resolved_api_key = if let Some(ref get_key) = config.get_api_key {
        get_key(config.model.provider.clone()).await.or(config.api_key.clone())
    } else {
        config.api_key.clone()
    };

    let stream_input = StreamFnInput {
        model: config.model.clone(),
        system_prompt: context.system_prompt.clone(),
        messages: llm_messages,
        tools: tools.to_vec(),
        api_key: resolved_api_key,
        signal: signal.clone(),
        session_id: config.session_id.clone(),
        thinking_budgets: config.thinking_budgets.clone(),
        transport: config.transport,
        max_retry_delay_ms: config.max_retry_delay_ms,
        reasoning: config.reasoning.clone(),
        temperature: config.temperature,
        max_tokens: config.max_tokens,
    };

    let response = match (stream_fn)(stream_input).await {
        Ok(stream) => stream,
        Err(err) => {
            let failure = make_failure_message(&config.model, "error", Some(&err));
            context.messages.push(failure.clone());
            emit(AgentEvent::MessageStart { message: failure.clone() }).await;
            emit(AgentEvent::MessageEnd { message: failure.clone() }).await;
            return failure;
        }
    };

    let mut partial_message: Option<AgentMessage> = None;
    let mut added_partial = false;

    loop {
        let events = response.take_events();
        let is_done = response.is_complete();

        if events.is_empty() && is_done {
            let final_message =
                response.wait_for_result_try().ok().flatten().unwrap_or_else(|| {
                    make_failure_message(
                        &config.model,
                        "error",
                        Some("Stream ended without result"),
                    )
                });
            if added_partial {
                let last = context.messages.len().saturating_sub(1);
                if last < context.messages.len() {
                    context.messages[last] = final_message.clone();
                }
            } else {
                context.messages.push(final_message.clone());
                emit(AgentEvent::MessageStart { message: final_message.clone() }).await;
            }
            emit(AgentEvent::MessageEnd { message: final_message.clone() }).await;
            return final_message;
        }

        for event in events {
            match event {
                AssistantMessageEvent::Start { partial } => {
                    partial_message = Some(partial.clone());
                    context.messages.push(partial.clone());
                    added_partial = true;
                    emit(AgentEvent::MessageStart { message: partial }).await;
                }
                ref ev @ (AssistantMessageEvent::TextStart { .. }
                | AssistantMessageEvent::TextDelta { .. }
                | AssistantMessageEvent::TextEnd { .. }
                | AssistantMessageEvent::ThinkingStart { .. }
                | AssistantMessageEvent::ThinkingDelta { .. }
                | AssistantMessageEvent::ThinkingEnd { .. }
                | AssistantMessageEvent::ToolCallStart { .. }
                | AssistantMessageEvent::ToolCallDelta { .. }
                | AssistantMessageEvent::ToolCallEnd { .. }) => {
                    if let Some(ref _pm) = partial_message {
                        let updated = partial_from_event(ev);
                        partial_message = Some(updated.clone());
                        let last = context.messages.len().saturating_sub(1);
                        if last < context.messages.len() {
                            context.messages[last] = updated.clone();
                        }
                        emit(AgentEvent::MessageUpdate {
                            message: updated,
                            assistant_message_event: ev.clone(),
                        })
                        .await;
                    }
                }
                AssistantMessageEvent::Done { message, .. }
                | AssistantMessageEvent::Error { error: message, .. } => {
                    if added_partial {
                        let last = context.messages.len().saturating_sub(1);
                        if last < context.messages.len() {
                            context.messages[last] = message.clone();
                        }
                    } else {
                        context.messages.push(message.clone());
                    }
                    if !added_partial {
                        emit(AgentEvent::MessageStart { message: message.clone() }).await;
                    }
                    emit(AgentEvent::MessageEnd { message: message.clone() }).await;
                    return message;
                }
            }
        }
    }
}

fn partial_from_event(event: &AssistantMessageEvent) -> AgentMessage {
    match event {
        AssistantMessageEvent::Start { partial }
        | AssistantMessageEvent::TextStart { partial, .. }
        | AssistantMessageEvent::TextDelta { partial, .. }
        | AssistantMessageEvent::TextEnd { partial, .. }
        | AssistantMessageEvent::ThinkingStart { partial, .. }
        | AssistantMessageEvent::ThinkingDelta { partial, .. }
        | AssistantMessageEvent::ThinkingEnd { partial, .. }
        | AssistantMessageEvent::ToolCallStart { partial, .. }
        | AssistantMessageEvent::ToolCallDelta { partial, .. }
        | AssistantMessageEvent::ToolCallEnd { partial, .. } => partial.clone(),
        _ => serde_json::json!({}),
    }
}

fn extract_tool_calls(message: &AgentMessage) -> Vec<AgentToolCall> {
    message
        .get("content")
        .and_then(|c| c.as_array())
        .map(|arr| {
            arr.iter()
                .filter(|block| block.get("type").and_then(|t| t.as_str()) == Some("toolCall"))
                .filter_map(|block| {
                    Some(AgentToolCall {
                        id: block.get("id")?.as_str()?.to_string(),
                        name: block.get("name")?.as_str()?.to_string(),
                        arguments: block.get("arguments")?.clone(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

// ============================================================================
// Tool execution dispatch
// ============================================================================

struct ToolBatch {
    messages: Vec<AgentMessage>,
    terminate: bool,
}

async fn execute_tool_calls(
    current_context: &AgentContext,
    tool_calls: &[AgentToolCall],
    config: &AgentLoopConfig,
    tools: &[Arc<dyn AgentTool>],
    emit: &AgentEventSink,
    signal: Option<CancellationToken>,
) -> ToolBatch {
    let has_sequential = tool_calls.iter().any(|tc| {
        find_tool(tools, &tc.name).and_then(|t| t.execution_mode())
            == Some(ToolExecutionMode::Sequential)
    });

    use crate::types::ToolExecutionMode;
    let effective_mode = match config.tool_execution {
        ToolExecutionMode::Sequential => ToolExecutionMode::Sequential,
        ToolExecutionMode::Parallel if has_sequential => ToolExecutionMode::Sequential,
        _ => ToolExecutionMode::Parallel,
    };

    match effective_mode {
        ToolExecutionMode::Sequential => {
            exec_sequential(current_context, tool_calls, config, tools, emit, signal).await
        }
        ToolExecutionMode::Parallel => {
            exec_parallel(current_context, tool_calls, config, tools, emit, signal).await
        }
    }
}

fn find_tool(tools: &[Arc<dyn AgentTool>], name: &str) -> Option<Arc<dyn AgentTool>> {
    tools.iter().find(|t| t.name() == name).cloned()
}

// ============================================================================
// Sequential execution
// ============================================================================

async fn exec_sequential(
    current_context: &AgentContext,
    tool_calls: &[AgentToolCall],
    config: &AgentLoopConfig,
    tools: &[Arc<dyn AgentTool>],
    emit: &AgentEventSink,
    signal: Option<CancellationToken>,
) -> ToolBatch {
    let mut finalized_calls: Vec<FinalizedToolCall> = vec![];
    let mut messages: Vec<AgentMessage> = vec![];

    for tool_call in tool_calls {
        emit(AgentEvent::ToolExecutionStart {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            args: tool_call.arguments.clone(),
        })
        .await;

        let prep = prepare(current_context, tool_call, config, tools, signal.clone()).await;
        let finalized = match prep {
            PreparedToolCall::Immediate { tool_call, result, is_error } => {
                FinalizedToolCall { tool_call, result, is_error }
            }
            PreparedToolCall::Ready { tool_call, tool, args } => {
                let exec = run_tool(&tool, &tool_call, &args, emit, signal.clone()).await;
                finalize(current_context, &tool_call, &args, exec, config, signal.clone()).await
            }
        };

        emit_tool_end(&finalized, emit).await;
        let msg = make_tool_result_msg(&finalized);
        emit_tool_result_msg(&msg, emit).await;
        finalized_calls.push(finalized);
        messages.push(msg);
    }

    ToolBatch {
        messages,
        terminate: should_terminate(&finalized_calls),
    }
}

// ============================================================================
// Parallel execution
// ============================================================================

async fn exec_parallel(
    current_context: &AgentContext,
    tool_calls: &[AgentToolCall],
    config: &AgentLoopConfig,
    tools: &[Arc<dyn AgentTool>],
    emit: &AgentEventSink,
    signal: Option<CancellationToken>,
) -> ToolBatch {
    // Phase 1: prepare all tools (sequential), emit start events
    // Immediate results are finalized immediately, Ready results spawn async tasks
    enum FinalizedEntry {
        Immediate(FinalizedToolCall),
        Deferred(tokio::task::JoinHandle<FinalizedToolCall>),
    }

    let mut entries: Vec<FinalizedEntry> = vec![];

    for tool_call in tool_calls {
        emit(AgentEvent::ToolExecutionStart {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            args: tool_call.arguments.clone(),
        })
        .await;

        let prep = prepare(current_context, tool_call, config, tools, signal.clone()).await;

        match prep {
            PreparedToolCall::Immediate { tool_call, result, is_error } => {
                // Immediate results: finalize and emit end immediately
                let finalized = FinalizedToolCall { tool_call, result, is_error };
                emit_tool_end(&finalized, emit).await;
                entries.push(FinalizedEntry::Immediate(finalized));
            }
            PreparedToolCall::Ready { tool_call, tool, args } => {
                // Ready results: spawn async task for parallel execution
                let emit = emit.clone();
                let sig = signal.clone();
                let ctx = current_context.clone();
                let cfg_after = config.after_tool_call.clone();

                let handle = tokio::spawn(async move {
                    let exec = run_tool(&tool, &tool_call, &args, &emit, sig.clone()).await;
                    // Use full finalize with proper context (not finalize_simple)
                    let finalized =
                        finalize_with_hooks(&ctx, &tool_call, &args, exec, &cfg_after, sig).await;
                    // Emit end in completion order (as tasks finish)
                    emit_tool_end(&finalized, &emit).await;
                    finalized
                });

                entries.push(FinalizedEntry::Deferred(handle));
            }
        }
    }

    // Phase 2: await all deferred tasks (maintains source order in array)
    let mut finalized_calls: Vec<FinalizedToolCall> = vec![];
    for entry in entries {
        match entry {
            FinalizedEntry::Immediate(f) => finalized_calls.push(f),
            FinalizedEntry::Deferred(handle) => {
                if let Ok(f) = handle.await {
                    finalized_calls.push(f);
                }
            }
        }
    }

    // Phase 3: emit tool result messages in source order
    let mut messages = vec![];
    for finalized in &finalized_calls {
        let msg = make_tool_result_msg(finalized);
        emit_tool_result_msg(&msg, emit).await;
        messages.push(msg);
    }

    ToolBatch {
        messages,
        terminate: should_terminate(&finalized_calls),
    }
}

// ============================================================================
// Tool preparation
// ============================================================================

async fn prepare(
    current_context: &AgentContext,
    tool_call: &AgentToolCall,
    config: &AgentLoopConfig,
    tools: &[Arc<dyn AgentTool>],
    signal: Option<CancellationToken>,
) -> PreparedToolCall {
    let Some(tool) = find_tool(tools, &tool_call.name) else {
        return PreparedToolCall::Immediate {
            tool_call: tool_call.clone(),
            result: AgentToolResult::error_text(format!("Tool {} not found", tool_call.name)),
            is_error: true,
        };
    };

    // Apply prepareArguments shim
    let prepared_args = tool.prepare_arguments(tool_call.arguments.clone());

    // Validate
    let validated = match validate_tool_arguments(tool.as_ref(), prepared_args) {
        Ok(args) => args,
        Err(e) => {
            return PreparedToolCall::Immediate {
                tool_call: tool_call.clone(),
                result: AgentToolResult::error_text(e),
                is_error: true,
            };
        }
    };

    // Check beforeToolCall
    if let Some(ref before_hook) = config.before_tool_call {
        let hook_ctx = crate::types::BeforeToolCallContext {
            assistant_message: serde_json::json!({}),
            tool_call: tool_call.clone(),
            args: validated.clone(),
            context: current_context.clone(),
        };
        if let Some(result) = before_hook(hook_ctx, signal.clone()).await {
            if result.block {
                return PreparedToolCall::Immediate {
                    tool_call: tool_call.clone(),
                    result: AgentToolResult::error_text(
                        result.reason.unwrap_or_else(|| "Tool execution was blocked".to_string()),
                    ),
                    is_error: true,
                };
            }
        }
    }

    PreparedToolCall::Ready {
        tool_call: tool_call.clone(),
        tool,
        args: validated,
    }
}

// ============================================================================
// Tool execution
// ============================================================================

async fn run_tool(
    tool: &Arc<dyn AgentTool>,
    tool_call: &AgentToolCall,
    args: &serde_json::Value,
    emit: &AgentEventSink,
    signal: Option<CancellationToken>,
) -> ExecutedToolCall {
    let tc_id = tool_call.id.clone();
    let tc_name = tool_call.name.clone();
    let tc_args = tool_call.arguments.clone();
    let emit = emit.clone();

    let on_update: Option<AgentToolUpdateCallback> = Some(Box::new(move |partial| {
        let emit = emit.clone();
        let id = tc_id.clone();
        let name = tc_name.clone();
        let args = tc_args.clone();
        tokio::spawn(async move {
            emit(AgentEvent::ToolExecutionUpdate {
                tool_call_id: id,
                tool_name: name,
                args,
                partial_result: serde_json::to_value(&partial.content).unwrap_or_default(),
            })
            .await;
        });
    }));

    match tool.execute(tool_call.id.clone(), args.clone(), signal, on_update).await {
        Ok(result) => ExecutedToolCall { result, is_error: false },
        Err(e) => ExecutedToolCall {
            result: AgentToolResult::error_text(e.to_string()),
            is_error: true,
        },
    }
}

// ============================================================================
// Tool finalization
// ============================================================================

async fn finalize(
    current_context: &AgentContext,
    tool_call: &AgentToolCall,
    args: &serde_json::Value,
    executed: ExecutedToolCall,
    config: &AgentLoopConfig,
    signal: Option<CancellationToken>,
) -> FinalizedToolCall {
    let mut result = executed.result;
    let mut is_error = executed.is_error;

    if let Some(after_hook) = &config.after_tool_call {
        let hook_ctx = AfterToolCallContext {
            assistant_message: serde_json::json!({}),
            tool_call: tool_call.clone(),
            args: args.clone(),
            result: result.clone(),
            is_error,
            context: current_context.clone(),
        };
        if let Some(override_result) = after_hook(hook_ctx, signal).await {
            if let Some(content) = override_result.content {
                result.content = content;
            }
            if let Some(details) = override_result.details {
                result.details = details;
            }
            if let Some(ie) = override_result.is_error {
                is_error = ie;
            }
            if let Some(term) = override_result.terminate {
                result.terminate = term;
            }
        }
    }

    FinalizedToolCall {
        tool_call: tool_call.clone(),
        result,
        is_error,
    }
}

/// Finalize with hooks for parallel execution (with proper context).
async fn finalize_with_hooks(
    current_context: &AgentContext,
    tool_call: &AgentToolCall,
    args: &serde_json::Value,
    executed: ExecutedToolCall,
    after_tool_call: &Option<
        std::sync::Arc<
            dyn Fn(
                    AfterToolCallContext,
                    Option<CancellationToken>,
                ) -> Pin<
                    Box<dyn Future<Output = Option<crate::types::AfterToolCallResult>> + Send>,
                > + Send
                + Sync,
        >,
    >,
    signal: Option<CancellationToken>,
) -> FinalizedToolCall {
    let mut result = executed.result;
    let mut is_error = executed.is_error;

    if let Some(after_hook) = after_tool_call {
        let hook_ctx = AfterToolCallContext {
            assistant_message: serde_json::json!({}),
            tool_call: tool_call.clone(),
            args: args.clone(),
            result: result.clone(),
            is_error,
            context: current_context.clone(),
        };
        if let Some(override_result) = after_hook(hook_ctx, signal).await {
            if let Some(content) = override_result.content {
                result.content = content;
            }
            if let Some(details) = override_result.details {
                result.details = details;
            }
            if let Some(ie) = override_result.is_error {
                is_error = ie;
            }
            if let Some(term) = override_result.terminate {
                result.terminate = term;
            }
        }
    }

    FinalizedToolCall {
        tool_call: tool_call.clone(),
        result,
        is_error,
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn should_terminate(finalized: &[FinalizedToolCall]) -> bool {
    !finalized.is_empty() && finalized.iter().all(|f| f.result.terminate)
}

async fn emit_tool_end(finalized: &FinalizedToolCall, emit: &AgentEventSink) {
    emit(AgentEvent::ToolExecutionEnd {
        tool_call_id: finalized.tool_call.id.clone(),
        tool_name: finalized.tool_call.name.clone(),
        result: serde_json::to_value(&finalized.result).unwrap_or_default(),
        is_error: finalized.is_error,
    })
    .await;
}

fn make_tool_result_msg(finalized: &FinalizedToolCall) -> AgentMessage {
    serde_json::json!({
        "role": "toolResult",
        "toolCallId": finalized.tool_call.id,
        "toolName": finalized.tool_call.name,
        "content": finalized.result.content,
        "details": finalized.result.details,
        "isError": finalized.is_error,
        "timestamp": chrono::Utc::now().timestamp_millis()
    })
}

async fn emit_tool_result_msg(message: &AgentMessage, emit: &AgentEventSink) {
    emit(AgentEvent::MessageStart { message: message.clone() }).await;
    emit(AgentEvent::MessageEnd { message: message.clone() }).await;
}

fn make_failure_message(
    model: &crate::types::ModelInfo,
    stop_reason: &str,
    error_message: Option<&str>,
) -> AgentMessage {
    serde_json::json!({
        "role": "assistant",
        "content": [{"type": "text", "text": ""}],
        "api": model.api,
        "provider": model.provider,
        "model": model.id,
        "usage": {
            "input": 0, "output": 0,
            "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}
        },
        "stopReason": stop_reason,
        "errorMessage": error_message.unwrap_or(""),
        "timestamp": chrono::Utc::now().timestamp_millis()
    })
}

async fn get_steering_messages(config: &AgentLoopConfig, _skip: bool) -> Vec<AgentMessage> {
    if let Some(ref getter) = config.get_steering_messages {
        getter().await
    } else {
        vec![]
    }
}

async fn get_follow_up_messages(config: &AgentLoopConfig) -> Vec<AgentMessage> {
    if let Some(ref getter) = config.get_follow_up_messages {
        getter().await
    } else {
        vec![]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_terminate_empty() {
        assert!(!should_terminate(&[]));
    }

    #[test]
    fn test_should_terminate_all_true() {
        let calls = vec![FinalizedToolCall {
            tool_call: AgentToolCall {
                id: "1".into(),
                name: "t".into(),
                arguments: serde_json::json!({}),
            },
            result: AgentToolResult {
                content: vec![],
                details: serde_json::Value::Null,
                terminate: true,
            },
            is_error: false,
        }];
        assert!(should_terminate(&calls));
    }

    #[test]
    fn test_should_terminate_mixed() {
        let calls = vec![
            FinalizedToolCall {
                tool_call: AgentToolCall {
                    id: "1".into(),
                    name: "t1".into(),
                    arguments: serde_json::json!({}),
                },
                result: AgentToolResult {
                    content: vec![],
                    details: serde_json::Value::Null,
                    terminate: true,
                },
                is_error: false,
            },
            FinalizedToolCall {
                tool_call: AgentToolCall {
                    id: "2".into(),
                    name: "t2".into(),
                    arguments: serde_json::json!({}),
                },
                result: AgentToolResult {
                    content: vec![],
                    details: serde_json::Value::Null,
                    terminate: false,
                },
                is_error: false,
            },
        ];
        assert!(!should_terminate(&calls));
    }

    #[test]
    fn test_make_tool_result_msg() {
        use crate::types::ContentBlock;

        let f = FinalizedToolCall {
            tool_call: AgentToolCall {
                id: "tc1".into(),
                name: "bash".into(),
                arguments: serde_json::json!({"command": "ls"}),
            },
            result: AgentToolResult {
                content: vec![ContentBlock::Text { text: "output".into() }],
                details: serde_json::Value::Null,
                terminate: false,
            },
            is_error: false,
        };
        let msg = make_tool_result_msg(&f);
        assert_eq!(msg["role"], "toolResult");
        assert_eq!(msg["toolCallId"], "tc1");
    }

    #[test]
    fn test_extract_tool_calls() {
        let msg = serde_json::json!({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me run that"},
                {"type": "toolCall", "id": "tc1", "name": "bash", "arguments": {"command": "ls"}}
            ]
        });
        let calls = extract_tool_calls(&msg);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
    }
}
