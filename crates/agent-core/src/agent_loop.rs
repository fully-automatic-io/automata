// The low-level agent loop: drives streamed assistant responses, dispatches
// tool calls (sequential or parallel), and orchestrates steering / follow-up
// queues.

use crate::event::{AgentEvent, AssistantMessageEvent, EventStream};
use crate::tool::{validate_tool_arguments, AgentTool};
use crate::types::{
    AfterToolCallContext, AfterToolCallFn, AgentContext, AgentLoopConfig, AgentMessage,
    AgentToolCall, AgentToolResult, AgentToolUpdateCallback, BeforeToolCallContext, ContentBlock,
    ModelInfo, PrepareNextTurnContext, ShouldStopAfterTurnContext, StopReason, ThinkingBudgets,
    ThinkingLevel, ToolDefinition, ToolExecutionMode, Transport, Usage,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

// ============================================================================
// Public types
// ============================================================================

pub type AgentEventSink =
    Arc<dyn Fn(AgentEvent) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

pub type AssistantMessageEventStream = EventStream<AssistantMessageEvent, AgentMessage>;

pub type StreamFn = Arc<
    dyn Fn(StreamFnInput)
            -> Pin<Box<dyn Future<Output = Result<AssistantMessageEventStream, String>> + Send>>
        + Send + Sync,
>;

#[derive(Clone)]
pub struct StreamFnInput {
    pub model: ModelInfo,
    pub system_prompt: String,
    pub messages: Vec<AgentMessage>,
    pub tools: Vec<Arc<dyn AgentTool>>,
    pub api_key: Option<String>,
    pub signal: Option<CancellationToken>,
    pub session_id: Option<String>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub transport: Transport,
    pub max_retry_delay_ms: Option<u64>,
    pub reasoning: Option<ThinkingLevel>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    /// Provider-specific overrides (e.g. Anthropic `forceAdaptiveThinking`).
    pub provider_options: Option<crate::types::ProviderOptions>,
}

// ============================================================================
// Tool batch execution machinery
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

struct ToolBatch {
    messages: Vec<AgentMessage>,
    terminate: bool,
}

// ============================================================================
// Public API
// ============================================================================

/// Bundles the long-lived collaborators that drive a single `run` / `run_continue`
/// invocation. `config`, `emit`, and `stream_fn` stay constant across the entire
/// loop; per-call inputs (prompts, context, signal) are passed to the methods.
pub struct AgentLoop<'a> {
    config: &'a AgentLoopConfig,
    emit: &'a AgentEventSink,
    stream_fn: &'a StreamFn,
}

impl<'a> AgentLoop<'a> {
    pub fn new(
        config: &'a AgentLoopConfig,
        emit: &'a AgentEventSink,
        stream_fn: &'a StreamFn,
    ) -> Self {
        Self { config, emit, stream_fn }
    }

    pub async fn run(
        &self,
        prompts: Vec<AgentMessage>,
        context: AgentContext,
        signal: Option<CancellationToken>,
    ) -> Vec<AgentMessage> {
        let mut new_messages: Vec<AgentMessage> = prompts.clone();
        let mut current_context = context;
        current_context.messages.extend(prompts.clone());

        self.emit(AgentEvent::AgentStart).await;
        self.emit(AgentEvent::TurnStart).await;
        for prompt in &prompts {
            self.emit(AgentEvent::MessageStart { message: prompt.clone() }).await;
            self.emit(AgentEvent::MessageEnd { message: prompt.clone() }).await;
        }

        self.run_loop(&mut current_context, &mut new_messages, signal).await;
        new_messages
    }

    pub async fn run_continue(
        &self,
        context: AgentContext,
        signal: Option<CancellationToken>,
    ) -> Vec<AgentMessage> {
        if context.messages.is_empty() {
            panic!("Cannot continue: no messages in context");
        }
        if let Some(AgentMessage::Assistant { .. }) = context.messages.last() {
            panic!("Cannot continue from message role: assistant");
        }

        let mut new_messages: Vec<AgentMessage> = vec![];
        let mut current_context = context;

        self.emit(AgentEvent::AgentStart).await;
        self.emit(AgentEvent::TurnStart).await;
        self.run_loop(&mut current_context, &mut new_messages, signal).await;
        new_messages
    }
}

// ============================================================================
// Main loop
// ============================================================================

impl AgentLoop<'_> {
    async fn emit(&self, event: AgentEvent) {
        (self.emit)(event).await;
    }

    async fn run_loop(
        &self,
        current_context: &mut AgentContext,
        new_messages: &mut Vec<AgentMessage>,
        signal: Option<CancellationToken>,
    ) {
        let mut first_turn = true;
        let mut pending_messages: Vec<AgentMessage> = self.drain_steer().await;

        loop {
            let mut has_more_tool_calls = true;

            while has_more_tool_calls || !pending_messages.is_empty() {
                if let Some(ref token) = signal {
                    if token.is_cancelled() {
                        self.emit(AgentEvent::AgentEnd { messages: new_messages.clone() }).await;
                        return;
                    }
                }

                if !first_turn {
                    self.emit(AgentEvent::TurnStart).await;
                } else {
                    first_turn = false;
                }

                if !pending_messages.is_empty() {
                    for message in &pending_messages {
                        self.emit(AgentEvent::MessageStart { message: message.clone() }).await;
                        self.emit(AgentEvent::MessageEnd { message: message.clone() }).await;
                        current_context.messages.push(message.clone());
                        new_messages.push(message.clone());
                    }
                    pending_messages.clear();
                }

                let assistant = self
                    .stream_assistant_response(current_context, signal.clone())
                    .await;
                new_messages.push(assistant.clone());

                let stop_reason = assistant.stop_reason().unwrap_or(StopReason::EndTurn);
                if matches!(stop_reason, StopReason::Error | StopReason::Aborted) {
                    self.emit(AgentEvent::TurnEnd { message: assistant, tool_results: vec![] })
                        .await;
                    self.emit(AgentEvent::AgentEnd { messages: new_messages.clone() }).await;
                    return;
                }

                let tool_calls = extract_tool_calls(&assistant);
                let mut tool_results: Vec<AgentMessage> = vec![];
                has_more_tool_calls = false;

                if !tool_calls.is_empty() {
                    let batch = self
                        .execute_tool_calls(current_context, &tool_calls, &assistant, signal.clone())
                        .await;
                    for result in &batch.messages {
                        current_context.messages.push(result.clone());
                        new_messages.push(result.clone());
                    }
                    tool_results = batch.messages;
                    has_more_tool_calls = !batch.terminate;
                }

                if let Some(ref hook) = self.config.should_stop_after_turn {
                    let ctx = ShouldStopAfterTurnContext {
                        assistant_message: assistant.clone(),
                        tool_results: tool_results.clone(),
                        context: current_context.clone(),
                        has_more_tool_calls,
                    };
                    if hook(ctx, signal.clone()).await {
                        has_more_tool_calls = false;
                    }
                }

                if has_more_tool_calls {
                    if let Some(ref hook) = self.config.prepare_next_turn {
                        let ctx = PrepareNextTurnContext {
                            last_assistant_message: assistant.clone(),
                            tool_results: tool_results.clone(),
                            context: current_context.clone(),
                        };
                        let extras = hook(ctx, signal.clone()).await;
                        pending_messages.extend(extras);
                    }
                }

                self.emit(AgentEvent::TurnEnd { message: assistant, tool_results }).await;
                pending_messages.extend(self.drain_steer().await);
            }

            let follow_up = self.drain_follow_up().await;
            if !follow_up.is_empty() {
                pending_messages = follow_up;
                continue;
            }
            break;
        }

        self.emit(AgentEvent::AgentEnd { messages: new_messages.clone() }).await;
    }
}

// ============================================================================
// Stream assistant response
// ============================================================================

impl AgentLoop<'_> {
    async fn stream_assistant_response(
        &self,
        context: &mut AgentContext,
        signal: Option<CancellationToken>,
    ) -> AgentMessage {
        let mut messages = context.messages.clone();
        if let Some(ref transform_fn) = self.config.transform_context {
            messages = transform_fn(messages, signal.clone()).await;
        }
        let llm_messages = (self.config.convert_to_llm)(messages).await;

        let resolved_api_key = if let Some(ref get_key) = self.config.get_api_key {
            get_key(self.config.model.provider.clone())
                .await
                .or(self.config.api_key.clone())
        } else {
            self.config.api_key.clone()
        };

        let stream_input = StreamFnInput {
            model: self.config.model.clone(),
            system_prompt: context.system_prompt.clone(),
            messages: llm_messages,
            tools: context.tools.clone(),
            api_key: resolved_api_key,
            signal: signal.clone(),
            session_id: self.config.session_id.clone(),
            thinking_budgets: self.config.thinking_budgets.clone(),
            transport: self.config.transport,
            max_retry_delay_ms: self.config.max_retry_delay_ms,
            reasoning: self.config.reasoning.clone(),
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
            provider_options: self.config.provider_options.clone(),
        };

        if let Some(ref hook) = self.config.on_payload {
            let payload = serde_json::json!({
                "model": stream_input.model.id,
                "system": stream_input.system_prompt,
                "messages": stream_input.messages,
                "transport": format!("{:?}", stream_input.transport).to_lowercase(),
            });
            hook(payload).await;
        }

        let response = match (self.stream_fn)(stream_input).await {
            Ok(stream) => stream,
            Err(err) => {
                let failure = make_failure_message(&self.config.model, StopReason::Error, Some(&err));
                context.messages.push(failure.clone());
                self.emit(AgentEvent::MessageStart { message: failure.clone() }).await;
                self.emit(AgentEvent::MessageEnd { message: failure.clone() }).await;
                return failure;
            }
        };

        let mut added_partial = false;

        loop {
            let events = response.take_events();
            let is_done = response.is_complete();

            if events.is_empty() {
                if is_done {
                    let final_message = match response.wait_for_result_try().ok().flatten() {
                        Some(m) => m,
                        None => make_failure_message(
                            &self.config.model,
                            StopReason::Error,
                            Some("Stream ended without result"),
                        ),
                    };
                    if added_partial {
                        if let Some(last) = context.messages.last_mut() {
                            *last = final_message.clone();
                        }
                    } else {
                        context.messages.push(final_message.clone());
                        self.emit(AgentEvent::MessageStart { message: final_message.clone() }).await;
                    }
                    if let Some(ref hook) = self.config.on_response {
                        hook(final_message.to_json()).await;
                    }
                    self.emit(AgentEvent::MessageEnd { message: final_message.clone() }).await;
                    return final_message;
                }
                // Park until the bridge task pushes more events or ends the stream.
                response.wait_for_more().await;
                continue;
            }

            for event in events {
                match event {
                    AssistantMessageEvent::Start { partial } => {
                        let msg = partial.into_finalized();
                        context.messages.push(msg.clone());
                        added_partial = true;
                        self.emit(AgentEvent::MessageStart { message: msg }).await;
                    }
                    AssistantMessageEvent::Done { message, .. } => {
                        let final_msg = message;
                        if added_partial {
                            if let Some(last) = context.messages.last_mut() {
                                *last = final_msg.clone();
                            }
                        } else {
                            context.messages.push(final_msg.clone());
                            self.emit(AgentEvent::MessageStart { message: final_msg.clone() }).await;
                        }
                        if let Some(ref hook) = self.config.on_response {
                            hook(final_msg.to_json()).await;
                        }
                        self.emit(AgentEvent::MessageEnd { message: final_msg.clone() }).await;
                        return final_msg;
                    }
                    AssistantMessageEvent::Error { error, .. } => {
                        let final_msg = error.into_finalized();
                        if added_partial {
                            if let Some(last) = context.messages.last_mut() {
                                *last = final_msg.clone();
                            }
                        } else {
                            context.messages.push(final_msg.clone());
                            self.emit(AgentEvent::MessageStart { message: final_msg.clone() }).await;
                        }
                        if let Some(ref hook) = self.config.on_response {
                            hook(final_msg.to_json()).await;
                        }
                        self.emit(AgentEvent::MessageEnd { message: final_msg.clone() }).await;
                        return final_msg;
                    }
                    ev => {
                        if let Some(partial) = ev.partial() {
                            let updated = partial.clone().into_finalized();
                            if let Some(last) = context.messages.last_mut() {
                                *last = updated;
                            }
                            let snapshot = partial.clone();
                            self.emit(AgentEvent::MessageUpdate {
                                partial: snapshot,
                                assistant_message_event: ev,
                            })
                            .await;
                        }
                    }
                }
            }
        }
    }
}

fn extract_tool_calls(message: &AgentMessage) -> Vec<AgentToolCall> {
    let Some(content) = message.assistant_content() else { return vec![] };
    content.iter().filter_map(|b| match b {
        ContentBlock::ToolCall { id, name, arguments } => Some(AgentToolCall {
            id: id.clone(),
            name: name.clone(),
            arguments: arguments.clone(),
        }),
        _ => None,
    }).collect()
}

// ============================================================================
// Tool execution dispatch
// ============================================================================

impl AgentLoop<'_> {
    async fn execute_tool_calls(
        &self,
        current_context: &AgentContext,
        tool_calls: &[AgentToolCall],
        assistant_message: &AgentMessage,
        signal: Option<CancellationToken>,
    ) -> ToolBatch {
        let has_sequential = tool_calls.iter().any(|tc| {
            find_tool(&current_context.tools, &tc.name)
                .and_then(|t| t.execution_mode()) == Some(ToolExecutionMode::Sequential)
        });

        let effective_mode = match self.config.tool_execution {
            ToolExecutionMode::Sequential => ToolExecutionMode::Sequential,
            ToolExecutionMode::Parallel if has_sequential => ToolExecutionMode::Sequential,
            _ => ToolExecutionMode::Parallel,
        };

        match effective_mode {
            ToolExecutionMode::Sequential => {
                self.exec_sequential(current_context, tool_calls, assistant_message, signal).await
            }
            ToolExecutionMode::Parallel => {
                self.exec_parallel(current_context, tool_calls, assistant_message, signal).await
            }
        }
    }
}

fn find_tool(tools: &[Arc<dyn AgentTool>], name: &str) -> Option<Arc<dyn AgentTool>> {
    tools.iter().find(|t| t.name() == name).cloned()
}

// ============================================================================
// Sequential execution
// ============================================================================

impl AgentLoop<'_> {
    async fn exec_sequential(
        &self,
        current_context: &AgentContext,
        tool_calls: &[AgentToolCall],
        assistant_message: &AgentMessage,
        signal: Option<CancellationToken>,
    ) -> ToolBatch {
        let mut finalized_calls: Vec<FinalizedToolCall> = vec![];
        let mut messages: Vec<AgentMessage> = vec![];

        for tool_call in tool_calls {
            self.emit(AgentEvent::ToolExecutionStart {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                args: tool_call.arguments.clone(),
            }).await;

            let prep = self.prepare(current_context, tool_call, assistant_message, signal.clone()).await;
            let finalized = match prep {
                PreparedToolCall::Immediate { tool_call, result, is_error } => {
                    FinalizedToolCall { tool_call, result, is_error }
                }
                PreparedToolCall::Ready { tool_call, tool, args } => {
                    let exec = run_tool(&tool, &tool_call, &args, self.emit, signal.clone()).await;
                    finalize(
                        current_context, &tool_call, &args, exec, assistant_message,
                        &self.config.after_tool_call, signal.clone(),
                    ).await
                }
            };

            emit_tool_end(&finalized, self.emit).await;
            let msg = make_tool_result_msg(&finalized);
            emit_tool_result_msg(&msg, self.emit).await;
            finalized_calls.push(finalized);
            messages.push(msg);
        }

        ToolBatch { messages, terminate: should_terminate(&finalized_calls) }
    }
}

// ============================================================================
// Parallel execution — preserve source order in result messages
// ============================================================================

impl AgentLoop<'_> {
    async fn exec_parallel(
        &self,
        current_context: &AgentContext,
        tool_calls: &[AgentToolCall],
        assistant_message: &AgentMessage,
        signal: Option<CancellationToken>,
    ) -> ToolBatch {
        enum Entry {
            Immediate(FinalizedToolCall),
            Deferred(tokio::task::JoinHandle<FinalizedToolCall>),
        }

        let mut entries: Vec<Entry> = vec![];

        for tool_call in tool_calls {
            self.emit(AgentEvent::ToolExecutionStart {
                tool_call_id: tool_call.id.clone(),
                tool_name: tool_call.name.clone(),
                args: tool_call.arguments.clone(),
            }).await;

            let prep = self.prepare(current_context, tool_call, assistant_message, signal.clone()).await;
            match prep {
                PreparedToolCall::Immediate { tool_call, result, is_error } => {
                    let f = FinalizedToolCall { tool_call, result, is_error };
                    emit_tool_end(&f, self.emit).await;
                    entries.push(Entry::Immediate(f));
                }
                PreparedToolCall::Ready { tool_call, tool, args } => {
                    let emit = self.emit.clone();
                    let sig = signal.clone();
                    let ctx = current_context.clone();
                    let after = self.config.after_tool_call.clone();
                    let assistant_msg = assistant_message.clone();
                    let handle = tokio::spawn(async move {
                        let exec = run_tool(&tool, &tool_call, &args, &emit, sig.clone()).await;
                        let f = finalize(&ctx, &tool_call, &args, exec, &assistant_msg, &after, sig).await;
                        emit_tool_end(&f, &emit).await;
                        f
                    });
                    entries.push(Entry::Deferred(handle));
                }
            }
        }

        let mut finalized_calls: Vec<FinalizedToolCall> = vec![];
        for entry in entries {
            match entry {
                Entry::Immediate(f) => finalized_calls.push(f),
                Entry::Deferred(h) => {
                    if let Ok(f) = h.await { finalized_calls.push(f); }
                }
            }
        }

        let mut messages = vec![];
        for f in &finalized_calls {
            let msg = make_tool_result_msg(f);
            emit_tool_result_msg(&msg, self.emit).await;
            messages.push(msg);
        }

        ToolBatch { messages, terminate: should_terminate(&finalized_calls) }
    }
}

// ============================================================================
// Tool preparation, execution, finalization
// ============================================================================

impl AgentLoop<'_> {
    async fn prepare(
        &self,
        current_context: &AgentContext,
        tool_call: &AgentToolCall,
        assistant_message: &AgentMessage,
        signal: Option<CancellationToken>,
    ) -> PreparedToolCall {
        let Some(tool) = find_tool(&current_context.tools, &tool_call.name) else {
            return PreparedToolCall::Immediate {
                tool_call: tool_call.clone(),
                result: AgentToolResult::error_text(format!("Tool {} not found", tool_call.name)),
                is_error: true,
            };
        };

        let prepared_args = tool.prepare_arguments(tool_call.arguments.clone());
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

        if let Some(ref before) = self.config.before_tool_call {
            let ctx = BeforeToolCallContext {
                assistant_message: assistant_message.clone(),
                tool_call: tool_call.clone(),
                args: validated.clone(),
                context: current_context.clone(),
            };
            if let Some(result) = before(ctx, signal.clone()).await {
                if result.block {
                    return PreparedToolCall::Immediate {
                        tool_call: tool_call.clone(),
                        result: AgentToolResult::error_text(
                            result.reason.unwrap_or_else(|| "Tool execution was blocked".into()),
                        ),
                        is_error: true,
                    };
                }
            }
        }

        PreparedToolCall::Ready { tool_call: tool_call.clone(), tool, args: validated }
    }
}

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
    let emit_for_update = emit.clone();
    let on_update: Option<AgentToolUpdateCallback> = Some(Box::new(move |partial| {
        let emit = emit_for_update.clone();
        let id = tc_id.clone();
        let name = tc_name.clone();
        let args = tc_args.clone();
        tokio::spawn(async move {
            emit(AgentEvent::ToolExecutionUpdate {
                tool_call_id: id,
                tool_name: name,
                args,
                partial_result: serde_json::to_value(&partial.content).unwrap_or_default(),
            }).await;
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

async fn finalize(
    current_context: &AgentContext,
    tool_call: &AgentToolCall,
    args: &serde_json::Value,
    executed: ExecutedToolCall,
    assistant_message: &AgentMessage,
    after_hook: &Option<AfterToolCallFn>,
    signal: Option<CancellationToken>,
) -> FinalizedToolCall {
    let mut result = executed.result;
    let mut is_error = executed.is_error;

    if let Some(hook) = after_hook {
        let ctx = AfterToolCallContext {
            assistant_message: assistant_message.clone(),
            tool_call: tool_call.clone(),
            args: args.clone(),
            result: result.clone(),
            is_error,
            context: current_context.clone(),
        };
        if let Some(over) = hook(ctx, signal).await {
            if let Some(content) = over.content { result.content = content; }
            if let Some(details) = over.details { result.details = details; }
            if let Some(ie) = over.is_error { is_error = ie; }
            if let Some(t) = over.terminate { result.terminate = t; }
        }
    }

    FinalizedToolCall { tool_call: tool_call.clone(), result, is_error }
}

// ============================================================================
// Helpers
// ============================================================================

fn should_terminate(finalized: &[FinalizedToolCall]) -> bool {
    !finalized.is_empty() && finalized.iter().all(|f| f.result.terminate)
}

async fn emit_tool_end(f: &FinalizedToolCall, emit: &AgentEventSink) {
    emit(AgentEvent::ToolExecutionEnd {
        tool_call_id: f.tool_call.id.clone(),
        tool_name: f.tool_call.name.clone(),
        result: f.result.clone(),
        is_error: f.is_error,
    }).await;
}

fn make_tool_result_msg(f: &FinalizedToolCall) -> AgentMessage {
    AgentMessage::ToolResult {
        tool_call_id: f.tool_call.id.clone(),
        tool_name: f.tool_call.name.clone(),
        content: f.result.content.clone(),
        details: serde_json::to_value(&f.result.details).ok(),
        is_error: f.is_error,
        timestamp: chrono::Utc::now().timestamp_millis().max(0) as u64,
    }
}

async fn emit_tool_result_msg(m: &AgentMessage, emit: &AgentEventSink) {
    emit(AgentEvent::MessageStart { message: m.clone() }).await;
    emit(AgentEvent::MessageEnd { message: m.clone() }).await;
}

fn make_failure_message(model: &ModelInfo, stop_reason: StopReason, error_message: Option<&str>) -> AgentMessage {
    AgentMessage::Assistant {
        content: vec![ContentBlock::Text { text: String::new() }],
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        usage: Usage::default(),
        stop_reason,
        error_message: error_message.map(|s| s.to_string()),
        timestamp: chrono::Utc::now().timestamp_millis().max(0) as u64,
    }
}

impl AgentLoop<'_> {
    async fn drain_steer(&self) -> Vec<AgentMessage> {
        if let Some(ref get) = self.config.get_steering_messages { get().await } else { vec![] }
    }

    async fn drain_follow_up(&self) -> Vec<AgentMessage> {
        if let Some(ref get) = self.config.get_follow_up_messages { get().await } else { vec![] }
    }
}

/// Build a `ToolDefinition` slice from the agent context. Useful when wiring
/// up llm-client requests; kept here so the loop doesn't expose it on every path.
pub fn tools_to_definitions(tools: &[Arc<dyn AgentTool>]) -> Vec<ToolDefinition> {
    tools.iter().map(|t| ToolDefinition {
        name: t.name().to_string(),
        description: t.description().to_string(),
        input_schema: t.parameters(),
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_terminate_empty() {
        assert!(!should_terminate(&[]));
    }

    #[test]
    fn test_extract_tool_calls() {
        let msg = AgentMessage::Assistant {
            content: vec![
                ContentBlock::Text { text: "x".into() },
                ContentBlock::ToolCall { id: "tc1".into(), name: "bash".into(), arguments: serde_json::json!({}) },
            ],
            api: crate::types::Api::Anthropic, provider: "".into(), model: "".into(),
            usage: Usage::default(), stop_reason: StopReason::EndTurn,
            error_message: None, timestamp: 0,
        };
        let calls = extract_tool_calls(&msg);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
    }
}
