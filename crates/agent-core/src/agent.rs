// High-level stateful wrapper around the agent loop.
//
// `Agent` owns the running transcript, dispatches lifecycle events to
// subscribers, and exposes the steering / follow-up queues. Callers must
// supply a real `StreamFn` (typically via
// `coding-agent::stream_bridge::create_stream_fn`).

use crate::agent_loop::{AgentEventSink, AgentLoop, StreamFn};
use crate::event::AgentEvent;
use crate::harness::messages::default_convert_to_llm;
use crate::queue::{MessageQueue, QueueMode};
use crate::tool::AgentTool;
use crate::types::{
    AfterToolCallFn, AgentContext, AgentLoopConfig, AgentMessage, AgentState, BeforeToolCallFn,
    ConvertToLlmFn, GetApiKeyFn, ModelInfo, OnPayloadFn, OnResponseFn, PrepareNextTurnFn,
    ShouldStopAfterTurnFn, ThinkingBudgets, ToolExecutionMode, TransformContextFn, Transport,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

type EventListener = Arc<
    dyn Fn(AgentEvent, Option<CancellationToken>) -> Pin<Box<dyn Future<Output = ()> + Send>>
        + Send
        + Sync,
>;

#[derive(Clone)]
struct ActiveRun {
    abort: CancellationToken,
}

pub struct Agent {
    state: Arc<Mutex<AgentState>>,
    listeners: Arc<Mutex<Vec<EventListener>>>,
    steering_queue: Arc<Mutex<MessageQueue>>,
    follow_up_queue: Arc<Mutex<MessageQueue>>,
    active_run: Arc<Mutex<Option<ActiveRun>>>,
    /// Notified when an active run finishes, so `wait_for_idle` can park
    /// instead of polling.
    idle_notify: Arc<tokio::sync::Notify>,

    pub stream_fn: StreamFn,
    pub convert_to_llm: ConvertToLlmFn,
    pub transform_context: Option<TransformContextFn>,
    pub get_api_key: Option<GetApiKeyFn>,
    pub before_tool_call: Option<BeforeToolCallFn>,
    pub after_tool_call: Option<AfterToolCallFn>,
    pub should_stop_after_turn: Option<ShouldStopAfterTurnFn>,
    pub prepare_next_turn: Option<PrepareNextTurnFn>,
    pub on_payload: Option<OnPayloadFn>,
    pub on_response: Option<OnResponseFn>,
    pub session_id: Option<String>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub transport: Transport,
    pub max_retry_delay_ms: Option<u64>,
    pub tool_execution: ToolExecutionMode,
}

pub struct AgentOptions {
    pub initial_state: Option<AgentState>,
    pub stream_fn: StreamFn,
    pub convert_to_llm: Option<ConvertToLlmFn>,
    pub transform_context: Option<TransformContextFn>,
    pub get_api_key: Option<GetApiKeyFn>,
    pub before_tool_call: Option<BeforeToolCallFn>,
    pub after_tool_call: Option<AfterToolCallFn>,
    pub should_stop_after_turn: Option<ShouldStopAfterTurnFn>,
    pub prepare_next_turn: Option<PrepareNextTurnFn>,
    pub on_payload: Option<OnPayloadFn>,
    pub on_response: Option<OnResponseFn>,
    pub steering_mode: Option<QueueMode>,
    pub follow_up_mode: Option<QueueMode>,
    pub session_id: Option<String>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub transport: Option<Transport>,
    pub max_retry_delay_ms: Option<u64>,
    pub tool_execution: Option<ToolExecutionMode>,
}

impl Agent {
    pub fn new(options: AgentOptions) -> Self {
        let state = options
            .initial_state
            .unwrap_or_else(|| AgentState::new(None, None, None, None, None));
        Self {
            state: Arc::new(Mutex::new(state)),
            listeners: Arc::new(Mutex::new(Vec::new())),
            steering_queue: Arc::new(Mutex::new(MessageQueue::new(
                options.steering_mode.unwrap_or(QueueMode::OneAtATime),
            ))),
            follow_up_queue: Arc::new(Mutex::new(MessageQueue::new(
                options.follow_up_mode.unwrap_or(QueueMode::OneAtATime),
            ))),
            active_run: Arc::new(Mutex::new(None)),
            idle_notify: Arc::new(tokio::sync::Notify::new()),
            stream_fn: options.stream_fn,
            convert_to_llm: options
                .convert_to_llm
                .unwrap_or_else(|| Arc::new(|msgs| Box::pin(default_convert_to_llm(msgs)))),
            transform_context: options.transform_context,
            get_api_key: options.get_api_key,
            before_tool_call: options.before_tool_call,
            after_tool_call: options.after_tool_call,
            should_stop_after_turn: options.should_stop_after_turn,
            prepare_next_turn: options.prepare_next_turn,
            on_payload: options.on_payload,
            on_response: options.on_response,
            session_id: options.session_id,
            thinking_budgets: options.thinking_budgets,
            transport: options.transport.unwrap_or(Transport::Sse),
            max_retry_delay_ms: options.max_retry_delay_ms,
            tool_execution: options.tool_execution.unwrap_or(ToolExecutionMode::Parallel),
        }
    }

    // -----------------------------------------------------------------------
    // State access
    // -----------------------------------------------------------------------

    pub fn with_state<R>(&self, f: impl FnOnce(&mut AgentState) -> R) -> R {
        f(&mut self.state.lock().unwrap())
    }

    pub fn snapshot(&self) -> AgentSnapshot {
        let s = self.state.lock().unwrap();
        AgentSnapshot {
            system_prompt: s.system_prompt().to_string(),
            model: s.model().clone(),
            tools: s.tools(),
            messages: s.messages(),
        }
    }

    // -----------------------------------------------------------------------
    // Event subscription
    // -----------------------------------------------------------------------

    pub fn subscribe<F, Fut>(&self, listener: F)
    where
        F: Fn(AgentEvent, Option<CancellationToken>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let wrapped: EventListener =
            Arc::new(move |event, signal| Box::pin(listener(event, signal)));
        self.listeners.lock().unwrap().push(wrapped);
    }

    // -----------------------------------------------------------------------
    // Queue API
    // -----------------------------------------------------------------------

    pub fn steer(&self, message: AgentMessage) {
        self.steering_queue.lock().unwrap().enqueue(message);
    }

    pub fn follow_up(&self, message: AgentMessage) {
        self.follow_up_queue.lock().unwrap().enqueue(message);
    }

    pub fn clear_steering_queue(&self) {
        self.steering_queue.lock().unwrap().clear();
    }
    pub fn clear_follow_up_queue(&self) {
        self.follow_up_queue.lock().unwrap().clear();
    }
    pub fn clear_all_queues(&self) {
        self.clear_steering_queue();
        self.clear_follow_up_queue();
    }

    /// Clear the transcript, runtime state (streaming / pending tool calls /
    /// error), and both queues. Mirrors pi-mono `Agent.reset`.
    pub fn reset(&self) {
        self.state.lock().unwrap().reset_runtime();
        self.clear_all_queues();
    }

    pub fn has_queued_messages(&self) -> bool {
        self.steering_queue.lock().unwrap().has_items()
            || self.follow_up_queue.lock().unwrap().has_items()
    }

    pub fn set_steering_mode(&self, mode: QueueMode) {
        self.steering_queue.lock().unwrap().set_mode(mode);
    }
    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        self.follow_up_queue.lock().unwrap().set_mode(mode);
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    pub fn signal(&self) -> Option<CancellationToken> {
        self.active_run.lock().unwrap().as_ref().map(|r| r.abort.clone())
    }

    pub fn abort(&self) {
        if let Some(ref run) = *self.active_run.lock().unwrap() {
            run.abort.cancel();
        }
    }

    pub async fn wait_for_idle(&self) {
        loop {
            // Register the waker before re-checking state, so a `notify_one`
            // that fires between the check and `await` still wakes us.
            let notified = self.idle_notify.notified();
            if self.active_run.lock().unwrap().is_none() {
                return;
            }
            notified.await;
        }
    }

    // -----------------------------------------------------------------------
    // Prompt / continue
    // -----------------------------------------------------------------------

    pub async fn prompt(&self, input: PromptInput) -> Result<Vec<AgentMessage>, String> {
        if self.active_run.lock().unwrap().is_some() {
            return Err("Agent is already processing".to_string());
        }
        let messages = match input {
            PromptInput::Messages(msgs) => msgs,
            PromptInput::Text(text) => vec![AgentMessage::user_text(text)],
        };
        self.run_prompt_messages(messages).await
    }

    pub async fn continue_agent(&self) -> Result<Vec<AgentMessage>, String> {
        if self.active_run.lock().unwrap().is_some() {
            return Err("Agent is already processing".to_string());
        }
        let last_role = self.state.lock().unwrap().messages().last().map(|m| m.role().to_string());
        match last_role.as_deref() {
            Some("assistant") => {
                let steering = self.steering_queue.lock().unwrap().drain();
                if !steering.is_empty() {
                    return self.run_prompt_messages(steering).await;
                }
                let follow_ups = self.follow_up_queue.lock().unwrap().drain();
                if !follow_ups.is_empty() {
                    return self.run_prompt_messages(follow_ups).await;
                }
                Err("Cannot continue from message role: assistant".to_string())
            }
            Some(_) => self.run_continuation().await,
            None => Err("No messages to continue from".to_string()),
        }
    }

    async fn run_prompt_messages(
        &self,
        messages: Vec<AgentMessage>,
    ) -> Result<Vec<AgentMessage>, String> {
        let snapshot = self.snapshot();
        let stream_fn = self.stream_fn.clone();
        let config = self.build_loop_config();
        self.run_with_lifecycle(move |signal, emit| {
            let context = AgentContext {
                system_prompt: snapshot.system_prompt.clone(),
                messages: snapshot.messages.clone(),
                tools: snapshot.tools.clone(),
            };
            let stream_fn = stream_fn.clone();
            let config = config.clone();
            Box::pin(async move {
                Ok(AgentLoop::new(&config, &emit, &stream_fn).run(messages, context, signal).await)
            })
        })
        .await
    }

    async fn run_continuation(&self) -> Result<Vec<AgentMessage>, String> {
        let snapshot = self.snapshot();
        let stream_fn = self.stream_fn.clone();
        let config = self.build_loop_config();
        self.run_with_lifecycle(move |signal, emit| {
            let context = AgentContext {
                system_prompt: snapshot.system_prompt.clone(),
                messages: snapshot.messages.clone(),
                tools: snapshot.tools.clone(),
            };
            let stream_fn = stream_fn.clone();
            let config = config.clone();
            Box::pin(async move {
                Ok(AgentLoop::new(&config, &emit, &stream_fn).run_continue(context, signal).await)
            })
        })
        .await
    }

    fn build_loop_config(&self) -> AgentLoopConfig {
        let model = self.state.lock().unwrap().model().clone();
        AgentLoopConfig {
            model,
            convert_to_llm: self.convert_to_llm.clone(),
            transform_context: self.transform_context.clone(),
            get_api_key: self.get_api_key.clone(),
            api_key: None,
            get_steering_messages: {
                let q = self.steering_queue.clone();
                Some(Arc::new(move || {
                    let q = q.clone();
                    Box::pin(async move { q.lock().unwrap().drain() })
                }))
            },
            get_follow_up_messages: {
                let q = self.follow_up_queue.clone();
                Some(Arc::new(move || {
                    let q = q.clone();
                    Box::pin(async move { q.lock().unwrap().drain() })
                }))
            },
            tool_execution: self.tool_execution,
            before_tool_call: self.before_tool_call.clone(),
            after_tool_call: self.after_tool_call.clone(),
            should_stop_after_turn: self.should_stop_after_turn.clone(),
            prepare_next_turn: self.prepare_next_turn.clone(),
            on_payload: self.on_payload.clone(),
            on_response: self.on_response.clone(),
            session_id: self.session_id.clone(),
            thinking_budgets: self.thinking_budgets.clone(),
            transport: self.transport,
            cache_retention: None,
            max_retry_delay_ms: self.max_retry_delay_ms,
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

    async fn run_with_lifecycle<F, Fut>(&self, executor: F) -> Result<Vec<AgentMessage>, String>
    where
        F: FnOnce(Option<CancellationToken>, AgentEventSink) -> Fut + Send + 'static,
        Fut: Future<Output = Result<Vec<AgentMessage>, String>> + Send + 'static,
    {
        if self.active_run.lock().unwrap().is_some() {
            return Err("Agent is already processing".to_string());
        }
        let cancel = CancellationToken::new();
        *self.active_run.lock().unwrap() = Some(ActiveRun { abort: cancel.clone() });

        let emit: AgentEventSink = {
            let listeners = self.listeners.clone();
            let state = self.state.clone();
            let cancel = cancel.clone();
            Arc::new(move |event: AgentEvent| {
                let listeners = listeners.clone();
                let state = state.clone();
                let cancel = cancel.clone();
                Box::pin(async move {
                    // Fold the event into the agent's own state before
                    // dispatching, mirroring pi-mono's `processEvents`.
                    state.lock().unwrap().apply_event(&event);
                    let list_copy: Vec<EventListener> = listeners.lock().unwrap().clone();
                    for listener in list_copy.iter() {
                        listener(event.clone(), Some(cancel.clone())).await;
                    }
                })
            })
        };

        let result = executor(Some(cancel.clone()), emit).await;
        *self.active_run.lock().unwrap() = None;
        self.idle_notify.notify_waiters();
        result
    }
}

/// Snapshot of agent state — owned, cheap to pass to the loop.
pub struct AgentSnapshot {
    pub system_prompt: String,
    pub model: ModelInfo,
    pub tools: Vec<Arc<dyn AgentTool>>,
    pub messages: Vec<AgentMessage>,
}

pub enum PromptInput {
    Messages(Vec<AgentMessage>),
    Text(String),
}

impl From<String> for PromptInput {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}
impl From<&str> for PromptInput {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}
impl From<Vec<AgentMessage>> for PromptInput {
    fn from(m: Vec<AgentMessage>) -> Self {
        Self::Messages(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_loop::StreamFnInput;
    use crate::event::EventStream;

    fn dummy_stream_fn() -> StreamFn {
        Arc::new(|_input: StreamFnInput| {
            Box::pin(async move {
                let stream = EventStream::new();
                stream.end(AgentMessage::assistant_text("ok"));
                Ok(stream)
            })
        })
    }

    fn make_agent() -> Agent {
        Agent::new(AgentOptions {
            initial_state: None,
            stream_fn: dummy_stream_fn(),
            convert_to_llm: None,
            transform_context: None,
            get_api_key: None,
            before_tool_call: None,
            after_tool_call: None,
            should_stop_after_turn: None,
            prepare_next_turn: None,
            on_payload: None,
            on_response: None,
            steering_mode: None,
            follow_up_mode: None,
            session_id: None,
            thinking_budgets: None,
            transport: None,
            max_retry_delay_ms: None,
            tool_execution: None,
        })
    }

    #[test]
    fn test_agent_queues() {
        let agent = make_agent();
        agent.steer(AgentMessage::user_text("steer me"));
        assert!(agent.has_queued_messages());
        agent.clear_all_queues();
        assert!(!agent.has_queued_messages());
    }

    #[test]
    fn test_prompt_input_from_string() {
        let input: PromptInput = "hello".into();
        match input {
            PromptInput::Text(t) => assert_eq!(t, "hello"),
            _ => panic!("Expected Text variant"),
        }
    }

    #[tokio::test]
    async fn test_agent_accumulates_transcript_and_resets() {
        let agent = make_agent();
        // Prompt runs the dummy stream fn, which ends with an assistant message.
        let _ = agent.prompt("hi".into()).await.unwrap();

        // The agent's own state should now reflect the transcript via events:
        // the user prompt plus the assistant reply.
        let msgs = agent.snapshot().messages;
        assert!(
            msgs.iter().any(|m| matches!(m, AgentMessage::Assistant { .. })),
            "agent state should contain the assistant reply after a run"
        );
        assert!(
            msgs.iter().any(|m| m.role() == "user"),
            "agent state should contain the user prompt"
        );

        agent.reset();
        assert!(agent.snapshot().messages.is_empty(), "reset clears the transcript");
        assert!(!agent.has_queued_messages());
    }
}
