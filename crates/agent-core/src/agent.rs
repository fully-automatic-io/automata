//
// Stateful wrapper around the low-level agent loop.
// Owns the transcript, emits lifecycle events, executes tools,
// and exposes queueing APIs for steering and follow-up messages.

use crate::agent_loop::{run_agent_loop, run_agent_loop_continue, AgentEventSink, StreamFn, StreamFnInput};
use crate::event::{AgentEvent, AssistantMessageEvent, EventStream};
use crate::hooks::default_convert_to_llm;
use crate::queue::{MessageQueue, QueueMode};
use crate::tool::AgentTool;
use crate::types::{
    AgentContext, AgentLoopConfig, AgentMessage, AgentState,
    ToolExecutionMode, Transport,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

// ============================================================================
// Default stream function (placeholder — apps provide real implementation)
// ============================================================================

fn default_stream_fn() -> StreamFn {
    Arc::new(|_input: StreamFnInput| {
        Box::pin(async move {
            let stream = EventStream::<AssistantMessageEvent, AgentMessage>::new();
            stream.push(AssistantMessageEvent::Error {
                reason: "error".to_string(),
                error: serde_json::json!({
                    "role": "assistant",
                    "content": [{"type": "text", "text": "No stream function configured"}],
                    "stopReason": "error",
                    "errorMessage": "No stream function configured"
                }),
            });
            stream.end(serde_json::json!({
                "role": "assistant",
                "content": [],
                "stopReason": "error",
                "errorMessage": "No stream function configured"
            }));
            Ok(stream)
        })
    })
}

// ============================================================================
// Agent
// ============================================================================

type EventListener = Arc<
    dyn Fn(AgentEvent, Option<CancellationToken>) -> Pin<Box<dyn Future<Output = ()> + Send>>
        + Send
        + Sync,
>;

#[derive(Clone)]
struct ActiveRun {
    abort: CancellationToken,
}

/// Stateful wrapper around the low-level agent loop.
pub struct Agent {
    state: Arc<Mutex<AgentState>>,
    listeners: Arc<Mutex<Vec<EventListener>>>,
    steering_queue: Arc<Mutex<MessageQueue>>,
    follow_up_queue: Arc<Mutex<MessageQueue>>,
    active_run: Arc<Mutex<Option<ActiveRun>>>,

    pub convert_to_llm: Arc<
        dyn Fn(Vec<AgentMessage>) -> Pin<Box<dyn Future<Output = Vec<crate::types::Message>> + Send>>
            + Send
            + Sync,
    >,
    pub transform_context: Option<
        Arc<
            dyn Fn(
                    Vec<AgentMessage>,
                    Option<CancellationToken>,
                ) -> Pin<Box<dyn Future<Output = Vec<AgentMessage>> + Send>>
                + Send
                + Sync,
        >,
    >,
    pub stream_fn: StreamFn,
    pub get_api_key: Option<
        Arc<dyn Fn(String) -> Pin<Box<dyn Future<Output = Option<String>> + Send>> + Send + Sync>,
    >,
    pub session_id: Option<String>,
    pub thinking_budgets: Option<crate::types::ThinkingBudgets>,
    pub transport: Transport,
    pub max_retry_delay_ms: Option<u64>,
    pub tool_execution: ToolExecutionMode,

    before_tool_call: Option<
        Arc<
            dyn Fn(
                    crate::types::BeforeToolCallContext,
                    Option<CancellationToken>,
                )
                    -> Pin<
                        Box<
                            dyn Future<Output = Option<crate::types::BeforeToolCallResult>>
                                + Send,
                        >,
                    > + Send
                    + Sync,
        >,
    >,
    after_tool_call: Option<
        Arc<
            dyn Fn(
                    crate::types::AfterToolCallContext,
                    Option<CancellationToken>,
                )
                    -> Pin<
                        Box<
                            dyn Future<Output = Option<crate::types::AfterToolCallResult>>
                                + Send,
                        >,
                    > + Send
                    + Sync,
        >,
    >,
}

/// Options for constructing an Agent.
pub struct AgentOptions {
    pub initial_state: Option<AgentState>,
    pub convert_to_llm: Option<
        Arc<
            dyn Fn(Vec<AgentMessage>) -> Pin<Box<dyn Future<Output = Vec<crate::types::Message>> + Send>>
                + Send
                + Sync,
        >,
    >,
    pub transform_context: Option<
        Arc<
            dyn Fn(
                    Vec<AgentMessage>,
                    Option<CancellationToken>,
                ) -> Pin<Box<dyn Future<Output = Vec<AgentMessage>> + Send>>
                + Send
                + Sync,
        >,
    >,
    pub stream_fn: Option<StreamFn>,
    pub get_api_key: Option<
        Arc<dyn Fn(String) -> Pin<Box<dyn Future<Output = Option<String>> + Send>> + Send + Sync>,
    >,
    pub before_tool_call: Option<
        Arc<
            dyn Fn(
                    crate::types::BeforeToolCallContext,
                    Option<CancellationToken>,
                )
                    -> Pin<
                        Box<
                            dyn Future<Output = Option<crate::types::BeforeToolCallResult>>
                                + Send,
                        >,
                    > + Send
                    + Sync,
        >,
    >,
    pub after_tool_call: Option<
        Arc<
            dyn Fn(
                    crate::types::AfterToolCallContext,
                    Option<CancellationToken>,
                )
                    -> Pin<
                        Box<
                            dyn Future<Output = Option<crate::types::AfterToolCallResult>>
                                + Send,
                        >,
                    > + Send
                    + Sync,
        >,
    >,
    pub steering_mode: Option<QueueMode>,
    pub follow_up_mode: Option<QueueMode>,
    pub session_id: Option<String>,
    pub thinking_budgets: Option<crate::types::ThinkingBudgets>,
    pub transport: Option<Transport>,
    pub max_retry_delay_ms: Option<u64>,
    pub tool_execution: Option<ToolExecutionMode>,
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            initial_state: None,
            convert_to_llm: None,
            transform_context: None,
            stream_fn: None,
            get_api_key: None,
            before_tool_call: None,
            after_tool_call: None,
            steering_mode: None,
            follow_up_mode: None,
            session_id: None,
            thinking_budgets: None,
            transport: None,
            max_retry_delay_ms: None,
            tool_execution: None,
        }
    }
}

impl Agent {
    pub fn new(options: AgentOptions) -> Self {
        let state = options.initial_state.unwrap_or_else(|| {
            AgentState::new(None, None, None, None, None)
        });

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
            convert_to_llm: options
                .convert_to_llm
                .unwrap_or_else(|| Arc::new(|msgs| Box::pin(default_convert_to_llm(msgs)))),
            transform_context: options.transform_context,
            stream_fn: options.stream_fn.unwrap_or_else(default_stream_fn),
            get_api_key: options.get_api_key,
            session_id: options.session_id,
            thinking_budgets: options.thinking_budgets,
            transport: options.transport.unwrap_or(Transport::Sse),
            max_retry_delay_ms: options.max_retry_delay_ms,
            tool_execution: options.tool_execution.unwrap_or(ToolExecutionMode::Parallel),
            before_tool_call: options.before_tool_call,
            after_tool_call: options.after_tool_call,
        }
    }

    // =========================================================================
    // State access
    // =========================================================================

    pub fn state(&self) -> std::sync::MutexGuard<'_, AgentState> {
        self.state.lock().unwrap()
    }

    pub fn read_state(&self) -> AgentState {
        // Return a snapshot — AgentState doesn't impl Clone with closures,
        // so we return a simplified view
        AgentState::new(
            Some(self.state.lock().unwrap().system_prompt().to_string()),
            Some(self.state.lock().unwrap().model().clone()),
            Some(self.state.lock().unwrap().thinking_level()),
            Some(self.state.lock().unwrap().tools()),
            Some(self.state.lock().unwrap().messages()),
        )
    }

    // =========================================================================
    // Event subscription
    // =========================================================================

    pub fn subscribe<F, Fut>(&self, listener: F)
    where
        F: Fn(AgentEvent, Option<CancellationToken>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let wrapped: EventListener =
            Arc::new(move |event, signal| Box::pin(listener(event, signal)));
        self.listeners.lock().unwrap().push(wrapped);
    }

    // =========================================================================
    // Queue management
    // =========================================================================

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

    pub fn has_queued_messages(&self) -> bool {
        self.steering_queue.lock().unwrap().has_items()
            || self.follow_up_queue.lock().unwrap().has_items()
    }

    pub fn set_steering_mode(&self, mode: QueueMode) {
        self.steering_queue.lock().unwrap().set_mode(mode);
    }

    pub fn steering_mode(&self) -> QueueMode {
        self.steering_queue.lock().unwrap().mode()
    }

    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        self.follow_up_queue.lock().unwrap().set_mode(mode);
    }

    pub fn follow_up_mode(&self) -> QueueMode {
        self.follow_up_queue.lock().unwrap().mode()
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    pub fn signal(&self) -> Option<CancellationToken> {
        self.active_run.lock().unwrap().as_ref().map(|r| r.abort.clone())
    }

    pub fn abort(&self) {
        if let Some(ref run) = *self.active_run.lock().unwrap() {
            run.abort.cancel();
        }
    }

    pub async fn wait_for_idle(&self) {
        // Poll until active_run is None
        loop {
            if self.active_run.lock().unwrap().is_none() {
                return;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
    }

    pub fn reset(&self) {
        let mut state = self.state.lock().unwrap();
        state.set_messages(&[]);
        *state = AgentState::new(
            Some(state.system_prompt().to_string()),
            Some(state.model().clone()),
            Some(state.thinking_level()),
            None,
            None,
        );
        self.clear_all_queues();
    }

    // =========================================================================
    // Prompt / Continue
    // =========================================================================

    pub async fn prompt(&self, input: PromptInput) -> Result<Vec<AgentMessage>, String> {
        if self.active_run.lock().unwrap().is_some() {
            return Err("Agent is already processing".to_string());
        }

        let messages = match input {
            PromptInput::Messages(msgs) => msgs,
            PromptInput::Text(text) => {
                vec![serde_json::json!({
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                    "timestamp": chrono::Utc::now().timestamp_millis()
                })]
            }
        };

        self.run_prompt_messages(messages).await
    }

    pub async fn continue_agent(&self) -> Result<Vec<AgentMessage>, String> {
        if self.active_run.lock().unwrap().is_some() {
            return Err("Agent is already processing".to_string());
        }

        let last_role = {
            let state = self.state.lock().unwrap();
            let msgs = state.messages();
            msgs.last()
                .and_then(|m| m.get("role"))
                .and_then(|r| r.as_str())
                .map(|s| s.to_string())
        };

        if let Some(role) = last_role {
            if role == "assistant" {
                // Try steering queue first
                let steering = {
                    let mut q = self.steering_queue.lock().unwrap();
                    q.drain()
                };
                if !steering.is_empty() {
                    return self.run_prompt_messages(steering).await;
                }

                // Then follow-up
                let follow_ups = {
                    let mut q = self.follow_up_queue.lock().unwrap();
                    q.drain()
                };
                if !follow_ups.is_empty() {
                    return self.run_prompt_messages(follow_ups).await;
                }

                return Err("Cannot continue from message role: assistant".to_string());
            }
        } else {
            return Err("No messages to continue from".to_string());
        }

        self.run_continuation().await
    }

    async fn run_prompt_messages(
        &self,
        messages: Vec<AgentMessage>,
    ) -> Result<Vec<AgentMessage>, String> {
        let system_prompt = self.state.lock().unwrap().system_prompt().to_string();
        let state_messages = self.state.lock().unwrap().messages();
        let state_tools = self.state.lock().unwrap().tools();
        let stream_fn = self.stream_fn.clone();
        let config = self.build_loop_config();

        self.run_with_lifecycle(move |signal, emit| {
            let context = AgentContext {
                system_prompt: system_prompt.clone(),
                messages: state_messages.clone(),
                tools: state_tools.clone(),
            };
            let tools: Vec<Arc<dyn AgentTool>> = vec![];
            let stream_fn = stream_fn.clone();
            let config = config.clone();
            Box::pin(async move {
                let result = run_agent_loop(
                    messages.clone(), context, &config, &tools, &emit, signal, &stream_fn,
                ).await;
                Ok(result)
            })
        }).await
    }

    async fn run_continuation(&self) -> Result<Vec<AgentMessage>, String> {
        let system_prompt = self.state.lock().unwrap().system_prompt().to_string();
        let state_messages = self.state.lock().unwrap().messages();
        let state_tools = self.state.lock().unwrap().tools();
        let stream_fn = self.stream_fn.clone();
        let config = self.build_loop_config();

        self.run_with_lifecycle(move |signal, emit| {
            let context = AgentContext {
                system_prompt: system_prompt.clone(),
                messages: state_messages.clone(),
                tools: state_tools.clone(),
            };
            let tools: Vec<Arc<dyn AgentTool>> = vec![];
            let stream_fn = stream_fn.clone();
            let config = config.clone();
            Box::pin(async move {
                let result = run_agent_loop_continue(
                    context, &config, &tools, &emit, signal, &stream_fn,
                ).await;
                Ok(result)
            })
        }).await
    }

    // =========================================================================
    // Internal
    // =========================================================================

    fn build_loop_config(&self) -> AgentLoopConfig {
        AgentLoopConfig {
            model: self.state.lock().unwrap().model().clone(),
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
            session_id: self.session_id.clone(),
            thinking_budgets: self.thinking_budgets.clone(),
            transport: self.transport,
            max_retry_delay_ms: self.max_retry_delay_ms,
            reasoning: None,
            temperature: None,
            max_tokens: None,
        }
    }

    async fn run_with_lifecycle<F, Fut>(
        &self,
        executor: F,
    ) -> Result<Vec<AgentMessage>, String>
    where
        F: FnOnce(Option<CancellationToken>, AgentEventSink) -> Fut + Send + 'static,
        Fut: Future<Output = Result<Vec<AgentMessage>, String>> + Send + 'static,
    {
        if self.active_run.lock().unwrap().is_some() {
            return Err("Agent is already processing".to_string());
        }

        let cancel = CancellationToken::new();
        *self.active_run.lock().unwrap() = Some(ActiveRun {
            abort: cancel.clone(),
        });

        {
            let mut state = self.state.lock().unwrap();
            *state = AgentState::new(
                Some(state.system_prompt().to_string()),
                Some(state.model().clone()),
                Some(state.thinking_level()),
                Some(state.tools()),
                Some(state.messages()),
            );
        }

        let emit: AgentEventSink = {
            let listeners = self.listeners.clone();
            let cancel = cancel.clone();
            Arc::new(move |event: AgentEvent| {
                let listeners = listeners.clone();
                let cancel = cancel.clone();
                Box::pin(async move {
                    let list_copy: Vec<EventListener> = listeners.lock().unwrap().clone();
                    for listener in list_copy.iter() {
                        listener(event.clone(), Some(cancel.clone())).await;
                    }
                })
            })
        };

        let result = executor(Some(cancel.clone()), emit).await;

        // Clean up
        {
            let mut state = self.state.lock().unwrap();
            *state = AgentState::new(
                Some(state.system_prompt().to_string()),
                Some(state.model().clone()),
                Some(state.thinking_level()),
                Some(state.tools()),
                Some(state.messages()),
            );
        }
        *self.active_run.lock().unwrap() = None;

        result
    }
}

// ============================================================================
// PromptInput
// ============================================================================

pub enum PromptInput {
    Messages(Vec<AgentMessage>),
    Text(String),
}

impl From<String> for PromptInput {
    fn from(s: String) -> Self {
        PromptInput::Text(s)
    }
}

impl From<&str> for PromptInput {
    fn from(s: &str) -> Self {
        PromptInput::Text(s.to_string())
    }
}

impl From<Vec<AgentMessage>> for PromptInput {
    fn from(msgs: Vec<AgentMessage>) -> Self {
        PromptInput::Messages(msgs)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_new() {
        let agent = Agent::new(AgentOptions::default());
        assert!(!agent.has_queued_messages());
    }

    #[test]
    fn test_agent_queues() {
        let agent = Agent::new(AgentOptions::default());
        agent.steer(serde_json::json!({"role": "user", "content": "steer msg"}));
        assert!(agent.has_queued_messages());

        let drained = agent.steering_queue.lock().unwrap().drain();
        assert_eq!(drained.len(), 1);
        assert!(!agent.has_queued_messages());
    }

    #[test]
    fn test_agent_follow_up() {
        let agent = Agent::new(AgentOptions::default());
        agent.follow_up(serde_json::json!({"role": "user", "content": "follow up"}));
        assert!(agent.has_queued_messages());

        let drained = agent.follow_up_queue.lock().unwrap().drain();
        assert_eq!(drained.len(), 1);
    }

    #[test]
    fn test_agent_clear_queues() {
        let agent = Agent::new(AgentOptions::default());
        agent.steer(serde_json::json!({"role": "user", "content": "s1"}));
        agent.follow_up(serde_json::json!({"role": "user", "content": "f1"}));
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

    #[test]
    fn test_prompt_input_from_messages() {
        let msgs = vec![serde_json::json!({"role": "user", "content": "hi"})];
        let input: PromptInput = msgs.clone().into();
        match input {
            PromptInput::Messages(m) => assert_eq!(m.len(), 1),
            _ => panic!("Expected Messages variant"),
        }
    }
}
