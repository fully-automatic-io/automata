// AgentHarness — high-level orchestrator wrapping the agent loop with session
// persistence, hook subscription, compaction, and stream-options management.
//
// Typed events, named hook closures, and simple concurrency using a single
// shared mutex on the inner state.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::agent_loop::{AgentEventSink, AgentLoop, StreamFn};
use crate::event::AgentEvent;
use crate::harness::compaction::{
    compact, prepare_compaction, CompactionError, CompactionResult, CompactionSettings,
    StreamFn as CompactionStreamFn,
};
use crate::harness::messages::default_convert_to_llm;
use crate::harness::session::{BranchSummaryOptions, Session, SessionContext, SessionError};
use crate::tool::AgentTool;
use crate::types::{
    AfterToolCallFn, AgentContext, AgentLoopConfig, AgentMessage, BeforeToolCallFn,
    ConvertToLlmFn, ModelInfo, OnPayloadFn, OnResponseFn, PrepareNextTurnFn,
    ShouldStopAfterTurnFn, ThinkingBudgets, ThinkingLevel, ToolExecutionMode, TransformContextFn,
    Transport,
};

// ============================================================================
// Phase machine
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarnessPhase {
    Idle,
    Turn,
    Compaction,
    BranchSummary,
}

#[derive(Debug, thiserror::Error)]
pub enum HarnessError {
    #[error("Harness is busy (phase: {0:?})")]
    Busy(HarnessPhase),
    #[error("Invalid state: {0}")]
    InvalidState(String),
    #[error("Session error: {0}")]
    Session(#[from] SessionError),
    #[error("Compaction error: {0}")]
    Compaction(#[from] CompactionError),
    #[error("Loop error: {0}")]
    Loop(String),
    #[error("{0}")]
    Other(String),
}

// ============================================================================
// Harness events
// ============================================================================

/// Events surfaced to harness subscribers. Includes the wrapped agent-loop
/// events plus harness-specific lifecycle events (save_point, compaction,
/// branch_summary).
#[derive(Debug, Clone)]
pub enum HarnessEvent {
    Agent(AgentEvent),
    Compaction { result: CompactionResult },
    SavePoint,
    Settled,
    Aborted,
}

pub type HarnessListener = Arc<
    dyn Fn(HarnessEvent, Option<CancellationToken>)
            -> Pin<Box<dyn Future<Output = ()> + Send>>
        + Send + Sync,
>;

// ============================================================================
// Pending writes
// ============================================================================

enum PendingWrite {
    Message(AgentMessage),
    ThinkingLevelChange(ThinkingLevel),
    ModelChange { provider: String, model_id: String },
    Compaction {
        summary: String,
        first_kept_entry_id: String,
        tokens_before: u64,
        details: Option<serde_json::Value>,
    },
    BranchSummary {
        from_id: String,
        summary: String,
        details: Option<serde_json::Value>,
    },
}

// ============================================================================
// Stream options + patch — what callers can adjust per turn.
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct StreamOptions {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub reasoning: Option<ThinkingLevel>,
    pub api_key: Option<String>,
    pub transport: Option<Transport>,
    pub session_id: Option<String>,
    pub headers: Option<HashMap<String, String>>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub max_retry_delay_ms: Option<u64>,
    pub timeout_ms: Option<u64>,
    pub max_retries: Option<u32>,
    pub tool_execution: Option<ToolExecutionMode>,
    pub provider_options: Option<crate::types::ProviderOptions>,
}

#[derive(Debug, Clone, Default)]
pub struct StreamOptionsPatch {
    pub temperature: Option<Option<f32>>,
    pub max_tokens: Option<Option<u32>>,
    pub reasoning: Option<Option<ThinkingLevel>>,
    pub transport: Option<Option<Transport>>,
}

impl StreamOptions {
    pub fn apply_patch(&mut self, patch: StreamOptionsPatch) {
        if let Some(v) = patch.temperature { self.temperature = v; }
        if let Some(v) = patch.max_tokens { self.max_tokens = v; }
        if let Some(v) = patch.reasoning { self.reasoning = v; }
        if let Some(v) = patch.transport { self.transport = v; }
    }
}

// ============================================================================
// Config
// ============================================================================

pub struct HarnessConfig {
    pub system_prompt: String,
    pub thinking_level: ThinkingLevel,
    pub model_provider: String,
    pub model_id: String,
}

// ============================================================================
// Resources — pluggable inputs the harness threads through to the loop.
// ============================================================================

#[derive(Default)]
pub struct HarnessResources {
    pub tools: Vec<Arc<dyn AgentTool>>,
    pub stream_options: StreamOptions,
    pub model_info: Option<ModelInfo>,
}

// ============================================================================
// AgentHarness
// ============================================================================

pub struct AgentHarness {
    session: Arc<Mutex<Session>>,
    config: Arc<Mutex<HarnessConfig>>,
    phase: Arc<Mutex<HarnessPhase>>,
    pending: Arc<Mutex<Vec<PendingWrite>>>,
    steer_queue: Arc<Mutex<Vec<AgentMessage>>>,
    follow_up_queue: Arc<Mutex<Vec<AgentMessage>>>,
    next_turn_queue: Arc<Mutex<Vec<AgentMessage>>>,
    listeners: Arc<Mutex<Vec<HarnessListener>>>,
    resources: Arc<Mutex<HarnessResources>>,
    abort: Arc<Mutex<CancellationToken>>,

    // Caller-supplied stream/convert hooks. Required.
    stream_fn: StreamFn,
    convert_to_llm: ConvertToLlmFn,

    // Optional hooks.
    transform_context: Option<TransformContextFn>,
    before_tool_call: Option<BeforeToolCallFn>,
    after_tool_call: Option<AfterToolCallFn>,
    should_stop_after_turn: Option<ShouldStopAfterTurnFn>,
    prepare_next_turn: Option<PrepareNextTurnFn>,
    on_payload: Option<OnPayloadFn>,
    on_response: Option<OnResponseFn>,
}

pub struct AgentHarnessOptions {
    pub stream_fn: StreamFn,
    pub convert_to_llm: Option<ConvertToLlmFn>,
    pub transform_context: Option<TransformContextFn>,
    pub before_tool_call: Option<BeforeToolCallFn>,
    pub after_tool_call: Option<AfterToolCallFn>,
    pub should_stop_after_turn: Option<ShouldStopAfterTurnFn>,
    pub prepare_next_turn: Option<PrepareNextTurnFn>,
    pub on_payload: Option<OnPayloadFn>,
    pub on_response: Option<OnResponseFn>,
}

impl AgentHarness {
    pub fn new(session: Session, config: HarnessConfig, options: AgentHarnessOptions) -> Self {
        Self {
            session: Arc::new(Mutex::new(session)),
            config: Arc::new(Mutex::new(config)),
            phase: Arc::new(Mutex::new(HarnessPhase::Idle)),
            pending: Arc::new(Mutex::new(vec![])),
            steer_queue: Arc::new(Mutex::new(vec![])),
            follow_up_queue: Arc::new(Mutex::new(vec![])),
            next_turn_queue: Arc::new(Mutex::new(vec![])),
            listeners: Arc::new(Mutex::new(vec![])),
            resources: Arc::new(Mutex::new(HarnessResources::default())),
            abort: Arc::new(Mutex::new(CancellationToken::new())),
            stream_fn: options.stream_fn,
            convert_to_llm: options.convert_to_llm
                .unwrap_or_else(|| Arc::new(|msgs| Box::pin(default_convert_to_llm(msgs)))),
            transform_context: options.transform_context,
            before_tool_call: options.before_tool_call,
            after_tool_call: options.after_tool_call,
            should_stop_after_turn: options.should_stop_after_turn,
            prepare_next_turn: options.prepare_next_turn,
            on_payload: options.on_payload,
            on_response: options.on_response,
        }
    }

    pub async fn phase(&self) -> HarnessPhase { *self.phase.lock().await }

    pub async fn signal(&self) -> CancellationToken { self.abort.lock().await.clone() }

    pub async fn abort(&self) {
        self.abort.lock().await.cancel();
        self.dispatch(HarnessEvent::Aborted).await;
    }

    /// Reset the abort token so a subsequent run can be cancelled cleanly.
    async fn refresh_abort(&self) -> CancellationToken {
        let new_token = CancellationToken::new();
        let mut slot = self.abort.lock().await;
        *slot = new_token.clone();
        new_token
    }

    // -----------------------------------------------------------------------
    // Subscription
    // -----------------------------------------------------------------------

    pub async fn subscribe<F, Fut>(&self, listener: F)
    where
        F: Fn(HarnessEvent, Option<CancellationToken>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let wrapped: HarnessListener =
            Arc::new(move |event, signal| Box::pin(listener(event, signal)));
        self.listeners.lock().await.push(wrapped);
    }

    async fn dispatch(&self, event: HarnessEvent) {
        let signal = Some(self.abort.lock().await.clone());
        let listeners = self.listeners.lock().await.clone();
        for l in &listeners {
            l(event.clone(), signal.clone()).await;
        }
    }

    // -----------------------------------------------------------------------
    // Queue API
    // -----------------------------------------------------------------------

    pub async fn steer(&self, message: AgentMessage) {
        self.steer_queue.lock().await.push(message);
    }

    pub async fn follow_up(&self, message: AgentMessage) {
        self.follow_up_queue.lock().await.push(message);
    }

    pub async fn next_turn(&self, message: AgentMessage) {
        self.next_turn_queue.lock().await.push(message);
    }

    pub async fn drain_steer(&self) -> Vec<AgentMessage> {
        std::mem::take(&mut *self.steer_queue.lock().await)
    }

    pub async fn drain_follow_up(&self) -> Vec<AgentMessage> {
        std::mem::take(&mut *self.follow_up_queue.lock().await)
    }

    pub async fn drain_next_turn(&self) -> Vec<AgentMessage> {
        std::mem::take(&mut *self.next_turn_queue.lock().await)
    }

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    pub async fn set_system_prompt(&self, prompt: String) {
        self.config.lock().await.system_prompt = prompt;
    }

    pub async fn set_thinking_level(&self, level: ThinkingLevel) {
        self.config.lock().await.thinking_level = level;
        self.pending.lock().await.push(PendingWrite::ThinkingLevelChange(level));
    }

    pub async fn set_model(&self, provider: String, model_id: String) {
        let mut cfg = self.config.lock().await;
        cfg.model_provider = provider.clone();
        cfg.model_id = model_id.clone();
        drop(cfg);
        self.pending.lock().await.push(PendingWrite::ModelChange { provider, model_id });
    }

    pub async fn set_active_tools(&self, tools: Vec<Arc<dyn AgentTool>>) {
        self.resources.lock().await.tools = tools;
    }

    pub async fn set_model_info(&self, info: ModelInfo) {
        self.resources.lock().await.model_info = Some(info);
    }

    pub async fn set_stream_options(&self, options: StreamOptions) {
        self.resources.lock().await.stream_options = options;
    }

    pub async fn patch_stream_options(&self, patch: StreamOptionsPatch) {
        self.resources.lock().await.stream_options.apply_patch(patch);
    }

    // -----------------------------------------------------------------------
    // Session access
    // -----------------------------------------------------------------------

    pub async fn build_context(&self) -> Result<SessionContext, HarnessError> {
        Ok(self.session.lock().await.build_context().await?)
    }

    pub async fn append_user_message(&self, text: &str) -> Result<String, HarnessError> {
        self.require_idle().await?;
        self.flush_pending().await?;
        let msg = AgentMessage::user_text(text);
        let id = self.session.lock().await.append_message(msg).await?;
        Ok(id)
    }

    pub async fn append_message(&self, message: AgentMessage) -> Result<String, HarnessError> {
        // If idle, write straight through. Otherwise queue for the next save point.
        if *self.phase.lock().await == HarnessPhase::Idle {
            self.flush_pending().await?;
            let id = self.session.lock().await.append_message(message).await?;
            return Ok(id);
        }
        self.pending.lock().await.push(PendingWrite::Message(message));
        Ok(String::new())
    }

    pub async fn record_compaction(
        &self,
        summary: &str,
        first_kept_entry_id: &str,
        tokens_before: u64,
        details: Option<serde_json::Value>,
    ) -> Result<String, HarnessError> {
        if *self.phase.lock().await != HarnessPhase::Idle {
            self.pending.lock().await.push(PendingWrite::Compaction {
                summary: summary.to_string(),
                first_kept_entry_id: first_kept_entry_id.to_string(),
                tokens_before,
                details,
            });
            return Ok(String::new());
        }
        Ok(self.session.lock().await
            .append_compaction(summary, first_kept_entry_id, tokens_before, details, None)
            .await?)
    }

    pub async fn record_branch_summary(
        &self,
        from_id: &str,
        summary: &str,
        details: Option<serde_json::Value>,
    ) -> Result<String, HarnessError> {
        if *self.phase.lock().await != HarnessPhase::Idle {
            self.pending.lock().await.push(PendingWrite::BranchSummary {
                from_id: from_id.to_string(),
                summary: summary.to_string(),
                details,
            });
            return Ok(String::new());
        }
        Ok(self.session.lock().await
            .append_branch_summary(from_id, summary, details, None)
            .await?)
    }

    pub async fn navigate_tree(
        &self,
        entry_id: Option<&str>,
        summary: Option<BranchSummaryOptions>,
    ) -> Result<Option<String>, HarnessError> {
        self.require_idle().await?;
        Ok(self.session.lock().await.move_to(entry_id, summary).await?)
    }

    // -----------------------------------------------------------------------
    // Compaction
    // -----------------------------------------------------------------------

    pub async fn compact(
        &self,
        settings: &CompactionSettings,
        stream_fn: &CompactionStreamFn,
    ) -> Result<Option<CompactionResult>, HarnessError> {
        self.require_idle().await?;
        *self.phase.lock().await = HarnessPhase::Compaction;
        self.flush_pending().await?;

        let result: Result<Option<CompactionResult>, HarnessError> = (async {
            let session = self.session.lock().await;
            let entries = session.get_branch().await?;
            drop(session);

            let prep = prepare_compaction(&entries, settings)?;
            let Some(prep) = prep else { return Ok(None) };

            let result = compact(&prep, stream_fn).await?;
            self.session.lock().await
                .append_compaction(
                    &result.summary,
                    &result.first_kept_entry_id,
                    result.tokens_before as u64,
                    Some(serde_json::json!({
                        "readFiles": result.read_files,
                        "modifiedFiles": result.modified_files,
                    })),
                    None,
                ).await?;
            Ok(Some(result))
        }).await;

        *self.phase.lock().await = HarnessPhase::Idle;

        if let Ok(Some(ref r)) = result {
            self.dispatch(HarnessEvent::Compaction { result: r.clone() }).await;
        }
        result
    }

    // -----------------------------------------------------------------------
    // Turn execution — drives the agent loop with current resources / hooks.
    // -----------------------------------------------------------------------

    pub async fn execute_turn(
        &self,
        prompts: Vec<AgentMessage>,
    ) -> Result<Vec<AgentMessage>, HarnessError> {
        self.require_idle().await?;
        *self.phase.lock().await = HarnessPhase::Turn;
        self.flush_pending().await?;

        let signal = self.refresh_abort().await;
        let result = self.run_turn(prompts, signal).await;

        // Save-point at turn end: flush pending writes accumulated during the turn.
        let _ = self.flush_pending().await;
        self.dispatch(HarnessEvent::SavePoint).await;
        *self.phase.lock().await = HarnessPhase::Idle;
        self.dispatch(HarnessEvent::Settled).await;
        result
    }

    async fn run_turn(
        &self,
        prompts: Vec<AgentMessage>,
        signal: CancellationToken,
    ) -> Result<Vec<AgentMessage>, HarnessError> {
        // Build context + tools snapshot
        let session_ctx = self.session.lock().await.build_context().await?;
        let messages: Vec<AgentMessage> = session_ctx.messages;

        let cfg = self.config.lock().await;
        let system_prompt = cfg.system_prompt.clone();
        let thinking_level = cfg.thinking_level;
        let model_provider = cfg.model_provider.clone();
        let model_id = cfg.model_id.clone();
        drop(cfg);

        let resources = self.resources.lock().await;
        let tools = resources.tools.clone();
        let opts = resources.stream_options.clone();
        let model_info = resources.model_info.clone().unwrap_or(ModelInfo {
            id: model_id.clone(),
            name: model_id.clone(),
            api: crate::types::Api::Anthropic,
            provider: model_provider,
            base_url: String::new(),
            reasoning: thinking_level != ThinkingLevel::Off,
            input: vec![],
            context_window: 200_000,
            max_tokens: 8192,
        });
        drop(resources);

        let context = AgentContext { system_prompt, messages, tools };

        let config = AgentLoopConfig {
            model: model_info,
            convert_to_llm: self.convert_to_llm.clone(),
            transform_context: self.transform_context.clone(),
            get_api_key: None,
            api_key: opts.api_key.clone(),
            get_steering_messages: {
                let q = self.steer_queue.clone();
                Some(Arc::new(move || {
                    let q = q.clone();
                    Box::pin(async move { std::mem::take(&mut *q.lock().await) })
                }))
            },
            get_follow_up_messages: {
                let q = self.follow_up_queue.clone();
                Some(Arc::new(move || {
                    let q = q.clone();
                    Box::pin(async move { std::mem::take(&mut *q.lock().await) })
                }))
            },
            tool_execution: opts.tool_execution.unwrap_or(ToolExecutionMode::Parallel),
            before_tool_call: self.before_tool_call.clone(),
            after_tool_call: self.after_tool_call.clone(),
            should_stop_after_turn: self.should_stop_after_turn.clone(),
            prepare_next_turn: self.prepare_next_turn.clone(),
            on_payload: self.on_payload.clone(),
            on_response: self.on_response.clone(),
            session_id: opts.session_id.clone(),
            thinking_budgets: opts.thinking_budgets.clone(),
            transport: opts.transport.unwrap_or(Transport::Sse),
            cache_retention: None,
            max_retry_delay_ms: opts.max_retry_delay_ms,
            timeout_ms: opts.timeout_ms,
            max_retries: opts.max_retries,
            headers: opts.headers.clone(),
            metadata: None,
            reasoning: opts.reasoning.clone(),
            temperature: opts.temperature,
            max_tokens: opts.max_tokens,
            provider_options: opts.provider_options.clone(),
        };

        let session = self.session.clone();
        let listeners = self.listeners.clone();
        let abort = self.abort.clone();
        let emit: AgentEventSink = Arc::new(move |event: AgentEvent| {
            let session = session.clone();
            let listeners = listeners.clone();
            let abort = abort.clone();
            Box::pin(async move {
                // Persist message_end events to the session.
                if let AgentEvent::MessageEnd { ref message } = event {
                    let _ = session.lock().await.append_message(message.clone()).await;
                }
                let signal = Some(abort.lock().await.clone());
                let listeners = listeners.lock().await.clone();
                for l in &listeners {
                    l(HarnessEvent::Agent(event.clone()), signal.clone()).await;
                }
            })
        });

        let mut prompts = prompts;
        prompts.extend(self.drain_next_turn().await);

        Ok(AgentLoop::new(&config, &emit, &self.stream_fn).run(prompts, context, Some(signal)).await)
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    async fn require_idle(&self) -> Result<(), HarnessError> {
        let phase = *self.phase.lock().await;
        if phase != HarnessPhase::Idle {
            return Err(HarnessError::Busy(phase));
        }
        Ok(())
    }

    async fn flush_pending(&self) -> Result<(), HarnessError> {
        let pending = std::mem::take(&mut *self.pending.lock().await);
        let mut session = self.session.lock().await;
        for write in pending {
            match write {
                PendingWrite::Message(msg) => { session.append_message(msg).await?; }
                PendingWrite::ThinkingLevelChange(level) => {
                    session.append_thinking_level_change(level).await?;
                }
                PendingWrite::ModelChange { provider, model_id } => {
                    session.append_model_change(&provider, &model_id).await?;
                }
                PendingWrite::Compaction { summary, first_kept_entry_id, tokens_before, details } => {
                    session.append_compaction(&summary, &first_kept_entry_id, tokens_before, details, None).await?;
                }
                PendingWrite::BranchSummary { from_id, summary, details } => {
                    session.append_branch_summary(&from_id, &summary, details, None).await?;
                }
            }
        }
        Ok(())
    }
}

/// Helper: convert the user-provided text into a `ThinkingLevel` enum.
pub fn parse_thinking_level(s: &str) -> ThinkingLevel {
    match s {
        "off" => ThinkingLevel::Off,
        "minimal" => ThinkingLevel::Minimal,
        "low" => ThinkingLevel::Low,
        "medium" => ThinkingLevel::Medium,
        "high" => ThinkingLevel::High,
        "xhigh" => ThinkingLevel::XHigh,
        _ => ThinkingLevel::Off,
    }
}
