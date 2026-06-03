// AgentHarness — high-level orchestrator wrapping the agent loop with session
// persistence, hook subscription, compaction, and stream-options management.
//
// Typed events, named hook closures, and simple concurrency using a single
// shared mutex on the inner state.

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::agent_loop::{AgentEventSink, AgentLoop, StreamFn};
use crate::auto_retry::{compute_retry_delay, is_retryable_error};
use crate::event::AgentEvent;
use crate::harness::compaction::{
    CompactionError, CompactionResult, CompactionSettings, StreamFn as CompactionStreamFn, compact,
    estimate_context_tokens_with_source, prepare_compaction,
};
use crate::harness::messages::default_convert_to_llm;
use crate::harness::session::{
    BranchSummaryOptions, Session, SessionContext, SessionError, SessionTreeEntry,
};
use crate::overflow::is_context_overflow;
use crate::tool::AgentTool;
use crate::types::{
    AfterToolCallFn, AgentContext, AgentLoopConfig, AgentMessage, BeforeToolCallFn, ConvertToLlmFn,
    ModelInfo, OnPayloadFn, OnResponseFn, PrepareNextTurnFn, ShouldStopAfterTurnFn, StopReason,
    ThinkingBudgets, ThinkingLevel, ToolExecutionMode, TransformContextFn, Transport,
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
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
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
// The `Agent` variant wraps the (intentionally unboxed) `AgentEvent`; boxing
// here would just push the indirection onto every consumer's match arm.
#[allow(clippy::large_enum_variant)]
pub enum HarnessEvent {
    Agent(AgentEvent),
    Compaction {
        result: CompactionResult,
    },
    CompactionStart {
        reason: CompactionReason,
    },
    CompactionEnd {
        reason: CompactionReason,
        result: Option<CompactionResult>,
        aborted: bool,
        will_retry: bool,
        error_message: Option<String>,
    },
    AutoRetryStart {
        attempt: u32,
        max_attempts: u32,
        delay_ms: u64,
        error_message: String,
    },
    AutoRetryEnd {
        success: bool,
        attempt: u32,
        final_error: Option<String>,
    },
    /// Model changed via `set_model`. `previous` is the prior model id.
    ModelSelect {
        provider: String,
        model_id: String,
        previous_provider: Option<String>,
        previous_model_id: Option<String>,
    },
    /// Thinking level changed via `set_thinking_level`.
    ThinkingLevelSelect {
        level: ThinkingLevel,
        previous: ThinkingLevel,
    },
    /// Steering / follow-up / next-turn queue contents changed.
    QueueUpdate {
        steer_count: usize,
        follow_up_count: usize,
        next_turn_count: usize,
    },
    SavePoint,
    Settled,
    Aborted {
        cleared_steer: Vec<AgentMessage>,
        cleared_follow_up: Vec<AgentMessage>,
    },
}

/// Why an auto-compaction was triggered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionReason {
    /// LLM returned a context-overflow error; harness will compact and
    /// retry the turn (subject to one-shot recovery limit).
    Overflow,
    /// Context tokens crossed the configured threshold; harness will
    /// compact but **not** retry — caller submits the next prompt manually.
    Threshold,
}

/// Outcome of an `abort()` call. Surfaces the queued messages that were
/// discarded so callers can warn the user / re-enqueue them.
#[derive(Debug, Clone, Default)]
pub struct AbortResult {
    pub cleared_steer: Vec<AgentMessage>,
    pub cleared_follow_up: Vec<AgentMessage>,
}

/// Wires a compaction strategy into the harness so [`execute_turn`] can
/// auto-run [`AgentHarness::check_pre_prompt`] before sending a new
/// prompt — recovering gracefully from the prior turn's aborted/error
/// state. Set via [`AgentHarness::set_auto_compaction`]; leave unset to
/// keep the harness behaviour identical to manual calls.
#[derive(Clone)]
pub struct AutoCompactionConfig {
    pub settings: CompactionSettings,
    /// Wrapped in `Arc` because [`CompactionStreamFn`] is a `Box<dyn Fn>`
    /// — the harness needs to keep one copy and hand another to
    /// `check_pre_prompt`, so we share ownership.
    pub stream_fn: Arc<CompactionStreamFn>,
}

impl AutoCompactionConfig {
    pub fn new(settings: CompactionSettings, stream_fn: CompactionStreamFn) -> Self {
        Self { settings, stream_fn: Arc::new(stream_fn) }
    }
}

impl std::fmt::Debug for AutoCompactionConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoCompactionConfig")
            .field("settings", &self.settings)
            .field("stream_fn", &"<fn>")
            .finish()
    }
}

pub type HarnessListener = Arc<
    dyn Fn(HarnessEvent, Option<CancellationToken>) -> Pin<Box<dyn Future<Output = ()> + Send>>
        + Send
        + Sync,
>;

// ============================================================================
// Pending writes
// ============================================================================

enum PendingWrite {
    Message(AgentMessage),
    ThinkingLevelChange(ThinkingLevel),
    ModelChange {
        provider: String,
        model_id: String,
    },
    ActiveToolsChange {
        active_tool_names: Vec<String>,
    },
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
    /// Plugin-defined custom entry (no message body).
    Custom {
        custom_type: String,
        data: Option<serde_json::Value>,
    },
    /// Plugin-defined custom message entry (carries content / display).
    CustomMessage {
        custom_type: String,
        content: serde_json::Value,
        display: bool,
        details: Option<serde_json::Value>,
    },
    /// Attach (or remove if `label` is `None`) a label on a target entry.
    Label {
        target_id: String,
        label: Option<String>,
    },
    /// Update session-info name.
    SessionInfo {
        name: Option<String>,
    },
    /// Move the leaf pointer (for tree navigation).
    Leaf {
        target_id: Option<String>,
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
        if let Some(v) = patch.temperature {
            self.temperature = v;
        }
        if let Some(v) = patch.max_tokens {
            self.max_tokens = v;
        }
        if let Some(v) = patch.reasoning {
            self.reasoning = v;
        }
        if let Some(v) = patch.transport {
            self.transport = v;
        }
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
    pub tools: HashMap<String, Arc<dyn AgentTool>>,
    pub active_tool_names: Vec<String>,
    pub stream_options: StreamOptions,
    pub model_info: Option<ModelInfo>,
}

fn validate_unique_names(names: &[String], message: &str) -> Result<(), HarnessError> {
    let mut seen = HashSet::with_capacity(names.len());
    let mut duplicates = Vec::new();
    for name in names {
        if !seen.insert(name.as_str()) && !duplicates.iter().any(|existing| existing == name) {
            duplicates.push(name.clone());
        }
    }
    if duplicates.is_empty() {
        Ok(())
    } else {
        Err(HarnessError::InvalidArgument(format!("{}: {}", message, duplicates.join(", "))))
    }
}

impl HarnessResources {
    fn replace_tools(
        &mut self,
        tools: Vec<Arc<dyn AgentTool>>,
        active_tool_names: Option<Vec<String>>,
    ) {
        let next_active = active_tool_names
            .unwrap_or_else(|| tools.iter().map(|tool| tool.name().to_string()).collect());
        self.tools = tools.into_iter().map(|tool| (tool.name().to_string(), tool)).collect();
        self.active_tool_names = next_active;
    }

    fn active_tools(&self) -> Vec<Arc<dyn AgentTool>> {
        self.tools_for_names(&self.active_tool_names)
    }

    fn tools_for_names(&self, names: &[String]) -> Vec<Arc<dyn AgentTool>> {
        names.iter().filter_map(|name| self.tools.get(name).cloned()).collect()
    }

    fn validate_active_tool_names(&self, names: &[String]) -> Result<(), HarnessError> {
        validate_unique_names(names, "Duplicate active tool name(s)")?;
        let missing: Vec<&str> = names
            .iter()
            .map(String::as_str)
            .filter(|name| !self.tools.contains_key(*name))
            .collect();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(HarnessError::InvalidArgument(format!(
                "Unknown tool(s): {}",
                missing.join(", ")
            )))
        }
    }
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
    steer_mode: Arc<Mutex<crate::queue::QueueMode>>,
    follow_up_mode: Arc<Mutex<crate::queue::QueueMode>>,
    listeners: Arc<Mutex<Vec<HarnessListener>>>,
    resources: Arc<Mutex<HarnessResources>>,
    abort: Arc<Mutex<CancellationToken>>,

    // Auto-compaction / retry state.
    overflow_recovery_attempted: Arc<Mutex<bool>>,
    retry_attempt: Arc<Mutex<u32>>,
    retry_settings: Arc<Mutex<crate::auto_retry::RetrySettings>>,
    last_assistant_message: Arc<Mutex<Option<AgentMessage>>>,

    /// When `Some`, `execute_turn` automatically runs `check_pre_prompt`
    /// using the stored settings + stream fn so an aborted / errored prior
    /// turn doesn't bleed into the new context. Caller opts in via
    /// [`AgentHarness::set_auto_compaction`].
    auto_compaction: Arc<Mutex<Option<AutoCompactionConfig>>>,

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
            steer_mode: Arc::new(Mutex::new(crate::queue::QueueMode::default())),
            follow_up_mode: Arc::new(Mutex::new(crate::queue::QueueMode::default())),
            listeners: Arc::new(Mutex::new(vec![])),
            resources: Arc::new(Mutex::new(HarnessResources::default())),
            abort: Arc::new(Mutex::new(CancellationToken::new())),
            overflow_recovery_attempted: Arc::new(Mutex::new(false)),
            retry_attempt: Arc::new(Mutex::new(0)),
            retry_settings: Arc::new(Mutex::new(crate::auto_retry::RetrySettings::default())),
            last_assistant_message: Arc::new(Mutex::new(None)),
            auto_compaction: Arc::new(Mutex::new(None)),
            stream_fn: options.stream_fn,
            convert_to_llm: options
                .convert_to_llm
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

    pub async fn phase(&self) -> HarnessPhase {
        *self.phase.lock().await
    }

    /// Test-only: force the phase to a specific value (e.g. simulate a concurrent
    /// turn / compaction). Production code should never call this.
    #[doc(hidden)]
    pub async fn set_phase_for_test(&self, phase: HarnessPhase) {
        *self.phase.lock().await = phase;
    }

    pub async fn signal(&self) -> CancellationToken {
        self.abort.lock().await.clone()
    }

    /// Cancel the in-flight turn (if any) and clear the steer / follow-up
    /// queues. Returns the messages that were discarded so callers can
    /// surface them to the user (e.g. "you had 3 queued steers, none
    /// reached the model").
    pub async fn abort(&self) -> AbortResult {
        let cleared_steer = std::mem::take(&mut *self.steer_queue.lock().await);
        let cleared_follow_up = std::mem::take(&mut *self.follow_up_queue.lock().await);
        self.abort.lock().await.cancel();
        if !cleared_steer.is_empty() || !cleared_follow_up.is_empty() {
            self.dispatch_queue_update().await;
        }
        self.dispatch(HarnessEvent::Aborted {
            cleared_steer: cleared_steer.clone(),
            cleared_follow_up: cleared_follow_up.clone(),
        })
        .await;
        AbortResult { cleared_steer, cleared_follow_up }
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
        self.dispatch_queue_update().await;
    }

    pub async fn follow_up(&self, message: AgentMessage) {
        self.follow_up_queue.lock().await.push(message);
        self.dispatch_queue_update().await;
    }

    pub async fn next_turn(&self, message: AgentMessage) {
        self.next_turn_queue.lock().await.push(message);
        self.dispatch_queue_update().await;
    }

    pub async fn drain_steer(&self) -> Vec<AgentMessage> {
        let v = std::mem::take(&mut *self.steer_queue.lock().await);
        if !v.is_empty() {
            self.dispatch_queue_update().await;
        }
        v
    }

    pub async fn drain_follow_up(&self) -> Vec<AgentMessage> {
        let v = std::mem::take(&mut *self.follow_up_queue.lock().await);
        if !v.is_empty() {
            self.dispatch_queue_update().await;
        }
        v
    }

    pub async fn drain_next_turn(&self) -> Vec<AgentMessage> {
        let v = std::mem::take(&mut *self.next_turn_queue.lock().await);
        if !v.is_empty() {
            self.dispatch_queue_update().await;
        }
        v
    }

    /// Whether any steer / follow-up / next-turn message is queued. Used by the
    /// post-run check to drain messages an `agent_end` listener enqueued.
    pub async fn has_queued_messages(&self) -> bool {
        !self.steer_queue.lock().await.is_empty()
            || !self.follow_up_queue.lock().await.is_empty()
            || !self.next_turn_queue.lock().await.is_empty()
    }

    /// Current drain policy for the steering queue.
    pub async fn steer_mode(&self) -> crate::queue::QueueMode {
        *self.steer_mode.lock().await
    }

    /// Set the drain policy for the steering queue.
    pub async fn set_steer_mode(&self, mode: crate::queue::QueueMode) {
        *self.steer_mode.lock().await = mode;
    }

    /// Current drain policy for the follow-up queue.
    pub async fn follow_up_mode(&self) -> crate::queue::QueueMode {
        *self.follow_up_mode.lock().await
    }

    /// Set the drain policy for the follow-up queue.
    pub async fn set_follow_up_mode(&self, mode: crate::queue::QueueMode) {
        *self.follow_up_mode.lock().await = mode;
    }

    /// Install (or remove with `None`) an auto-compaction strategy. When
    /// installed, [`execute_turn`] will run [`check_pre_prompt`] before
    /// sending the prompt so an aborted / errored prior turn doesn't bleed
    /// into the new context.
    pub async fn set_auto_compaction(&self, config: Option<AutoCompactionConfig>) {
        *self.auto_compaction.lock().await = config;
    }

    async fn dispatch_queue_update(&self) {
        let steer = self.steer_queue.lock().await.len();
        let follow_up = self.follow_up_queue.lock().await.len();
        let next_turn = self.next_turn_queue.lock().await.len();
        self.dispatch(HarnessEvent::QueueUpdate {
            steer_count: steer,
            follow_up_count: follow_up,
            next_turn_count: next_turn,
        })
        .await;
    }

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    pub async fn set_system_prompt(&self, prompt: String) {
        self.config.lock().await.system_prompt = prompt;
    }

    pub async fn set_thinking_level(&self, level: ThinkingLevel) {
        let previous = self.config.lock().await.thinking_level;
        self.config.lock().await.thinking_level = level;
        self.pending.lock().await.push(PendingWrite::ThinkingLevelChange(level));
        self.dispatch(HarnessEvent::ThinkingLevelSelect { level, previous }).await;
    }

    pub async fn set_model(&self, provider: String, model_id: String) {
        let mut cfg = self.config.lock().await;
        let prev_provider = cfg.model_provider.clone();
        let prev_model = cfg.model_id.clone();
        cfg.model_provider = provider.clone();
        cfg.model_id = model_id.clone();
        drop(cfg);
        self.pending.lock().await.push(PendingWrite::ModelChange {
            provider: provider.clone(),
            model_id: model_id.clone(),
        });
        self.dispatch(HarnessEvent::ModelSelect {
            provider,
            model_id,
            previous_provider: Some(prev_provider),
            previous_model_id: Some(prev_model),
        })
        .await;
    }

    /// Install the available tool set and mark every supplied tool active.
    ///
    /// This setup-oriented method preserves the pre-migration behavior and does
    /// not write an `active_tools_change` entry. Use
    /// [`AgentHarness::set_tools`] or [`AgentHarness::set_active_tool_names`]
    /// for user-visible changes that should be persisted in the session tree.
    pub async fn set_active_tools(&self, tools: Vec<Arc<dyn AgentTool>>) {
        self.resources.lock().await.replace_tools(tools, None);
    }

    pub async fn set_tools(
        &self,
        tools: Vec<Arc<dyn AgentTool>>,
        active_tool_names: Option<Vec<String>>,
    ) -> Result<(), HarnessError> {
        validate_unique_names(
            &tools.iter().map(|tool| tool.name().to_string()).collect::<Vec<_>>(),
            "Duplicate tool name(s)",
        )?;
        let active_tool_names = active_tool_names
            .unwrap_or_else(|| tools.iter().map(|tool| tool.name().to_string()).collect());
        let mut next = HarnessResources::default();
        next.replace_tools(tools, Some(active_tool_names.clone()));
        next.validate_active_tool_names(&active_tool_names)?;

        if *self.phase.lock().await == HarnessPhase::Idle {
            self.flush_pending().await?;
            self.session
                .lock()
                .await
                .append_active_tools_change(active_tool_names.clone())
                .await?;
        } else {
            self.pending.lock().await.push(PendingWrite::ActiveToolsChange {
                active_tool_names: active_tool_names.clone(),
            });
        }

        let mut resources = self.resources.lock().await;
        resources.tools = next.tools;
        resources.active_tool_names = active_tool_names;
        Ok(())
    }

    pub async fn set_active_tool_names(&self, names: Vec<String>) -> Result<(), HarnessError> {
        validate_unique_names(&names, "Duplicate active tool name(s)")?;
        {
            let resources = self.resources.lock().await;
            resources.validate_active_tool_names(&names)?;
        }

        if *self.phase.lock().await == HarnessPhase::Idle {
            self.flush_pending().await?;
            self.session.lock().await.append_active_tools_change(names.clone()).await?;
        } else {
            self.pending
                .lock()
                .await
                .push(PendingWrite::ActiveToolsChange { active_tool_names: names.clone() });
        }
        self.resources.lock().await.active_tool_names = names;
        Ok(())
    }

    pub async fn registered_tool_names(&self) -> Vec<String> {
        self.resources.lock().await.tools.keys().cloned().collect()
    }

    pub async fn active_tool_names(&self) -> Vec<String> {
        self.resources.lock().await.active_tool_names.clone()
    }

    pub async fn active_tools(&self) -> Vec<Arc<dyn AgentTool>> {
        self.resources.lock().await.active_tools()
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

    /// Test / inspection accessor for the underlying session. Hold the lock
    /// briefly — the harness needs to acquire it during turns and writes.
    pub fn session(&self) -> Arc<Mutex<Session>> {
        self.session.clone()
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
        Ok(self
            .session
            .lock()
            .await
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
        Ok(self
            .session
            .lock()
            .await
            .append_branch_summary(from_id, summary, details, None)
            .await?)
    }

    /// Append (or queue, if not idle) a custom plugin entry.
    pub async fn record_custom(
        &self,
        custom_type: &str,
        data: Option<serde_json::Value>,
    ) -> Result<String, HarnessError> {
        if *self.phase.lock().await != HarnessPhase::Idle {
            self.pending.lock().await.push(PendingWrite::Custom {
                custom_type: custom_type.to_string(),
                data,
            });
            return Ok(String::new());
        }
        Ok(self.session.lock().await.append_custom(custom_type, data).await?)
    }

    /// Append (or queue) a custom plugin message.
    pub async fn record_custom_message(
        &self,
        custom_type: &str,
        content: serde_json::Value,
        display: bool,
        details: Option<serde_json::Value>,
    ) -> Result<String, HarnessError> {
        if *self.phase.lock().await != HarnessPhase::Idle {
            self.pending.lock().await.push(PendingWrite::CustomMessage {
                custom_type: custom_type.to_string(),
                content,
                display,
                details,
            });
            return Ok(String::new());
        }
        Ok(self
            .session
            .lock()
            .await
            .append_custom_message(custom_type, content, display, details)
            .await?)
    }

    /// Attach (or queue, or remove if `label` is `None`) a label on `target_id`.
    pub async fn record_label(
        &self,
        target_id: &str,
        label: Option<String>,
    ) -> Result<String, HarnessError> {
        if *self.phase.lock().await != HarnessPhase::Idle {
            self.pending
                .lock()
                .await
                .push(PendingWrite::Label { target_id: target_id.to_string(), label });
            return Ok(String::new());
        }
        Ok(self.session.lock().await.append_label(target_id, label).await?)
    }

    /// Update (or queue) the session-info name.
    pub async fn set_session_name(&self, name: &str) -> Result<String, HarnessError> {
        if *self.phase.lock().await != HarnessPhase::Idle {
            self.pending
                .lock()
                .await
                .push(PendingWrite::SessionInfo { name: Some(name.to_string()) });
            return Ok(String::new());
        }
        Ok(self.session.lock().await.append_session_name(name).await?)
    }

    /// Move (or queue a move of) the session leaf pointer.
    pub async fn move_leaf(&self, target_id: Option<&str>) -> Result<(), HarnessError> {
        if *self.phase.lock().await != HarnessPhase::Idle {
            self.pending.lock().await.push(PendingWrite::Leaf {
                target_id: target_id.map(|s| s.to_string()),
            });
            return Ok(());
        }
        self.session.lock().await.move_to(target_id, None).await?;
        Ok(())
    }

    pub async fn navigate_tree(
        &self,
        entry_id: Option<&str>,
        summary: Option<BranchSummaryOptions>,
    ) -> Result<Option<String>, HarnessError> {
        self.require_idle().await?;
        // Hold the BranchSummary phase for the duration of the call so a
        // concurrent turn / compaction can't race with branch summarization.
        *self.phase.lock().await = HarnessPhase::BranchSummary;
        self.flush_pending().await?;

        let result = self.session.lock().await.move_to(entry_id, summary).await;

        *self.phase.lock().await = HarnessPhase::Idle;
        self.dispatch(HarnessEvent::Settled).await;
        Ok(result?)
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
            self.session
                .lock()
                .await
                .append_compaction(
                    &result.summary,
                    &result.first_kept_entry_id,
                    result.tokens_before as u64,
                    Some(serde_json::json!({
                        "readFiles": result.read_files,
                        "modifiedFiles": result.modified_files,
                    })),
                    None,
                )
                .await?;
            Ok(Some(result))
        })
        .await;

        *self.phase.lock().await = HarnessPhase::Idle;

        if let Ok(Some(ref r)) = result {
            self.dispatch(HarnessEvent::Compaction { result: r.clone() }).await;
        }
        result
    }

    // -----------------------------------------------------------------------
    // Auto-compaction trigger + retry decision (the post-agent-run state
    // machine).
    // -----------------------------------------------------------------------

    /// Cached last assistant message from the just-completed turn. Returns
    /// `None` until a turn produces an assistant message.
    pub async fn last_assistant_message(&self) -> Option<AgentMessage> {
        self.last_assistant_message.lock().await.clone()
    }

    /// Test-only: override the cached last-assistant-message.
    #[doc(hidden)]
    pub async fn set_last_assistant_for_test(&self, msg: Option<AgentMessage>) {
        *self.last_assistant_message.lock().await = msg;
    }

    /// Test-only: read overflow_recovery_attempted flag.
    #[doc(hidden)]
    pub async fn overflow_recovery_attempted_for_test(&self) -> bool {
        *self.overflow_recovery_attempted.lock().await
    }

    /// Test-only: read retry_attempt counter.
    #[doc(hidden)]
    pub async fn retry_attempt_for_test(&self) -> u32 {
        *self.retry_attempt.lock().await
    }

    /// Configure the auto-retry settings (enabled / max_retries / base_delay_ms).
    pub async fn set_retry_settings(&self, settings: crate::auto_retry::RetrySettings) {
        *self.retry_settings.lock().await = settings;
    }

    /// Reset the overflow / retry state machines. Called automatically when a
    /// new prompt is submitted; callers rarely need to invoke directly.
    pub async fn reset_recovery_state(&self) {
        *self.overflow_recovery_attempted.lock().await = false;
        *self.retry_attempt.lock().await = 0;
    }

    /// Run the post-turn dispatch logic against the just-completed assistant
    /// message. Returns a `PostRunDecision` describing whether the caller
    /// should re-run the turn (compact-and-retry / auto-retry) or stop.
    ///
    /// Combines the auto-retry, context-overflow recovery, and threshold
    /// compaction checks. The harness itself does not loop — callers (or a
    /// higher-level wrapper) decide whether to continue based on the returned
    /// decision.
    pub async fn check_post_run(
        &self,
        compaction_settings: &CompactionSettings,
        compaction_stream_fn: &CompactionStreamFn,
    ) -> Result<PostRunDecision, HarnessError> {
        let decision = self
            .check_post_run_inner(compaction_settings, compaction_stream_fn, true)
            .await?;
        // The agent loop drains both queues before emitting agent_end; any
        // messages still queued here were enqueued by agent_end listeners and
        // need a continuation to be processed. (pi-mono fix a29a7902.)
        if decision == PostRunDecision::Stop && self.has_queued_messages().await {
            return Ok(PostRunDecision::DrainQueues);
        }
        Ok(decision)
    }

    /// Inner check used by both `check_post_run` (post-turn) and
    /// `check_pre_prompt` (pre-prompt). When `skip_aborted` is true, an
    /// aborted prior turn short-circuits to `Stop`. When false, aborted
    /// turns still flow into the threshold-compaction path so the
    /// follow-up prompt sees a clean context.
    async fn check_post_run_inner(
        &self,
        compaction_settings: &CompactionSettings,
        compaction_stream_fn: &CompactionStreamFn,
        skip_aborted: bool,
    ) -> Result<PostRunDecision, HarnessError> {
        let Some(msg) = self.last_assistant_message().await else {
            return Ok(PostRunDecision::Stop);
        };
        let AgentMessage::Assistant {
            stop_reason,
            timestamp: msg_timestamp,
            provider: msg_provider,
            model: msg_model,
            usage: msg_usage,
            ..
        } = &msg
        else {
            return Ok(PostRunDecision::Stop);
        };

        // Skip aborted / cancelled turns unless the caller (pre-prompt
        // path) explicitly opted in to handling them.
        if skip_aborted && *stop_reason == StopReason::Aborted {
            return Ok(PostRunDecision::Stop);
        }

        let cfg = self.config.lock().await;
        let cur_provider = cfg.model_provider.clone();
        let cur_model_id = cfg.model_id.clone();
        drop(cfg);

        let resources = self.resources.lock().await;
        let context_window = resources.model_info.as_ref().map(|m| m.context_window).unwrap_or(0);
        drop(resources);

        // Stale-message protection: a pre-compaction usage record must not
        // re-trigger compaction on the next prompt. Compare timestamps.
        if let Some(latest_compaction_ts) = self.latest_compaction_timestamp().await
            && (*msg_timestamp as i64) <= latest_compaction_ts
        {
            return Ok(PostRunDecision::Stop);
        }

        let same_model = msg_provider == &cur_provider && msg_model == &cur_model_id;

        // ── 1. Auto-retry on transient errors (5xx / 429 / network) ──
        // Tried before compaction because retry is cheaper. Overflow errors
        // are filtered out by `is_retryable_error`.
        if *stop_reason == StopReason::Error {
            let settings = *self.retry_settings.lock().await;
            if same_model && is_retryable_error(&msg, Some(context_window)) {
                let next_attempt = *self.retry_attempt.lock().await + 1;
                if let Some(delay_ms) = compute_retry_delay(next_attempt, &settings) {
                    *self.retry_attempt.lock().await = next_attempt;
                    let err = msg.assistant_error_message().unwrap_or_default();
                    self.dispatch(HarnessEvent::AutoRetryStart {
                        attempt: next_attempt,
                        max_attempts: settings.max_retries,
                        delay_ms,
                        error_message: err.to_string(),
                    })
                    .await;
                    // Drop the error message from session state so the retry
                    // doesn't re-include it as context.
                    self.drop_last_assistant_from_session().await;
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                    return Ok(PostRunDecision::Retry { attempt: next_attempt });
                } else {
                    // Exhausted; emit retry_end with the final failure.
                    let attempt = *self.retry_attempt.lock().await;
                    *self.retry_attempt.lock().await = 0;
                    self.dispatch(HarnessEvent::AutoRetryEnd {
                        success: false,
                        attempt,
                        final_error: msg.assistant_error_message().map(|s| s.to_string()),
                    })
                    .await;
                }
            }
        } else if *self.retry_attempt.lock().await > 0 {
            // Successful response after a retry — emit retry_end.
            let attempt = *self.retry_attempt.lock().await;
            *self.retry_attempt.lock().await = 0;
            self.dispatch(HarnessEvent::AutoRetryEnd {
                success: true,
                attempt,
                final_error: None,
            })
            .await;
        }

        // ── 2. Overflow path (drop error msg, compact, retry once) ──
        if same_model && is_context_overflow(&msg, Some(context_window)) {
            if *self.overflow_recovery_attempted.lock().await {
                self.dispatch(HarnessEvent::CompactionEnd {
                    reason: CompactionReason::Overflow,
                    result: None,
                    aborted: false,
                    will_retry: false,
                    error_message: Some(
                        "Context overflow recovery failed after one compact-and-retry attempt. \
                         Try reducing context or switching to a larger-context model."
                            .into(),
                    ),
                })
                .await;
                return Ok(PostRunDecision::Stop);
            }
            *self.overflow_recovery_attempted.lock().await = true;
            self.drop_last_assistant_from_session().await;
            self.run_auto_compaction(
                CompactionReason::Overflow,
                true,
                compaction_settings,
                compaction_stream_fn,
            )
            .await?;
            return Ok(PostRunDecision::CompactedRetry);
        }

        // ── 3. Threshold path — compact but don't retry. ──
        // For error-stop messages with no usage, fall back to estimating
        // from the last successful assistant in the branch.
        let context_tokens = if *stop_reason == StopReason::Error {
            let session_ctx = self.session.lock().await.build_context().await?;
            let est = estimate_context_tokens_with_source(&session_ctx.messages);
            let Some(idx) = est.last_usage_index else {
                return Ok(PostRunDecision::Stop); // No usage data anywhere.
            };
            // Verify the source isn't pre-compaction either.
            if let Some(latest_compaction_ts) = self.latest_compaction_timestamp().await
                && let Some(AgentMessage::Assistant { timestamp, .. }) =
                    session_ctx.messages.get(idx)
                && (*timestamp as i64) <= latest_compaction_ts
            {
                return Ok(PostRunDecision::Stop);
            }
            est.tokens
        } else {
            // Successful response — use the message's own usage.
            let total = msg_usage.total_tokens.max(
                msg_usage.input + msg_usage.output + msg_usage.cache_read + msg_usage.cache_write,
            );
            total as usize
        };

        if should_compact_at_threshold(context_tokens, context_window as usize, compaction_settings)
        {
            self.run_auto_compaction(
                CompactionReason::Threshold,
                false,
                compaction_settings,
                compaction_stream_fn,
            )
            .await?;
            // Threshold-triggered compaction does not auto-retry the turn.
        }
        Ok(PostRunDecision::Stop)
    }

    /// Internal helper that runs a compaction with `compaction_start` /
    /// `compaction_end` events.
    async fn run_auto_compaction(
        &self,
        reason: CompactionReason,
        will_retry: bool,
        settings: &CompactionSettings,
        stream_fn: &CompactionStreamFn,
    ) -> Result<(), HarnessError> {
        self.dispatch(HarnessEvent::CompactionStart { reason }).await;
        let result = self.compact(settings, stream_fn).await;
        match result {
            Ok(r) => {
                self.dispatch(HarnessEvent::CompactionEnd {
                    reason,
                    result: r,
                    aborted: false,
                    will_retry,
                    error_message: None,
                })
                .await;
                Ok(())
            }
            Err(e) => {
                let msg = e.to_string();
                self.dispatch(HarnessEvent::CompactionEnd {
                    reason,
                    result: None,
                    aborted: matches!(e, HarnessError::Compaction(CompactionError::Aborted)),
                    will_retry: false,
                    error_message: Some(msg),
                })
                .await;
                Err(e)
            }
        }
    }

    /// Walk the current branch in reverse to find the most recent compaction
    /// timestamp (millis since epoch), if any.
    async fn latest_compaction_timestamp(&self) -> Option<i64> {
        let branch = self.session.lock().await.get_branch().await.ok()?;
        for entry in branch.iter().rev() {
            if let SessionTreeEntry::Compaction { timestamp, .. } = entry {
                return chrono::DateTime::parse_from_rfc3339(timestamp)
                    .ok()
                    .map(|dt| dt.timestamp_millis());
            }
        }
        None
    }

    /// Drop the last assistant entry from the in-memory session view. Used
    /// before retry / overflow recovery so the context doesn't include the
    /// error message on the re-run.
    async fn drop_last_assistant_from_session(&self) {
        // The session model is append-only; we add a leaf-move write that
        // reverts to the parent of the last entry (equivalent to dropping the
        // last message from the in-memory transcript).
        let branch = match self.session.lock().await.get_branch().await {
            Ok(b) => b,
            Err(_) => return,
        };
        // Find the last assistant message entry.
        let target_parent = branch.iter().rev().find_map(|e| {
            if let SessionTreeEntry::Message { message, parent_id, .. } = e
                && matches!(message, AgentMessage::Assistant { .. })
            {
                return Some(parent_id.clone());
            }
            None
        });
        if let Some(parent) = target_parent.flatten() {
            let _ = self.session.lock().await.move_to(Some(&parent), None).await;
        }
    }
}

/// Decision returned by [`AgentHarness::check_post_run`].
#[derive(Debug, Clone, PartialEq)]
pub enum PostRunDecision {
    /// Caller should re-run the same turn — compaction completed (overflow
    /// recovery), error message has been dropped from the session view.
    CompactedRetry,
    /// Caller should re-run the same turn — auto-retry triggered after a
    /// transient error, error message dropped, backoff slept.
    Retry { attempt: u32 },
    /// Caller should run one more (empty-prompt) turn to drain steer /
    /// follow-up / next-turn messages an `agent_end` listener enqueued.
    DrainQueues,
    /// No further action; the turn is settled.
    Stop,
}

fn should_compact_at_threshold(
    context_tokens: usize,
    context_window: usize,
    settings: &CompactionSettings,
) -> bool {
    if !settings.enabled || context_window == 0 {
        return false;
    }
    // Soft cap is `(context_window - reserve_tokens)`: trigger compaction once
    // the estimated context tokens cross it.
    let cap = context_window.saturating_sub(settings.reserve_tokens);
    context_tokens >= cap
}

impl AgentHarness {
    // Turn execution — drives the agent loop with current resources / hooks.
    // -----------------------------------------------------------------------

    pub async fn execute_turn(
        &self,
        prompts: Vec<AgentMessage>,
    ) -> Result<Vec<AgentMessage>, HarnessError> {
        self.require_idle().await?;
        // Reset overflow + retry state at the start of every fresh user turn.
        self.reset_recovery_state().await;

        // Pre-prompt compaction check: if an auto-compaction strategy was
        // installed, check whether the prior turn left aborted / error state
        // and, if so, compact before the new prompt arrives.
        if let Some(cfg) = self.auto_compaction.lock().await.clone() {
            // `check_pre_prompt` does its own no-op early-exit when there's
            // nothing to clean up, so this is cheap on the happy path.
            self.check_pre_prompt(&cfg.settings, &cfg.stream_fn).await?;
        }

        self.continue_turn(prompts).await
    }

    /// Like [`execute_turn`] but does **not** reset overflow / retry counters.
    /// Used internally after `check_post_run` returns `CompactedRetry` /
    /// `Retry` so the recovery state machine carries across the re-run.
    ///
    /// Also accepts an optional compaction stream fn for the pre-prompt check:
    /// if the last assistant message is an aborted / overflow response, compact
    /// before running the new turn.
    pub async fn continue_turn(
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

    /// Pre-prompt compaction check: if the last assistant message is an
    /// aborted / overflow response, compact before the new turn so the
    /// context is clean. Returns `true` if compaction ran (caller may want
    /// to `continue_turn` with empty prompts to let the model re-run).
    pub async fn check_pre_prompt(
        &self,
        compaction_settings: &CompactionSettings,
        compaction_stream_fn: &CompactionStreamFn,
    ) -> Result<bool, HarnessError> {
        let Some(msg) = self.last_assistant_message().await else {
            return Ok(false);
        };
        let AgentMessage::Assistant { stop_reason, .. } = &msg else {
            return Ok(false);
        };
        // Only act on aborted or error-terminated turns.
        if !matches!(stop_reason, StopReason::Aborted | StopReason::Error) {
            return Ok(false);
        }
        // Pre-prompt path: process aborted messages too (skip_aborted=false).
        let dec = self
            .check_post_run_inner(compaction_settings, compaction_stream_fn, false)
            .await?;
        Ok(dec != PostRunDecision::Stop)
    }

    async fn run_turn(
        &self,
        prompts: Vec<AgentMessage>,
        signal: CancellationToken,
    ) -> Result<Vec<AgentMessage>, HarnessError> {
        // Build context + tools snapshot
        let session_ctx = self.session.lock().await.build_context().await?;
        let restored_active_tool_names = session_ctx.active_tool_names.clone();
        let messages: Vec<AgentMessage> = session_ctx.messages;

        let cfg = self.config.lock().await;
        let system_prompt = cfg.system_prompt.clone();
        let thinking_level = cfg.thinking_level;
        let model_provider = cfg.model_provider.clone();
        let model_id = cfg.model_id.clone();
        drop(cfg);

        let resources = self.resources.lock().await;
        let tools = restored_active_tool_names
            .as_deref()
            .map(|names| resources.tools_for_names(names))
            .unwrap_or_else(|| resources.active_tools());
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
                let mode = self.steer_mode.clone();
                Some(Arc::new(move || {
                    let q = q.clone();
                    let mode = mode.clone();
                    Box::pin(async move {
                        let mode = *mode.lock().await;
                        let mut guard = q.lock().await;
                        match mode {
                            crate::queue::QueueMode::All => std::mem::take(&mut *guard),
                            crate::queue::QueueMode::OneAtATime => {
                                if guard.is_empty() {
                                    Vec::new()
                                } else {
                                    vec![guard.remove(0)]
                                }
                            }
                        }
                    })
                }))
            },
            get_follow_up_messages: {
                let q = self.follow_up_queue.clone();
                let mode = self.follow_up_mode.clone();
                Some(Arc::new(move || {
                    let q = q.clone();
                    let mode = mode.clone();
                    Box::pin(async move {
                        let mode = *mode.lock().await;
                        let mut guard = q.lock().await;
                        match mode {
                            crate::queue::QueueMode::All => std::mem::take(&mut *guard),
                            crate::queue::QueueMode::OneAtATime => {
                                if guard.is_empty() {
                                    Vec::new()
                                } else {
                                    vec![guard.remove(0)]
                                }
                            }
                        }
                    })
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
            reasoning: opts.reasoning,
            temperature: opts.temperature,
            max_tokens: opts.max_tokens,
            provider_options: opts.provider_options.clone(),
        };

        let session = self.session.clone();
        let listeners = self.listeners.clone();
        let abort = self.abort.clone();
        let last_assistant = self.last_assistant_message.clone();
        let emit: AgentEventSink = Arc::new(move |event: AgentEvent| {
            let session = session.clone();
            let listeners = listeners.clone();
            let abort = abort.clone();
            let last_assistant = last_assistant.clone();
            Box::pin(async move {
                // Persist message_end events to the session and cache the
                // most recent assistant message for `_handlePostAgentRun`.
                if let AgentEvent::MessageEnd { ref message } = event {
                    let _ = session.lock().await.append_message(message.clone()).await;
                    if matches!(message, AgentMessage::Assistant { .. }) {
                        *last_assistant.lock().await = Some(message.clone());
                    }
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

        Ok(AgentLoop::new(&config, &emit, &self.stream_fn)
            .run(prompts, context, Some(signal))
            .await)
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
                PendingWrite::Message(msg) => {
                    session.append_message(msg).await?;
                }
                PendingWrite::ThinkingLevelChange(level) => {
                    session.append_thinking_level_change(level).await?;
                }
                PendingWrite::ModelChange { provider, model_id } => {
                    session.append_model_change(&provider, &model_id).await?;
                }
                PendingWrite::ActiveToolsChange { active_tool_names } => {
                    session.append_active_tools_change(active_tool_names).await?;
                }
                PendingWrite::Compaction {
                    summary,
                    first_kept_entry_id,
                    tokens_before,
                    details,
                } => {
                    session
                        .append_compaction(
                            &summary,
                            &first_kept_entry_id,
                            tokens_before,
                            details,
                            None,
                        )
                        .await?;
                }
                PendingWrite::BranchSummary { from_id, summary, details } => {
                    session.append_branch_summary(&from_id, &summary, details, None).await?;
                }
                PendingWrite::Custom { custom_type, data } => {
                    session.append_custom(&custom_type, data).await?;
                }
                PendingWrite::CustomMessage { custom_type, content, display, details } => {
                    session.append_custom_message(&custom_type, content, display, details).await?;
                }
                PendingWrite::Label { target_id, label } => {
                    session.append_label(&target_id, label).await?;
                }
                PendingWrite::SessionInfo { name } => {
                    if let Some(n) = name {
                        session.append_session_name(&n).await?;
                    }
                }
                PendingWrite::Leaf { target_id } => {
                    session.move_to(target_id.as_deref(), None).await?;
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
