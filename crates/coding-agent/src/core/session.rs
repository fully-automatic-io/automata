// CodingAgentSession — the end-to-end entry point for the coding agent.
//
// Wires together everything the lower layers provide: an `LlmProvider`
// (via the stream bridge), the stateful `AgentHarness` (session persistence,
// steering/follow-up queues, auto-compaction + auto-retry), the built-in
// coding tools, and the system prompt. Callers drive it with `prompt(text)`
// and observe progress by subscribing to `HarnessEvent`s.

use std::sync::Arc;

use agent_core::harness::compaction::{CompactionError, CompactionSettings, StreamFn as CompactionStreamFn};
use agent_core::harness::session::{InMemorySessionStorage, Session};
use agent_core::harness::{
    AgentHarness, AgentHarnessOptions, AutoCompactionConfig, HarnessConfig, HarnessError,
    HarnessEvent, PostRunDecision, StreamOptions,
};
use agent_core::auto_retry::RetrySettings;
use agent_core::tool::AgentTool;
use agent_core::types::{AgentMessage, ContentBlock, LlmRequest, Model, ModelInfo, ThinkingLevel};
use llm_client::provider::LlmProvider;
use tokio_util::sync::CancellationToken;

use super::provider::{build_provider, Auth, ProviderBuild};
use crate::stream_bridge::create_stream_fn;
use crate::tools::{
    BashTool, BashToolOptions, EditTool, EditToolOptions, FindTool, GrepTool, LsTool, ReadTool,
    ReadToolOptions, WriteTool, WriteToolOptions,
};

/// The default tool set, in display order.
pub const DEFAULT_TOOL_NAMES: &[&str] = &["read", "bash", "edit", "write", "grep", "find", "ls"];

/// Build the built-in coding tools for `cwd`, filtered to `names`.
pub fn build_tools(cwd: &str, names: &[&str]) -> Vec<Arc<dyn AgentTool>> {
    let mut tools: Vec<Arc<dyn AgentTool>> = Vec::with_capacity(names.len());
    for name in names {
        let tool: Arc<dyn AgentTool> = match *name {
            "read" => Arc::new(ReadTool::new(cwd.to_string(), ReadToolOptions::default())),
            "bash" => Arc::new(BashTool::new(cwd.to_string(), BashToolOptions::default())),
            "edit" => Arc::new(EditTool::new(cwd.to_string(), EditToolOptions::default())),
            "write" => Arc::new(WriteTool::new(cwd.to_string(), WriteToolOptions::default())),
            "grep" => Arc::new(GrepTool::new(cwd.to_string())),
            "find" => Arc::new(FindTool::new(cwd.to_string())),
            "ls" => Arc::new(LsTool::new(cwd.to_string())),
            _ => continue,
        };
        tools.push(tool);
    }
    tools
}

/// Build a compaction summarization callback backed by a provider's
/// non-streaming `complete` endpoint. Returns the concatenated text content.
fn make_compaction_stream_fn(provider: Arc<dyn LlmProvider>, model_id: String) -> CompactionStreamFn {
    Box::new(move |messages: Vec<AgentMessage>, system: &str| {
        let provider = provider.clone();
        let model_id = model_id.clone();
        let system = system.to_string();
        Box::pin(async move {
            let request = LlmRequest {
                model: model_id,
                messages,
                system: Some(system),
                max_tokens: Some(8192),
                ..Default::default()
            };
            let response = provider
                .complete(request)
                .await
                .map_err(|e| CompactionError::SummarizationFailed(e.to_string()))?;
            let text = response
                .content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
            Ok(text)
        })
    })
}

/// Maximum number of automatic re-runs (compaction recovery / transient retry)
/// per `prompt` call. The harness's own counters terminate sooner; this is a
/// defensive ceiling against an unexpected `Retry`/`CompactedRetry` cycle.
const MAX_AUTO_RERUNS: u32 = 16;

/// Options for [`CodingAgentSession::builder`].
pub struct SessionOptions {
    /// Working directory the tools operate in.
    pub cwd: String,
    /// The model to drive the turn loop with.
    pub model: Model,
    /// Credential for the model's provider.
    pub api_key: String,
    /// Override the provider's default endpoint (full URL).
    pub base_url: Option<String>,
    /// Authentication scheme override.
    pub auth: Auth,
    /// System prompt sent on every turn.
    pub system_prompt: String,
    /// Reasoning level (clamped to the model's capability).
    pub thinking_level: ThinkingLevel,
    /// Tool names to enable. Defaults to [`DEFAULT_TOOL_NAMES`] when `None`.
    pub tools: Option<Vec<String>>,
    /// Compaction policy. Auto-compaction is wired only when `Some`.
    pub compaction: Option<CompactionSettings>,
    /// Transient-error retry policy.
    pub retry: RetrySettings,
}

impl SessionOptions {
    /// Minimal options: everything but `cwd`, `model`, and `api_key` defaulted.
    pub fn new(cwd: impl Into<String>, model: Model, api_key: impl Into<String>) -> Self {
        Self {
            cwd: cwd.into(),
            model,
            api_key: api_key.into(),
            base_url: None,
            auth: Auth::Native,
            system_prompt: String::new(),
            thinking_level: ThinkingLevel::Off,
            tools: None,
            compaction: Some(CompactionSettings::default()),
            retry: RetrySettings::default(),
        }
    }
}

/// The end-to-end coding agent: a configured [`AgentHarness`] plus the
/// compaction strategy needed to drive `prompt` to completion.
pub struct CodingAgentSession {
    harness: AgentHarness,
    compaction_settings: CompactionSettings,
    compaction_stream_fn: Arc<CompactionStreamFn>,
}

impl CodingAgentSession {
    /// Build a session over an in-memory transcript. For a persisted session,
    /// construct a [`Session`] from a `JsonlSessionRepo` and call
    /// [`CodingAgentSession::with_session`].
    pub async fn builder(options: SessionOptions) -> Self {
        let session = Session::new(Box::new(InMemorySessionStorage::new(None)));
        Self::with_session(session, options).await
    }

    /// Build a session over a caller-provided [`Session`] (e.g. a persisted
    /// JSONL session opened or created via a repo).
    pub async fn with_session(session: Session, options: SessionOptions) -> Self {
        let provider = build_provider(ProviderBuild {
            model: &options.model,
            api_key: options.api_key.clone(),
            base_url: options.base_url.clone(),
            auth: options.auth,
        });
        Self::with_provider(session, provider, options).await
    }

    /// Build a session over a caller-supplied provider. Useful for tests
    /// (inject a stub provider) or custom transports. `options.api_key` /
    /// `base_url` / `auth` are ignored here since the provider is built.
    pub async fn with_provider(
        session: Session,
        provider: Arc<dyn LlmProvider>,
        options: SessionOptions,
    ) -> Self {
        let SessionOptions {
            cwd,
            model,
            system_prompt,
            thinking_level,
            tools,
            compaction,
            retry,
            ..
        } = options;

        let stream_fn = create_stream_fn(provider.clone(), model.clone());
        let compaction_stream_fn = make_compaction_stream_fn(provider, model.id.clone());

        let harness = AgentHarness::new(
            session,
            HarnessConfig {
                system_prompt,
                thinking_level: model
                    .clamp_reasoning(Some(thinking_level))
                    .unwrap_or(ThinkingLevel::Off),
                model_provider: model.provider.clone(),
                model_id: model.id.clone(),
            },
            AgentHarnessOptions {
                stream_fn,
                convert_to_llm: None,
                transform_context: None,
                before_tool_call: None,
                after_tool_call: None,
                should_stop_after_turn: None,
                prepare_next_turn: None,
                on_payload: None,
                on_response: None,
            },
        );

        let tool_names: Vec<&str> = tools
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect())
            .unwrap_or_else(|| DEFAULT_TOOL_NAMES.to_vec());
        let built_tools = build_tools(&cwd, &tool_names);

        let compaction_settings = compaction.clone().unwrap_or_default();
        let compaction_stream_fn = Arc::new(compaction_stream_fn);

        // Thread resources + recovery policy through the harness.
        harness.set_active_tools(built_tools).await;
        harness.set_model_info(ModelInfo::from(&model)).await;
        harness.set_retry_settings(retry).await;
        // Wire the pre-prompt auto-compaction check only when enabled, so an
        // aborted/overflow prior turn is cleaned up before the next prompt.
        if compaction.is_some() {
            harness
                .set_auto_compaction(Some(AutoCompactionConfig {
                    settings: compaction_settings.clone(),
                    stream_fn: compaction_stream_fn.clone(),
                }))
                .await;
        }

        Self {
            harness,
            compaction_settings,
            compaction_stream_fn,
        }
    }

    /// The harness powering this session. Use it to subscribe to events,
    /// inspect the session tree, or steer/follow-up.
    pub fn harness(&self) -> &AgentHarness {
        &self.harness
    }

    /// Subscribe to harness events (assistant deltas, tool execution,
    /// compaction, auto-retry, ...).
    pub async fn subscribe<F, Fut>(&self, listener: F)
    where
        F: Fn(HarnessEvent, Option<CancellationToken>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        self.harness.subscribe(listener).await;
    }

    /// Run a full prompt to completion: execute the user turn, then consult the
    /// harness's post-run state machine (auto-retry on transient errors,
    /// compact-and-retry on context overflow, threshold compaction) and re-run
    /// as directed until the turn settles.
    ///
    /// Returns the messages produced by the final (settled) turn.
    pub async fn prompt(&self, text: impl Into<String>) -> Result<Vec<AgentMessage>, HarnessError> {
        let prompt = AgentMessage::user_text(text);
        let mut messages = self.harness.execute_turn(vec![prompt]).await?;

        for _ in 0..MAX_AUTO_RERUNS {
            let decision = self
                .harness
                .check_post_run(&self.compaction_settings, &self.compaction_stream_fn)
                .await?;
            match decision {
                PostRunDecision::Stop => break,
                // Both recovery paths re-run the same turn with no new prompt;
                // the harness has already dropped the failed assistant message
                // and (for overflow) compacted before we get here.
                PostRunDecision::Retry { .. } | PostRunDecision::CompactedRetry => {
                    messages = self.harness.continue_turn(vec![]).await?;
                }
            }
        }

        Ok(messages)
    }

    /// Manually compact the session history now (ignores the threshold).
    pub async fn compact(&self) -> Result<Option<agent_core::harness::CompactionResult>, HarnessError> {
        self.harness
            .compact(&self.compaction_settings, &self.compaction_stream_fn)
            .await
    }

    /// Queue a steering message to inject after the current turn's tool batch.
    pub async fn steer(&self, text: impl Into<String>) {
        self.harness.steer(AgentMessage::user_text(text)).await;
    }

    /// Queue a follow-up message to run once the agent would otherwise stop.
    pub async fn follow_up(&self, text: impl Into<String>) {
        self.harness.follow_up(AgentMessage::user_text(text)).await;
    }

    /// Cancel the in-flight turn and clear pending queues.
    pub async fn abort(&self) -> agent_core::harness::AbortResult {
        self.harness.abort().await
    }

    /// Set/override stream options (temperature, max_tokens, headers, ...).
    pub async fn set_stream_options(&self, options: StreamOptions) {
        self.harness.set_stream_options(options).await;
    }
}

