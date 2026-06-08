// CodingAgentSession — the end-to-end entry point for the coding agent.
//
// Wires together everything the lower layers provide: an `LlmProvider`
// (via the stream bridge), the stateful `AgentHarness` (session persistence,
// steering/follow-up queues, auto-compaction + auto-retry), the built-in
// coding tools, and the system prompt. Callers drive it with `prompt(text)`
// and observe progress by subscribing to `HarnessEvent`s.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use agent_core::auto_retry::RetrySettings;
use agent_core::harness::compaction::{
    CompactionError, CompactionSettings, StreamFn as CompactionStreamFn,
};
use agent_core::harness::session::{InMemorySessionStorage, Session};
use agent_core::harness::{
    AgentHarness, AgentHarnessOptions, AutoCompactionConfig, HarnessConfig, HarnessError,
    HarnessEvent, PostRunDecision, StreamOptions,
};
use agent_core::tool::AgentTool;
use agent_core::types::{AgentMessage, ContentBlock, LlmRequest, Model, ModelInfo, ThinkingLevel};
use llm_client::provider::LlmProvider;
use tokio_util::sync::CancellationToken;

use super::provider::{Auth, ProviderBuild, build_provider};
use crate::extensions::{
    ExtensionEvent, ExtensionRunner, LoadExtensionsResult, SessionLifecycleReason,
    extension_after_tool_call_hook, extension_agent_tools, extension_before_tool_call_hook,
    extension_on_payload_hook, extension_on_response_hook, extension_transform_context_hook,
    subscribe_extension_harness_events,
};
use crate::stream_bridge::create_stream_fn;
use crate::tools::{
    BashTool, BashToolOptions, EditTool, EditToolOptions, FindTool, GrepTool, LsTool, ReadTool,
    ReadToolOptions, WriteTool, WriteToolOptions,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinTool {
    Read,
    Bash,
    Edit,
    Write,
    Grep,
    Find,
    Ls,
}

impl BuiltinTool {
    pub fn name(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Bash => "bash",
            Self::Edit => "edit",
            Self::Write => "write",
            Self::Grep => "grep",
            Self::Find => "find",
            Self::Ls => "ls",
        }
    }
}

impl std::fmt::Display for BuiltinTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl TryFrom<&str> for BuiltinTool {
    type Error = ToolSelectionError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "read" => Ok(Self::Read),
            "bash" => Ok(Self::Bash),
            "edit" => Ok(Self::Edit),
            "write" => Ok(Self::Write),
            "grep" => Ok(Self::Grep),
            "find" => Ok(Self::Find),
            "ls" => Ok(Self::Ls),
            other => Err(ToolSelectionError::UnknownTool(other.to_string())),
        }
    }
}

pub const ALL_BUILTIN_TOOLS: [BuiltinTool; 7] = [
    BuiltinTool::Read,
    BuiltinTool::Bash,
    BuiltinTool::Edit,
    BuiltinTool::Write,
    BuiltinTool::Grep,
    BuiltinTool::Find,
    BuiltinTool::Ls,
];

pub const DEFAULT_ACTIVE_TOOLS: [BuiltinTool; 4] =
    [BuiltinTool::Read, BuiltinTool::Bash, BuiltinTool::Edit, BuiltinTool::Write];

pub const READ_ONLY_TOOLS: [BuiltinTool; 4] =
    [BuiltinTool::Read, BuiltinTool::Grep, BuiltinTool::Find, BuiltinTool::Ls];

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ToolName(String);

impl ToolName {
    pub fn new(name: impl Into<String>) -> Result<Self, ToolSelectionError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(ToolSelectionError::EmptyToolName);
        }
        Ok(Self(name))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl From<BuiltinTool> for ToolName {
    fn from(tool: BuiltinTool) -> Self {
        Self(tool.name().to_string())
    }
}

impl std::fmt::Display for ToolName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolPreset {
    Coding,
    ReadOnly,
    All,
    None,
}

impl ToolPreset {
    fn tools(self) -> &'static [BuiltinTool] {
        match self {
            Self::Coding => &DEFAULT_ACTIVE_TOOLS,
            Self::ReadOnly => &READ_ONLY_TOOLS,
            Self::All => &ALL_BUILTIN_TOOLS,
            Self::None => &[],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSelection {
    preset: ToolPreset,
    allow: Option<Vec<ToolName>>,
    exclude: Vec<ToolName>,
    restore_session: bool,
}

impl Default for ToolSelection {
    fn default() -> Self {
        Self {
            preset: ToolPreset::Coding,
            allow: None,
            exclude: Vec::new(),
            restore_session: true,
        }
    }
}

impl ToolSelection {
    pub fn preset(preset: ToolPreset) -> Self {
        Self {
            preset,
            allow: None,
            exclude: Vec::new(),
            restore_session: false,
        }
    }

    pub fn restore_or(preset: ToolPreset) -> Self {
        Self {
            restore_session: true,
            ..Self::preset(preset)
        }
    }

    pub fn coding() -> Self {
        Self::preset(ToolPreset::Coding)
    }

    pub fn read_only() -> Self {
        Self::preset(ToolPreset::ReadOnly)
    }

    pub fn all() -> Self {
        Self::preset(ToolPreset::All)
    }

    pub fn none() -> Self {
        Self::preset(ToolPreset::None)
    }

    pub fn only<I, T>(tools: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<ToolName>,
    {
        Self {
            preset: ToolPreset::None,
            allow: Some(tools.into_iter().map(Into::into).collect()),
            exclude: Vec::new(),
            restore_session: false,
        }
    }

    pub fn from_names<I, S>(names: I) -> Result<Self, ToolSelectionError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let names = names
            .into_iter()
            .map(|name| ToolName::new(name.as_ref().to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        validate_unique_tool_names(&names)?;
        Ok(Self {
            preset: ToolPreset::None,
            allow: Some(names),
            exclude: Vec::new(),
            restore_session: false,
        })
    }

    pub fn exclude<I, T>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<ToolName>,
    {
        self.exclude = tools.into_iter().map(Into::into).collect();
        self
    }

    pub fn restore_session(mut self, restore_session: bool) -> Self {
        self.restore_session = restore_session;
        self
    }

    pub fn active_names(
        &self,
        restored_names: Option<Vec<String>>,
    ) -> Result<Vec<ToolName>, ToolSelectionError> {
        self.active_names_with_extensions(restored_names, &[])
    }

    pub fn active_names_with_extensions(
        &self,
        restored_names: Option<Vec<String>>,
        extension_names: &[String],
    ) -> Result<Vec<ToolName>, ToolSelectionError> {
        let base = if self.restore_session
            && self.allow.is_none()
            && let Some(restored_names) = restored_names
            && !restored_names.is_empty()
        {
            restored_names.into_iter().map(ToolName::new).collect::<Result<Vec<_>, _>>()?
        } else if let Some(allow) = &self.allow {
            allow.clone()
        } else {
            let mut names =
                self.preset.tools().iter().copied().map(ToolName::from).collect::<Vec<_>>();
            if matches!(self.preset, ToolPreset::Coding | ToolPreset::All) {
                names.extend(
                    extension_names
                        .iter()
                        .map(|name| ToolName::new(name.clone()))
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            names
        };

        let excluded: HashSet<&str> = self.exclude.iter().map(ToolName::as_str).collect();
        let active = base
            .into_iter()
            .filter(|name| !excluded.contains(name.as_str()))
            .collect::<Vec<_>>();
        validate_unique_tool_names(&active)?;
        Ok(active)
    }

    pub fn should_write_active_tools(&self, has_restored_tools: bool) -> bool {
        self.allow.is_some() || !self.restore_session || !has_restored_tools
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolSelectionError {
    #[error("tool name cannot be empty")]
    EmptyToolName,
    #[error("duplicate tool name: {0}")]
    DuplicateTool(String),
    #[error("unknown tool: {0}")]
    UnknownTool(String),
}

#[derive(Debug, thiserror::Error)]
pub enum SessionBuildError {
    #[error(transparent)]
    ToolSelection(#[from] ToolSelectionError),
    #[error(transparent)]
    Harness(#[from] HarnessError),
    #[error("no model selected")]
    NoModelSelected,
    #[error("no API key available for provider: {0}")]
    NoApiKey(String),
}

#[derive(Clone)]
pub struct BuiltTools {
    pub tools: Vec<Arc<dyn AgentTool>>,
    pub names: Vec<String>,
}

#[derive(Clone, Default)]
pub struct BuildToolsOptions {
    pub shell_path: Option<String>,
    pub shell_command_prefix: Option<String>,
    pub extension_tools: Vec<Arc<dyn AgentTool>>,
}

impl std::fmt::Debug for BuildToolsOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuildToolsOptions")
            .field("shell_path", &self.shell_path)
            .field("shell_command_prefix", &self.shell_command_prefix)
            .field("extension_tool_count", &self.extension_tools.len())
            .finish()
    }
}

/// Build the selected built-in coding tools for `cwd`.
pub fn build_tools(cwd: &str, selection: &ToolSelection) -> Result<BuiltTools, ToolSelectionError> {
    build_tools_with_options(cwd, selection, None, BuildToolsOptions::default())
}

/// Build the built-in coding tools with runtime settings applied.
pub fn build_tools_with_options(
    cwd: &str,
    selection: &ToolSelection,
    restored_names: Option<Vec<String>>,
    options: BuildToolsOptions,
) -> Result<BuiltTools, ToolSelectionError> {
    let extension_names = options
        .extension_tools
        .iter()
        .map(|tool| tool.name().to_string())
        .collect::<Vec<_>>();
    let names = selection.active_names_with_extensions(restored_names, &extension_names)?;
    build_tools_from_names(cwd, &names, options)
}

pub fn build_tools_from_names(
    cwd: &str,
    names: &[ToolName],
    options: BuildToolsOptions,
) -> Result<BuiltTools, ToolSelectionError> {
    let mut tools: Vec<Arc<dyn AgentTool>> = Vec::with_capacity(names.len());
    let mut tool_names = Vec::with_capacity(names.len());
    let extension_tools = options
        .extension_tools
        .iter()
        .map(|tool| (tool.name().to_string(), tool.clone()))
        .collect::<HashMap<_, _>>();
    for name in names {
        let tool: Arc<dyn AgentTool> = match BuiltinTool::try_from(name.as_str()) {
            Ok(builtin) => match builtin {
                BuiltinTool::Read => {
                    Arc::new(ReadTool::new(cwd.to_string(), ReadToolOptions::default()))
                }
                BuiltinTool::Bash => Arc::new(BashTool::new(
                    cwd.to_string(),
                    BashToolOptions {
                        shell_path: options.shell_path.clone(),
                        command_prefix: options.shell_command_prefix.clone(),
                        ..Default::default()
                    },
                )),
                BuiltinTool::Edit => {
                    Arc::new(EditTool::new(cwd.to_string(), EditToolOptions::default()))
                }
                BuiltinTool::Write => {
                    Arc::new(WriteTool::new(cwd.to_string(), WriteToolOptions::default()))
                }
                BuiltinTool::Grep => Arc::new(GrepTool::new(cwd.to_string())),
                BuiltinTool::Find => Arc::new(FindTool::new(cwd.to_string())),
                BuiltinTool::Ls => Arc::new(LsTool::new(cwd.to_string())),
            },
            Err(ToolSelectionError::UnknownTool(_)) => extension_tools
                .get(name.as_str())
                .cloned()
                .ok_or_else(|| ToolSelectionError::UnknownTool(name.as_str().to_string()))?,
            Err(err) => return Err(err),
        };
        tools.push(tool);
        tool_names.push(name.as_str().to_string());
    }
    Ok(BuiltTools { tools, names: tool_names })
}

fn validate_unique_tool_names(names: &[ToolName]) -> Result<(), ToolSelectionError> {
    let mut seen = HashSet::new();
    for name in names {
        if !seen.insert(name.as_str()) {
            return Err(ToolSelectionError::DuplicateTool(name.as_str().to_string()));
        }
    }
    Ok(())
}

/// Build a compaction summarization callback backed by a provider's
/// non-streaming `complete` endpoint. Returns the concatenated text content.
fn make_compaction_stream_fn(
    provider: Arc<dyn LlmProvider>,
    model_id: String,
) -> CompactionStreamFn {
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
    /// Built-in tool selection. Defaults to the coding preset and restores
    /// active tools from persisted sessions when available.
    pub tools: ToolSelection,
    /// Optional shell executable used by the bash tool.
    pub shell_path: Option<String>,
    /// Optional prefix prepended to every bash command.
    pub shell_command_prefix: Option<String>,
    /// Wasmtime component extensions already loaded by the resource layer.
    pub extensions: Option<LoadExtensionsResult>,
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
            tools: ToolSelection::default(),
            shell_path: None,
            shell_command_prefix: None,
            extensions: None,
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
    pub async fn builder(options: SessionOptions) -> Result<Self, SessionBuildError> {
        let session = Session::new(Box::new(InMemorySessionStorage::new(None)));
        Self::with_session(session, options).await
    }

    /// Build a session over a caller-provided [`Session`] (e.g. a persisted
    /// JSONL session opened or created via a repo).
    pub async fn with_session(
        session: Session,
        options: SessionOptions,
    ) -> Result<Self, SessionBuildError> {
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
    ) -> Result<Self, SessionBuildError> {
        let SessionOptions {
            cwd,
            model,
            system_prompt,
            thinking_level,
            tools,
            shell_path,
            shell_command_prefix,
            extensions,
            compaction,
            retry,
            ..
        } = options;

        let stream_fn = create_stream_fn(provider.clone(), model.clone());
        let compaction_stream_fn = make_compaction_stream_fn(provider, model.id.clone());
        let extension_tools = extensions.as_ref().map(extension_agent_tools).unwrap_or_default();
        let extension_runner = extensions.map(|extensions| {
            let mut runner = ExtensionRunner::new();
            runner.load(extensions);
            Arc::new(runner)
        });

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
                transform_context: extension_runner
                    .as_ref()
                    .map(|runner| extension_transform_context_hook(runner.clone())),
                before_tool_call: extension_runner
                    .as_ref()
                    .map(|runner| extension_before_tool_call_hook(runner.clone())),
                after_tool_call: extension_runner
                    .as_ref()
                    .map(|runner| extension_after_tool_call_hook(runner.clone())),
                should_stop_after_turn: None,
                prepare_next_turn: None,
                on_payload: extension_runner
                    .as_ref()
                    .map(|runner| extension_on_payload_hook(runner.clone())),
                on_response: extension_runner
                    .as_ref()
                    .map(|runner| extension_on_response_hook(runner.clone())),
            },
        );

        let built_tools = build_tools_with_options(
            &cwd,
            &tools,
            None,
            BuildToolsOptions {
                extension_tools,
                shell_path,
                shell_command_prefix,
            },
        )?;

        let compaction_settings = compaction.clone().unwrap_or_default();
        let compaction_stream_fn = Arc::new(compaction_stream_fn);

        // Thread resources + recovery policy through the harness.
        harness.set_tools(built_tools.tools, Some(built_tools.names)).await?;
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
        if let Some(runner) = &extension_runner {
            let _ = runner.dispatch_event(&ExtensionEvent::SessionStart {
                reason: SessionLifecycleReason::Startup,
                previous_session_file: None,
            });
            subscribe_extension_harness_events(&harness, runner.clone()).await;
        }

        Ok(Self {
            harness,
            compaction_settings,
            compaction_stream_fn,
        })
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
                // All re-run paths continue the same turn with no new prompt;
                // the loop drains steer / follow-up queues itself, and the
                // harness has already dropped any failed assistant message
                // (and, for overflow, compacted) before we get here.
                PostRunDecision::Retry { .. }
                | PostRunDecision::CompactedRetry
                | PostRunDecision::DrainQueues => {
                    messages = self.harness.continue_turn(vec![]).await?;
                }
            }
        }

        Ok(messages)
    }

    /// Manually compact the session history now (ignores the threshold).
    pub async fn compact(
        &self,
    ) -> Result<Option<agent_core::harness::CompactionResult>, HarnessError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tool_selection_uses_coding_preset() {
        let names = ToolSelection::default().active_names(None).unwrap();
        assert_eq!(
            names.iter().map(ToolName::as_str).collect::<Vec<_>>(),
            vec!["read", "bash", "edit", "write"]
        );
    }

    #[test]
    fn coding_preset_includes_extension_tools() {
        let names = ToolSelection::default()
            .active_names_with_extensions(None, &["project_status".to_string()])
            .unwrap();
        assert_eq!(
            names.iter().map(ToolName::as_str).collect::<Vec<_>>(),
            vec!["read", "bash", "edit", "write", "project_status"]
        );
    }

    #[test]
    fn read_only_preset_excludes_extension_tools_by_default() {
        let names = ToolSelection::read_only()
            .active_names_with_extensions(None, &["project_status".to_string()])
            .unwrap();
        assert_eq!(
            names.iter().map(ToolName::as_str).collect::<Vec<_>>(),
            vec!["read", "grep", "find", "ls"]
        );
    }

    #[test]
    fn tool_selection_rejects_unknown_names() {
        let selection = ToolSelection::from_names(["read", "missing"]).unwrap();
        let err = match build_tools("/tmp", &selection) {
            Ok(_) => panic!("expected unknown tool to fail"),
            Err(err) => err,
        };
        assert!(matches!(err, ToolSelectionError::UnknownTool(name) if name == "missing"));
    }
}
