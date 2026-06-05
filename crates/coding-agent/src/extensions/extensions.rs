// Extension system — Wasmtime-based plugin model (types + loader + runner merged).

use agent_core::tool::AgentTool;
use agent_core::types::{
    AfterToolCallResult, AgentMessage, AgentToolResult, BeforeToolCallResult, ContentBlock,
    ToolExecutionMode,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use wasmtime::component::{Component, HasSelf, Linker};
use wasmtime::{Config, Engine, Store};

mod component_abi {
    wasmtime::component::bindgen!({
        world: "extension",
        inline: r#"
            package automata:plugin;

            world extension {
                import host-action: func(action-json: string) -> result<string, string>;

                export register: func() -> result<string, string>;
                export on-event: func(event-json: string) -> result<option<string>, string>;
                export invoke-tool: func(request-json: string) -> result<string, string>;
            }
        "#,
    });
}

// ── Types ─────────────────────────────────────────────────────────────────────

/// Lifecycle reason for session-level extension events. Closed set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionLifecycleReason {
    Startup,
    Reload,
    New,
    Resume,
    Fork,
    Quit,
}

/// Events dispatched to WASM extensions.
///
/// Several variants carry `serde_json::Value` (or `Vec<Value>`) for fields like
/// `preparation`, `payload`, `message`, `messages`, `branchEntries` —
/// these cross a WASM boundary as raw JSON and have to remain shape-flexible
/// so plugins built against older / newer schemas keep deserializing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExtensionEvent {
    #[serde(rename = "resources_discover")]
    ResourcesDiscover {
        cwd: String,
        reason: SessionLifecycleReason,
    },
    #[serde(rename = "session_start")]
    SessionStart {
        reason: SessionLifecycleReason,
        #[serde(
            rename = "previousSessionFile",
            skip_serializing_if = "Option::is_none"
        )]
        previous_session_file: Option<String>,
    },
    #[serde(rename = "session_before_switch")]
    SessionBeforeSwitch {
        reason: SessionLifecycleReason,
        #[serde(rename = "targetSessionFile", skip_serializing_if = "Option::is_none")]
        target_session_file: Option<String>,
    },
    #[serde(rename = "session_before_fork")]
    SessionBeforeFork {
        #[serde(rename = "entryId")]
        entry_id: String,
        position: String,
    },
    #[serde(rename = "session_before_compact")]
    SessionBeforeCompact {
        preparation: serde_json::Value,
        #[serde(rename = "branchEntries")]
        branch_entries: Vec<serde_json::Value>,
        #[serde(rename = "customInstructions", skip_serializing_if = "Option::is_none")]
        custom_instructions: Option<String>,
    },
    #[serde(rename = "session_compact")]
    SessionCompact {
        #[serde(rename = "compactionEntry")]
        compaction_entry: serde_json::Value,
        #[serde(rename = "fromExtension")]
        from_extension: bool,
    },
    #[serde(rename = "session_shutdown")]
    SessionShutdown {
        reason: SessionLifecycleReason,
        #[serde(rename = "targetSessionFile", skip_serializing_if = "Option::is_none")]
        target_session_file: Option<String>,
    },
    #[serde(rename = "session_before_tree")]
    SessionBeforeTree { preparation: serde_json::Value },
    #[serde(rename = "session_tree")]
    SessionTree {
        #[serde(rename = "newLeafId")]
        new_leaf_id: Option<String>,
        #[serde(rename = "oldLeafId")]
        old_leaf_id: Option<String>,
        #[serde(rename = "summaryEntry", skip_serializing_if = "Option::is_none")]
        summary_entry: Option<serde_json::Value>,
        #[serde(rename = "fromExtension", skip_serializing_if = "Option::is_none")]
        from_extension: Option<bool>,
    },
    #[serde(rename = "context")]
    Context { messages: Vec<serde_json::Value> },
    #[serde(rename = "before_provider_request")]
    BeforeProviderRequest { payload: serde_json::Value },
    #[serde(rename = "after_provider_response")]
    AfterProviderResponse {
        status: u16,
        headers: HashMap<String, String>,
    },
    #[serde(rename = "before_agent_start")]
    BeforeAgentStart {
        prompt: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<serde_json::Value>>,
        #[serde(rename = "systemPrompt")]
        system_prompt: String,
        #[serde(rename = "systemPromptOptions")]
        system_prompt_options: serde_json::Value,
    },
    #[serde(rename = "agent_start")]
    AgentStart,
    #[serde(rename = "agent_end")]
    AgentEnd { messages: Vec<serde_json::Value> },
    #[serde(rename = "turn_start")]
    TurnStart {
        #[serde(rename = "turnIndex")]
        turn_index: u64,
        timestamp: u64,
    },
    #[serde(rename = "turn_end")]
    TurnEnd {
        #[serde(rename = "turnIndex")]
        turn_index: u64,
        message: serde_json::Value,
        #[serde(rename = "toolResults")]
        tool_results: Vec<serde_json::Value>,
    },
    #[serde(rename = "message_start")]
    MessageStart { message: serde_json::Value },
    #[serde(rename = "message_update")]
    MessageUpdate {
        message: serde_json::Value,
        #[serde(rename = "assistantMessageEvent")]
        assistant_message_event: serde_json::Value,
    },
    #[serde(rename = "message_end")]
    MessageEnd { message: serde_json::Value },
    #[serde(rename = "tool_execution_start")]
    ToolExecutionStart {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        args: serde_json::Value,
    },
    #[serde(rename = "tool_execution_update")]
    ToolExecutionUpdate {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        args: serde_json::Value,
        #[serde(rename = "partialResult")]
        partial_result: serde_json::Value,
    },
    #[serde(rename = "tool_execution_end")]
    ToolExecutionEnd {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        result: serde_json::Value,
        #[serde(rename = "isError")]
        is_error: bool,
    },
    #[serde(rename = "model_select")]
    ModelSelect {
        model: serde_json::Value,
        #[serde(rename = "previousModel", skip_serializing_if = "Option::is_none")]
        previous_model: Option<serde_json::Value>,
        source: String,
    },
    #[serde(rename = "user_bash")]
    UserBash {
        command: String,
        #[serde(rename = "excludeFromContext")]
        exclude_from_context: bool,
        cwd: String,
    },
    #[serde(rename = "input")]
    Input {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<serde_json::Value>>,
        source: String,
    },
    #[serde(rename = "tool_call")]
    ToolCall {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        input: serde_json::Value,
        content: Vec<serde_json::Value>,
        #[serde(rename = "isError")]
        is_error: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<serde_json::Value>,
    },
    #[serde(rename = "event_bus")]
    EventBus { event: String, data: serde_json::Value },
}

#[derive(Debug, Clone)]
pub struct ExtensionContext {
    pub cwd: String,
    pub model: Option<serde_json::Value>,
    pub signal: Option<tokio_util::sync::CancellationToken>,
}

impl ExtensionContext {
    pub fn new(cwd: String) -> Self {
        Self { cwd, model: None, signal: None }
    }
    pub fn with_model(mut self, model: serde_json::Value) -> Self {
        self.model = Some(model);
        self
    }
}

#[derive(Debug, Clone)]
pub struct RegisteredTool {
    pub definition: ToolDefinition,
    pub source_info: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    #[serde(default)]
    pub label: String,
    pub description: String,
    #[serde(default)]
    pub parameters: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_snippet: Option<String>,
    #[serde(default)]
    pub prompt_guidelines: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_mode: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RegisteredCommand {
    pub name: String,
    pub source_info: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RegisteredShortcut {
    pub shortcut: String,
    pub description: Option<String>,
    pub extension_path: String,
}

#[derive(Debug, Clone)]
pub struct ExtensionFlag {
    pub name: String,
    pub description: Option<String>,
    pub flag_type: String,
    pub default: Option<serde_json::Value>,
    pub extension_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExtensionManifest {
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    #[serde(default)]
    pub commands: Vec<RegisteredCommandDef>,
    #[serde(default)]
    pub flags: Vec<ExtensionFlagDef>,
    #[serde(default)]
    pub shortcuts: Vec<RegisteredShortcutDef>,
    #[serde(default)]
    pub event_subscriptions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredCommandDef {
    pub name: String,
    pub description: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionFlagDef {
    pub name: String,
    pub description: Option<String>,
    pub flag_type: String,
    pub default: Option<serde_json::Value>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredShortcutDef {
    pub shortcut: String,
    pub description: Option<String>,
}

pub struct Extension {
    pub path: String,
    pub resolved_path: String,
    pub source_info: String,
    pub event_subscriptions: HashSet<String>,
    pub tools: HashMap<String, RegisteredTool>,
    pub commands: HashMap<String, RegisteredCommand>,
    pub flags: HashMap<String, ExtensionFlag>,
    pub shortcuts: HashMap<String, RegisteredShortcut>,
    vm: Arc<Mutex<WasmtimeExtensionVm>>,
}

impl Extension {
    fn from_manifest(
        path: String,
        resolved_path: String,
        manifest: ExtensionManifest,
        vm: WasmtimeExtensionVm,
    ) -> Self {
        let tools = manifest
            .tools
            .into_iter()
            .map(|t| (t.name.clone(), RegisteredTool { definition: t, source_info: path.clone() }))
            .collect();
        let commands = manifest
            .commands
            .into_iter()
            .map(|c| {
                (
                    c.name.clone(),
                    RegisteredCommand {
                        name: c.name,
                        source_info: path.clone(),
                        description: c.description,
                    },
                )
            })
            .collect();
        let flags = manifest
            .flags
            .into_iter()
            .map(|f| {
                (
                    f.name.clone(),
                    ExtensionFlag {
                        name: f.name,
                        description: f.description,
                        flag_type: f.flag_type,
                        default: f.default,
                        extension_path: path.clone(),
                    },
                )
            })
            .collect();
        let shortcuts = manifest
            .shortcuts
            .into_iter()
            .map(|s| {
                (
                    s.shortcut.clone(),
                    RegisteredShortcut {
                        shortcut: s.shortcut,
                        description: s.description,
                        extension_path: path.clone(),
                    },
                )
            })
            .collect();
        Self {
            path: path.clone(),
            resolved_path,
            source_info: path,
            event_subscriptions: manifest.event_subscriptions.into_iter().collect(),
            tools,
            commands,
            flags,
            shortcuts,
            vm: Arc::new(Mutex::new(vm)),
        }
    }

    pub fn host_snapshot(&self) -> ExtensionHostSnapshot {
        self.vm.lock().map(|vm| vm.host_snapshot()).unwrap_or_default()
    }

    fn invoke_tool_json(
        &self,
        name: &str,
        args: serde_json::Value,
        tool_call_id: Option<String>,
    ) -> Result<serde_json::Value, String> {
        let input_str = serde_json::to_string(&serde_json::json!({
            "name": name,
            "args": args,
            "toolCallId": tool_call_id,
        }))
        .map_err(|e| e.to_string())?;
        let mut vm = self.vm.lock().map_err(|e| e.to_string())?;
        let raw = vm.invoke_tool(&input_str)?;
        serde_json::from_str(&raw).map_err(|e| e.to_string())
    }
}

impl std::fmt::Debug for Extension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Extension")
            .field("path", &self.path)
            .field("tools_count", &self.tools.len())
            .finish()
    }
}

#[derive(Clone)]
pub struct ExtensionAgentTool {
    definition: ToolDefinition,
    extension_path: String,
    vm: Arc<Mutex<WasmtimeExtensionVm>>,
}

impl ExtensionAgentTool {
    fn new(
        definition: ToolDefinition,
        extension_path: String,
        vm: Arc<Mutex<WasmtimeExtensionVm>>,
    ) -> Self {
        Self { definition, extension_path, vm }
    }

    fn parse_output(value: serde_json::Value) -> Result<AgentToolResult, String> {
        #[derive(Deserialize)]
        struct PluginToolOutput {
            #[serde(default)]
            content: Vec<ContentBlock>,
            #[serde(default)]
            details: serde_json::Value,
            #[serde(default)]
            terminate: bool,
            #[serde(rename = "isError", default)]
            is_error: bool,
            #[serde(default)]
            error: Option<String>,
            #[serde(default)]
            text: Option<String>,
        }

        let output: PluginToolOutput =
            serde_json::from_value(value).map_err(|err| err.to_string())?;
        if output.is_error {
            let message = output
                .error
                .or(output.text)
                .or_else(|| {
                    output.content.iter().find_map(|block| match block {
                        ContentBlock::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                })
                .unwrap_or_else(|| "extension tool returned an error".to_string());
            return Err(message);
        }
        Ok(AgentToolResult {
            content: output.content,
            details: output.details,
            terminate: output.terminate,
        })
    }
}

impl std::fmt::Debug for ExtensionAgentTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionAgentTool")
            .field("name", &self.definition.name)
            .field("extension_path", &self.extension_path)
            .finish()
    }
}

#[async_trait]
impl AgentTool for ExtensionAgentTool {
    fn name(&self) -> &str {
        &self.definition.name
    }

    fn label(&self) -> &str {
        if self.definition.label.is_empty() {
            &self.definition.name
        } else {
            &self.definition.label
        }
    }

    fn description(&self) -> &str {
        &self.definition.description
    }

    fn parameters(&self) -> serde_json::Value {
        self.definition.parameters.clone()
    }

    async fn execute(
        &self,
        tool_call_id: String,
        params: serde_json::Value,
        _signal: Option<tokio_util::sync::CancellationToken>,
        _on_update: Option<agent_core::types::AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>> {
        let input_str = serde_json::to_string(&serde_json::json!({
            "name": self.definition.name,
            "args": params,
            "toolCallId": tool_call_id,
        }))?;
        let raw = {
            let mut vm = self.vm.lock().map_err(|err| err.to_string())?;
            vm.invoke_tool(&input_str)?
        };
        let value = serde_json::from_str(&raw)?;
        Self::parse_output(value).map_err(|err| err.into())
    }

    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        match self.definition.execution_mode.as_deref() {
            Some("sequential") => Some(ToolExecutionMode::Sequential),
            Some("parallel") => Some(ToolExecutionMode::Parallel),
            _ => None,
        }
    }
}

pub fn extension_agent_tools(result: &LoadExtensionsResult) -> Vec<Arc<dyn AgentTool>> {
    let mut seen = HashSet::new();
    let mut tools: Vec<Arc<dyn AgentTool>> = vec![];
    for extension in &result.extensions {
        for tool in extension.tools.values() {
            if !seen.insert(tool.definition.name.clone()) {
                continue;
            }
            tools.push(Arc::new(ExtensionAgentTool::new(
                tool.definition.clone(),
                extension.path.clone(),
                extension.vm.clone(),
            )));
        }
    }
    tools
}

pub fn collect_registered_providers(
    result: &LoadExtensionsResult,
) -> HashMap<String, ProviderConfig> {
    let mut providers = HashMap::new();
    for extension in &result.extensions {
        providers.extend(extension.host_snapshot().providers);
    }
    providers
}

pub fn dispatch_loaded_extensions(
    result: &LoadExtensionsResult,
    event: &ExtensionEvent,
) -> Vec<(String, serde_json::Value)> {
    dispatch_extensions(&result.extensions, event)
}

#[derive(Debug)]
pub struct LoadExtensionsResult {
    pub extensions: Vec<Extension>,
    pub errors: Vec<ExtensionLoadError>,
}

#[derive(Debug, Clone)]
pub struct ExtensionLoadError {
    pub path: String,
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    #[serde(rename = "baseUrl", skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(rename = "apiKey", skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    #[serde(rename = "authHeader", skip_serializing_if = "Option::is_none")]
    pub auth_header: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<ProviderModelConfig>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderModelConfig {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    pub reasoning: bool,
    pub input: Vec<String>,
    pub cost: ModelCost,
    #[serde(rename = "contextWindow")]
    pub context_window: u64,
    #[serde(rename = "maxTokens")]
    pub max_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
}

pub use agent_core::types::ModelCost;

// ── Loader / Component Runtime ────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct ExtensionHostSnapshot {
    pub logs: Vec<serde_json::Value>,
    pub entries: Vec<serde_json::Value>,
    pub messages: Vec<serde_json::Value>,
    pub outgoing_messages: Vec<serde_json::Value>,
    pub session_name: Option<String>,
    pub labels: Vec<String>,
    pub active_tools: Vec<String>,
    pub model: Option<serde_json::Value>,
    pub thinking_level: Option<String>,
    pub providers: HashMap<String, ProviderConfig>,
    pub commands: HashMap<String, RegisteredCommandDef>,
    pub refresh_tools_requested: bool,
    pub compact_requested: bool,
    pub abort_requested: bool,
    pub shutdown_requested: bool,
    pub reload_requested: bool,
    pub ui_actions: Vec<serde_json::Value>,
}

#[derive(Debug, Default)]
struct ExtensionHostState {
    cwd: String,
    logs: Vec<serde_json::Value>,
    entries: Vec<serde_json::Value>,
    messages: Vec<serde_json::Value>,
    outgoing_messages: Vec<serde_json::Value>,
    session_name: Option<String>,
    labels: HashSet<String>,
    active_tools: HashSet<String>,
    model: Option<serde_json::Value>,
    thinking_level: Option<String>,
    providers: HashMap<String, ProviderConfig>,
    commands: HashMap<String, RegisteredCommandDef>,
    refresh_tools_requested: bool,
    compact_requested: bool,
    abort_requested: bool,
    shutdown_requested: bool,
    reload_requested: bool,
    ui_actions: Vec<serde_json::Value>,
}

impl ExtensionHostState {
    fn new(cwd: impl Into<String>) -> Self {
        Self { cwd: cwd.into(), ..Self::default() }
    }

    fn snapshot(&self) -> ExtensionHostSnapshot {
        let mut labels = self.labels.iter().cloned().collect::<Vec<_>>();
        labels.sort();
        let mut active_tools = self.active_tools.iter().cloned().collect::<Vec<_>>();
        active_tools.sort();
        ExtensionHostSnapshot {
            logs: self.logs.clone(),
            entries: self.entries.clone(),
            messages: self.messages.clone(),
            outgoing_messages: self.outgoing_messages.clone(),
            session_name: self.session_name.clone(),
            labels,
            active_tools,
            model: self.model.clone(),
            thinking_level: self.thinking_level.clone(),
            providers: self.providers.clone(),
            commands: self.commands.clone(),
            refresh_tools_requested: self.refresh_tools_requested,
            compact_requested: self.compact_requested,
            abort_requested: self.abort_requested,
            shutdown_requested: self.shutdown_requested,
            reload_requested: self.reload_requested,
            ui_actions: self.ui_actions.clone(),
        }
    }

    fn handle_action(&mut self, raw: &str) -> Result<String, String> {
        let action: serde_json::Value = serde_json::from_str(raw).map_err(|err| err.to_string())?;
        let kind = action.get("kind").and_then(|v| v.as_str()).unwrap_or_default();
        let response = match kind {
            "log" => {
                self.logs.push(action);
                serde_json::json!({ "ok": true })
            }
            "append_entry" => {
                self.entries
                    .push(action.get("entry").cloned().unwrap_or(serde_json::Value::Null));
                serde_json::json!({ "ok": true })
            }
            "append_message" => {
                self.messages.push(action.clone());
                serde_json::json!({ "ok": true })
            }
            "send_message" | "send_user_message" => {
                self.outgoing_messages.push(action.clone());
                serde_json::json!({ "ok": true })
            }
            "get_session_name" => serde_json::json!({ "name": self.session_name }),
            "set_session_name" => {
                self.session_name =
                    action.get("name").and_then(|v| v.as_str()).map(ToOwned::to_owned);
                serde_json::json!({ "ok": true })
            }
            "get_labels" => {
                let mut labels = self.labels.iter().cloned().collect::<Vec<_>>();
                labels.sort();
                serde_json::json!({ "labels": labels })
            }
            "set_labels" => {
                self.labels = string_array(action.get("labels")).into_iter().collect();
                serde_json::json!({ "ok": true })
            }
            "add_label" => {
                if let Some(label) = action.get("label").and_then(|v| v.as_str()) {
                    self.labels.insert(label.to_string());
                }
                serde_json::json!({ "ok": true })
            }
            "remove_label" => {
                if let Some(label) = action.get("label").and_then(|v| v.as_str()) {
                    self.labels.remove(label);
                }
                serde_json::json!({ "ok": true })
            }
            "get_active_tools" => {
                let mut tools = self.active_tools.iter().cloned().collect::<Vec<_>>();
                tools.sort();
                serde_json::json!({ "tools": tools })
            }
            "set_active_tools" => {
                self.active_tools = string_array(action.get("tools")).into_iter().collect();
                serde_json::json!({ "ok": true })
            }
            "refresh_tools" => {
                self.refresh_tools_requested = true;
                serde_json::json!({ "ok": true })
            }
            "get_model" => serde_json::json!({ "model": self.model }),
            "set_model" => {
                self.model = action.get("model").cloned();
                serde_json::json!({ "ok": true })
            }
            "get_thinking" => serde_json::json!({ "thinkingLevel": self.thinking_level }),
            "set_thinking" => {
                self.thinking_level = action
                    .get("thinkingLevel")
                    .or_else(|| action.get("level"))
                    .and_then(|v| v.as_str())
                    .map(ToOwned::to_owned);
                serde_json::json!({ "ok": true })
            }
            "compact" => {
                self.compact_requested = true;
                serde_json::json!({ "ok": true, "available": false })
            }
            "abort" => {
                self.abort_requested = true;
                serde_json::json!({ "ok": true })
            }
            "shutdown" => {
                self.shutdown_requested = true;
                serde_json::json!({ "ok": true })
            }
            "reload" => {
                self.reload_requested = true;
                serde_json::json!({ "ok": true })
            }
            "register_provider" => {
                let name = action
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "register_provider action requires name".to_string())?;
                let config_value = action
                    .get("config")
                    .cloned()
                    .ok_or_else(|| "register_provider action requires config".to_string())?;
                let config = serde_json::from_value(config_value).map_err(|err| err.to_string())?;
                self.providers.insert(name.to_string(), config);
                serde_json::json!({ "ok": true })
            }
            "unregister_provider" => {
                if let Some(name) = action.get("name").and_then(|v| v.as_str()) {
                    self.providers.remove(name);
                }
                serde_json::json!({ "ok": true })
            }
            "register_command" => {
                let name = action
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "register_command action requires name".to_string())?;
                let description =
                    action.get("description").and_then(|v| v.as_str()).map(ToOwned::to_owned);
                self.commands.insert(
                    name.to_string(),
                    RegisteredCommandDef { name: name.to_string(), description },
                );
                serde_json::json!({ "ok": true })
            }
            "unregister_command" => {
                if let Some(name) = action.get("name").and_then(|v| v.as_str()) {
                    self.commands.remove(name);
                }
                serde_json::json!({ "ok": true })
            }
            "cwd" => serde_json::json!({ "cwd": self.cwd }),
            "ui" => {
                self.ui_actions
                    .push(action.get("action").cloned().unwrap_or(serde_json::Value::Null));
                serde_json::json!({ "ok": true, "available": false, "headless": true })
            }
            other => return Err(format!("unknown host action kind '{}'", other)),
        };
        serde_json::to_string(&response).map_err(|err| err.to_string())
    }
}

fn string_array(value: Option<&serde_json::Value>) -> Vec<String> {
    value
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|v| v.as_str().map(ToOwned::to_owned))
        .collect()
}

#[derive(Clone)]
struct ExtensionStore {
    host: Arc<Mutex<ExtensionHostState>>,
}

impl component_abi::ExtensionImports for ExtensionStore {
    fn host_action(&mut self, action_json: String) -> Result<String, String> {
        self.host.lock().map_err(|err| err.to_string())?.handle_action(&action_json)
    }
}

struct WasmtimeExtensionVm {
    store: Store<ExtensionStore>,
    bindings: component_abi::Extension,
    host: Arc<Mutex<ExtensionHostState>>,
}

impl WasmtimeExtensionVm {
    fn load(path: &str, cwd: &str) -> Result<Self, String> {
        let mut config = Config::new();
        config.wasm_component_model(true);
        let engine = Engine::new(&config).map_err(|err| err.to_string())?;
        let component = Component::from_file(&engine, path)
            .map_err(|err| format!("Failed to compile component: {:#}", err))?;
        let mut linker = Linker::new(&engine);
        component_abi::Extension::add_to_linker::<_, HasSelf<_>>(&mut linker, |state| state)
            .map_err(|err| err.to_string())?;
        let host = Arc::new(Mutex::new(ExtensionHostState::new(cwd)));
        let mut store = Store::new(&engine, ExtensionStore { host: host.clone() });
        let bindings = component_abi::Extension::instantiate(&mut store, &component, &linker)
            .map_err(|err| format!("Failed to instantiate component: {:#}", err))?;
        Ok(Self { store, bindings, host })
    }

    fn register(&mut self) -> Result<String, String> {
        self.bindings
            .call_register(&mut self.store)
            .map_err(|err| format!("register() trapped: {}", err))?
    }

    fn on_event(&mut self, input: &str) -> Result<Option<String>, String> {
        self.bindings
            .call_on_event(&mut self.store, input)
            .map_err(|err| format!("on-event trapped: {}", err))?
    }

    fn invoke_tool(&mut self, input: &str) -> Result<String, String> {
        self.bindings
            .call_invoke_tool(&mut self.store, input)
            .map_err(|err| format!("invoke-tool trapped: {}", err))?
    }

    fn host_snapshot(&self) -> ExtensionHostSnapshot {
        self.host.lock().map(|state| state.snapshot()).unwrap_or_default()
    }
}

pub fn load_extensions(paths: &[String], cwd: &str) -> LoadExtensionsResult {
    let mut extensions = vec![];
    let mut errors = vec![];
    for ext_path in paths {
        let resolved = resolve_ext_path(ext_path, cwd);
        match load_single_extension(ext_path, &resolved, cwd) {
            Ok(ext) => extensions.push(ext),
            Err(e) => errors.push(ExtensionLoadError { path: ext_path.clone(), error: e }),
        }
    }
    LoadExtensionsResult { extensions, errors }
}

fn resolve_ext_path(ext_path: &str, cwd: &str) -> String {
    let expanded = if let Some(rest) = ext_path.strip_prefix("~/") {
        let home = dirs_next::home_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        format!("{}/{}", home, rest)
    } else {
        ext_path.to_string()
    };
    if Path::new(&expanded).is_absolute() {
        expanded
    } else {
        Path::new(cwd).join(&expanded).to_string_lossy().to_string()
    }
}

fn load_single_extension(
    original_path: &str,
    resolved: &str,
    cwd: &str,
) -> Result<Extension, String> {
    let component_path = resolve_component_file(resolved)?;
    if !component_path.exists() {
        return Err(format!("Extension file not found: {}", resolved));
    }
    let component_path_string = component_path.to_string_lossy().to_string();
    let mut vm = WasmtimeExtensionVm::load(&component_path_string, cwd)?;
    let manifest_json = vm.register().map_err(|e| format!("register() failed: {}", e))?;
    let ext_manifest: ExtensionManifest = serde_json::from_str(&manifest_json)
        .map_err(|e| format!("Invalid manifest JSON: {}", e))?;
    Ok(Extension::from_manifest(
        original_path.to_string(),
        component_path_string,
        ext_manifest,
        vm,
    ))
}

#[derive(Debug, Deserialize)]
struct ExtensionDescriptor {
    #[serde(alias = "wasm", alias = "componentPath")]
    component: Option<String>,
}

fn resolve_component_file(path: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(path);
    if path.is_dir() {
        let descriptor = path.join("automata-extension.json");
        if descriptor.exists() {
            return resolve_descriptor_component(&descriptor);
        }
        return Ok(path.join("extension.wasm"));
    }
    if path.file_name().and_then(|name| name.to_str()) == Some("automata-extension.json") {
        return resolve_descriptor_component(&path);
    }
    Ok(path)
}

fn resolve_descriptor_component(path: &Path) -> Result<PathBuf, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read extension descriptor: {}", err))?;
    let descriptor: ExtensionDescriptor = serde_json::from_str(&raw)
        .map_err(|err| format!("invalid extension descriptor JSON: {}", err))?;
    let component = descriptor
        .component
        .ok_or_else(|| "extension descriptor requires component".to_string())?;
    let component = PathBuf::from(component);
    if component.is_absolute() {
        Ok(component)
    } else {
        Ok(path.parent().unwrap_or_else(|| Path::new(".")).join(component))
    }
}

pub fn discover_and_load_extensions(
    configured_paths: &[String],
    cwd: &str,
    agent_dir: Option<&str>,
) -> LoadExtensionsResult {
    let mut all_paths = Vec::new();
    let mut seen = HashSet::new();
    let default_agent_dir = dirs_next::home_dir()
        .map(|p| p.join(".automata").join("agent").to_string_lossy().to_string())
        .unwrap_or_default();
    let agent_dir = agent_dir.unwrap_or(&default_agent_dir);
    discover_in_dir(
        &Path::new(cwd).join(".automata").join("extensions"),
        &mut all_paths,
        &mut seen,
    );
    discover_in_dir(&Path::new(agent_dir).join("extensions"), &mut all_paths, &mut seen);
    for p in configured_paths {
        let resolved = resolve_ext_path(p, cwd);
        if seen.insert(resolved.clone()) {
            all_paths.push(resolved);
        }
    }
    load_extensions(&all_paths, cwd)
}

fn discover_in_dir(dir: &Path, paths: &mut Vec<String>, seen: &mut HashSet<String>) {
    if !dir.exists() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            let is_wasm = path.extension().and_then(|e| e.to_str()) == Some("wasm");
            let is_descriptor =
                path.file_name().and_then(|name| name.to_str()) == Some("automata-extension.json");
            if is_wasm || is_descriptor {
                let s = path.to_string_lossy().to_string();
                if seen.insert(s.clone()) {
                    paths.push(s);
                }
            }
        } else if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            let descriptor = path.join("automata-extension.json");
            let index = path.join("extension.wasm");
            let selected = if descriptor.exists() {
                Some(descriptor)
            } else if index.exists() {
                Some(index)
            } else {
                None
            };
            if let Some(selected) = selected {
                let s = selected.to_string_lossy().to_string();
                if seen.insert(s.clone()) {
                    paths.push(s);
                }
            }
        }
    }
}

// ── Runner ────────────────────────────────────────────────────────────────────

pub struct ExtensionRunner {
    extensions: Vec<Extension>,
}

impl ExtensionRunner {
    pub fn new() -> Self {
        Self { extensions: vec![] }
    }

    pub fn load(&mut self, result: LoadExtensionsResult) {
        self.extensions.extend(result.extensions);
    }

    pub fn collect_tools(&self) -> HashMap<String, RegisteredTool> {
        self.extensions
            .iter()
            .flat_map(|e| e.tools.iter().map(|(k, v)| (k.clone(), v.clone())))
            .collect()
    }

    pub fn collect_commands(&self) -> HashMap<String, RegisteredCommand> {
        self.extensions
            .iter()
            .flat_map(|e| {
                e.commands.iter().map(|(k, v)| {
                    (
                        k.clone(),
                        RegisteredCommand {
                            name: v.name.clone(),
                            source_info: v.source_info.clone(),
                            description: v.description.clone(),
                        },
                    )
                })
            })
            .collect()
    }

    pub fn collect_flags(&self) -> HashMap<String, ExtensionFlag> {
        self.extensions
            .iter()
            .flat_map(|e| e.flags.iter().map(|(k, v)| (k.clone(), v.clone())))
            .collect()
    }

    pub fn collect_host_snapshots(&self) -> HashMap<String, ExtensionHostSnapshot> {
        self.extensions
            .iter()
            .map(|extension| (extension.path.clone(), extension.host_snapshot()))
            .collect()
    }

    pub fn collect_registered_providers(&self) -> HashMap<String, ProviderConfig> {
        let mut providers = HashMap::new();
        for extension in &self.extensions {
            providers.extend(extension.host_snapshot().providers);
        }
        providers
    }

    pub fn dispatch_event(&self, event: &ExtensionEvent) -> Vec<(String, serde_json::Value)> {
        dispatch_extensions(&self.extensions, event)
    }

    pub fn invoke_tool(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let ext = self
            .extensions
            .iter()
            .find(|e| e.tools.contains_key(name))
            .ok_or_else(|| format!("No extension provides tool '{}'", name))?;
        ext.invoke_tool_json(name, args, None)
    }

    pub fn run_command(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<Option<serde_json::Value>, String> {
        let Some(ext) = self.extensions.iter().find(|e| e.commands.contains_key(name)) else {
            return Ok(None);
        };
        let event_json = serde_json::to_string(&serde_json::json!({
            "type": "command",
            "name": name,
            "args": args,
        }))
        .map_err(|err| err.to_string())?;
        let mut vm = ext.vm.lock().map_err(|err| err.to_string())?;
        let Some(raw) = vm.on_event(&event_json)? else {
            return Ok(None);
        };
        let value = serde_json::from_str(&raw).map_err(|err| err.to_string())?;
        Ok(Some(value))
    }

    pub fn extension_count(&self) -> usize {
        self.extensions.len()
    }
}

impl Default for ExtensionRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Alias for backwards compatibility.
pub type ExtensionService = ExtensionRunner;

pub fn extension_transform_context_hook(
    runner: Arc<ExtensionRunner>,
) -> agent_core::types::TransformContextFn {
    Arc::new(move |messages, _signal| {
        let runner = runner.clone();
        Box::pin(async move {
            let mut current = messages;
            let event = ExtensionEvent::Context {
                messages: current.iter().map(message_to_value).collect(),
            };
            for (_, value) in runner.dispatch_event(&event) {
                if let Some(next) = parse_messages_update(value) {
                    current = next;
                }
            }
            current
        })
    })
}

pub fn extension_before_tool_call_hook(
    runner: Arc<ExtensionRunner>,
) -> agent_core::types::BeforeToolCallFn {
    Arc::new(move |ctx, _signal| {
        let runner = runner.clone();
        Box::pin(async move {
            let event = ExtensionEvent::ToolCall {
                tool_call_id: ctx.tool_call.id,
                tool_name: ctx.tool_call.name,
                input: ctx.args,
            };
            for (_, value) in runner.dispatch_event(&event) {
                if value.get("block").and_then(|v| v.as_bool()).unwrap_or(false) {
                    return Some(BeforeToolCallResult {
                        block: true,
                        reason: value.get("reason").and_then(|v| v.as_str()).map(ToOwned::to_owned),
                    });
                }
            }
            None
        })
    })
}

pub fn extension_after_tool_call_hook(
    runner: Arc<ExtensionRunner>,
) -> agent_core::types::AfterToolCallFn {
    Arc::new(move |ctx, _signal| {
        let runner = runner.clone();
        Box::pin(async move {
            let event = ExtensionEvent::ToolResult {
                tool_call_id: ctx.tool_call.id,
                tool_name: ctx.tool_call.name,
                input: ctx.args,
                content: ctx.result.content.iter().map(content_to_value).collect(),
                is_error: ctx.is_error,
                details: Some(ctx.result.details),
            };
            let mut update = AfterToolCallResult::default();
            let mut changed = false;
            for (_, value) in runner.dispatch_event(&event) {
                if let Some(content) = value.get("content")
                    && let Ok(parsed) = serde_json::from_value::<Vec<ContentBlock>>(content.clone())
                {
                    update.content = Some(parsed);
                    changed = true;
                }
                if let Some(details) = value.get("details") {
                    update.details = Some(details.clone());
                    changed = true;
                }
                if let Some(is_error) = value.get("isError").and_then(|v| v.as_bool()) {
                    update.is_error = Some(is_error);
                    changed = true;
                }
                if let Some(terminate) = value.get("terminate").and_then(|v| v.as_bool()) {
                    update.terminate = Some(terminate);
                    changed = true;
                }
            }
            changed.then_some(update)
        })
    })
}

pub fn extension_on_payload_hook(runner: Arc<ExtensionRunner>) -> agent_core::types::OnPayloadFn {
    Arc::new(move |payload| {
        let runner = runner.clone();
        Box::pin(async move {
            let event = ExtensionEvent::BeforeProviderRequest { payload: payload.clone() };
            let mut current = payload;
            let mut changed = false;
            for (_, value) in runner.dispatch_event(&event) {
                if let Some(next) = value.get("payload") {
                    current = next.clone();
                    changed = true;
                } else if value.is_object() {
                    current = value;
                    changed = true;
                }
            }
            changed.then_some(current)
        })
    })
}

pub fn extension_on_response_hook(runner: Arc<ExtensionRunner>) -> agent_core::types::OnResponseFn {
    Arc::new(move |response| {
        let runner = runner.clone();
        Box::pin(async move {
            let _ = runner.dispatch_event(&ExtensionEvent::AfterProviderResponse {
                status: 200,
                headers: HashMap::new(),
            });
            let _ = runner.dispatch_event(&ExtensionEvent::EventBus {
                event: "provider_response".to_string(),
                data: response,
            });
        })
    })
}

pub async fn subscribe_extension_harness_events(
    harness: &agent_core::harness::AgentHarness,
    runner: Arc<ExtensionRunner>,
) {
    let turn_index = Arc::new(std::sync::atomic::AtomicU64::new(0));
    harness
        .subscribe(move |event, _signal| {
            let runner = runner.clone();
            let turn_index = turn_index.clone();
            async move {
                dispatch_harness_event(&runner, &turn_index, event);
            }
        })
        .await;
}

fn dispatch_harness_event(
    runner: &ExtensionRunner,
    turn_index: &std::sync::atomic::AtomicU64,
    event: agent_core::harness::HarnessEvent,
) {
    use agent_core::event::AgentEvent;
    use agent_core::harness::HarnessEvent;
    use std::sync::atomic::Ordering;

    match event {
        HarnessEvent::Agent(agent_event) => match agent_event {
            AgentEvent::AgentStart => {
                let _ = runner.dispatch_event(&ExtensionEvent::AgentStart);
            }
            AgentEvent::AgentEnd { messages } => {
                let _ = runner.dispatch_event(&ExtensionEvent::AgentEnd {
                    messages: messages.iter().map(message_to_value).collect(),
                });
            }
            AgentEvent::TurnStart => {
                let index = turn_index.fetch_add(1, Ordering::Relaxed) + 1;
                let _ = runner.dispatch_event(&ExtensionEvent::TurnStart {
                    turn_index: index,
                    timestamp: now_millis(),
                });
            }
            AgentEvent::TurnEnd { message, tool_results } => {
                let index = turn_index.load(Ordering::Relaxed);
                let _ = runner.dispatch_event(&ExtensionEvent::TurnEnd {
                    turn_index: index,
                    message: message_to_value(&message),
                    tool_results: tool_results.iter().map(message_to_value).collect(),
                });
            }
            AgentEvent::MessageStart { message } => {
                let _ = runner.dispatch_event(&ExtensionEvent::MessageStart {
                    message: message_to_value(&message),
                });
            }
            AgentEvent::MessageUpdate { partial, assistant_message_event } => {
                let _ = runner.dispatch_event(&ExtensionEvent::MessageUpdate {
                    message: serde_json::to_value(partial).unwrap_or(serde_json::Value::Null),
                    assistant_message_event: serde_json::to_value(assistant_message_event)
                        .unwrap_or(serde_json::Value::Null),
                });
            }
            AgentEvent::MessageEnd { message } => {
                let _ = runner.dispatch_event(&ExtensionEvent::MessageEnd {
                    message: message_to_value(&message),
                });
            }
            AgentEvent::ToolExecutionStart { tool_call_id, tool_name, args } => {
                let _ = runner.dispatch_event(&ExtensionEvent::ToolExecutionStart {
                    tool_call_id,
                    tool_name,
                    args,
                });
            }
            AgentEvent::ToolExecutionUpdate {
                tool_call_id,
                tool_name,
                args,
                partial_result,
            } => {
                let _ = runner.dispatch_event(&ExtensionEvent::ToolExecutionUpdate {
                    tool_call_id,
                    tool_name,
                    args,
                    partial_result,
                });
            }
            AgentEvent::ToolExecutionEnd {
                tool_call_id,
                tool_name,
                result,
                is_error,
            } => {
                let _ = runner.dispatch_event(&ExtensionEvent::ToolExecutionEnd {
                    tool_call_id,
                    tool_name,
                    result: serde_json::json!({
                        "content": result.content.iter().map(content_to_value).collect::<Vec<_>>(),
                        "details": result.details,
                        "terminate": result.terminate,
                    }),
                    is_error,
                });
            }
        },
        HarnessEvent::ModelSelect {
            provider,
            model_id,
            previous_provider,
            previous_model_id,
        } => {
            let previous_model = if previous_provider.is_some() || previous_model_id.is_some() {
                Some(serde_json::json!({
                    "provider": previous_provider,
                    "modelId": previous_model_id,
                }))
            } else {
                None
            };
            let _ = runner.dispatch_event(&ExtensionEvent::ModelSelect {
                model: serde_json::json!({ "provider": provider, "modelId": model_id }),
                previous_model,
                source: "harness".to_string(),
            });
        }
        HarnessEvent::CompactionStart { .. } => {
            let _ = runner.dispatch_event(&ExtensionEvent::SessionBeforeCompact {
                preparation: serde_json::Value::Null,
                branch_entries: vec![],
                custom_instructions: None,
            });
        }
        HarnessEvent::Compaction { result } => {
            let _ = runner.dispatch_event(&ExtensionEvent::SessionCompact {
                compaction_entry: serde_json::json!({
                    "summary": result.summary,
                    "firstKeptEntryId": result.first_kept_entry_id,
                    "tokensBefore": result.tokens_before,
                    "readFiles": result.read_files,
                    "modifiedFiles": result.modified_files,
                }),
                from_extension: false,
            });
        }
        HarnessEvent::Aborted { .. } => {
            let _ = runner.dispatch_event(&ExtensionEvent::EventBus {
                event: "abort".to_string(),
                data: serde_json::Value::Null,
            });
        }
        _ => {}
    }
}

fn message_to_value(message: &AgentMessage) -> serde_json::Value {
    serde_json::to_value(message).unwrap_or(serde_json::Value::Null)
}

fn content_to_value(content: &ContentBlock) -> serde_json::Value {
    serde_json::to_value(content).unwrap_or(serde_json::Value::Null)
}

fn parse_messages_update(value: serde_json::Value) -> Option<Vec<AgentMessage>> {
    if let Some(messages) = value.get("messages") {
        serde_json::from_value(messages.clone()).ok()
    } else if value.is_array() {
        serde_json::from_value(value).ok()
    } else {
        None
    }
}

fn now_millis() -> u64 {
    chrono::Utc::now().timestamp_millis().max(0) as u64
}

fn event_type_str(event: &ExtensionEvent) -> &'static str {
    match event {
        ExtensionEvent::ResourcesDiscover { .. } => "resources_discover",
        ExtensionEvent::SessionStart { .. } => "session_start",
        ExtensionEvent::SessionBeforeSwitch { .. } => "session_before_switch",
        ExtensionEvent::SessionBeforeFork { .. } => "session_before_fork",
        ExtensionEvent::SessionBeforeCompact { .. } => "session_before_compact",
        ExtensionEvent::SessionCompact { .. } => "session_compact",
        ExtensionEvent::SessionShutdown { .. } => "session_shutdown",
        ExtensionEvent::SessionBeforeTree { .. } => "session_before_tree",
        ExtensionEvent::SessionTree { .. } => "session_tree",
        ExtensionEvent::Context { .. } => "context",
        ExtensionEvent::BeforeProviderRequest { .. } => "before_provider_request",
        ExtensionEvent::AfterProviderResponse { .. } => "after_provider_response",
        ExtensionEvent::BeforeAgentStart { .. } => "before_agent_start",
        ExtensionEvent::AgentStart => "agent_start",
        ExtensionEvent::AgentEnd { .. } => "agent_end",
        ExtensionEvent::TurnStart { .. } => "turn_start",
        ExtensionEvent::TurnEnd { .. } => "turn_end",
        ExtensionEvent::MessageStart { .. } => "message_start",
        ExtensionEvent::MessageUpdate { .. } => "message_update",
        ExtensionEvent::MessageEnd { .. } => "message_end",
        ExtensionEvent::ToolExecutionStart { .. } => "tool_execution_start",
        ExtensionEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
        ExtensionEvent::ToolExecutionEnd { .. } => "tool_execution_end",
        ExtensionEvent::ModelSelect { .. } => "model_select",
        ExtensionEvent::UserBash { .. } => "user_bash",
        ExtensionEvent::Input { .. } => "input",
        ExtensionEvent::ToolCall { .. } => "tool_call",
        ExtensionEvent::ToolResult { .. } => "tool_result",
        ExtensionEvent::EventBus { .. } => "event_bus",
    }
}

fn dispatch_extensions(
    extensions: &[Extension],
    event: &ExtensionEvent,
) -> Vec<(String, serde_json::Value)> {
    let event_type = event_type_str(event);
    let Ok(event_json) = serde_json::to_string(event) else {
        return vec![];
    };
    let mut results = vec![];
    for ext in extensions {
        if !ext.event_subscriptions.contains(event_type) {
            continue;
        }
        let Ok(mut vm) = ext.vm.lock() else {
            continue;
        };
        if let Ok(Some(raw)) = vm.on_event(&event_json)
            && let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw)
            && !v.is_null()
        {
            results.push((ext.path.clone(), v));
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_extension_wasm() -> Vec<u8> {
        let manifest = r#"{"tools":[{"name":"echo","description":"Echo input","parameters":{}}],"event_subscriptions":["agent_start"]}"#;
        let event_result = r#"{"handled":true}"#;
        let tool_result = r#"{"content":[{"type":"text","text":"ok"}],"isError":false}"#;
        let manifest_ptr = 2048;
        let event_ptr = manifest_ptr + manifest.len() + 16;
        let tool_ptr = event_ptr + event_result.len() + 16;
        let wat = format!(
            r#"
            (component
              (type $string-result (result string (error string)))
              (type $register-ty (func (result $string-result)))
              (type $event-ok (option string))
              (type $event-result (result $event-ok (error string)))
              (type $event-ty (func (param "event-json" string) (result $event-result)))
              (type $tool-ty (func (param "request-json" string) (result $string-result)))
              (core module $module
                (memory (export "memory") 1)
                (global $heap (mut i32) (i32.const 4096))
                (func $realloc (export "cabi_realloc")
                  (param $old i32) (param $old_align i32) (param $new_align i32) (param $len i32)
                  (result i32)
                  (local $ptr i32)
                  global.get $heap
                  local.set $ptr
                  global.get $heap
                  local.get $len
                  i32.add
                  global.set $heap
                  local.get $ptr)
                (func $write-ok-string (param $result i32) (param $ptr i32) (param $len i32)
                  local.get $result
                  i32.const 0
                  i32.store8
                  local.get $result
                  i32.const 4
                  i32.add
                  local.get $ptr
                  i32.store
                  local.get $result
                  i32.const 8
                  i32.add
                  local.get $len
                  i32.store)
                (func $write-ok-some-string (param $result i32) (param $ptr i32) (param $len i32)
                  local.get $result
                  i32.const 0
                  i32.store8
                  local.get $result
                  i32.const 4
                  i32.add
                  i32.const 1
                  i32.store8
                  local.get $result
                  i32.const 8
                  i32.add
                  local.get $ptr
                  i32.store
                  local.get $result
                  i32.const 12
                  i32.add
                  local.get $len
                  i32.store)
                (func $register (export "register") (result i32)
                  i32.const 1024
                  i32.const {manifest_ptr}
                  i32.const {manifest_len}
                  call $write-ok-string
                  i32.const 1024)
                (func $on-event (export "on-event") (param i32 i32) (result i32)
                  i32.const 1040
                  i32.const {event_ptr}
                  i32.const {event_len}
                  call $write-ok-some-string
                  i32.const 1040)
                (func $invoke-tool (export "invoke-tool") (param i32 i32) (result i32)
                  i32.const 1064
                  i32.const {tool_ptr}
                  i32.const {tool_len}
                  call $write-ok-string
                  i32.const 1064)
                (data (i32.const {manifest_ptr}) {manifest:?})
                (data (i32.const {event_ptr}) {event_result:?})
                (data (i32.const {tool_ptr}) {tool_result:?})
              )
              (core instance $instance (instantiate $module))
              (alias core export $instance "memory" (core memory $memory))
              (alias core export $instance "cabi_realloc" (core func $realloc))
              (alias core export $instance "register" (core func $register))
              (alias core export $instance "on-event" (core func $on-event))
              (alias core export $instance "invoke-tool" (core func $invoke-tool))
              (func $register-lift (type $register-ty)
                (canon lift (core func $register) (memory $memory) (realloc $realloc) string-encoding=utf8))
              (func $on-event-lift (type $event-ty)
                (canon lift (core func $on-event) (memory $memory) (realloc $realloc) string-encoding=utf8))
              (func $invoke-tool-lift (type $tool-ty)
                (canon lift (core func $invoke-tool) (memory $memory) (realloc $realloc) string-encoding=utf8))
              (export "register" (func $register-lift))
              (export "on-event" (func $on-event-lift))
              (export "invoke-tool" (func $invoke-tool-lift))
            )
            "#,
            manifest_len = manifest.len(),
            event_len = event_result.len(),
            tool_len = tool_result.len(),
        );
        wat::parse_str(wat).unwrap()
    }

    fn write_test_extension() -> (tempfile::TempDir, String) {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("extension.wasm");
        std::fs::write(&path, test_extension_wasm()).unwrap();
        (dir, path.to_string_lossy().to_string())
    }

    #[test]
    fn test_extension_event_serde() {
        let event = ExtensionEvent::AgentStart;
        let json = serde_json::to_string(&event).unwrap();
        assert_eq!(json, r#"{"type":"agent_start"}"#);
    }

    #[test]
    fn test_extension_manifest_default() {
        let m: ExtensionManifest = serde_json::from_str("{}").unwrap();
        assert!(m.tools.is_empty());
    }

    #[test]
    fn test_runner_empty() {
        let runner = ExtensionRunner::new();
        assert_eq!(runner.extension_count(), 0);
        assert!(runner.collect_tools().is_empty());
    }

    #[test]
    fn test_load_extensions_empty() {
        let result = load_extensions(&[], "/tmp");
        assert!(result.extensions.is_empty());
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_load_extension_not_found() {
        let result = load_extensions(&["/nonexistent/path.wasm".to_string()], "/tmp");
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].error.contains("not found"));
    }

    #[test]
    fn test_load_wasmtime_extension_manifest() {
        let (_dir, path) = write_test_extension();
        let result = load_extensions(&[path], "/tmp");

        assert!(result.errors.is_empty(), "{:?}", result.errors);
        assert_eq!(result.extensions.len(), 1);
        assert_eq!(result.extensions[0].tools.len(), 1);
        assert!(result.extensions[0].event_subscriptions.contains("agent_start"));
    }

    #[test]
    fn test_wasmtime_extension_runner_event_and_tool() {
        let (_dir, path) = write_test_extension();
        let mut runner = ExtensionRunner::new();
        runner.load(load_extensions(&[path], "/tmp"));

        let event_results = runner.dispatch_event(&ExtensionEvent::AgentStart);
        assert_eq!(event_results.len(), 1);
        assert_eq!(event_results[0].1, serde_json::json!({"handled": true}));

        let tool_result = runner.invoke_tool("echo", serde_json::json!({"text": "hello"})).unwrap();
        assert_eq!(
            tool_result,
            serde_json::json!({
                "content": [{"type": "text", "text": "ok"}],
                "isError": false
            })
        );
    }

    #[tokio::test]
    async fn test_extension_agent_tool_executes_component_tool() {
        let (_dir, path) = write_test_extension();
        let result = load_extensions(&[path], "/tmp");
        assert!(result.errors.is_empty(), "{:?}", result.errors);

        let tools = extension_agent_tools(&result);
        assert_eq!(tools.len(), 1);
        let output = tools[0]
            .execute("call-1".to_string(), serde_json::json!({"text": "hello"}), None, None)
            .await
            .unwrap();
        assert!(matches!(
            output.content.as_slice(),
            [ContentBlock::Text { text }] if text == "ok"
        ));
    }

    #[test]
    fn test_host_action_registers_provider_in_headless_state() {
        let mut state = ExtensionHostState::new("/tmp/project");
        state
            .handle_action(
                r#"{
                    "kind":"register_provider",
                    "name":"local",
                    "config":{
                        "baseUrl":"http://localhost:11434",
                        "api":"openai",
                        "models":[{
                            "id":"local-model",
                            "name":"Local Model",
                            "reasoning":false,
                            "input":["text"],
                            "cost":{"input":0.0,"output":0.0,"cacheRead":0.0,"cacheWrite":0.0},
                            "contextWindow":8192,
                            "maxTokens":2048
                        }]
                    }
                }"#,
            )
            .unwrap();
        let snapshot = state.snapshot();
        assert!(snapshot.providers.contains_key("local"));
        assert_eq!(snapshot.providers["local"].models.as_ref().unwrap()[0].id, "local-model");
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionPreparation {
    #[serde(rename = "firstKeptEntryId")]
    pub first_kept_entry_id: String,
    #[serde(rename = "messagesToSummarize")]
    pub messages_to_summarize: Vec<serde_json::Value>,
    #[serde(rename = "turnPrefixMessages")]
    pub turn_prefix_messages: Vec<serde_json::Value>,
    #[serde(rename = "isSplitTurn")]
    pub is_split_turn: bool,
    #[serde(rename = "tokensBefore")]
    pub tokens_before: u64,
    #[serde(rename = "previousSummary", skip_serializing_if = "Option::is_none")]
    pub previous_summary: Option<String>,
    #[serde(rename = "fileOps")]
    pub file_ops: serde_json::Value,
    pub settings: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionResult {
    pub summary: String,
    #[serde(rename = "firstKeptEntryId")]
    pub first_kept_entry_id: String,
    #[serde(rename = "tokensBefore")]
    pub tokens_before: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}
