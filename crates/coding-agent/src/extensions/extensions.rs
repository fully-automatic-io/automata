// Extension system — WASM-based plugin model (types + loader + runner merged).

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};
use extism::{Manifest, Plugin, Wasm};

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
    ResourcesDiscover { cwd: String, reason: SessionLifecycleReason },
    #[serde(rename = "session_start")]
    SessionStart { reason: SessionLifecycleReason, #[serde(rename = "previousSessionFile", skip_serializing_if = "Option::is_none")] previous_session_file: Option<String> },
    #[serde(rename = "session_before_switch")]
    SessionBeforeSwitch { reason: SessionLifecycleReason, #[serde(rename = "targetSessionFile", skip_serializing_if = "Option::is_none")] target_session_file: Option<String> },
    #[serde(rename = "session_before_fork")]
    SessionBeforeFork { #[serde(rename = "entryId")] entry_id: String, position: String },
    #[serde(rename = "session_before_compact")]
    SessionBeforeCompact { preparation: serde_json::Value, #[serde(rename = "branchEntries")] branch_entries: Vec<serde_json::Value>, #[serde(rename = "customInstructions", skip_serializing_if = "Option::is_none")] custom_instructions: Option<String> },
    #[serde(rename = "session_compact")]
    SessionCompact { #[serde(rename = "compactionEntry")] compaction_entry: serde_json::Value, #[serde(rename = "fromExtension")] from_extension: bool },
    #[serde(rename = "session_shutdown")]
    SessionShutdown { reason: SessionLifecycleReason, #[serde(rename = "targetSessionFile", skip_serializing_if = "Option::is_none")] target_session_file: Option<String> },
    #[serde(rename = "session_before_tree")]
    SessionBeforeTree { preparation: serde_json::Value },
    #[serde(rename = "session_tree")]
    SessionTree { #[serde(rename = "newLeafId")] new_leaf_id: Option<String>, #[serde(rename = "oldLeafId")] old_leaf_id: Option<String>, #[serde(rename = "summaryEntry", skip_serializing_if = "Option::is_none")] summary_entry: Option<serde_json::Value>, #[serde(rename = "fromExtension", skip_serializing_if = "Option::is_none")] from_extension: Option<bool> },
    #[serde(rename = "context")]
    Context { messages: Vec<serde_json::Value> },
    #[serde(rename = "before_provider_request")]
    BeforeProviderRequest { payload: serde_json::Value },
    #[serde(rename = "after_provider_response")]
    AfterProviderResponse { status: u16, headers: HashMap<String, String> },
    #[serde(rename = "before_agent_start")]
    BeforeAgentStart { prompt: String, #[serde(skip_serializing_if = "Option::is_none")] images: Option<Vec<serde_json::Value>>, #[serde(rename = "systemPrompt")] system_prompt: String, #[serde(rename = "systemPromptOptions")] system_prompt_options: serde_json::Value },
    #[serde(rename = "agent_start")]
    AgentStart,
    #[serde(rename = "agent_end")]
    AgentEnd { messages: Vec<serde_json::Value> },
    #[serde(rename = "turn_start")]
    TurnStart { #[serde(rename = "turnIndex")] turn_index: u64, timestamp: u64 },
    #[serde(rename = "turn_end")]
    TurnEnd { #[serde(rename = "turnIndex")] turn_index: u64, message: serde_json::Value, #[serde(rename = "toolResults")] tool_results: Vec<serde_json::Value> },
    #[serde(rename = "message_start")]
    MessageStart { message: serde_json::Value },
    #[serde(rename = "message_update")]
    MessageUpdate { message: serde_json::Value, #[serde(rename = "assistantMessageEvent")] assistant_message_event: serde_json::Value },
    #[serde(rename = "message_end")]
    MessageEnd { message: serde_json::Value },
    #[serde(rename = "tool_execution_start")]
    ToolExecutionStart { #[serde(rename = "toolCallId")] tool_call_id: String, #[serde(rename = "toolName")] tool_name: String, args: serde_json::Value },
    #[serde(rename = "tool_execution_update")]
    ToolExecutionUpdate { #[serde(rename = "toolCallId")] tool_call_id: String, #[serde(rename = "toolName")] tool_name: String, args: serde_json::Value, #[serde(rename = "partialResult")] partial_result: serde_json::Value },
    #[serde(rename = "tool_execution_end")]
    ToolExecutionEnd { #[serde(rename = "toolCallId")] tool_call_id: String, #[serde(rename = "toolName")] tool_name: String, result: serde_json::Value, #[serde(rename = "isError")] is_error: bool },
    #[serde(rename = "model_select")]
    ModelSelect { model: serde_json::Value, #[serde(rename = "previousModel", skip_serializing_if = "Option::is_none")] previous_model: Option<serde_json::Value>, source: String },
    #[serde(rename = "user_bash")]
    UserBash { command: String, #[serde(rename = "excludeFromContext")] exclude_from_context: bool, cwd: String },
    #[serde(rename = "input")]
    Input { text: String, #[serde(skip_serializing_if = "Option::is_none")] images: Option<Vec<serde_json::Value>>, source: String },
    #[serde(rename = "tool_call")]
    ToolCall { #[serde(rename = "toolCallId")] tool_call_id: String, #[serde(rename = "toolName")] tool_name: String, input: serde_json::Value },
    #[serde(rename = "tool_result")]
    ToolResult { #[serde(rename = "toolCallId")] tool_call_id: String, #[serde(rename = "toolName")] tool_name: String, input: serde_json::Value, content: Vec<serde_json::Value>, #[serde(rename = "isError")] is_error: bool, #[serde(skip_serializing_if = "Option::is_none")] details: Option<serde_json::Value> },
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
    pub fn new(cwd: String) -> Self { Self { cwd, model: None, signal: None } }
    pub fn with_model(mut self, model: serde_json::Value) -> Self { self.model = Some(model); self }
}

#[derive(Debug, Clone)]
pub struct RegisteredTool {
    pub definition: ToolDefinition,
    pub source_info: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    #[serde(default)] pub label: String,
    pub description: String,
    #[serde(default)] pub parameters: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")] pub prompt_snippet: Option<String>,
    #[serde(default)] pub prompt_guidelines: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub execution_mode: Option<String>,
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
    #[serde(default)] pub tools: Vec<ToolDefinition>,
    #[serde(default)] pub commands: Vec<RegisteredCommandDef>,
    #[serde(default)] pub flags: Vec<ExtensionFlagDef>,
    #[serde(default)] pub shortcuts: Vec<RegisteredShortcutDef>,
    #[serde(default)] pub event_subscriptions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredCommandDef { pub name: String, pub description: Option<String> }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionFlagDef { pub name: String, pub description: Option<String>, pub flag_type: String, pub default: Option<serde_json::Value> }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredShortcutDef { pub shortcut: String, pub description: Option<String> }

pub struct Extension {
    pub path: String,
    pub resolved_path: String,
    pub source_info: String,
    pub event_subscriptions: HashSet<String>,
    pub tools: HashMap<String, RegisteredTool>,
    pub commands: HashMap<String, RegisteredCommand>,
    pub flags: HashMap<String, ExtensionFlag>,
    pub shortcuts: HashMap<String, RegisteredShortcut>,
    pub plugin: Arc<Mutex<Plugin>>,
}

impl Extension {
    pub fn from_manifest(path: String, resolved_path: String, manifest: ExtensionManifest, plugin: Plugin) -> Self {
        let tools = manifest.tools.into_iter()
            .map(|t| (t.name.clone(), RegisteredTool { definition: t, source_info: path.clone() }))
            .collect();
        let commands = manifest.commands.into_iter()
            .map(|c| (c.name.clone(), RegisteredCommand { name: c.name, source_info: path.clone(), description: c.description }))
            .collect();
        let flags = manifest.flags.into_iter()
            .map(|f| (f.name.clone(), ExtensionFlag { name: f.name, description: f.description, flag_type: f.flag_type, default: f.default, extension_path: path.clone() }))
            .collect();
        let shortcuts = manifest.shortcuts.into_iter()
            .map(|s| (s.shortcut.clone(), RegisteredShortcut { shortcut: s.shortcut, description: s.description, extension_path: path.clone() }))
            .collect();
        Self {
            path: path.clone(), resolved_path, source_info: path,
            event_subscriptions: manifest.event_subscriptions.into_iter().collect(),
            tools, commands, flags, shortcuts,
            plugin: Arc::new(Mutex::new(plugin)),
        }
    }
}

impl std::fmt::Debug for Extension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Extension").field("path", &self.path).field("tools_count", &self.tools.len()).finish()
    }
}

#[derive(Debug)]
pub struct LoadExtensionsResult {
    pub extensions: Vec<Extension>,
    pub errors: Vec<ExtensionLoadError>,
}

#[derive(Debug, Clone)]
pub struct ExtensionLoadError { pub path: String, pub error: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    #[serde(rename = "baseUrl", skip_serializing_if = "Option::is_none")] pub base_url: Option<String>,
    #[serde(rename = "apiKey", skip_serializing_if = "Option::is_none")] pub api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub api: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub headers: Option<HashMap<String, String>>,
    #[serde(rename = "authHeader", skip_serializing_if = "Option::is_none")] pub auth_header: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")] pub models: Option<Vec<ProviderModelConfig>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderModelConfig {
    pub id: String, pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")] pub api: Option<String>,
    pub reasoning: bool, pub input: Vec<String>, pub cost: ModelCost,
    #[serde(rename = "contextWindow")] pub context_window: u64,
    #[serde(rename = "maxTokens")] pub max_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")] pub headers: Option<HashMap<String, String>>,
}

pub use agent_core::types::ModelCost;

// ── Loader ────────────────────────────────────────────────────────────────────

pub fn load_extensions(paths: &[String], cwd: &str) -> LoadExtensionsResult {
    let mut extensions = vec![];
    let mut errors = vec![];
    for ext_path in paths {
        let resolved = resolve_ext_path(ext_path, cwd);
        match load_single_extension(ext_path, &resolved) {
            Ok(ext) => extensions.push(ext),
            Err(e) => errors.push(ExtensionLoadError { path: ext_path.clone(), error: e }),
        }
    }
    LoadExtensionsResult { extensions, errors }
}

fn resolve_ext_path(ext_path: &str, cwd: &str) -> String {
    let expanded = if let Some(rest) = ext_path.strip_prefix("~/") {
        let home = dirs_next::home_dir().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
        format!("{}/{}", home, rest)
    } else {
        ext_path.to_string()
    };
    if Path::new(&expanded).is_absolute() { expanded }
    else { Path::new(cwd).join(&expanded).to_string_lossy().to_string() }
}

fn load_single_extension(original_path: &str, resolved: &str) -> Result<Extension, String> {
    if !Path::new(resolved).exists() {
        return Err(format!("Extension file not found: {}", resolved));
    }
    let manifest = Manifest::new([Wasm::file(resolved)]);
    let mut plugin = Plugin::new(&manifest, [], true)
        .map_err(|e| format!("Failed to load WASM plugin: {}", e))?;
    let manifest_json = plugin.call::<&str, &str>("register", "")
        .map_err(|e| format!("register() failed: {}", e))?;
    let ext_manifest: ExtensionManifest = serde_json::from_str(manifest_json)
        .map_err(|e| format!("Invalid manifest JSON: {}", e))?;
    Ok(Extension::from_manifest(original_path.to_string(), resolved.to_string(), ext_manifest, plugin))
}

pub fn discover_and_load_extensions(configured_paths: &[String], cwd: &str, agent_dir: Option<&str>) -> LoadExtensionsResult {
    let mut all_paths = Vec::new();
    let mut seen = HashSet::new();
    let default_agent_dir = dirs_next::home_dir()
        .map(|p| p.join(".automata").join("agent").to_string_lossy().to_string())
        .unwrap_or_default();
    let agent_dir = agent_dir.unwrap_or(&default_agent_dir);
    discover_in_dir(&Path::new(cwd).join(".automata").join("extensions"), &mut all_paths, &mut seen);
    discover_in_dir(&Path::new(agent_dir).join("extensions"), &mut all_paths, &mut seen);
    for p in configured_paths {
        let resolved = resolve_ext_path(p, cwd);
        if seen.insert(resolved.clone()) { all_paths.push(resolved); }
    }
    load_extensions(&all_paths, cwd)
}

fn discover_in_dir(dir: &Path, paths: &mut Vec<String>, seen: &mut HashSet<String>) {
    if !dir.exists() { return; }
    let Ok(entries) = std::fs::read_dir(dir) else { return; };
    for entry in entries.flatten() {
        let path = entry.path();
        if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            if path.extension().and_then(|e| e.to_str()) == Some("wasm") {
                let s = path.to_string_lossy().to_string();
                if seen.insert(s.clone()) { paths.push(s); }
            }
        } else if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            let index = path.join("extension.wasm");
            if index.exists() {
                let s = index.to_string_lossy().to_string();
                if seen.insert(s.clone()) { paths.push(s); }
            }
        }
    }
}

// ── Runner ────────────────────────────────────────────────────────────────────

pub struct ExtensionRunner {
    extensions: Vec<Extension>,
}

impl ExtensionRunner {
    pub fn new() -> Self { Self { extensions: vec![] } }

    pub fn load(&mut self, result: LoadExtensionsResult) {
        self.extensions.extend(result.extensions);
    }

    pub fn collect_tools(&self) -> HashMap<String, RegisteredTool> {
        self.extensions.iter().flat_map(|e| e.tools.iter().map(|(k, v)| (k.clone(), v.clone()))).collect()
    }

    pub fn collect_commands(&self) -> HashMap<String, RegisteredCommand> {
        self.extensions.iter().flat_map(|e| e.commands.iter().map(|(k, v)| (k.clone(), RegisteredCommand { name: v.name.clone(), source_info: v.source_info.clone(), description: v.description.clone() }))).collect()
    }

    pub fn collect_flags(&self) -> HashMap<String, ExtensionFlag> {
        self.extensions.iter().flat_map(|e| e.flags.iter().map(|(k, v)| (k.clone(), v.clone()))).collect()
    }

    pub fn dispatch_event(&self, event: &ExtensionEvent) -> Vec<(String, serde_json::Value)> {
        let event_type = event_type_str(event);
        let Ok(event_json) = serde_json::to_string(event) else { return vec![]; };
        let mut results = vec![];
        for ext in &self.extensions {
            if !ext.event_subscriptions.contains(event_type) { continue; }
            let Ok(mut plugin) = ext.plugin.lock() else { continue; };
            if let Ok(raw) = plugin.call::<&str, &str>("on_event", &event_json)
                && let Ok(v) = serde_json::from_str::<serde_json::Value>(raw)
                    && !v.is_null() { results.push((ext.path.clone(), v)); }
        }
        results
    }

    pub fn invoke_tool(&self, name: &str, args: serde_json::Value) -> Result<serde_json::Value, String> {
        let ext = self.extensions.iter().find(|e| e.tools.contains_key(name))
            .ok_or_else(|| format!("No extension provides tool '{}'", name))?;
        let input_str = serde_json::to_string(&serde_json::json!({ "name": name, "args": args })).map_err(|e| e.to_string())?;
        let mut plugin = ext.plugin.lock().map_err(|e| e.to_string())?;
        let raw = plugin.call::<&str, &str>("invoke_tool", &input_str).map_err(|e| e.to_string())?;
        serde_json::from_str(raw).map_err(|e| e.to_string())
    }

    pub fn extension_count(&self) -> usize { self.extensions.len() }
}

impl Default for ExtensionRunner { fn default() -> Self { Self::new() } }

/// Alias for backwards compatibility.
pub type ExtensionService = ExtensionRunner;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionPreparation {
    #[serde(rename = "firstKeptEntryId")] pub first_kept_entry_id: String,
    #[serde(rename = "messagesToSummarize")] pub messages_to_summarize: Vec<serde_json::Value>,
    #[serde(rename = "turnPrefixMessages")] pub turn_prefix_messages: Vec<serde_json::Value>,
    #[serde(rename = "isSplitTurn")] pub is_split_turn: bool,
    #[serde(rename = "tokensBefore")] pub tokens_before: u64,
    #[serde(rename = "previousSummary", skip_serializing_if = "Option::is_none")] pub previous_summary: Option<String>,
    #[serde(rename = "fileOps")] pub file_ops: serde_json::Value,
    pub settings: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionResult {
    pub summary: String,
    #[serde(rename = "firstKeptEntryId")] pub first_kept_entry_id: String,
    #[serde(rename = "tokensBefore")] pub tokens_before: u64,
    #[serde(skip_serializing_if = "Option::is_none")] pub details: Option<serde_json::Value>,
}
