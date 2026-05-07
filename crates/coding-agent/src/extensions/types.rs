// Extension system types — WASM-based plugin model.
//
// Extensions are WASM modules (produced by e.g. `extism-pdk`) that export:
//   - register() -> JSON ExtensionManifest
//   - on_event(JSON ExtensionEvent) -> JSON (null or handler result)
//   - invoke_tool(JSON { name, args }) -> JSON tool result

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

// ============================================================================
// Session Events
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExtensionEvent {
    // Resources
    #[serde(rename = "resources_discover")]
    ResourcesDiscover {
        cwd: String,
        reason: String,
    },

    // Session lifecycle
    #[serde(rename = "session_start")]
    SessionStart {
        reason: String,
        #[serde(rename = "previousSessionFile", skip_serializing_if = "Option::is_none")]
        previous_session_file: Option<String>,
    },
    #[serde(rename = "session_before_switch")]
    SessionBeforeSwitch {
        reason: String,
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
        reason: String,
        #[serde(rename = "targetSessionFile", skip_serializing_if = "Option::is_none")]
        target_session_file: Option<String>,
    },
    #[serde(rename = "session_before_tree")]
    SessionBeforeTree {
        preparation: serde_json::Value,
    },
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

    // Agent events
    #[serde(rename = "context")]
    Context {
        messages: Vec<serde_json::Value>,
    },
    #[serde(rename = "before_provider_request")]
    BeforeProviderRequest {
        payload: serde_json::Value,
    },
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
    AgentEnd {
        messages: Vec<serde_json::Value>,
    },
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
    MessageStart {
        message: serde_json::Value,
    },
    #[serde(rename = "message_update")]
    MessageUpdate {
        message: serde_json::Value,
        #[serde(rename = "assistantMessageEvent")]
        assistant_message_event: serde_json::Value,
    },
    #[serde(rename = "message_end")]
    MessageEnd {
        message: serde_json::Value,
    },
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

    // Model
    #[serde(rename = "model_select")]
    ModelSelect {
        model: serde_json::Value,
        #[serde(rename = "previousModel", skip_serializing_if = "Option::is_none")]
        previous_model: Option<serde_json::Value>,
        source: String,
    },

    // User bash
    #[serde(rename = "user_bash")]
    UserBash {
        command: String,
        #[serde(rename = "excludeFromContext")]
        exclude_from_context: bool,
        cwd: String,
    },

    // Input
    #[serde(rename = "input")]
    Input {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        images: Option<Vec<serde_json::Value>>,
        source: String,
    },

    // Tool call/result (extension event versions)
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

    // Event bus
    #[serde(rename = "event_bus")]
    EventBus {
        event: String,
        data: serde_json::Value,
    },
}

// ============================================================================
// Extension Context
// ============================================================================

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

// ============================================================================
// Registered Extension
// ============================================================================

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

// ============================================================================
// Extension manifest (returned by the WASM register() export)
// ============================================================================

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
    /// Event type strings this extension subscribes to (e.g. "agent_start").
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

// ============================================================================
// Extension (loaded)
// ============================================================================

pub struct Extension {
    pub path: String,
    pub resolved_path: String,
    pub source_info: String,
    /// Event types this extension handles.
    pub event_subscriptions: HashSet<String>,
    pub tools: HashMap<String, RegisteredTool>,
    pub commands: HashMap<String, RegisteredCommand>,
    pub flags: HashMap<String, ExtensionFlag>,
    pub shortcuts: HashMap<String, RegisteredShortcut>,
    pub plugin: Arc<Mutex<extism::Plugin>>,
}

impl Extension {
    pub fn from_manifest(
        path: String,
        resolved_path: String,
        manifest: ExtensionManifest,
        plugin: extism::Plugin,
    ) -> Self {
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
            path: path.clone(),
            resolved_path,
            source_info: path,
            event_subscriptions: manifest.event_subscriptions.into_iter().collect(),
            tools,
            commands,
            flags,
            shortcuts,
            plugin: Arc::new(Mutex::new(plugin)),
        }
    }
}

impl std::fmt::Debug for Extension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Extension")
            .field("path", &self.path)
            .field("tools_count", &self.tools.len())
            .field("commands_count", &self.commands.len())
            .finish()
    }
}

// ============================================================================
// Handler result
// ============================================================================

#[derive(Debug, Clone)]
pub enum ExtensionHandlerResult {
    None,
    Value(serde_json::Value),
}

// ============================================================================
// Load extensions result
// ============================================================================

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

// ============================================================================
// Provider config (for extensions that register providers)
// ============================================================================

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCost {
    pub input: f64,
    pub output: f64,
    #[serde(rename = "cacheRead")]
    pub cache_read: f64,
    #[serde(rename = "cacheWrite")]
    pub cache_write: f64,
}

// ============================================================================
// Compaction types (for extension events)
// ============================================================================

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
    #[serde(rename = "settings")]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extension_event_serde() {
        let event = ExtensionEvent::AgentStart;
        let json = serde_json::to_string(&event).unwrap();
        assert_eq!(json, r#"{"type":"agent_start"}"#);

        let event = ExtensionEvent::ToolExecutionEnd {
            tool_call_id: "tc1".into(),
            tool_name: "bash".into(),
            result: serde_json::json!({"output": "ok"}),
            is_error: false,
        };
        let json = serde_json::to_string(&event).unwrap();
        let deser: ExtensionEvent = serde_json::from_str(&json).unwrap();
        match deser {
            ExtensionEvent::ToolExecutionEnd { tool_call_id, .. } => {
                assert_eq!(tool_call_id, "tc1");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_extension_manifest_default() {
        let m: ExtensionManifest = serde_json::from_str("{}").unwrap();
        assert!(m.tools.is_empty());
        assert!(m.event_subscriptions.is_empty());
    }
}
