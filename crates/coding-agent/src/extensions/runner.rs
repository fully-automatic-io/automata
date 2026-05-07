use super::types::*;
use std::collections::HashMap;

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
        let mut tools = HashMap::new();
        for ext in &self.extensions {
            for (name, tool) in &ext.tools {
                tools.insert(name.clone(), tool.clone());
            }
        }
        tools
    }

    pub fn collect_commands(&self) -> HashMap<String, RegisteredCommand> {
        let mut cmds = HashMap::new();
        for ext in &self.extensions {
            for (name, cmd) in &ext.commands {
                cmds.insert(name.clone(), RegisteredCommand {
                    name: cmd.name.clone(),
                    source_info: cmd.source_info.clone(),
                    description: cmd.description.clone(),
                });
            }
        }
        cmds
    }

    pub fn collect_flags(&self) -> HashMap<String, ExtensionFlag> {
        let mut flags = HashMap::new();
        for ext in &self.extensions {
            for (name, flag) in &ext.flags {
                flags.insert(name.clone(), flag.clone());
            }
        }
        flags
    }

    /// Dispatch an event to all subscribed extensions. Returns (path, result_value) pairs.
    pub fn dispatch_event(
        &self,
        event: &ExtensionEvent,
    ) -> Vec<(String, serde_json::Value)> {
        let event_type = event_type_str(event);
        let event_json = match serde_json::to_string(event) {
            Ok(j) => j,
            Err(_) => return vec![],
        };

        let mut results = vec![];
        for ext in &self.extensions {
            if !ext.event_subscriptions.contains(event_type) { continue; }
            let mut plugin = match ext.plugin.lock() {
                Ok(p) => p,
                Err(_) => continue,
            };
            if let Ok(raw) = plugin.call::<&str, &str>("on_event", &event_json) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) {
                    if !v.is_null() {
                        results.push((ext.path.clone(), v));
                    }
                }
            }
        }
        results
    }

    /// Invoke a tool registered by an extension.
    pub fn invoke_tool(&self, name: &str, args: serde_json::Value) -> Result<serde_json::Value, String> {
        let ext = self.extensions.iter()
            .find(|e| e.tools.contains_key(name))
            .ok_or_else(|| format!("No extension provides tool '{}'", name))?;

        let input = serde_json::json!({ "name": name, "args": args });
        let input_str = serde_json::to_string(&input).map_err(|e| e.to_string())?;

        let mut plugin = ext.plugin.lock().map_err(|e| e.to_string())?;
        let raw = plugin.call::<&str, &str>("invoke_tool", &input_str)
            .map_err(|e| e.to_string())?;
        serde_json::from_str(raw).map_err(|e| e.to_string())
    }

    pub fn extension_count(&self) -> usize {
        self.extensions.len()
    }
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
        ExtensionEvent::EventBus { event, .. } => {
            // EventBus uses a dynamic string; we can't return &'static str here.
            // Callers that need the exact event name should handle EventBus separately.
            let _ = event;
            "event_bus"
        }
    }
}

impl Default for ExtensionRunner {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runner_empty() {
        let runner = ExtensionRunner::new();
        assert_eq!(runner.extension_count(), 0);
        assert!(runner.collect_tools().is_empty());
    }
}
