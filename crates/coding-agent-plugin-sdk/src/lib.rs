//! Rust SDK for Automata coding-agent Wasmtime component plugins.
//!
//! Plugins implement [`ExtensionPlugin`] and call [`export_plugin!`]. The SDK
//! owns the WIT glue and keeps plugin code in ordinary Rust data structures.

use serde::{Deserialize, Serialize};

pub mod bindings {
    wit_bindgen::generate!({
        world: "extension",
        path: "wit",
    });
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExtensionManifest {
    #[serde(default)]
    pub tools: Vec<ToolDefinition>,
    #[serde(default)]
    pub commands: Vec<CommandDefinition>,
    #[serde(default)]
    pub flags: Vec<FlagDefinition>,
    #[serde(default)]
    pub shortcuts: Vec<ShortcutDefinition>,
    #[serde(default)]
    pub event_subscriptions: Vec<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlagDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub flag_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortcutDefinition {
    pub shortcut: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInvocation {
    pub name: String,
    #[serde(default)]
    pub args: serde_json::Value,
    #[serde(rename = "toolCallId", skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    #[serde(default)]
    pub content: Vec<serde_json::Value>,
    #[serde(default)]
    pub details: serde_json::Value,
    #[serde(default)]
    pub terminate: bool,
    #[serde(rename = "isError", default)]
    pub is_error: bool,
}

impl ToolOutput {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![serde_json::json!({ "type": "text", "text": text.into() })],
            details: serde_json::Value::Null,
            terminate: false,
            is_error: false,
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self { is_error: true, ..Self::text(text) }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionEvent {
    #[serde(flatten)]
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, Default)]
pub struct Registrar {
    manifest: ExtensionManifest,
}

impl Registrar {
    pub fn tool(&mut self, tool: ToolDefinition) -> &mut Self {
        self.manifest.tools.push(tool);
        self
    }

    pub fn command(
        &mut self,
        name: impl Into<String>,
        description: impl Into<Option<String>>,
    ) -> &mut Self {
        self.manifest.commands.push(CommandDefinition {
            name: name.into(),
            description: description.into(),
        });
        self
    }

    pub fn flag(&mut self, flag: FlagDefinition) -> &mut Self {
        self.manifest.flags.push(flag);
        self
    }

    pub fn shortcut(
        &mut self,
        shortcut: impl Into<String>,
        description: impl Into<Option<String>>,
    ) -> &mut Self {
        self.manifest.shortcuts.push(ShortcutDefinition {
            shortcut: shortcut.into(),
            description: description.into(),
        });
        self
    }

    pub fn subscribe(&mut self, event: impl Into<String>) -> &mut Self {
        self.manifest.event_subscriptions.push(event.into());
        self
    }

    pub fn into_manifest(self) -> ExtensionManifest {
        self.manifest
    }
}

#[derive(Debug, Clone, Default)]
pub struct Context;

impl Context {
    pub fn host_action<T: Serialize>(&mut self, action: &T) -> Result<serde_json::Value, String> {
        let json = serde_json::to_string(action).map_err(|err| err.to_string())?;
        let response = bindings::host_action(&json)?;
        serde_json::from_str(&response).map_err(|err| err.to_string())
    }

    pub fn log(&mut self, level: &str, message: impl Into<String>) -> Result<(), String> {
        let _ = self.host_action(&serde_json::json!({
            "kind": "log",
            "level": level,
            "message": message.into(),
        }))?;
        Ok(())
    }

    pub fn append_message(
        &mut self,
        role: impl Into<String>,
        content: serde_json::Value,
    ) -> Result<(), String> {
        let _ = self.host_action(&serde_json::json!({
            "kind": "append_message",
            "role": role.into(),
            "content": content,
        }))?;
        Ok(())
    }

    pub fn set_session_name(&mut self, name: impl Into<String>) -> Result<(), String> {
        let _ = self.host_action(&serde_json::json!({
            "kind": "set_session_name",
            "name": name.into(),
        }))?;
        Ok(())
    }

    pub fn set_active_tools(&mut self, tools: Vec<String>) -> Result<(), String> {
        let _ = self.host_action(&serde_json::json!({
            "kind": "set_active_tools",
            "tools": tools,
        }))?;
        Ok(())
    }

    pub fn register_provider(
        &mut self,
        name: impl Into<String>,
        config: serde_json::Value,
    ) -> Result<(), String> {
        let _ = self.host_action(&serde_json::json!({
            "kind": "register_provider",
            "name": name.into(),
            "config": config,
        }))?;
        Ok(())
    }

    pub fn headless_ui(&mut self, action: serde_json::Value) -> Result<serde_json::Value, String> {
        self.host_action(&serde_json::json!({
            "kind": "ui",
            "action": action,
        }))
    }
}

pub trait ExtensionPlugin {
    fn register(registrar: &mut Registrar) -> Result<(), String>;

    fn on_event(
        _ctx: &mut Context,
        _event: serde_json::Value,
    ) -> Result<Option<serde_json::Value>, String> {
        Ok(None)
    }

    fn invoke_tool(
        _ctx: &mut Context,
        request: ToolInvocation,
    ) -> Result<serde_json::Value, String> {
        Err(format!("plugin does not implement tool '{}'", request.name))
    }
}

#[macro_export]
macro_rules! export_plugin {
    ($plugin:ty) => {
        struct __AutomataPlugin;

        impl $crate::bindings::Guest for __AutomataPlugin {
            fn register() -> Result<String, String> {
                let mut registrar = $crate::Registrar::default();
                <$plugin as $crate::ExtensionPlugin>::register(&mut registrar)?;
                serde_json::to_string(&registrar.into_manifest()).map_err(|err| err.to_string())
            }

            fn on_event(event_json: String) -> Result<Option<String>, String> {
                let event = serde_json::from_str(&event_json).map_err(|err| err.to_string())?;
                let mut ctx = $crate::Context::default();
                <$plugin as $crate::ExtensionPlugin>::on_event(&mut ctx, event)?
                    .map(|value| serde_json::to_string(&value).map_err(|err| err.to_string()))
                    .transpose()
            }

            fn invoke_tool(request_json: String) -> Result<String, String> {
                let request = serde_json::from_str(&request_json).map_err(|err| err.to_string())?;
                let mut ctx = $crate::Context::default();
                let output = <$plugin as $crate::ExtensionPlugin>::invoke_tool(&mut ctx, request)?;
                serde_json::to_string(&output).map_err(|err| err.to_string())
            }
        }

        $crate::bindings::export!(__AutomataPlugin with_types_in $crate::bindings);
    };
}
