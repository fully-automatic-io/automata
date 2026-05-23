
use crate::types::{
    AgentToolResult, AgentToolUpdateCallback, ContentBlock, ToolExecutionMode,
};
use async_trait::async_trait;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

// ============================================================================
// AgentTool trait — matches TS AgentTool<TParameters, TDetails>
// ============================================================================

/// Tool definition used by the agent runtime.
///
/// TParameters: the validated input type (deserialized from JSON Schema params).
/// TDetails: the structured details type for tool results.
#[async_trait]
pub trait AgentTool: Send + Sync {
    /// Tool name (used in LLM tool calls).
    fn name(&self) -> &str;

    /// Human-readable label for UI display.
    fn label(&self) -> &str;

    /// Description for the LLM.
    fn description(&self) -> &str;

    /// JSON Schema for the tool's input parameters.
    /// Returns `Value` because every tool defines its own schema shape.
    fn parameters(&self) -> serde_json::Value;

    /// Optional compatibility shim for raw tool-call arguments before schema validation.
    /// Must return an object that matches the parameter schema.
    fn prepare_arguments(&self, args: serde_json::Value) -> serde_json::Value {
        args
    }

    /// Execute the tool call. Throw on failure instead of encoding errors in `content`.
    /// `params` is the validated argument value; its concrete shape is defined
    /// by `parameters()` and is therefore typed as `Value` at the trait boundary.
    async fn execute(
        &self,
        tool_call_id: String,
        params: serde_json::Value,
        signal: Option<CancellationToken>,
        on_update: Option<AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>>;

    /// Per-tool execution mode override.
    /// - Sequential: this tool must execute one at a time.
    /// - Parallel: this tool can execute concurrently.
    /// If None, the default execution mode applies.
    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        None
    }
}

// ============================================================================
// Tool definition for extension-registered tools
// ============================================================================

/// Wraps a ToolDefinition into the AgentTool trait.
pub struct ToolDefinitionWrapper {
    pub name: String,
    pub label: String,
    pub description: String,
    /// JSON Schema for the tool's input parameters.
    pub parameters_schema: serde_json::Value,
    pub execution_mode_override: Option<ToolExecutionMode>,
    pub prepare_arguments_fn: Option<
        Arc<dyn Fn(serde_json::Value) -> serde_json::Value + Send + Sync>,
    >,
    pub execute_fn: Arc<
        dyn Fn(
                String,
                serde_json::Value,
                Option<CancellationToken>,
                Option<AgentToolUpdateCallback>,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<
                            Output = Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>>,
                        > + Send,
                >,
            > + Send
            + Sync,
    >,
}

#[async_trait]
impl AgentTool for ToolDefinitionWrapper {
    fn name(&self) -> &str {
        &self.name
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> serde_json::Value {
        self.parameters_schema.clone()
    }

    fn prepare_arguments(&self, args: serde_json::Value) -> serde_json::Value {
        if let Some(ref f) = self.prepare_arguments_fn {
            f(args)
        } else {
            args
        }
    }

    async fn execute(
        &self,
        tool_call_id: String,
        params: serde_json::Value,
        signal: Option<CancellationToken>,
        on_update: Option<AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>> {
        (self.execute_fn)(tool_call_id, params, signal, on_update).await
    }

    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        self.execution_mode_override
    }
}

// ============================================================================
// Tool Result helpers
// ============================================================================

/// Create an error tool result from a message.
pub fn create_error_tool_result(message: impl Into<String>) -> AgentToolResult {
    AgentToolResult::error_text(message)
}

/// Create a successful tool result.
pub fn create_success_tool_result(
    content: Vec<ContentBlock>,
    details: serde_json::Value,
) -> AgentToolResult {
    AgentToolResult {
        content,
        details,
        terminate: false,
    }
}

/// Downcast a tool result `details` Value into a tool-specific typed struct.
/// Returns `Err` if the JSON shape doesn't match — typically because the
/// caller assumed the wrong tool's `*Details` type.
pub fn downcast_details<T: serde::de::DeserializeOwned>(
    details: &serde_json::Value,
) -> Result<T, serde_json::Error> {
    serde_json::from_value(details.clone())
}

// ============================================================================
// Validation utilities
// ============================================================================

/// Validate tool arguments against the tool's JSON Schema.
/// Returns the validated arguments on success, or an error on failure.
pub fn validate_tool_arguments(
    tool: &dyn AgentTool,
    prepared_args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let schema = tool.parameters();

    // If schema has no properties or is empty, skip validation
    if schema.get("properties").and_then(|p| p.as_object()).map_or(true, |o| o.is_empty()) {
        return Ok(prepared_args);
    }

    // Basic validation: check required fields
    if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
        let obj = prepared_args
            .as_object()
            .ok_or_else(|| format!("Tool {}: arguments must be an object", tool.name()))?;
        for req in required {
            let key = req.as_str().unwrap_or("");
            if !obj.contains_key(key) {
                return Err(format!(
                    "Tool {}: missing required field '{}'",
                    tool.name(),
                    key
                ));
            }
        }
    }

    Ok(prepared_args)
}

// ============================================================================
// Tool Registry
// ============================================================================

/// Registry of available tools, keyed by name.
#[derive(Default)]
pub struct ToolRegistry {
    tools: Vec<Arc<dyn AgentTool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn register(&mut self, tool: Arc<dyn AgentTool>) {
        // Replace existing tool with same name if present
        let name = tool.name().to_string();
        self.tools.retain(|t| t.name() != name);
        self.tools.push(tool);
    }

    pub fn find(&self, name: &str) -> Option<Arc<dyn AgentTool>> {
        self.tools.iter().find(|t| t.name() == name).cloned()
    }

    pub fn all(&self) -> Vec<Arc<dyn AgentTool>> {
        self.tools.clone()
    }

    pub fn names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name().to_string()).collect()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Clone for ToolRegistry {
    fn clone(&self) -> Self {
        Self {
            tools: self.tools.clone(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTool {
        name: String,
        label: String,
        description: String,
        executed: std::sync::Mutex<bool>,
    }

    impl MockTool {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                label: format!("{} label", name),
                description: format!("{} description", name),
                executed: std::sync::Mutex::new(false),
            }
        }
    }

    #[async_trait]
    impl AgentTool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn label(&self) -> &str {
            &self.label
        }

        fn description(&self) -> &str {
            &self.description
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                },
                "required": ["input"]
            })
        }

        async fn execute(
            &self,
            _tool_call_id: String,
            params: serde_json::Value,
            _signal: Option<CancellationToken>,
            _on_update: Option<AgentToolUpdateCallback>,
        ) -> Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>> {
            *self.executed.lock().unwrap() = true;
            Ok(AgentToolResult {
                content: vec![ContentBlock::Text {
                    text: format!("done: {}", params),
                }],
                details: serde_json::Value::Null,
                terminate: false,
            })
        }
    }

    #[tokio::test]
    async fn test_tool_execution() {
        let tool = MockTool::new("mock");
        let result = tool
            .execute(
                "tc1".to_string(),
                serde_json::json!({"input": "test"}),
                None,
                None,
            )
            .await
            .unwrap();
        assert!(!result.terminate);
    }

    #[test]
    fn test_validate_arguments() {
        let tool = MockTool::new("mock");
        let result = validate_tool_arguments(
            &tool,
            serde_json::json!({"input": "hello"}),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_arguments_missing_required() {
        let tool = MockTool::new("mock");
        let result = validate_tool_arguments(&tool, serde_json::json!({}));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("input"));
    }

    #[test]
    fn test_tool_registry() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(MockTool::new("bash")));
        reg.register(Arc::new(MockTool::new("edit")));

        assert_eq!(reg.len(), 2);
        assert!(reg.find("bash").is_some());
        assert!(reg.find("nonexistent").is_none());
        assert_eq!(reg.names(), vec!["bash", "edit"]);
    }

    #[test]
    fn test_tool_registry_replace() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(MockTool::new("bash")));
        reg.register(Arc::new(MockTool::new("bash")));
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_error_tool_result() {
        let result = create_error_tool_result("something went wrong");
        assert!(!result.terminate);
        match &result.content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "something went wrong"),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn test_downcast_details_round_trip() {
        #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        struct MyDetails { count: u32, ok: bool }
        let original = MyDetails { count: 42, ok: true };
        let value = serde_json::to_value(&original).unwrap();
        let recovered: MyDetails = downcast_details(&value).unwrap();
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_downcast_details_shape_mismatch() {
        #[derive(Debug, serde::Deserialize)]
        struct Wanted { #[allow(dead_code)] needed_field: u32 }
        let value = serde_json::json!({"unrelated": "shape"});
        assert!(downcast_details::<Wanted>(&value).is_err());
    }
}
