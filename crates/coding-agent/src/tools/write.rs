
use agent_core::tool::AgentTool;
use agent_core::types::{AgentToolResult, AgentToolUpdateCallback, ContentBlock};
use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteToolDetails {
    pub bytes: usize,
    #[serde(rename = "fileExists")]
    pub file_exists: bool,
    #[serde(rename = "emojisFiltered")]
    pub emojis_filtered: bool,
    #[serde(rename = "isDocumentation")]
    pub is_documentation: bool,
}

// ============================================================================
// WriteOperations trait — pluggable file I/O
// ============================================================================

#[async_trait]
pub trait WriteOperations: Send + Sync {
    async fn write_file(&self, path: &str, content: &str) -> Result<(), String>;
    async fn read_file(&self, path: &str) -> Result<Vec<u8>, String>;
    async fn mkdir(&self, dir: &str) -> Result<(), String>;
    async fn exists(&self, path: &str) -> bool;
}

pub struct LocalWriteOperations;

#[async_trait]
impl WriteOperations for LocalWriteOperations {
    async fn write_file(&self, path: &str, content: &str) -> Result<(), String> {
        tokio::fs::write(path, content)
            .await
            .map_err(|e| e.to_string())
    }

    async fn read_file(&self, path: &str) -> Result<Vec<u8>, String> {
        tokio::fs::read(path).await.map_err(|e| e.to_string())
    }

    async fn mkdir(&self, dir: &str) -> Result<(), String> {
        tokio::fs::create_dir_all(dir)
            .await
            .map_err(|e| e.to_string())
    }

    async fn exists(&self, path: &str) -> bool {
        tokio::fs::metadata(path).await.is_ok()
    }
}

// ============================================================================
// Utility functions
// ============================================================================

/// Check if file is a documentation file
fn is_documentation_file(path: &str) -> bool {
    let path_lower = path.to_lowercase();

    // Check for README files
    if path_lower.contains("readme") {
        return true;
    }

    // Check for .md extension
    if path_lower.ends_with(".md") {
        return true;
    }

    false
}

/// Filter emojis from content (unless explicitly requested)
fn filter_emojis(content: &str) -> String {
    lazy_static::lazy_static! {
        // Match emoji characters (basic emoji range)
        static ref EMOJI_REGEX: Regex = Regex::new(
            r"[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]"
        ).unwrap();
    }

    EMOJI_REGEX.replace_all(content, "").to_string()
}

/// Check if content contains emojis
fn contains_emojis(content: &str) -> bool {
    lazy_static::lazy_static! {
        static ref EMOJI_REGEX: Regex = Regex::new(
            r"[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]"
        ).unwrap();
    }

    EMOJI_REGEX.is_match(content)
}

// ============================================================================
// Write Tool
// ============================================================================

#[derive(Default)]
pub struct WriteToolOptions {
    pub operations: Option<Arc<dyn WriteOperations>>,
}


pub struct WriteTool {
    cwd: String,
    operations: Arc<dyn WriteOperations>,
}

impl WriteTool {
    pub fn new(cwd: String, options: WriteToolOptions) -> Self {
        let ops = options
            .operations
            .unwrap_or_else(|| Arc::new(LocalWriteOperations));
        Self {
            cwd,
            operations: ops,
        }
    }

    fn resolve_path(&self, path: &str) -> String {
        if Path::new(path).is_absolute() {
            path.to_string()
        } else {
            Path::new(&self.cwd)
                .join(path)
                .to_string_lossy()
                .to_string()
        }
    }
}

#[async_trait]
impl AgentTool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn label(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates parent directories if needed. Overwrites existing files."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(
        &self,
        _tool_call_id: String,
        params: serde_json::Value,
        _signal: Option<CancellationToken>,
        on_update: Option<AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>> {
        let path = params
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' parameter")?;

        let content = params
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'content' parameter")?;

        let absolute_path = self.resolve_path(path);

        // Check if file exists (for read-before-write validation)
        let file_exists = self.operations.exists(&absolute_path).await;

        // Validate: if file exists, ensure we can read it first
        if file_exists {
            // Try to read the file to ensure it's accessible
            match self.operations.read_file(&absolute_path).await {
                Ok(_) => {
                    // File is readable, proceed
                }
                Err(e) => {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        format!("Cannot read existing file before writing: {}", e),
                    )));
                }
            }
        }

        // Check for emojis and filter them
        let mut final_content = content.to_string();
        let mut warnings = Vec::new();

        if contains_emojis(content) {
            let filtered = filter_emojis(content);
            if filtered != content {
                warnings.push("Note: Emojis were filtered from the content. \
                    If you need emojis, please explicitly mention it in your request.".to_string());
                final_content = filtered;
            }
        }

        // Warn about documentation files
        if is_documentation_file(path) {
            warnings.push(format!(
                "Warning: Writing to documentation file '{}'. \
                Ensure this is intentional.",
                path
            ));
        }

        // Emit streaming partial: full content preview before disk write hits.
        if let Some(cb) = on_update.as_ref() {
            let partial = AgentToolResult {
                content: vec![ContentBlock::Text {
                    text: final_content.clone(),
                }],
                details: serde_json::to_value(WriteToolDetails {
                    bytes: final_content.len(),
                    file_exists,
                    emojis_filtered: content != final_content,
                    is_documentation: is_documentation_file(path),
                })
                .unwrap_or_default(),
                terminate: false,
            };
            cb(partial);
        }

        // Create parent directories
        if let Some(parent) = Path::new(&absolute_path).parent() {
            self.operations
                .mkdir(&parent.to_string_lossy())
                .await
                .map_err(|e| {
                    Box::new(std::io::Error::other(e)) as Box<_>
                })?;
        }

        // Write file
        self.operations
            .write_file(&absolute_path, &final_content)
            .await
            .map_err(|e| {
                Box::new(std::io::Error::other(e)) as Box<_>
            })?;

        // Build result message
        let mut message = format!(
            "Successfully wrote {} bytes to {}",
            final_content.len(),
            path
        );

        if !warnings.is_empty() {
            message.push_str("\n\n");
            message.push_str(&warnings.join("\n"));
        }

        Ok(AgentToolResult {
            content: vec![ContentBlock::Text { text: message }],
            details: serde_json::to_value(WriteToolDetails {
                bytes: final_content.len(),
                file_exists,
                emojis_filtered: content != final_content,
                is_documentation: is_documentation_file(path),
            }).unwrap_or_default(),
            terminate: false,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_tool_details_wire_shape() {
        let d = WriteToolDetails {
            bytes: 42,
            file_exists: true,
            emojis_filtered: false,
            is_documentation: true,
        };
        let v = serde_json::to_value(&d).unwrap();
        assert_eq!(v["bytes"], 42);
        assert_eq!(v["fileExists"], true);
        assert_eq!(v["emojisFiltered"], false);
        assert_eq!(v["isDocumentation"], true);
    }

    #[test]
    fn test_write_tool_name() {
        let tool = WriteTool::new("/tmp".into(), WriteToolOptions::default());
        assert_eq!(tool.name(), "write");
        assert_eq!(tool.label(), "write");
    }

    #[test]
    fn test_is_documentation_file() {
        assert!(is_documentation_file("README.md"));
        assert!(is_documentation_file("readme.txt"));
        assert!(is_documentation_file("docs/guide.md"));
        assert!(is_documentation_file("CONTRIBUTING.md"));
        assert!(!is_documentation_file("src/main.rs"));
        assert!(!is_documentation_file("test.txt"));
    }

    #[test]
    fn test_contains_emojis() {
        assert!(contains_emojis("Hello 😀 World"));
        assert!(contains_emojis("🚀 Rocket"));
        assert!(!contains_emojis("Hello World"));
        assert!(!contains_emojis("No emojis here!"));
    }

    #[test]
    fn test_filter_emojis() {
        assert_eq!(filter_emojis("Hello 😀 World"), "Hello  World");
        assert_eq!(filter_emojis("🚀 Rocket"), " Rocket");
        assert_eq!(filter_emojis("No emojis"), "No emojis");
    }

    #[tokio::test]
    async fn test_write_tool_missing_path() {
        let tool = WriteTool::new("/tmp".into(), WriteToolOptions::default());
        let params = serde_json::json!({
            "content": "test"
        });
        let result = tool.execute("test".into(), params, None, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_write_tool_missing_content() {
        let tool = WriteTool::new("/tmp".into(), WriteToolOptions::default());
        let params = serde_json::json!({
            "path": "test.txt"
        });
        let result = tool.execute("test".into(), params, None, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn streaming_callback_receives_preview_before_write() {
        use std::sync::Mutex;
        let dir = tempfile::tempdir().unwrap();
        let tool = WriteTool::new(
            dir.path().to_string_lossy().into_owned(),
            WriteToolOptions::default(),
        );

        let updates: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let updates_clone = updates.clone();
        let on_update: AgentToolUpdateCallback = Box::new(move |partial: AgentToolResult| {
            for block in &partial.content {
                if let ContentBlock::Text { text } = block {
                    updates_clone.lock().unwrap().push(text.clone());
                }
            }
        });

        let params = serde_json::json!({
            "path": "preview.txt",
            "content": "hello streaming preview"
        });
        let result = tool.execute("test".into(), params, None, Some(on_update)).await;
        assert!(result.is_ok());

        let collected = updates.lock().unwrap();
        assert_eq!(collected.len(), 1, "expected one preview update");
        assert_eq!(collected[0], "hello streaming preview");
    }
}
