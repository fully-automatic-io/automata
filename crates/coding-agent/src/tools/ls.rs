
use agent_core::tool::AgentTool;
use agent_core::types::{AgentToolResult, AgentToolUpdateCallback, ContentBlock};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

// Output limits
const DEFAULT_MAX_ENTRIES: usize = 500;
const DEFAULT_MAX_BYTES: usize = 50_000;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LsToolDetails {
    pub truncated: bool,
    pub total_entries: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub displayed_entries: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_entries: Option<usize>,
}

// ============================================================================
// Ls Tool
// ============================================================================

pub struct LsTool {
    cwd: String,
}

impl LsTool {
    pub fn new(cwd: String) -> Self {
        Self { cwd }
    }

    fn resolve_path(&self, path: &str) -> PathBuf {
        let p = Path::new(path);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            Path::new(&self.cwd).join(p)
        }
    }
}

#[async_trait]
impl AgentTool for LsTool {
    fn name(&self) -> &str {
        "ls"
    }

    fn label(&self) -> &str {
        "ls"
    }

    fn description(&self) -> &str {
        "List files and directories in a path. Directories are marked with trailing /."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list (default: current directory)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of entries to return (default: 500)"
                }
            }
        })
    }

    async fn execute(
        &self,
        _tool_call_id: String,
        params: serde_json::Value,
        _signal: Option<CancellationToken>,
        _on_update: Option<AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>> {
        let path = params
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or(".");

        let limit = params
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_MAX_ENTRIES as u64) as usize;

        // Resolve path
        let target = self.resolve_path(path);

        // Check if path exists
        if !target.exists() {
            return Ok(AgentToolResult {
                content: vec![ContentBlock::Text {
                    text: format!("Path does not exist: {}", path),
                }],
                details: serde_json::to_value(LsToolDetails::default()).unwrap_or_default(),
                terminate: false,
            });
        }

        // Check if it's a directory
        if !target.is_dir() {
            return Ok(AgentToolResult {
                content: vec![ContentBlock::Text {
                    text: format!("Not a directory: {}", path),
                }],
                details: serde_json::to_value(LsToolDetails::default()).unwrap_or_default(),
                terminate: false,
            });
        }

        // Read directory entries
        let read_dir = std::fs::read_dir(&target)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

        // Collect entries
        let mut entries: Vec<(String, bool)> = Vec::new();

        for entry in read_dir {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue, // Skip entries we can't read
            };

            let name = entry.file_name().to_string_lossy().to_string();
            let is_dir = entry
                .file_type()
                .map(|ft| ft.is_dir())
                .unwrap_or(false);

            entries.push((name, is_dir));
        }

        // Sort entries alphabetically (case-insensitive)
        entries.sort_by(|a, b| {
            a.0.to_lowercase().cmp(&b.0.to_lowercase())
        });

        // Apply limits and format
        let mut result_lines: Vec<String> = Vec::new();
        let mut total_bytes = 0;
        let mut truncated = false;

        for (name, is_dir) in entries.iter() {
            // Check entry limit
            if result_lines.len() >= limit {
                truncated = true;
                break;
            }

            // Format entry (add trailing / for directories)
            let display_name = if *is_dir {
                format!("{}/", name)
            } else {
                name.clone()
            };

            // Check byte limit
            if total_bytes + display_name.len() + 1 > DEFAULT_MAX_BYTES {
                truncated = true;
                break;
            }

            total_bytes += display_name.len() + 1; // +1 for newline
            result_lines.push(display_name);
        }

        // Build result text
        let result_text = if result_lines.is_empty() {
            format!("Directory is empty: {}", path)
        } else {
            let mut text = result_lines.join("\n");

            if truncated {
                text.push_str(&format!(
                    "\n\n... (truncated at {} entries out of {})",
                    result_lines.len(),
                    entries.len()
                ));
            }

            text
        };

        Ok(AgentToolResult {
            content: vec![ContentBlock::Text { text: result_text }],
            details: serde_json::to_value(LsToolDetails {
                truncated,
                total_entries: entries.len(),
                displayed_entries: Some(result_lines.len()),
                max_entries: Some(limit),
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
    fn test_ls_tool_details_wire_shape() {
        let d = LsToolDetails {
            truncated: true,
            total_entries: 600,
            displayed_entries: Some(500),
            max_entries: Some(500),
        };
        let v = serde_json::to_value(&d).unwrap();
        assert_eq!(v["truncated"], true);
        assert_eq!(v["total_entries"], 600);
        assert_eq!(v["displayed_entries"], 500);
        assert_eq!(v["max_entries"], 500);
    }

    #[test]
    fn test_ls_tool_name() {
        let tool = LsTool::new("/tmp".into());
        assert_eq!(tool.name(), "ls");
        assert_eq!(tool.label(), "ls");
    }

    #[test]
    fn test_resolve_path() {
        let tool = LsTool::new("/home/user".into());

        // Relative path
        let rel = tool.resolve_path("src");
        assert_eq!(rel, PathBuf::from("/home/user/src"));

        // Absolute path
        let abs = tool.resolve_path("/tmp");
        assert_eq!(abs, PathBuf::from("/tmp"));

        // Current directory
        let cur = tool.resolve_path(".");
        assert_eq!(cur, PathBuf::from("/home/user/."));
    }

    #[tokio::test]
    async fn test_ls_nonexistent_path() {
        let tool = LsTool::new("/tmp".into());
        let params = serde_json::json!({
            "path": "/nonexistent/path/12345"
        });
        let result = tool.execute("test".into(), params, None, None).await.unwrap();
        let text = match &result.content[0] {
            ContentBlock::Text { text } => text,
            _ => panic!("Expected text content"),
        };
        assert!(text.contains("does not exist"));
    }
}
