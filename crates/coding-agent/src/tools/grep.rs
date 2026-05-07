
use agent_core::tool::AgentTool;
use agent_core::types::{AgentToolResult, AgentToolUpdateCallback, ContentBlock};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

// Constants matching TypeScript implementation
const DEFAULT_MAX_MATCHES: usize = 100;
const DEFAULT_MAX_BYTES: usize = 50_000;
const MAX_LINE_LENGTH: usize = 500;

#[derive(Debug, Serialize, Deserialize)]
pub struct GrepToolDetails {
    pub truncated: bool,
    pub total_matches: usize,
    pub max_matches: usize,
}

pub struct GrepTool {
    cwd: String,
}

impl GrepTool {
    pub fn new(cwd: String) -> Self {
        Self { cwd }
    }

    /// Truncate a single line to MAX_LINE_LENGTH
    fn truncate_line(line: &str) -> String {
        if line.len() <= MAX_LINE_LENGTH {
            line.to_string()
        } else {
            format!("{}... (line truncated)", &line[..MAX_LINE_LENGTH])
        }
    }

    /// Resolve path relative to cwd
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
impl AgentTool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn label(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search for a pattern in files using ripgrep. Supports regex patterns, context lines, and glob filtering."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Pattern to search for (regex by default, or literal if literal=true)"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in (default: current directory)"
                },
                "glob": {
                    "type": "string",
                    "description": "File glob pattern to filter (e.g., '*.ts', '**/*.json')"
                },
                "ignoreCase": {
                    "type": "boolean",
                    "description": "Case-insensitive search"
                },
                "literal": {
                    "type": "boolean",
                    "description": "Treat pattern as literal string instead of regex"
                },
                "context": {
                    "type": "integer",
                    "description": "Number of context lines to show before and after matches (default: 0)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matches to return (default: 100)"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(
        &self,
        _tool_call_id: String,
        params: serde_json::Value,
        signal: Option<CancellationToken>,
        _on_update: Option<AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>> {
        // Parse parameters
        let pattern = params
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'pattern' parameter")?;

        let search_path = params
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or(".");

        let glob = params.get("glob").and_then(|v| v.as_str());
        let ignore_case = params.get("ignoreCase").and_then(|v| v.as_bool()).unwrap_or(false);
        let literal = params.get("literal").and_then(|v| v.as_bool()).unwrap_or(false);
        let context = params.get("context").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(DEFAULT_MAX_MATCHES as u64) as usize;

        // Resolve path
        let resolved_path = self.resolve_path(search_path);

        // Check if path exists
        if !resolved_path.exists() {
            return Ok(AgentToolResult {
                content: vec![ContentBlock::Text {
                    text: format!("Path does not exist: {}", search_path),
                }],
                details: serde_json::json!({
                    "truncated": false,
                    "total_matches": 0,
                    "max_matches": limit
                }),
                terminate: false,
            });
        }

        // Build ripgrep command
        let mut cmd = Command::new("rg");

        // Output format: JSON lines for easier parsing
        cmd.arg("--json");

        // Pattern
        if literal {
            cmd.arg("--fixed-strings");
        }
        if ignore_case {
            cmd.arg("--ignore-case");
        }
        cmd.arg(pattern);

        // Path
        cmd.arg(&resolved_path);

        // Glob filter
        if let Some(g) = glob {
            cmd.arg("--glob").arg(g);
        }

        // Context lines
        if context > 0 {
            cmd.arg("--context").arg(context.to_string());
        }

        // Max count (limit + 1 to detect truncation)
        cmd.arg("--max-count").arg((limit + 1).to_string());

        // Apply .gitignore rules
        cmd.arg("--no-require-git");

        // Hidden files
        cmd.arg("--hidden");

        // Execute command
        let output = if let Some(token) = signal {
            tokio::select! {
                result = cmd.output() => result?,
                _ = token.cancelled() => {
                    return Ok(AgentToolResult {
                        content: vec![ContentBlock::Text {
                            text: "Search cancelled".to_string(),
                        }],
                        details: serde_json::json!({}),
                        terminate: false,
                    });
                }
            }
        } else {
            cmd.output().await?
        };

        // Parse JSON output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut matches: Vec<String> = Vec::new();
        let mut total_matches = 0;
        let mut current_bytes = 0;
        let mut truncated = false;

        for line in stdout.lines() {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
                if json["type"] == "match" {
                    total_matches += 1;

                    // Check match limit
                    if matches.len() >= limit {
                        truncated = true;
                        break;
                    }

                    // Extract match information
                    let path = json["data"]["path"]["text"].as_str().unwrap_or("");
                    let line_number = json["data"]["line_number"].as_u64().unwrap_or(0);
                    let line_text = json["data"]["lines"]["text"].as_str().unwrap_or("");

                    // Truncate long lines
                    let truncated_line = Self::truncate_line(line_text.trim_end());

                    // Format: path:line_number: content
                    let match_line = format!("{}:{}:{}", path, line_number, truncated_line);

                    // Check byte limit
                    if current_bytes + match_line.len() > DEFAULT_MAX_BYTES {
                        truncated = true;
                        break;
                    }

                    current_bytes += match_line.len() + 1; // +1 for newline
                    matches.push(match_line);
                } else if json["type"] == "context" {
                    // Context lines (before/after matches)
                    if matches.len() >= limit {
                        continue;
                    }

                    let path = json["data"]["path"]["text"].as_str().unwrap_or("");
                    let line_number = json["data"]["line_number"].as_u64().unwrap_or(0);
                    let line_text = json["data"]["lines"]["text"].as_str().unwrap_or("");

                    let truncated_line = Self::truncate_line(line_text.trim_end());
                    let context_line = format!("{}-{}-{}", path, line_number, truncated_line);

                    if current_bytes + context_line.len() <= DEFAULT_MAX_BYTES {
                        current_bytes += context_line.len() + 1;
                        matches.push(context_line);
                    }
                }
            }
        }

        // Build result text
        let result_text = if matches.is_empty() {
            if !output.stderr.is_empty() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                format!("Error: {}", stderr.trim())
            } else {
                "No matches found".to_string()
            }
        } else {
            let mut text = matches.join("\n");

            if truncated {
                text.push_str(&format!(
                    "\n\n... (truncated at {} matches out of {}+ total)",
                    matches.len(),
                    total_matches
                ));
                if matches.len() < limit {
                    text.push_str(&format!(
                        "\nConsider increasing the limit parameter (current: {})",
                        limit
                    ));
                }
            }

            text
        };

        Ok(AgentToolResult {
            content: vec![ContentBlock::Text { text: result_text }],
            details: serde_json::json!({
                "truncated": truncated,
                "total_matches": total_matches,
                "max_matches": limit
            }),
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

    #[tokio::test]
    async fn test_grep_tool_name() {
        let tool = GrepTool::new("/tmp".into());
        assert_eq!(tool.name(), "grep");
        assert_eq!(tool.label(), "grep");
    }

    #[tokio::test]
    async fn test_grep_missing_pattern() {
        let tool = GrepTool::new("/tmp".into());
        let params = serde_json::json!({});
        let result = tool.execute("test".into(), params, None, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_grep_nonexistent_path() {
        let tool = GrepTool::new("/tmp".into());
        let params = serde_json::json!({
            "pattern": "test",
            "path": "/nonexistent/path/12345"
        });
        let result = tool.execute("test".into(), params, None, None).await.unwrap();
        let text = match &result.content[0] {
            ContentBlock::Text { text } => text,
            _ => panic!("Expected text content"),
        };
        assert!(text.contains("does not exist"));
    }

    #[test]
    fn test_truncate_line() {
        let short = "short line";
        assert_eq!(GrepTool::truncate_line(short), short);

        let long = "a".repeat(600);
        let truncated = GrepTool::truncate_line(&long);
        assert!(truncated.len() < long.len());
        assert!(truncated.contains("truncated"));
    }

    #[test]
    fn test_resolve_path() {
        let tool = GrepTool::new("/home/user".into());

        // Relative path
        let rel = tool.resolve_path("src/main.rs");
        assert_eq!(rel, PathBuf::from("/home/user/src/main.rs"));

        // Absolute path
        let abs = tool.resolve_path("/tmp/test.txt");
        assert_eq!(abs, PathBuf::from("/tmp/test.txt"));
    }
}
