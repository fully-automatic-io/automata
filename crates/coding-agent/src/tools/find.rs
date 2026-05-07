
use agent_core::tool::AgentTool;
use agent_core::types::{AgentToolResult, AgentToolUpdateCallback, ContentBlock};
use async_trait::async_trait;
use globset::{Glob, GlobMatcher};
use ignore::WalkBuilder;
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

// Constants matching TypeScript implementation
const DEFAULT_MAX_RESULTS: usize = 1000;
const DEFAULT_MAX_BYTES: usize = 50_000;

// ============================================================================
// Find Tool
// ============================================================================

pub struct FindTool {
    cwd: String,
}

impl FindTool {
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

    /// Convert glob pattern to GlobMatcher
    fn compile_glob(pattern: &str) -> Result<GlobMatcher, String> {
        Glob::new(pattern)
            .map(|g| g.compile_matcher())
            .map_err(|e| format!("Invalid glob pattern: {}", e))
    }

    /// Check if pattern contains path separator (needs full-path matching)
    fn needs_full_path_match(pattern: &str) -> bool {
        pattern.contains('/') || pattern.contains('\\')
    }
}

#[async_trait]
impl AgentTool for FindTool {
    fn name(&self) -> &str {
        "find"
    }

    fn label(&self) -> &str {
        "find"
    }

    fn description(&self) -> &str {
        "Find files by glob pattern. Supports ** for recursive matching, respects .gitignore."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g., '*.ts', '**/*.json', 'src/**/*.spec.ts')"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 1000)"
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
        let pattern = params
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'pattern' parameter")?;

        let search_path = params
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or(".");

        let limit = params
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_MAX_RESULTS as u64) as usize;

        // Resolve search path
        let base_dir = self.resolve_path(search_path);

        // Check if path exists
        if !base_dir.exists() {
            return Ok(AgentToolResult {
                content: vec![ContentBlock::Text {
                    text: format!("Path does not exist: {}", search_path),
                }],
                details: serde_json::json!({
                    "truncated": false,
                    "total_results": 0
                }),
                terminate: false,
            });
        }

        // Compile glob pattern
        let glob_matcher = Self::compile_glob(pattern)
            .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, e)) as Box<_>)?;

        // Check if we need full path matching
        let full_path_match = Self::needs_full_path_match(pattern);

        // Build walker with .gitignore support
        let mut walker = WalkBuilder::new(&base_dir);
        walker
            .hidden(true) // Include hidden files
            .git_ignore(true) // Respect .gitignore
            .git_global(true) // Respect global .gitignore
            .git_exclude(true) // Respect .git/info/exclude
            .require_git(false) // Apply .gitignore rules even outside git repos
            .max_depth(None); // No depth limit

        // Collect results
        let mut results: Vec<String> = Vec::new();
        let mut total_bytes = 0;
        let mut truncated = false;

        for entry in walker.build() {
            // Check for cancellation
            if signal.as_ref().map_or(false, |s| s.is_cancelled()) {
                return Ok(AgentToolResult {
                    content: vec![ContentBlock::Text {
                        text: "Search cancelled".to_string(),
                    }],
                    details: serde_json::json!({}),
                    terminate: false,
                });
            }

            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue, // Skip entries we can't read
            };

            // Skip the base directory itself
            if entry.path() == base_dir {
                continue;
            }

            // Get relative path
            let rel_path = entry
                .path()
                .strip_prefix(&base_dir)
                .unwrap_or(entry.path());

            // Convert to POSIX-style path (forward slashes)
            let path_str = rel_path
                .to_string_lossy()
                .replace('\\', "/");

            // Add trailing slash for directories
            let display_path = if entry.file_type().map_or(false, |ft| ft.is_dir()) {
                format!("{}/", path_str)
            } else {
                path_str.clone()
            };

            // Match against pattern
            let matches = if full_path_match {
                // Match against full relative path
                glob_matcher.is_match(&path_str)
            } else {
                // Match against filename only
                if let Some(filename) = rel_path.file_name() {
                    glob_matcher.is_match(filename.to_string_lossy().as_ref())
                } else {
                    false
                }
            };

            if matches {
                // Check result limit
                if results.len() >= limit {
                    truncated = true;
                    break;
                }

                // Check byte limit
                if total_bytes + display_path.len() + 1 > DEFAULT_MAX_BYTES {
                    truncated = true;
                    break;
                }

                total_bytes += display_path.len() + 1; // +1 for newline
                results.push(display_path);
            }
        }

        // Build result text
        let result_text = if results.is_empty() {
            "No files found".to_string()
        } else {
            let mut text = results.join("\n");

            if truncated {
                text.push_str(&format!(
                    "\n\n... (truncated at {} results)",
                    results.len()
                ));
                if results.len() < limit {
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
                "total_results": results.len(),
                "max_results": limit
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

    #[test]
    fn test_find_tool_name() {
        let tool = FindTool::new("/tmp".into());
        assert_eq!(tool.name(), "find");
        assert_eq!(tool.label(), "find");
    }

    #[test]
    fn test_needs_full_path_match() {
        assert!(FindTool::needs_full_path_match("src/**/*.ts"));
        assert!(FindTool::needs_full_path_match("**/*.json"));
        assert!(FindTool::needs_full_path_match("foo/bar.txt"));
        assert!(!FindTool::needs_full_path_match("*.ts"));
        assert!(!FindTool::needs_full_path_match("test.*"));
    }

    #[test]
    fn test_compile_glob() {
        assert!(FindTool::compile_glob("*.ts").is_ok());
        assert!(FindTool::compile_glob("**/*.json").is_ok());
        assert!(FindTool::compile_glob("test.?").is_ok());
        assert!(FindTool::compile_glob("[invalid").is_err());
    }

    #[tokio::test]
    async fn test_find_missing_pattern() {
        let tool = FindTool::new("/tmp".into());
        let params = serde_json::json!({});
        let result = tool.execute("test".into(), params, None, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_find_nonexistent_path() {
        let tool = FindTool::new("/tmp".into());
        let params = serde_json::json!({
            "pattern": "*.txt",
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
    fn test_resolve_path() {
        let tool = FindTool::new("/home/user".into());

        // Relative path
        let rel = tool.resolve_path("src/main.rs");
        assert_eq!(rel, PathBuf::from("/home/user/src/main.rs"));

        // Absolute path
        let abs = tool.resolve_path("/tmp/test.txt");
        assert_eq!(abs, PathBuf::from("/tmp/test.txt"));
    }
}
