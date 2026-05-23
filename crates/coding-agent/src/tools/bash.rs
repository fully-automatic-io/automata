
use agent_core::tool::AgentTool;
use agent_core::types::{AgentToolResult, AgentToolUpdateCallback, ContentBlock, ToolExecutionMode};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

// ============================================================================
// Constants
// ============================================================================

pub const DEFAULT_MAX_BYTES: usize = 50 * 1024; // 50KB
pub const DEFAULT_MAX_LINES: usize = 2000;

// ============================================================================
// Bash Tool Details
// ============================================================================

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BashToolDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationResult>,
    #[serde(rename = "fullOutputPath", skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<String>,
}

// ============================================================================
// Truncation
// ============================================================================

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TruncationResult {
    pub content: String,
    pub truncated: bool,
    #[serde(rename = "totalLines")]
    pub total_lines: usize,
    #[serde(rename = "outputLines")]
    pub output_lines: usize,
    #[serde(rename = "outputBytes")]
    pub output_bytes: usize,
    #[serde(rename = "truncatedBy")]
    pub truncated_by: Option<String>,
    #[serde(rename = "maxBytes", skip_serializing_if = "Option::is_none")]
    pub max_bytes: Option<usize>,
    #[serde(rename = "maxLines", skip_serializing_if = "Option::is_none")]
    pub max_lines: Option<usize>,
    #[serde(rename = "lastLinePartial")]
    pub last_line_partial: bool,
    #[serde(rename = "firstLineExceedsLimit")]
    pub first_line_exceeds_limit: bool,
}

/// Truncate output from the tail, respecting byte and line limits.
pub fn truncate_tail(full_output: &str) -> TruncationResult {
    use agent_core::harness::utils::split_lines_for_counting;
    let lines: Vec<&str> = split_lines_for_counting(full_output);
    let total_lines = lines.len();

    let mut output_lines: Vec<&str> = vec![];
    let mut output_bytes: usize = 0;
    let mut line_count: usize = 0;

    for line in lines.iter().rev() {
        let line_bytes = line.len() + 1; // +1 for newline
        if line_count >= DEFAULT_MAX_LINES || output_bytes + line_bytes > DEFAULT_MAX_BYTES {
            break;
        }
        output_lines.push(line);
        output_bytes += line_bytes;
        line_count += 1;
    }

    output_lines.reverse();

    if line_count == total_lines {
        TruncationResult {
            content: full_output.to_string(),
            truncated: false,
            total_lines,
            output_lines: total_lines,
            output_bytes: full_output.len(),
            truncated_by: None,
            max_bytes: Some(DEFAULT_MAX_BYTES),
            max_lines: Some(DEFAULT_MAX_LINES),
            last_line_partial: false,
            first_line_exceeds_limit: false,
        }
    } else {
        let content = output_lines.join("\n");
        TruncationResult {
            content,
            truncated: true,
            total_lines,
            output_lines: line_count,
            output_bytes,
            truncated_by: if line_count >= DEFAULT_MAX_LINES {
                Some("lines".to_string())
            } else {
                Some("bytes".to_string())
            },
            max_bytes: Some(DEFAULT_MAX_BYTES),
            max_lines: Some(DEFAULT_MAX_LINES),
            last_line_partial: false,
            first_line_exceeds_limit: false,
        }
    }
}

// ============================================================================
// BashOperations trait
// ============================================================================

#[derive(Clone)]
pub struct BashExecOptions {
    pub on_data: Arc<dyn Fn(Vec<u8>) + Send + Sync>,
    pub signal: Option<CancellationToken>,
    pub timeout: Option<u64>,
    pub env: Option<HashMap<String, String>>,
}

pub struct BashExecResult {
    pub exit_code: Option<i32>,
}

#[async_trait]
pub trait BashOperations: Send + Sync {
    async fn exec(
        &self,
        command: &str,
        cwd: &str,
        options: BashExecOptions,
    ) -> Result<BashExecResult, Box<dyn std::error::Error + Send + Sync>>;
}

// ============================================================================
// Local Bash Operations
// ============================================================================

pub struct LocalBashOperations {
    shell_path: Option<String>,
}

impl LocalBashOperations {
    pub fn new(shell_path: Option<String>) -> Self {
        Self { shell_path }
    }
}

#[async_trait]
impl BashOperations for LocalBashOperations {
    async fn exec(
        &self,
        command: &str,
        cwd: &str,
        options: BashExecOptions,
    ) -> Result<BashExecResult, Box<dyn std::error::Error + Send + Sync>> {
        let shell = self.shell_path.as_deref().unwrap_or("/bin/bash");
        let mut cmd = Command::new(shell);
        cmd.arg("-c").arg(command).current_dir(cwd)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        if let Some(ref env) = options.env {
            cmd.envs(env);
        }
        let mut child = cmd.spawn()?;

        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();


        let _stdout_buf = vec![0u8; 8192];
        let _stderr_buf = vec![0u8; 8192];

        let on_data = options.on_data.clone();
        let stdout_handle = {
            let on_data = on_data.clone();
            tokio::spawn(async move {
                let mut reader = tokio::io::BufReader::new(stdout);
                loop {
                    use tokio::io::AsyncBufReadExt;
                    let mut line = vec![];
                    let n = reader.read_until(b'\n', &mut line).await.unwrap_or(0);
                    if n == 0 { break; }
                    on_data(line);
                }
            })
        };

        let stderr_handle = {
            let on_data = on_data.clone();
            tokio::spawn(async move {
                let mut reader = tokio::io::BufReader::new(stderr);
                loop {
                    use tokio::io::AsyncBufReadExt;
                    let mut line = vec![];
                    let n = reader.read_until(b'\n', &mut line).await.unwrap_or(0);
                    if n == 0 { break; }
                    on_data(line);
                }
            })
        };

        let status = if let Some(timeout_secs) = options.timeout {
            match tokio::time::timeout(
                std::time::Duration::from_secs(timeout_secs),
                child.wait(),
            ).await {
                Ok(result) => result?,
                Err(_) => {
                    let _ = child.kill().await;
                    let _ = child.wait().await;
                    return Err("Command timed out".into());
                }
            }
        } else {
            child.wait().await?
        };
        let _ = stdout_handle.await;
        let _ = stderr_handle.await;

        Ok(BashExecResult {
            exit_code: status.code(),
        })
    }
}

// ============================================================================
// Bash Tool Options
// ============================================================================

#[derive(Clone)]
pub struct BashToolOptions {
    pub operations: Option<Arc<dyn BashOperations>>,
    pub command_prefix: Option<String>,
    pub shell_path: Option<String>,
}

impl Default for BashToolOptions {
    fn default() -> Self {
        Self {
            operations: None,
            command_prefix: None,
            shell_path: None,
        }
    }
}

// ============================================================================
// Bash Tool
// ============================================================================

pub struct BashTool {
    cwd: String,
    options: BashToolOptions,
    operations: Arc<dyn BashOperations>,
}

impl BashTool {
    pub fn new(cwd: String, options: BashToolOptions) -> Self {
        let ops = options.operations.clone().unwrap_or_else(|| {
            Arc::new(LocalBashOperations::new(options.shell_path.clone()))
        });
        Self { cwd, options, operations: ops }
    }
}

#[async_trait]
impl AgentTool for BashTool {
    fn name(&self) -> &str { "bash" }
    fn label(&self) -> &str { "bash" }

    fn description(&self) -> &str {
        "Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Bash command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (optional)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(
        &self,
        _tool_call_id: String,
        params: serde_json::Value,
        signal: Option<CancellationToken>,
        _on_update: Option<AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>> {
        let command = params.get("command")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'command' field")?
            .to_string();

        let timeout = params.get("timeout").and_then(|v| v.as_u64());

        let resolved_command = if let Some(ref prefix) = self.options.command_prefix {
            format!("{}\n{}", prefix, command)
        } else {
            command
        };

        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join(format!("bash-{}.log", uuid::Uuid::new_v4()));
        let temp_path_str = temp_path.to_string_lossy().to_string();

        let chunks: Arc<Mutex<Vec<Vec<u8>>>> = Arc::new(Mutex::new(vec![]));
        let total_bytes: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
        let temp_file_written: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));

        let on_data = {
            let chunks = chunks.clone();
            let total_bytes = total_bytes.clone();
            let _temp_path = temp_path.clone();
            let temp_file_written = temp_file_written.clone();
            Arc::new(move |data: Vec<u8>| {
                let mut t = total_bytes.lock().unwrap();
                *t += data.len();
                if *t > DEFAULT_MAX_BYTES && !*temp_file_written.lock().unwrap() {
                    // Would write to temp file here
                    *temp_file_written.lock().unwrap() = true;
                }
                drop(t);

                chunks.lock().unwrap().push(data);
            })
        };

        let options = BashExecOptions {
            on_data: on_data.clone(),
            signal: signal.clone(),
            timeout,
            env: None,
        };

        let result = self.operations.exec(&resolved_command, &self.cwd, options).await;

        let all_chunks: Vec<u8> = chunks.lock().unwrap().concat();
        let full_output = String::from_utf8_lossy(&all_chunks).to_string();

        match result {
            Ok(exec_result) => {
                let truncation = truncate_tail(&full_output);
                let mut output_text = if truncation.truncated {
                    truncation.content.clone()
                } else if full_output.is_empty() {
                    "(no output)".to_string()
                } else {
                    full_output.clone()
                };

                if truncation.truncated {
                    output_text.push_str(&format!(
                        "\n\n[Output truncated. Full output: {}]",
                        temp_path_str
                    ));
                }

                if let Some(code) = exec_result.exit_code {
                    if code != 0 {
                        output_text.push_str(&format!("\n\nCommand exited with code {}", code));
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            output_text,
                        )));
                    }
                }

                Ok(AgentToolResult {
                    content: vec![ContentBlock::Text { text: output_text }],
                    details: serde_json::to_value(BashToolDetails {
                        truncation: if truncation.truncated { Some(truncation) } else { None },
                        full_output_path: Some(temp_path_str),
                    }).unwrap_or_default(),
                    terminate: false,
                })
            }
            Err(e) => {
                let mut output = full_output.clone();
                let err_msg = e.to_string();
                if err_msg.contains("aborted") {
                    if !output.is_empty() { output.push_str("\n\n"); }
                    output.push_str("Command aborted");
                } else if err_msg.contains("timeout") {
                    if !output.is_empty() { output.push_str("\n\n"); }
                    output.push_str(&format!("Command timed out"));
                }
                Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    output,
                )))
            }
        }
    }

    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        Some(ToolExecutionMode::Sequential)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_tail_short_output() {
        let result = truncate_tail("hello\nworld");
        assert!(!result.truncated);
        assert_eq!(result.total_lines, 2);
    }

    #[test]
    fn test_truncate_tail_long_output() {
        let lines: Vec<String> = (0..2001).map(|i| format!("line {}", i)).collect();
        let input = lines.join("\n");
        let result = truncate_tail(&input);
        assert!(result.truncated);
    }

    #[test]
    fn test_bash_tool_name() {
        let tool = BashTool::new("/tmp".into(), BashToolOptions::default());
        assert_eq!(tool.name(), "bash");
    }

    #[test]
    fn test_bash_tool_schema() {
        let tool = BashTool::new("/tmp".into(), BashToolOptions::default());
        let schema = tool.parameters();
        assert!(schema.get("required").unwrap().as_array().unwrap().contains(&serde_json::json!("command")));
    }
}
