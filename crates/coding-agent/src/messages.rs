//
// Custom message types and transformers for the coding agent.

use serde::{Deserialize, Serialize};

// ============================================================================
// Constants
// ============================================================================

pub const COMPACTION_SUMMARY_PREFIX: &str = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
pub const COMPACTION_SUMMARY_SUFFIX: &str = "\n</summary>";

pub const BRANCH_SUMMARY_PREFIX: &str = "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n";
pub const BRANCH_SUMMARY_SUFFIX: &str = "</summary>";

// ============================================================================
// Custom Message Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BashExecutionMessage {
    pub role: String, // "bashExecution"
    pub command: String,
    pub output: String,
    #[serde(rename = "exitCode")]
    pub exit_code: Option<i32>,
    pub cancelled: bool,
    pub truncated: bool,
    #[serde(rename = "fullOutputPath", skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<String>,
    pub timestamp: u64,
    #[serde(rename = "excludeFromContext", default, skip_serializing_if = "std::ops::Not::not")]
    pub exclude_from_context: bool,
}

impl BashExecutionMessage {
    pub fn to_llm_text(&self) -> String {
        let mut text = format!("Ran `{}`\n", self.command);
        if !self.output.is_empty() {
            text.push_str(&format!("```\n{}\n```", self.output));
        } else {
            text.push_str("(no output)");
        }
        if self.cancelled {
            text.push_str("\n\n(command cancelled)");
        } else if let Some(code) = self.exit_code {
            if code != 0 {
                text.push_str(&format!("\n\nCommand exited with code {}", code));
            }
        }
        if self.truncated {
            if let Some(ref path) = self.full_output_path {
                text.push_str(&format!("\n\n[Output truncated. Full output: {}]", path));
            }
        }
        text
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMessage<T = serde_json::Value> {
    pub role: String, // "custom"
    #[serde(rename = "customType")]
    pub custom_type: String,
    pub content: serde_json::Value, // string or array of content blocks
    pub display: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<T>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchSummaryMessage {
    pub role: String, // "branchSummary"
    pub summary: String,
    #[serde(rename = "fromId")]
    pub from_id: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionSummaryMessage {
    pub role: String, // "compactionSummary"
    pub summary: String,
    #[serde(rename = "tokensBefore")]
    pub tokens_before: u64,
    pub timestamp: u64,
}

// ============================================================================
// Factory functions
// ============================================================================

pub fn create_branch_summary_message(
    summary: String,
    from_id: String,
    timestamp: i64,
) -> BranchSummaryMessage {
    BranchSummaryMessage {
        role: "branchSummary".to_string(),
        summary,
        from_id,
        timestamp: timestamp as u64,
    }
}

pub fn create_compaction_summary_message(
    summary: String,
    tokens_before: u64,
    timestamp: i64,
) -> CompactionSummaryMessage {
    CompactionSummaryMessage {
        role: "compactionSummary".to_string(),
        summary,
        tokens_before,
        timestamp: timestamp as u64,
    }
}

pub fn create_custom_message<T: Serialize>(
    custom_type: String,
    content: serde_json::Value,
    display: bool,
    details: Option<T>,
    timestamp: i64,
) -> CustomMessage<T> {
    CustomMessage {
        role: "custom".to_string(),
        custom_type,
        content,
        display,
        details,
        timestamp: timestamp as u64,
    }
}

// ============================================================================
// convertToLlm — AgentMessage[] → LLM Message[]
// ============================================================================

/// Convert coding-agent AgentMessages to LLM-compatible Messages.
/// Handles bashExecution, custom, branchSummary, compactionSummary roles.
pub fn convert_to_llm(
    messages: &[serde_json::Value],
) -> Vec<serde_json::Value> {
    messages
        .iter()
        .filter_map(|m| {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
            match role {
                "bashExecution" => {
                    if m.get("excludeFromContext")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                    {
                        return None;
                    }
                    let msg: BashExecutionMessage =
                        serde_json::from_value(m.clone()).ok()?;
                    Some(serde_json::json!({
                        "role": "user",
                        "content": [{"type": "text", "text": msg.to_llm_text()}],
                        "timestamp": msg.timestamp,
                    }))
                }
                "custom" => {
                    let content = m.get("content").cloned().unwrap_or(serde_json::Value::Null);
                    let content_blocks = if content.is_string() {
                        vec![serde_json::json!({"type": "text", "text": content.as_str().unwrap()})]
                    } else if content.is_array() {
                        content.as_array().unwrap().clone()
                    } else {
                        vec![serde_json::json!({"type": "text", "text": content.to_string()})]
                    };
                    Some(serde_json::json!({
                        "role": "user",
                        "content": content_blocks,
                        "timestamp": m.get("timestamp").and_then(|t| t.as_u64()).unwrap_or(0),
                    }))
                }
                "branchSummary" => {
                    let summary = m.get("summary").and_then(|s| s.as_str()).unwrap_or("");
                    Some(serde_json::json!({
                        "role": "user",
                        "content": [{"type": "text", "text": format!("{}{}{}", BRANCH_SUMMARY_PREFIX, summary, BRANCH_SUMMARY_SUFFIX)}],
                        "timestamp": m.get("timestamp").and_then(|t| t.as_u64()).unwrap_or(0),
                    }))
                }
                "compactionSummary" => {
                    let summary = m.get("summary").and_then(|s| s.as_str()).unwrap_or("");
                    Some(serde_json::json!({
                        "role": "user",
                        "content": [{"type": "text", "text": format!("{}{}{}", COMPACTION_SUMMARY_PREFIX, summary, COMPACTION_SUMMARY_SUFFIX)}],
                        "timestamp": m.get("timestamp").and_then(|t| t.as_u64()).unwrap_or(0),
                    }))
                }
                "user" | "assistant" | "toolResult" => Some(m.clone()),
                _ => None,
            }
        })
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bash_execution_to_llm_text() {
        let msg = BashExecutionMessage {
            role: "bashExecution".into(),
            command: "ls -la".into(),
            output: "file1\nfile2".into(),
            exit_code: Some(0),
            cancelled: false,
            truncated: false,
            full_output_path: None,
            timestamp: 1000,
            exclude_from_context: false,
        };
        let text = msg.to_llm_text();
        assert!(text.contains("ls -la"));
        assert!(text.contains("file1"));
    }

    #[test]
    fn test_bash_execution_exit_code() {
        let msg = BashExecutionMessage {
            role: "bashExecution".into(),
            command: "bad".into(),
            output: "".into(),
            exit_code: Some(1),
            cancelled: false,
            truncated: false,
            full_output_path: None,
            timestamp: 1000,
            exclude_from_context: false,
        };
        let text = msg.to_llm_text();
        assert!(text.contains("exited with code 1"));
    }

    #[test]
    fn test_convert_to_llm_bash_execution() {
        let msg = serde_json::json!({
            "role": "bashExecution",
            "command": "echo hi",
            "output": "hi",
            "exitCode": 0,
            "cancelled": false,
            "truncated": false,
            "timestamp": 1000,
        });
        let result = convert_to_llm(&[msg]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "user");
    }

    #[test]
    fn test_convert_to_llm_exclude_from_context() {
        let msg = serde_json::json!({
            "role": "bashExecution",
            "command": "secret",
            "output": "",
            "exitCode": 0,
            "cancelled": false,
            "truncated": false,
            "timestamp": 1000,
            "excludeFromContext": true,
        });
        let result = convert_to_llm(&[msg]);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_convert_to_llm_custom() {
        let msg = serde_json::json!({
            "role": "custom",
            "customType": "mytype",
            "content": [{"type": "text", "text": "hello"}],
            "display": true,
            "timestamp": 1000,
        });
        let result = convert_to_llm(&[msg]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "user");
    }

    #[test]
    fn test_convert_to_llm_branch_summary() {
        let msg = serde_json::json!({
            "role": "branchSummary",
            "summary": "Did some work",
            "fromId": "abc",
            "timestamp": 1000,
        });
        let result = convert_to_llm(&[msg]);
        assert_eq!(result.len(), 1);
        assert!(result[0]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("branch"));
    }

    #[test]
    fn test_convert_to_llm_compaction_summary() {
        let msg = serde_json::json!({
            "role": "compactionSummary",
            "summary": "Previous context",
            "tokensBefore": 5000u64,
            "timestamp": 1000,
        });
        let result = convert_to_llm(&[msg]);
        assert_eq!(result.len(), 1);
        assert!(result[0]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("compacted"));
    }

    #[test]
    fn test_convert_to_llm_passthrough() {
        let msg = serde_json::json!({
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
            "timestamp": 1000,
        });
        let result = convert_to_llm(&[msg]);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_create_messages() {
        let branch = create_branch_summary_message("test".into(), "id1".into(), 1000);
        assert_eq!(branch.role, "branchSummary");

        let comp = create_compaction_summary_message("test".into(), 5000, 1000);
        assert_eq!(comp.role, "compactionSummary");
        assert_eq!(comp.tokens_before, 5000);

        let custom = create_custom_message::<serde_json::Value>(
            "test_type".into(),
            serde_json::json!("content"),
            true,
            None,
            1000,
        );
        assert_eq!(custom.role, "custom");
        assert_eq!(custom.custom_type, "test_type");
    }
}
