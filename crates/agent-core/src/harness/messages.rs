// Custom message types and conversion utilities used by the
// harness, compaction, and session-storage layers.
//
// The typed enum form lives on `AgentMessage` (User / Assistant / ToolResult /
// Custom / BashExecution / BranchSummary / CompactionSummary). These helpers
// fold the custom roles (BashExecution / Custom / BranchSummary /
// CompactionSummary) down to plain User content so a provider can consume
// the result.

use crate::types::{AgentMessage, ContentBlock, MessageContent};
use serde::{Deserialize, Serialize};

pub const COMPACTION_SUMMARY_PREFIX: &str =
    "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
pub const COMPACTION_SUMMARY_SUFFIX: &str = "\n</summary>";
pub const BRANCH_SUMMARY_PREFIX: &str =
    "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n";
pub const BRANCH_SUMMARY_SUFFIX: &str = "</summary>";

// ============================================================================
// Backwards-compatible plain-data wrappers (used by external callers / tests)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BashExecutionMessage {
    pub role: String,
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
        bash_execution_to_text(
            &self.command,
            &self.output,
            self.exit_code,
            self.cancelled,
            self.truncated,
            self.full_output_path.as_deref(),
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchSummaryMessage {
    pub role: String,
    pub summary: String,
    #[serde(rename = "fromId")]
    pub from_id: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionSummaryMessage {
    pub role: String,
    pub summary: String,
    #[serde(rename = "tokensBefore")]
    pub tokens_before: u64,
    pub timestamp: u64,
}

// ============================================================================
// Custom-role collapse helpers
// ============================================================================

pub fn bash_execution_to_text(
    command: &str,
    output: &str,
    exit_code: Option<i32>,
    cancelled: bool,
    truncated: bool,
    full_output_path: Option<&str>,
) -> String {
    let mut text = format!("Ran `{}`\n", command);
    if !output.is_empty() {
        text.push_str(&format!("```\n{}\n```", output));
    } else {
        text.push_str("(no output)");
    }
    if cancelled {
        text.push_str("\n\n(command cancelled)");
    } else if let Some(code) = exit_code
        && code != 0 {
            text.push_str(&format!("\n\nCommand exited with code {}", code));
        }
    if truncated
        && let Some(path) = full_output_path {
            text.push_str(&format!("\n\n[Output truncated. Full output: {}]", path));
        }
    text
}

/// Collapse custom roles (BashExecution / Custom / BranchSummary /
/// CompactionSummary) into plain user/assistant/toolResult `AgentMessage`s
/// suitable for an LLM provider.
pub fn convert_to_llm(messages: &[AgentMessage]) -> Vec<AgentMessage> {
    messages.iter().filter_map(|m| match m {
        AgentMessage::User { .. } | AgentMessage::Assistant { .. } | AgentMessage::ToolResult { .. } => {
            Some(m.clone())
        }
        AgentMessage::BashExecution {
            command, output, exit_code, cancelled, truncated, full_output_path,
            timestamp, exclude_from_context,
        } => {
            if *exclude_from_context { return None; }
            let text = bash_execution_to_text(
                command, output, *exit_code, *cancelled, *truncated, full_output_path.as_deref(),
            );
            Some(AgentMessage::User {
                content: MessageContent::Blocks(vec![ContentBlock::Text { text }]),
                timestamp: *timestamp,
                metadata: None,
            })
        }
        AgentMessage::Custom { content, timestamp, .. } => {
            let blocks = if let Some(s) = content.as_str() {
                vec![ContentBlock::Text { text: s.to_string() }]
            } else if let Ok(parsed) = serde_json::from_value::<Vec<ContentBlock>>(content.clone()) {
                parsed
            } else {
                // fallback: stringify
                vec![ContentBlock::Text { text: content.to_string() }]
            };
            Some(AgentMessage::User {
                content: MessageContent::Blocks(blocks),
                timestamp: *timestamp,
                metadata: None,
            })
        }
        AgentMessage::BranchSummary { summary, timestamp, .. } => {
            let text = format!("{}{}{}", BRANCH_SUMMARY_PREFIX, summary, BRANCH_SUMMARY_SUFFIX);
            Some(AgentMessage::User {
                content: MessageContent::Blocks(vec![ContentBlock::Text { text }]),
                timestamp: *timestamp,
                metadata: None,
            })
        }
        AgentMessage::CompactionSummary { summary, timestamp, .. } => {
            let text = format!("{}{}{}", COMPACTION_SUMMARY_PREFIX, summary, COMPACTION_SUMMARY_SUFFIX);
            Some(AgentMessage::User {
                content: MessageContent::Blocks(vec![ContentBlock::Text { text }]),
                timestamp: *timestamp,
                metadata: None,
            })
        }
    }).collect()
}

/// Default `convert_to_llm` callback for `AgentLoopConfig`.
pub async fn default_convert_to_llm(messages: Vec<AgentMessage>) -> Vec<AgentMessage> {
    convert_to_llm(&messages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_default_convert_to_llm_drops_custom_roles() {
        let messages = vec![
            AgentMessage::user_text("hello"),
            AgentMessage::assistant_text("hi"),
            AgentMessage::Custom {
                custom_type: "artifact".into(),
                content: serde_json::json!("payload"),
                display: false,
                details: None,
                timestamp: 1,
            },
        ];
        let result = default_convert_to_llm(messages).await;
        assert_eq!(result.len(), 3); // custom collapses to user — preserved
        assert_eq!(result[2].role(), "user");
    }

    #[test]
    fn test_convert_to_llm_excludes_excluded_bash() {
        let messages = vec![AgentMessage::BashExecution {
            command: "ls".into(),
            output: String::new(),
            exit_code: Some(0),
            cancelled: false,
            truncated: false,
            full_output_path: None,
            timestamp: 0,
            exclude_from_context: true,
        }];
        assert!(convert_to_llm(&messages).is_empty());
    }

    #[test]
    fn test_convert_to_llm_compaction_summary() {
        let messages = vec![AgentMessage::CompactionSummary {
            summary: "prior work".into(),
            tokens_before: 1000,
            timestamp: 0,
        }];
        let llm = convert_to_llm(&messages);
        assert_eq!(llm.len(), 1);
        match &llm[0] {
            AgentMessage::User { content: MessageContent::Blocks(b), .. } => {
                if let ContentBlock::Text { text } = &b[0] {
                    assert!(text.contains("prior work"));
                    assert!(text.contains("<summary>"));
                } else { panic!("expected text block"); }
            }
            _ => panic!("expected user message"),
        }
    }
}
