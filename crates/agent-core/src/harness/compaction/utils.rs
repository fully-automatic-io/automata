use crate::types::{AgentMessage, ContentBlock, MessageContent};
use serde_json::Value;

pub const TOOL_RESULT_MAX_CHARS: usize = 2000;

#[derive(Debug, Clone, Default)]
pub struct FileOperations {
    pub read: std::collections::HashSet<String>,
    pub written: std::collections::HashSet<String>,
    pub edited: std::collections::HashSet<String>,
}

pub fn create_file_ops() -> FileOperations {
    FileOperations::default()
}

pub fn extract_file_ops_from_message(message: &AgentMessage, file_ops: &mut FileOperations) {
    let AgentMessage::Assistant { content, .. } = message else {
        return;
    };
    for block in content {
        let ContentBlock::ToolCall { name, arguments, .. } = block else {
            continue;
        };
        let Some(path) = arguments.get("path").and_then(|p| p.as_str()) else {
            continue;
        };
        match name.as_str() {
            "read" => {
                file_ops.read.insert(path.to_string());
            }
            "write" => {
                file_ops.written.insert(path.to_string());
            }
            "edit" => {
                file_ops.edited.insert(path.to_string());
            }
            _ => {}
        }
    }
}

pub fn compute_file_lists(file_ops: &FileOperations) -> (Vec<String>, Vec<String>) {
    let modified: std::collections::HashSet<_> =
        file_ops.edited.iter().chain(file_ops.written.iter()).collect();
    let mut read_files: Vec<String> =
        file_ops.read.iter().filter(|f| !modified.contains(*f)).cloned().collect();
    let mut modified_files: Vec<String> = modified.into_iter().cloned().collect();
    read_files.sort();
    modified_files.sort();
    (read_files, modified_files)
}

pub fn format_file_operations(read_files: &[String], modified_files: &[String]) -> String {
    let mut sections = vec![];
    if !read_files.is_empty() {
        sections.push(format!("<read-files>\n{}\n</read-files>", read_files.join("\n")));
    }
    if !modified_files.is_empty() {
        sections
            .push(format!("<modified-files>\n{}\n</modified-files>", modified_files.join("\n")));
    }
    if sections.is_empty() {
        return String::new();
    }
    format!("\n\n{}", sections.join("\n\n"))
}

fn truncate_for_summary(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }
    let truncated_chars = text.len() - max_chars;
    format!("{}\n\n[... {} more characters truncated]", &text[..max_chars], truncated_chars)
}

fn safe_json_stringify(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "[unserializable]".to_string())
}

pub fn serialize_conversation(messages: &[AgentMessage]) -> String {
    let mut parts = vec![];
    for msg in messages {
        match msg {
            AgentMessage::User { content, .. } => {
                let text = match content {
                    MessageContent::String(s) => s.clone(),
                    MessageContent::Blocks(blocks) => blocks
                        .iter()
                        .filter_map(|b| {
                            if let ContentBlock::Text { text } = b {
                                Some(text.as_str())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(""),
                };
                if !text.is_empty() {
                    parts.push(format!("[User]: {}", text));
                }
            }
            AgentMessage::Assistant { content, .. } => {
                let mut text_parts = vec![];
                let mut thinking_parts = vec![];
                let mut tool_calls = vec![];
                for block in content {
                    match block {
                        ContentBlock::Text { text } => text_parts.push(text.clone()),
                        ContentBlock::Thinking { thinking } => {
                            thinking_parts.push(thinking.clone())
                        }
                        ContentBlock::ToolCall { name, arguments, .. } => {
                            let args_str = if let Some(obj) = arguments.as_object() {
                                obj.iter()
                                    .map(|(k, v)| format!("{}={}", k, safe_json_stringify(v)))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            } else {
                                safe_json_stringify(arguments)
                            };
                            tool_calls.push(format!("{}({})", name, args_str));
                        }
                        _ => {}
                    }
                }
                if !thinking_parts.is_empty() {
                    parts.push(format!("[Assistant thinking]: {}", thinking_parts.join("\n")));
                }
                if !text_parts.is_empty() {
                    parts.push(format!("[Assistant]: {}", text_parts.join("\n")));
                }
                if !tool_calls.is_empty() {
                    parts.push(format!("[Assistant tool calls]: {}", tool_calls.join("; ")));
                }
            }
            AgentMessage::ToolResult { content, .. } => {
                let text: String = content
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::Text { text } = b {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("");
                if !text.is_empty() {
                    parts.push(format!(
                        "[Tool result]: {}",
                        truncate_for_summary(&text, TOOL_RESULT_MAX_CHARS)
                    ));
                }
            }
            _ => {}
        }
    }
    parts.join("\n\n")
}
