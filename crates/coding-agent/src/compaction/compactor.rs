
use agent_core::types::ContentBlock;
use crate::session::manager::SessionEntry;

// ============================================================================
// Token estimation
// ============================================================================

/// Estimate token count for text (chars / 4 heuristic, conservative).
pub fn estimate_tokens(text: &str) -> u64 {
    (text.len() as u64).div_ceil(4)
}

/// Estimate tokens from a ContentBlock.
pub fn estimate_block_tokens(block: &ContentBlock) -> u64 {
    match block {
        ContentBlock::Text { text } => estimate_tokens(text),
        ContentBlock::ToolCall { name, arguments, .. } => {
            estimate_tokens(name) + estimate_tokens(&arguments.to_string())
        }
        ContentBlock::ToolResult { content, .. } => {
            content.iter().map(estimate_block_tokens).sum()
        }
        ContentBlock::Thinking { thinking } => estimate_tokens(thinking),
        ContentBlock::Image { .. } => 1200, // ~4800 chars / 4
    }
}

/// Estimate tokens for a single session entry message.
pub fn estimate_entry_tokens(entry: &SessionEntry) -> u64 {
    match entry {
        SessionEntry::Message { message, .. } => {
            let content = message.get("content");
            if let Some(arr) = content.and_then(|c| c.as_array()) {
                arr.iter()
                    .filter_map(|b| serde_json::from_value::<ContentBlock>(b.clone()).ok())
                    .map(|b| estimate_block_tokens(&b))
                    .sum()
            } else if let Some(s) = content.and_then(|c| c.as_str()) {
                estimate_tokens(s)
            } else {
                estimate_tokens(&message.to_string())
            }
        }
        SessionEntry::Compaction { summary, .. } => estimate_tokens(summary),
        SessionEntry::BranchSummary { summary, .. } => estimate_tokens(summary),
        SessionEntry::CustomMessage { content, .. } => {
            estimate_tokens(&content.to_string())
        }
        _ => 0,
    }
}

/// Calculate total tokens for a slice of messages (JSON values).
pub fn calculate_context_tokens(messages: &[serde_json::Value]) -> u64 {
    messages.iter().map(|m| {
        let content = m.get("content");
        if let Some(arr) = content.and_then(|c| c.as_array()) {
            arr.iter()
                .filter_map(|b| serde_json::from_value::<ContentBlock>(b.clone()).ok())
                .map(|b| estimate_block_tokens(&b))
                .sum()
        } else if let Some(s) = content.and_then(|c| c.as_str()) {
            estimate_tokens(s)
        } else {
            estimate_tokens(&m.to_string())
        }
    }).sum()
}

// ============================================================================
// Compaction settings
// ============================================================================

#[derive(Debug, Clone)]
pub struct CompactionSettings {
    pub enabled: bool,
    /// Tokens to reserve for prompt + response
    pub reserve_tokens: u64,
    /// Keep this many recent tokens uncompacted
    pub keep_recent_tokens: u64,
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            reserve_tokens: 16384,
            keep_recent_tokens: 20000,
        }
    }
}

/// Check if compaction should trigger.
pub fn should_compact(
    context_tokens: u64,
    context_window: u64,
    settings: &CompactionSettings,
) -> bool {
    if !settings.enabled { return false; }
    context_tokens > context_window.saturating_sub(settings.reserve_tokens)
}

// ============================================================================
// Cut point detection
// ============================================================================

#[derive(Debug, Clone)]
pub struct CutPointResult {
    /// Index of the first entry to keep (not compact)
    pub first_kept_entry_index: usize,
    /// Index of the turn start if we're splitting a turn
    pub turn_start_index: Option<usize>,
    /// Whether we're splitting in the middle of a turn
    pub is_split_turn: bool,
}

/// Find valid cut points in entries (positions where we can safely cut).
/// Never cut at a toolResult — it must follow its toolCall.
fn find_valid_cut_points(entries: &[SessionEntry], start: usize, end: usize) -> Vec<usize> {
    let mut cut_points = Vec::new();
    for i in start..end {
        match &entries[i] {
            SessionEntry::Message { message, .. } => {
                let role = message.get("role").and_then(|r| r.as_str()).unwrap_or("");
                match role {
                    "toolResult" => {} // Never cut at tool results
                    _ => cut_points.push(i),
                }
            }
            SessionEntry::BranchSummary { .. } | SessionEntry::CustomMessage { .. } => {
                cut_points.push(i);
            }
            _ => {}
        }
    }
    cut_points
}

/// Find the start of the turn containing the entry at `idx`.
fn find_turn_start_index(entries: &[SessionEntry], idx: usize, start: usize) -> Option<usize> {
    // Walk backwards to find the user message that started this turn
    let mut i = idx;
    while i > start {
        i -= 1;
        if let SessionEntry::Message { message, .. } = &entries[i] {
            let role = message.get("role").and_then(|r| r.as_str()).unwrap_or("");
            if role == "user" {
                return Some(i);
            }
        }
    }
    None
}

/// Find the optimal cut point to keep `keep_recent_tokens` of recent context.
pub fn find_cut_point(
    entries: &[SessionEntry],
    start: usize,
    end: usize,
    keep_recent_tokens: u64,
) -> CutPointResult {
    let cut_points = find_valid_cut_points(entries, start, end);

    if cut_points.is_empty() {
        return CutPointResult {
            first_kept_entry_index: start,
            turn_start_index: None,
            is_split_turn: false,
        };
    }

    // Walk backwards accumulating tokens until we exceed the budget
    let mut accumulated = 0u64;
    let mut cut_index = cut_points[0]; // default: keep from first valid point

    for i in (start..end).rev() {
        let tokens = estimate_entry_tokens(&entries[i]);
        accumulated += tokens;

        if accumulated >= keep_recent_tokens {
            // Find the nearest valid cut point at or after i
            for &cp in &cut_points {
                if cp >= i {
                    cut_index = cp;
                    break;
                }
            }
            break;
        }
    }

    // Check if we're splitting a turn (cut point is not at a user message)
    let is_user_message = matches!(&entries[cut_index],
        SessionEntry::Message { message, .. }
        if message.get("role").and_then(|r| r.as_str()) == Some("user")
    );

    let (turn_start_index, is_split_turn) = if !is_user_message {
        let turn_start = find_turn_start_index(entries, cut_index, start);
        (turn_start, turn_start.is_some())
    } else {
        (None, false)
    };

    CutPointResult {
        first_kept_entry_index: cut_index,
        turn_start_index,
        is_split_turn,
    }
}

// ============================================================================
// Compaction preparation
// ============================================================================

#[derive(Debug)]
pub struct CompactionPreparation {
    pub first_kept_entry_id: String,
    pub messages_to_summarize: Vec<serde_json::Value>,
    pub turn_prefix_messages: Vec<serde_json::Value>,
    pub is_split_turn: bool,
    pub tokens_before: u64,
    pub previous_summary: Option<String>,
}

/// Prepare compaction: find cut point, collect messages to summarize.
pub fn prepare_compaction(
    entries: &[SessionEntry],
    settings: &CompactionSettings,
) -> Option<CompactionPreparation> {
    // Don't compact if last entry is already a compaction
    if matches!(entries.last(), Some(SessionEntry::Compaction { .. })) {
        return None;
    }

    // Find previous compaction boundary and previous summary
    let (boundary_start, previous_summary) = {
        let mut start = 0usize;
        let mut prev_summary: Option<String> = None;
        for (i, e) in entries.iter().enumerate().rev() {
            if let SessionEntry::Compaction { summary: s, first_kept_entry_id, .. } = e {
                prev_summary = Some(s.clone());
                start = entries.iter().position(|e| e.id() == first_kept_entry_id)
                    .unwrap_or(i + 1);
                break;
            }
        }
        (start, prev_summary)
    };

    let boundary_end = entries.len();

    // Estimate total tokens
    let tokens_before: u64 = entries[boundary_start..boundary_end]
        .iter()
        .map(estimate_entry_tokens)
        .sum();

    // Find cut point
    let cut = find_cut_point(entries, boundary_start, boundary_end, settings.keep_recent_tokens);

    let first_kept_entry = entries.get(cut.first_kept_entry_index)?;
    let first_kept_entry_id = first_kept_entry.id().to_string();

    let history_end = if cut.is_split_turn {
        cut.turn_start_index.unwrap_or(cut.first_kept_entry_index)
    } else {
        cut.first_kept_entry_index
    };

    // Collect messages to summarize
    let messages_to_summarize: Vec<serde_json::Value> = entries[boundary_start..history_end]
        .iter()
        .filter_map(entry_to_message)
        .collect();

    // Collect turn prefix messages if split turn
    let turn_prefix_messages: Vec<serde_json::Value> = if cut.is_split_turn {
        let turn_start = cut.turn_start_index.unwrap_or(cut.first_kept_entry_index);
        entries[turn_start..cut.first_kept_entry_index]
            .iter()
            .filter_map(entry_to_message)
            .collect()
    } else {
        vec![]
    };

    Some(CompactionPreparation {
        first_kept_entry_id,
        messages_to_summarize,
        turn_prefix_messages,
        is_split_turn: cut.is_split_turn,
        tokens_before,
        previous_summary,
    })
}

fn entry_to_message(entry: &SessionEntry) -> Option<serde_json::Value> {
    match entry {
        SessionEntry::Message { message, .. } => Some(message.clone()),
        SessionEntry::CustomMessage { content, display, .. } if *display => {
            Some(serde_json::json!({
                "role": "user",
                "content": content
            }))
        }
        SessionEntry::BranchSummary { summary, .. } if !summary.is_empty() => {
            Some(serde_json::json!({
                "role": "user",
                "content": [{"type": "text", "text": summary}]
            }))
        }
        _ => None,
    }
}

// ============================================================================
// Summary generation prompts
// ============================================================================

pub const SUMMARIZATION_SYSTEM_PROMPT: &str =
    "You are a context summarization assistant. Your task is to read a conversation \
     between a user and an AI coding assistant, then produce a structured summary \
     following the exact format specified.\n\n\
     Do NOT continue the conversation. Do NOT respond to any questions in the \
     conversation. ONLY output the structured summary.";

pub const SUMMARIZATION_PROMPT: &str = "The messages above are a conversation to summarize. \
Create a structured context checkpoint summary that another LLM will use to continue the work.\n\n\
Use this EXACT format:\n\n\
## Goal\n[What is the user trying to accomplish?]\n\n\
## Constraints & Preferences\n- [Any constraints or preferences]\n\n\
## Progress\n### Done\n- [x] [Completed tasks]\n\n### In Progress\n- [ ] [Current work]\n\n\
## Key Decisions\n- **[Decision]**: [Brief rationale]\n\n\
## Next Steps\n1. [Ordered list of what should happen next]\n\n\
## Critical Context\n- [Any data or references needed to continue]\n\n\
Keep each section concise. Preserve exact file paths, function names, and error messages.";

pub const UPDATE_SUMMARIZATION_PROMPT: &str = "The messages above are NEW conversation messages \
to incorporate into the existing summary provided in <previous-summary> tags.\n\n\
Update the existing structured summary with new information. RULES:\n\
- PRESERVE all existing information from the previous summary\n\
- ADD new progress, decisions, and context from the new messages\n\
- UPDATE the Progress section: move items from \"In Progress\" to \"Done\" when completed\n\
- UPDATE \"Next Steps\" based on what was accomplished\n\
- PRESERVE exact file paths, function names, and error messages";

/// Serialize conversation messages to text for summarization.
/// Prevents the model from treating it as a live conversation.
pub fn serialize_conversation(messages: &[serde_json::Value]) -> String {
    const TOOL_RESULT_MAX_CHARS: usize = 2000;

    messages.iter().map(|msg| {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("unknown");
        let content = msg.get("content");

        match role {
            "user" => {
                let text = extract_text_content(content);
                format!("[User]\n{}\n", text)
            }
            "assistant" => {
                let thinking = extract_thinking_content(content);
                let text = extract_text_content(content);
                let tool_calls = extract_tool_calls_text(content);

                let mut parts = Vec::new();
                if !thinking.is_empty() {
                    parts.push(format!("[Assistant thinking]\n{}", thinking));
                }
                if !text.is_empty() {
                    parts.push(format!("[Assistant]\n{}", text));
                }
                if !tool_calls.is_empty() {
                    parts.push(format!("[Assistant tool calls]\n{}", tool_calls));
                }
                parts.join("\n")
            }
            "toolResult" => {
                let tool_name = msg.get("toolName").and_then(|n| n.as_str()).unwrap_or("unknown");
                let text = extract_text_content(content);
                let truncated = if text.len() > TOOL_RESULT_MAX_CHARS {
                    format!("{}... (truncated)", &text[..TOOL_RESULT_MAX_CHARS])
                } else {
                    text
                };
                format!("[Tool result: {}]\n{}\n", tool_name, truncated)
            }
            _ => String::new(),
        }
    }).collect::<Vec<_>>().join("\n")
}

fn extract_text_content(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(arr)) => {
            arr.iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n")
        }
        _ => String::new(),
    }
}

fn extract_thinking_content(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::Array(arr)) => {
            arr.iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("thinking"))
                .filter_map(|b| b.get("thinking").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n")
        }
        _ => String::new(),
    }
}

fn extract_tool_calls_text(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::Array(arr)) => {
            arr.iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("toolCall"))
                .map(|b| {
                    let name = b.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
                    let args = b.get("arguments").map(|a| a.to_string()).unwrap_or_default();
                    format!("{}({})", name, args)
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
        _ => String::new(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("hello world"), 3); // 11/4 = 3 (ceil)
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
    }

    #[test]
    fn test_should_compact() {
        let settings = CompactionSettings::default();
        assert!(should_compact(190000, 200000, &settings));
        assert!(!should_compact(100000, 200000, &settings));
        assert!(!should_compact(190000, 200000, &CompactionSettings {
            enabled: false, ..Default::default()
        }));
    }

    #[test]
    fn test_find_valid_cut_points() {
        let entries = vec![
            make_msg("user", "u1"),
            make_msg("assistant", "a1"),
            make_msg("toolResult", "tr1"),
            make_msg("user", "u2"),
        ];
        let cuts = find_valid_cut_points(&entries, 0, entries.len());
        // toolResult should not be a cut point
        assert!(!cuts.contains(&2));
        assert!(cuts.contains(&0));
        assert!(cuts.contains(&1));
        assert!(cuts.contains(&3));
    }

    #[test]
    fn test_serialize_conversation() {
        let messages = vec![
            serde_json::json!({"role": "user", "content": "hello"}),
            serde_json::json!({"role": "assistant", "content": [{"type": "text", "text": "hi"}]}),
        ];
        let text = serialize_conversation(&messages);
        assert!(text.contains("[User]"));
        assert!(text.contains("[Assistant]"));
        assert!(text.contains("hello"));
        assert!(text.contains("hi"));
    }

    fn make_msg(role: &str, id: &str) -> SessionEntry {
        SessionEntry::Message {
            id: id.to_string(),
            parent_id: None,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            message: serde_json::json!({"role": role, "content": "test"}),
        }
    }
}
