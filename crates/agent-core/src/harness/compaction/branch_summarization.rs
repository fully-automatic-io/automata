use crate::harness::session::types::SessionTreeEntry;
use crate::types::AgentMessage;
use super::compaction::{estimate_tokens, StreamFn, CompactionError, SUMMARIZATION_SYSTEM_PROMPT};
use super::utils::{
    FileOperations, create_file_ops, extract_file_ops_from_message,
    compute_file_lists, format_file_operations, serialize_conversation,
};
use crate::harness::messages::convert_to_llm;

const BRANCH_SUMMARY_PREAMBLE: &str = "The user explored a different conversation branch before returning here.\nSummary of that exploration:\n\n";

const BRANCH_SUMMARY_PROMPT: &str = "Create a structured summary of this conversation branch for context when returning later.\n\nUse this EXACT format:\n\n## Goal\n[What was the user trying to accomplish in this branch?]\n\n## Constraints & Preferences\n- [Any constraints, preferences, or requirements mentioned]\n- [Or \"(none)\" if none were mentioned]\n\n## Progress\n### Done\n- [x] [Completed tasks/changes]\n\n### In Progress\n- [ ] [Work that was started but not finished]\n\n### Blocked\n- [Issues preventing progress, if any]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale]\n\n## Next Steps\n1. [What should happen next to continue this work]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

#[derive(Debug, Clone)]
pub struct BranchSummaryResult {
    pub summary: String,
    pub read_files: Vec<String>,
    pub modified_files: Vec<String>,
}

fn parse_ts(s: &str) -> u64 {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.timestamp_millis().max(0) as u64)
        .unwrap_or(0)
}

fn get_message_from_entry(entry: &SessionTreeEntry) -> Option<AgentMessage> {
    match entry {
        SessionTreeEntry::Message { message, .. } => {
            if matches!(message, AgentMessage::ToolResult { .. }) { None } else { Some(message.clone()) }
        }
        SessionTreeEntry::CustomMessage { custom_type, content, display, details, timestamp, .. } => {
            Some(AgentMessage::Custom {
                custom_type: custom_type.clone(),
                content: content.clone(),
                display: *display,
                details: details.clone(),
                timestamp: parse_ts(timestamp),
            })
        }
        SessionTreeEntry::BranchSummary { summary, from_id, timestamp, .. } => {
            Some(AgentMessage::BranchSummary {
                summary: summary.clone(),
                from_id: from_id.clone(),
                timestamp: parse_ts(timestamp),
            })
        }
        SessionTreeEntry::Compaction { summary, tokens_before, timestamp, .. } => {
            Some(AgentMessage::CompactionSummary {
                summary: summary.clone(),
                tokens_before: *tokens_before,
                timestamp: parse_ts(timestamp),
            })
        }
        _ => None,
    }
}

pub fn prepare_branch_entries(entries: &[SessionTreeEntry], token_budget: usize) -> (Vec<AgentMessage>, FileOperations) {
    let mut file_ops = create_file_ops();

    // Collect file ops from branch_summary details first
    for entry in entries {
        if let SessionTreeEntry::BranchSummary { details: Some(details), from_hook, .. } = entry
            && from_hook != &Some(true) {
                if let Some(read) = details.get("readFiles").and_then(|r| r.as_array()) {
                    for f in read { if let Some(s) = f.as_str() { file_ops.read.insert(s.to_string()); } }
                }
                if let Some(modified) = details.get("modifiedFiles").and_then(|r| r.as_array()) {
                    for f in modified { if let Some(s) = f.as_str() { file_ops.edited.insert(s.to_string()); } }
                }
            }
    }

    let mut messages: Vec<AgentMessage> = vec![];
    let mut total_tokens = 0usize;

    for entry in entries.iter().rev() {
        let Some(msg) = get_message_from_entry(entry) else { continue };
        extract_file_ops_from_message(&msg, &mut file_ops);
        let tokens = estimate_tokens(&msg);

        if token_budget > 0 && total_tokens + tokens > token_budget {
            if matches!(entry, SessionTreeEntry::Compaction { .. } | SessionTreeEntry::BranchSummary { .. })
                && total_tokens < (token_budget as f64 * 0.9) as usize {
                    messages.insert(0, msg);
                }
            break;
        }
        messages.insert(0, msg);
        total_tokens += tokens;
    }

    (messages, file_ops)
}

pub async fn generate_branch_summary(
    entries: &[SessionTreeEntry],
    stream_fn: &StreamFn,
    custom_instructions: Option<&str>,
    replace_instructions: bool,
    reserve_tokens: usize,
    context_window: usize,
) -> Result<BranchSummaryResult, CompactionError> {
    let token_budget = context_window.saturating_sub(reserve_tokens);
    let (messages, file_ops) = prepare_branch_entries(entries, token_budget);

    if messages.is_empty() {
        return Ok(BranchSummaryResult {
            summary: "No content to summarize".to_string(),
            read_files: vec![],
            modified_files: vec![],
        });
    }

    let llm_messages = convert_to_llm(&messages);
    let conversation_text = serialize_conversation(&llm_messages);

    let instructions = match custom_instructions {
        Some(ci) if replace_instructions => ci.to_string(),
        Some(ci) => format!("{}\n\nAdditional focus: {}", BRANCH_SUMMARY_PROMPT, ci),
        None => BRANCH_SUMMARY_PROMPT.to_string(),
    };

    let prompt_text = format!("<conversation>\n{}\n</conversation>\n\n{}", conversation_text, instructions);
    let summarization_messages = vec![AgentMessage::user_text(prompt_text)];

    let text = stream_fn(summarization_messages, SUMMARIZATION_SYSTEM_PROMPT).await?;
    let (read_files, modified_files) = compute_file_lists(&file_ops);
    let summary = format!("{}{}{}", BRANCH_SUMMARY_PREAMBLE, text, format_file_operations(&read_files, &modified_files));

    Ok(BranchSummaryResult { summary, read_files, modified_files })
}

/// Collect entries that should be summarized before navigating to a different leaf.
pub async fn collect_entries_for_branch_summary(
    session: &crate::harness::session::types::Session,
    old_leaf_id: Option<&str>,
    target_id: &str,
) -> Result<(Vec<SessionTreeEntry>, Option<String>), crate::harness::session::types::SessionError> {
    let Some(old_leaf) = old_leaf_id else {
        return Ok((vec![], None));
    };

    let old_path: std::collections::HashSet<String> = session
        .get_branch_from(old_leaf).await?
        .iter().map(|e| e.id().to_string()).collect();

    let target_path = session.get_branch_from(target_id).await?;
    let mut common_ancestor_id: Option<String> = None;
    for entry in target_path.iter().rev() {
        if old_path.contains(entry.id()) {
            common_ancestor_id = Some(entry.id().to_string());
            break;
        }
    }

    let mut entries = vec![];
    let mut current: Option<String> = Some(old_leaf.to_string());
    while let Some(ref id) = current.clone() {
        if Some(id.as_str()) == common_ancestor_id.as_deref() { break; }
        let entry = session.storage().get_entry(id).await
            .ok_or_else(|| crate::harness::session::types::SessionError::InvalidSession(format!("Entry {} not found", id)))?;
        let parent = entry.parent_id().map(|s| s.to_string());
        entries.push(entry);
        current = parent;
    }
    entries.reverse();

    Ok((entries, common_ancestor_id))
}
