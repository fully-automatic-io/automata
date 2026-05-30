use crate::harness::session::types::{SessionTreeEntry, build_session_context};
use crate::types::{AgentMessage, ContentBlock, MessageContent, StopReason};
use super::utils::{
    FileOperations, create_file_ops, extract_file_ops_from_message,
    compute_file_lists, format_file_operations, serialize_conversation,
};

pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "You are a context summarization assistant. Your task is to read a conversation between a user and an AI coding assistant, then produce a structured summary following the exact format specified.\n\nDo NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.";

const SUMMARIZATION_PROMPT: &str = "The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.\n\nUse this EXACT format:\n\n## Goal\n[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]\n\n## Constraints & Preferences\n- [Any constraints, preferences, or requirements mentioned by user]\n- [Or \"(none)\" if none were mentioned]\n\n## Progress\n### Done\n- [x] [Completed tasks/changes]\n\n### In Progress\n- [ ] [Current work]\n\n### Blocked\n- [Issues preventing progress, if any]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale]\n\n## Next Steps\n1. [Ordered list of what should happen next]\n\n## Critical Context\n- [Any data, examples, or references needed to continue]\n- [Or \"(none)\" if not applicable]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

const UPDATE_SUMMARIZATION_PROMPT: &str = "The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.\n\nUpdate the existing structured summary with new information. RULES:\n- PRESERVE all existing information from the previous summary\n- ADD new progress, decisions, and context from the new messages\n- UPDATE the Progress section: move items from \"In Progress\" to \"Done\" when completed\n- UPDATE \"Next Steps\" based on what was accomplished\n- PRESERVE exact file paths, function names, and error messages\n- If something is no longer relevant, you may remove it\n\nUse this EXACT format:\n\n## Goal\n[Preserve existing goals, add new ones if the task expanded]\n\n## Constraints & Preferences\n- [Preserve existing, add new ones discovered]\n\n## Progress\n### Done\n- [x] [Include previously done items AND newly completed items]\n\n### In Progress\n- [ ] [Current work - update based on progress]\n\n### Blocked\n- [Current blockers - remove if resolved]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale] (preserve all previous, add new)\n\n## Next Steps\n1. [Update based on current state]\n\n## Critical Context\n- [Preserve important context, add new if needed]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

const TURN_PREFIX_SUMMARIZATION_PROMPT: &str = "This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.\n\nSummarize the prefix to provide context for the retained suffix:\n\n## Original Request\n[What did the user ask for in this turn?]\n\n## Early Progress\n- [Key decisions and work done in the prefix]\n\n## Context for Suffix\n- [Information needed to understand the retained recent work]\n\nBe concise. Focus on what's needed to understand the kept suffix.";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct CompactionSettings {
    pub enabled: bool,
    pub reserve_tokens: usize,
    pub keep_recent_tokens: usize,
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self { enabled: true, reserve_tokens: 16384, keep_recent_tokens: 20000 }
    }
}

#[derive(Debug, Clone)]
pub struct CompactionPreparation {
    pub first_kept_entry_id: String,
    pub messages_to_summarize: Vec<AgentMessage>,
    pub turn_prefix_messages: Vec<AgentMessage>,
    pub is_split_turn: bool,
    pub tokens_before: usize,
    pub previous_summary: Option<String>,
    pub file_ops: FileOperations,
    pub settings: CompactionSettings,
}

#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub summary: String,
    pub first_kept_entry_id: String,
    pub tokens_before: usize,
    pub read_files: Vec<String>,
    pub modified_files: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum CompactionError {
    #[error("Aborted")]
    Aborted,
    #[error("Summarization failed: {0}")]
    SummarizationFailed(String),
    #[error("Invalid session: {0}")]
    InvalidSession(String),
}

pub fn estimate_tokens(message: &AgentMessage) -> usize {
    let chars = match message {
        AgentMessage::User { content, .. } => match content {
            MessageContent::String(s) => s.len(),
            MessageContent::Blocks(blocks) => blocks.iter()
                .filter_map(|b| if let ContentBlock::Text { text } = b { Some(text.len()) } else { None })
                .sum(),
        },
        AgentMessage::Assistant { content, .. } => content.iter().map(|b| match b {
            ContentBlock::Text { text } => text.len(),
            ContentBlock::Thinking { thinking } => thinking.len(),
            ContentBlock::ToolCall { name, arguments, .. } => {
                name.len() + serde_json::to_string(arguments).map(|s| s.len()).unwrap_or(0)
            }
            _ => 0,
        }).sum(),
        AgentMessage::ToolResult { content, .. } => content.iter().filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.len()),
            ContentBlock::Image { .. } => Some(4800),
            _ => None,
        }).sum(),
        AgentMessage::Custom { content, .. } => {
            // Walk text-typed blocks if it's an array; otherwise stringified length.
            if let Some(arr) = content.as_array() {
                arr.iter().filter_map(|b| {
                    match b.get("type").and_then(|t| t.as_str()) {
                        Some("text") => b.get("text").and_then(|t| t.as_str()).map(|s| s.len()),
                        Some("image") => Some(4800),
                        _ => None,
                    }
                }).sum()
            } else if let Some(s) = content.as_str() {
                s.len()
            } else {
                content.to_string().len()
            }
        }
        AgentMessage::BashExecution { command, output, .. } => command.len() + output.len(),
        AgentMessage::BranchSummary { summary, .. }
        | AgentMessage::CompactionSummary { summary, .. } => summary.len(),
    };
    chars.div_ceil(4)
}

pub fn estimate_context_tokens(messages: &[AgentMessage]) -> usize {
    // Use last assistant usage if available
    for msg in messages.iter().rev() {
        if let AgentMessage::Assistant { stop_reason, usage, .. } = msg
            && !matches!(stop_reason, StopReason::Aborted | StopReason::Error) {
                if usage.total_tokens > 0 { return usage.total_tokens as usize; }
                let total = usage.input + usage.output + usage.cache_read + usage.cache_write;
                return total as usize;
            }
    }
    messages.iter().map(estimate_tokens).sum()
}

/// Like [`estimate_context_tokens`] but also reports which message contributed
/// the usage data (or `None` if no successful assistant exists). Used by
/// `_checkCompaction` to detect stale post-compaction usage data.
pub struct ContextTokenEstimate {
    pub tokens: usize,
    pub last_usage_index: Option<usize>,
}

pub fn estimate_context_tokens_with_source(messages: &[AgentMessage]) -> ContextTokenEstimate {
    for (i, msg) in messages.iter().enumerate().rev() {
        if let AgentMessage::Assistant { stop_reason, usage, .. } = msg
            && !matches!(stop_reason, StopReason::Aborted | StopReason::Error) {
                let tokens = if usage.total_tokens > 0 {
                    usage.total_tokens as usize
                } else {
                    (usage.input + usage.output + usage.cache_read + usage.cache_write) as usize
                };
                return ContextTokenEstimate { tokens, last_usage_index: Some(i) };
            }
    }
    ContextTokenEstimate {
        tokens: messages.iter().map(estimate_tokens).sum(),
        last_usage_index: None,
    }
}

fn get_message_from_entry(entry: &SessionTreeEntry) -> Option<AgentMessage> {
    match entry {
        SessionTreeEntry::Message { message, .. } => Some(message.clone()),
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

fn parse_ts(s: &str) -> u64 {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.timestamp_millis().max(0) as u64)
        .unwrap_or(0)
}

fn get_message_from_entry_for_compaction(entry: &SessionTreeEntry) -> Option<AgentMessage> {
    if matches!(entry, SessionTreeEntry::Compaction { .. }) { return None; }
    get_message_from_entry(entry)
}

fn find_valid_cut_points(entries: &[SessionTreeEntry], start: usize, end: usize) -> Vec<usize> {
    let mut cut_points = vec![];
    for (i, entry) in entries.iter().enumerate().take(end).skip(start) {
        match entry {
            SessionTreeEntry::Message { message, .. }
                // Tool results aren't valid cut points (they're attached to a turn).
                if !matches!(message, AgentMessage::ToolResult { .. }) => {
                    cut_points.push(i);
                }
            SessionTreeEntry::BranchSummary { .. } | SessionTreeEntry::CustomMessage { .. } => {
                cut_points.push(i);
            }
            _ => {}
        }
    }
    cut_points
}

pub fn find_turn_start_index(entries: &[SessionTreeEntry], entry_index: usize, start_index: usize) -> Option<usize> {
    let mut i = entry_index as isize;
    while i >= start_index as isize {
        let entry = &entries[i as usize];
        match entry {
            SessionTreeEntry::BranchSummary { .. } | SessionTreeEntry::CustomMessage { .. } => return Some(i as usize),
            SessionTreeEntry::Message {
                message: AgentMessage::User { .. } | AgentMessage::BashExecution { .. },
                ..
            } => return Some(i as usize),
            _ => {}
        }
        i -= 1;
    }
    None
}

struct CutPoint {
    first_kept_entry_index: usize,
    turn_start_index: Option<usize>,
    is_split_turn: bool,
}

fn find_cut_point(entries: &[SessionTreeEntry], start: usize, end: usize, keep_recent_tokens: usize) -> CutPoint {
    let cut_points = find_valid_cut_points(entries, start, end);
    if cut_points.is_empty() {
        return CutPoint { first_kept_entry_index: start, turn_start_index: None, is_split_turn: false };
    }
    let mut accumulated = 0usize;
    let mut cut_index = cut_points[0];

    'outer: for i in (start..end).rev() {
        if let SessionTreeEntry::Message { message, .. } = &entries[i] {
            accumulated += estimate_tokens(message);
            if accumulated >= keep_recent_tokens {
                for &cp in &cut_points {
                    if cp >= i { cut_index = cp; break 'outer; }
                }
                break;
            }
        }
    }

    while cut_index > start {
        match &entries[cut_index - 1] {
            SessionTreeEntry::Compaction { .. } | SessionTreeEntry::Message { .. } => break,
            _ => cut_index -= 1,
        }
    }

    let cut_entry = &entries[cut_index];
    let is_user_message = matches!(cut_entry, SessionTreeEntry::Message { message, .. }
        if matches!(message, AgentMessage::User { .. }));

    let turn_start_index = if is_user_message {
        None
    } else {
        find_turn_start_index(entries, cut_index, start)
    };

    CutPoint {
        first_kept_entry_index: cut_index,
        is_split_turn: !is_user_message && turn_start_index.is_some(),
        turn_start_index,
    }
}

fn extract_file_operations(messages: &[AgentMessage], entries: &[SessionTreeEntry], prev_compaction_index: Option<usize>) -> FileOperations {
    let mut file_ops = create_file_ops();
    if let Some(idx) = prev_compaction_index
        && let SessionTreeEntry::Compaction { details: Some(details), from_hook, .. } = &entries[idx]
            && from_hook != &Some(true) {
                if let Some(read) = details.get("readFiles").and_then(|r| r.as_array()) {
                    for f in read { if let Some(s) = f.as_str() { file_ops.read.insert(s.to_string()); } }
                }
                if let Some(modified) = details.get("modifiedFiles").and_then(|r| r.as_array()) {
                    for f in modified { if let Some(s) = f.as_str() { file_ops.edited.insert(s.to_string()); } }
                }
            }
    for msg in messages {
        extract_file_ops_from_message(msg, &mut file_ops);
    }
    file_ops
}

pub fn prepare_compaction(path_entries: &[SessionTreeEntry], settings: &CompactionSettings) -> Result<Option<CompactionPreparation>, CompactionError> {
    if path_entries.is_empty() || matches!(path_entries.last(), Some(SessionTreeEntry::Compaction { .. })) {
        return Ok(None);
    }

    let mut prev_compaction_index: Option<usize> = None;
    for (i, e) in path_entries.iter().enumerate().rev() {
        if matches!(e, SessionTreeEntry::Compaction { .. }) { prev_compaction_index = Some(i); break; }
    }

    let mut previous_summary: Option<String> = None;
    let mut boundary_start = 0usize;
    if let Some(cidx) = prev_compaction_index
        && let SessionTreeEntry::Compaction { summary, first_kept_entry_id, .. } = &path_entries[cidx] {
            previous_summary = Some(summary.clone());
            let first_kept_pos = path_entries.iter().position(|e| e.id() == first_kept_entry_id);
            boundary_start = first_kept_pos.unwrap_or(cidx + 1);
        }

    let ctx = build_session_context(path_entries);
    let tokens_before = estimate_context_tokens(&ctx.messages);

    let cut = find_cut_point(path_entries, boundary_start, path_entries.len(), settings.keep_recent_tokens);
    let first_kept_entry = &path_entries[cut.first_kept_entry_index];
    let first_kept_entry_id = first_kept_entry.id().to_string();
    if first_kept_entry_id.is_empty() {
        return Err(CompactionError::InvalidSession("First kept entry has no UUID".to_string()));
    }

    let history_end = if cut.is_split_turn { cut.turn_start_index.unwrap_or(cut.first_kept_entry_index) } else { cut.first_kept_entry_index };
    let messages_to_summarize: Vec<AgentMessage> = path_entries[boundary_start..history_end]
        .iter()
        .filter_map(get_message_from_entry_for_compaction)
        .collect();

    let turn_prefix_messages: Vec<AgentMessage> = if cut.is_split_turn {
        let ts = cut.turn_start_index.unwrap_or(cut.first_kept_entry_index);
        path_entries[ts..cut.first_kept_entry_index]
            .iter()
            .filter_map(get_message_from_entry_for_compaction)
            .collect()
    } else {
        vec![]
    };

    let mut file_ops = extract_file_operations(&messages_to_summarize, path_entries, prev_compaction_index);
    for msg in &turn_prefix_messages {
        extract_file_ops_from_message(msg, &mut file_ops);
    }

    Ok(Some(CompactionPreparation {
        first_kept_entry_id,
        messages_to_summarize,
        turn_prefix_messages,
        is_split_turn: cut.is_split_turn,
        tokens_before,
        previous_summary,
        file_ops,
        settings: settings.clone(),
    }))
}

/// StreamFn: takes (messages, system_prompt) and returns a summary string.
pub type StreamFn = Box<
    dyn Fn(Vec<AgentMessage>, &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, CompactionError>> + Send>>
        + Send
        + Sync,
>;

pub async fn compact(
    preparation: &CompactionPreparation,
    stream_fn: &StreamFn,
) -> Result<CompactionResult, CompactionError> {
    let summary = if preparation.is_split_turn && !preparation.turn_prefix_messages.is_empty() {
        let (history_result, prefix_result) = tokio::join!(
            generate_summary(&preparation.messages_to_summarize, preparation.previous_summary.as_deref(), stream_fn),
            generate_turn_prefix_summary(&preparation.turn_prefix_messages, stream_fn),
        );
        let history = history_result?;
        let prefix = prefix_result?;
        format!("{}\n\n---\n\n**Turn Context (split turn):**\n\n{}", history, prefix)
    } else {
        generate_summary(&preparation.messages_to_summarize, preparation.previous_summary.as_deref(), stream_fn).await?
    };

    let (read_files, modified_files) = compute_file_lists(&preparation.file_ops);
    let summary = format!("{}{}", summary, format_file_operations(&read_files, &modified_files));

    Ok(CompactionResult {
        summary,
        first_kept_entry_id: preparation.first_kept_entry_id.clone(),
        tokens_before: preparation.tokens_before,
        read_files,
        modified_files,
    })
}

async fn generate_summary(messages: &[AgentMessage], previous_summary: Option<&str>, stream_fn: &StreamFn) -> Result<String, CompactionError> {
    if messages.is_empty() && previous_summary.is_none() {
        return Ok("No prior history.".to_string());
    }
    let llm_messages = crate::harness::messages::convert_to_llm(messages);
    let conversation_text = serialize_conversation(&llm_messages);
    let base_prompt = if previous_summary.is_some() { UPDATE_SUMMARIZATION_PROMPT } else { SUMMARIZATION_PROMPT };
    let mut prompt_text = format!("<conversation>\n{}\n</conversation>\n\n", conversation_text);
    if let Some(prev) = previous_summary {
        prompt_text.push_str(&format!("<previous-summary>\n{}\n</previous-summary>\n\n", prev));
    }
    prompt_text.push_str(base_prompt);

    let summarization_messages = vec![AgentMessage::user_text(prompt_text)];
    stream_fn(summarization_messages, SUMMARIZATION_SYSTEM_PROMPT).await
}

async fn generate_turn_prefix_summary(messages: &[AgentMessage], stream_fn: &StreamFn) -> Result<String, CompactionError> {
    let llm_messages = crate::harness::messages::convert_to_llm(messages);
    let conversation_text = serialize_conversation(&llm_messages);
    let prompt_text = format!("<conversation>\n{}\n</conversation>\n\n{}", conversation_text, TURN_PREFIX_SUMMARIZATION_PROMPT);
    let summarization_messages = vec![AgentMessage::user_text(prompt_text)];
    stream_fn(summarization_messages, SUMMARIZATION_SYSTEM_PROMPT).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Api, Usage};

    fn assistant_with_total(total: u64) -> AgentMessage {
        AgentMessage::Assistant {
            content: vec![],
            api: Api::Anthropic,
            provider: "p".into(),
            model: "m".into(),
            usage: Usage { total_tokens: total, ..Default::default() },
            stop_reason: StopReason::EndTurn,
            error_message: None,
            timestamp: 0,
        }
    }

    fn errored_assistant() -> AgentMessage {
        AgentMessage::Assistant {
            content: vec![],
            api: Api::Anthropic,
            provider: "p".into(),
            model: "m".into(),
            usage: Usage::default(),
            stop_reason: StopReason::Error,
            error_message: Some("oops".into()),
            timestamp: 0,
        }
    }

    #[test]
    fn estimate_with_source_tracks_index() {
        let messages = vec![
            AgentMessage::user_text("a"),     // 0
            assistant_with_total(1500),       // 1 — successful
            AgentMessage::user_text("b"),     // 2
            errored_assistant(),              // 3 — skipped (error)
        ];
        let est = estimate_context_tokens_with_source(&messages);
        assert_eq!(est.tokens, 1500);
        assert_eq!(est.last_usage_index, Some(1));
    }

    #[test]
    fn estimate_with_source_falls_back_when_no_assistant() {
        let messages = vec![AgentMessage::user_text("hello world")];
        let est = estimate_context_tokens_with_source(&messages);
        assert_eq!(est.last_usage_index, None);
        assert!(est.tokens > 0);  // Estimated from text length
    }

    #[test]
    fn estimate_with_source_skips_aborted_and_error() {
        let messages = vec![
            assistant_with_total(900),        // 0 — successful, picked
            AgentMessage::Assistant {         // 1 — Aborted, skipped
                content: vec![],
                api: Api::Anthropic, provider: "p".into(), model: "m".into(),
                usage: Usage { total_tokens: 9999, ..Default::default() },
                stop_reason: StopReason::Aborted,
                error_message: None, timestamp: 0,
            },
            errored_assistant(),              // 2 — Error, skipped
        ];
        let est = estimate_context_tokens_with_source(&messages);
        assert_eq!(est.tokens, 900);
        assert_eq!(est.last_usage_index, Some(0));
    }
}
