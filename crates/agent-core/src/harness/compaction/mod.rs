pub mod branch_summarization;
#[allow(clippy::module_inception)]
pub mod compaction;
pub mod utils;

pub use compaction::{
    compact, estimate_context_tokens, estimate_context_tokens_with_source, estimate_tokens,
    find_turn_start_index, prepare_compaction, CompactionError, CompactionPreparation,
    CompactionResult, CompactionSettings, ContextTokenEstimate, StreamFn,
    SUMMARIZATION_SYSTEM_PROMPT,
};
pub use branch_summarization::{
    collect_entries_for_branch_summary, generate_branch_summary, BranchSummaryResult,
};
pub use utils::{
    compute_file_lists, create_file_ops, extract_file_ops_from_message,
    format_file_operations, serialize_conversation, FileOperations,
};
