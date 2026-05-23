pub mod agent_harness;
pub mod compaction;
pub mod env;
pub mod messages;
pub mod prompt_templates;
pub mod session;
pub mod skills;
pub mod utils;

pub use agent_harness::{AgentHarness, HarnessConfig, HarnessError, HarnessPhase};
pub use compaction::{
    compact, estimate_context_tokens, estimate_tokens, prepare_compaction,
    CompactionError, CompactionPreparation, CompactionResult, CompactionSettings, StreamFn,
    collect_entries_for_branch_summary, generate_branch_summary, BranchSummaryResult,
    FileOperations, create_file_ops, extract_file_ops_from_message,
    compute_file_lists, format_file_operations, serialize_conversation,
};
pub use env::{EnvError, ExecResult, FileInfo, FileKind, NativeEnv};
pub use messages::{
    bash_execution_to_text, convert_to_llm, BashExecutionMessage, BranchSummaryMessage,
    CompactionSummaryMessage, BRANCH_SUMMARY_PREFIX, BRANCH_SUMMARY_SUFFIX,
    COMPACTION_SUMMARY_PREFIX, COMPACTION_SUMMARY_SUFFIX,
};
pub use prompt_templates::{
    load_prompt_templates_from_dir, parse_command_args, substitute_args, PromptTemplate,
};
pub use session::{
    build_session_context, now_iso, uuidv7, BranchSummaryOptions, InMemorySessionRepo,
    InMemorySessionStorage, JsonlSessionMetadata, JsonlSessionRepo, JsonlSessionStorage,
    Session, SessionContext, SessionError, SessionMetadata, SessionStorage, SessionTreeEntry,
};
pub use skills::{
    format_skill_invocation, format_skills_for_system_prompt, load_skills_from_dir, Skill,
};
pub use utils::{
    execute_shell_with_capture, sanitize_binary_output, ShellCaptureResult,
    truncate_head, truncate_line, truncate_tail, TruncationResult, DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES, GREP_MAX_LINE_LENGTH,
};
