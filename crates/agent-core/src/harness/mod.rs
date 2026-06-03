pub mod agent_harness;
pub mod compaction;
pub mod env;
pub mod messages;
pub mod prompt_templates;
pub mod session;
pub mod skills;
pub mod utils;

pub use agent_harness::{
    AbortResult, AgentHarness, AgentHarnessOptions, AutoCompactionConfig, CompactionReason,
    HarnessConfig, HarnessError, HarnessEvent, HarnessPhase, HarnessResources, PostRunDecision,
    StreamOptions, StreamOptionsPatch,
};
pub use compaction::{
    BranchSummaryResult, CompactionError, CompactionPreparation, CompactionResult,
    CompactionSettings, FileOperations, StreamFn, collect_entries_for_branch_summary, compact,
    compute_file_lists, create_file_ops, estimate_context_tokens, estimate_tokens,
    extract_file_ops_from_message, format_file_operations, generate_branch_summary,
    prepare_compaction, serialize_conversation,
};
pub use env::{EnvError, ExecResult, FileInfo, FileKind, NativeEnv};
pub use messages::{
    BRANCH_SUMMARY_PREFIX, BRANCH_SUMMARY_SUFFIX, BashExecutionMessage, BranchSummaryMessage,
    COMPACTION_SUMMARY_PREFIX, COMPACTION_SUMMARY_SUFFIX, CompactionSummaryMessage,
    bash_execution_to_text, convert_to_llm,
};
pub use prompt_templates::{
    PromptTemplate, load_prompt_templates_from_dir, parse_command_args, substitute_args,
};
pub use session::{
    BranchSummaryOptions, InMemorySessionRepo, InMemorySessionStorage, JsonlSessionMetadata,
    JsonlSessionRepo, JsonlSessionStorage, Session, SessionContext, SessionError, SessionMetadata,
    SessionStorage, SessionTreeEntry, build_session_context, now_iso, uuidv7,
};
pub use skills::{
    Skill, format_skill_invocation, format_skills_for_system_prompt, load_skills_from_dir,
};
pub use utils::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, GREP_MAX_LINE_LENGTH, ShellCaptureResult,
    TruncationResult, execute_shell_with_capture, sanitize_binary_output, truncate_head,
    truncate_line, truncate_tail,
};
