pub mod compaction;
pub mod extensions;
pub mod messages;
pub mod models;
pub mod resource_loader;
pub mod session;
pub mod settings;
pub mod stream_bridge;
pub mod tools;
pub mod utils;

pub use messages::{
    BashExecutionMessage, BranchSummaryMessage, CompactionSummaryMessage, CustomMessage,
    convert_to_llm, create_branch_summary_message, create_compaction_summary_message,
    create_custom_message, COMPACTION_SUMMARY_PREFIX, COMPACTION_SUMMARY_SUFFIX,
    BRANCH_SUMMARY_PREFIX, BRANCH_SUMMARY_SUFFIX,
};
pub use models::{Model, ModelRegistry};
pub use session::{AgentSession, SessionManager, SessionEntry, SessionContext, SessionInfo};
pub use settings::{Settings, SettingsManager};
pub use tools::{
    BashTool, BashToolOptions, BashOperations, LocalBashOperations, BashToolDetails, BashExecOptions,
    EditTool, EditToolOptions, EditOperations, LocalEditOperations, EditToolDetails, Edit,
    ReadTool, ReadToolOptions, ReadOperations, LocalReadOperations,
    WriteTool, WriteToolOptions, WriteOperations, LocalWriteOperations,
    GrepTool, FindTool, LsTool,
    apply_edits_to_normalized_content, detect_line_ending, generate_diff_string,
    normalize_to_lf, restore_line_endings, strip_bom,
    truncate_tail, truncate_head, DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES,
};
pub use extensions::{
    Extension, ExtensionContext, ExtensionEvent, ExtensionManifest, ExtensionRunner, ExtensionService,
    LoadExtensionsResult, RegisteredTool, ToolDefinition,
};
pub use compaction::{CompactionSettings, estimate_tokens, should_compact};
