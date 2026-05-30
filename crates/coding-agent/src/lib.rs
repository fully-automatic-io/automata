pub mod core;
pub mod extensions;
pub mod models;
pub mod settings;
pub mod stream_bridge;
pub mod tools;

// End-to-end session entry point.
pub use core::session::{build_tools, CodingAgentSession, SessionOptions, DEFAULT_TOOL_NAMES};
pub use core::provider::{build_provider, Auth, ProviderBuild};

// Custom message types and conversion now live in agent-core.
pub use agent_core::harness::messages::{
    bash_execution_to_text, convert_to_llm,
    BashExecutionMessage, BranchSummaryMessage, CompactionSummaryMessage,
    BRANCH_SUMMARY_PREFIX, BRANCH_SUMMARY_SUFFIX,
    COMPACTION_SUMMARY_PREFIX, COMPACTION_SUMMARY_SUFFIX,
};
pub use agent_core::types::Model;
pub use models::ModelRegistry;
pub use settings::{Settings, SettingsManager};
pub use tools::{
    BashTool, BashToolOptions, BashOperations, LocalBashOperations, BashToolDetails, BashExecOptions,
    EditTool, EditToolOptions, EditOperations, LocalEditOperations, EditToolDetails, Edit,
    ReadTool, ReadToolOptions, ReadOperations, LocalReadOperations, ReadToolDetails, ImageDimensions,
    WriteTool, WriteToolOptions, WriteOperations, LocalWriteOperations, WriteToolDetails,
    GrepTool, GrepToolDetails, FindTool, FindToolDetails, LsTool, LsToolDetails,
    apply_edits_to_normalized_content, detect_line_ending, generate_diff_string,
    generate_unified_patch, normalize_to_lf, restore_line_endings, strip_bom,
    truncate_tail, truncate_head, DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES,
};
pub use extensions::{
    Extension, ExtensionContext, ExtensionEvent, ExtensionManifest, ExtensionRunner, ExtensionService,
    SessionLifecycleReason,
    LoadExtensionsResult, RegisteredTool, ToolDefinition,
};
