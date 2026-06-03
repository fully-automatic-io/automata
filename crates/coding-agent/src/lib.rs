pub mod core;
pub mod extensions;
pub mod models;
pub mod settings;
pub mod stream_bridge;
pub mod tools;

// End-to-end session entry point.
pub use core::provider::{Auth, ProviderBuild, build_provider};
pub use core::sdk::{AgentSessionHandle, CreateAgentSessionOptions, create_agent_session};
pub use core::session::{
    BuildToolsOptions, CodingAgentSession, DEFAULT_TOOL_NAMES, SessionOptions, build_tools,
    build_tools_with_options,
};

// Custom message types and conversion now live in agent-core.
pub use agent_core::harness::messages::{
    BRANCH_SUMMARY_PREFIX, BRANCH_SUMMARY_SUFFIX, BashExecutionMessage, BranchSummaryMessage,
    COMPACTION_SUMMARY_PREFIX, COMPACTION_SUMMARY_SUFFIX, CompactionSummaryMessage,
    bash_execution_to_text, convert_to_llm,
};
pub use agent_core::types::Model;
pub use core::resource::{
    DefaultResourceLoader, ResourceDiagnostic, ResourceDiagnosticKind, ResourceLoaderOptions,
    ResourceSet, context_files_for_prompt, load_project_context_files,
};
pub use extensions::{
    Extension, ExtensionContext, ExtensionEvent, ExtensionManifest, ExtensionRunner,
    ExtensionService, LoadExtensionsResult, RegisteredTool, SessionLifecycleReason, ToolDefinition,
};
pub use models::ModelRegistry;
pub use settings::{Settings, SettingsManager};
pub use tools::{
    BashExecOptions, BashOperations, BashTool, BashToolDetails, BashToolOptions, DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES, Edit, EditOperations, EditTool, EditToolDetails, EditToolOptions, FindTool,
    FindToolDetails, GrepTool, GrepToolDetails, ImageDimensions, LocalBashOperations,
    LocalEditOperations, LocalReadOperations, LocalWriteOperations, LsTool, LsToolDetails,
    ReadOperations, ReadTool, ReadToolDetails, ReadToolOptions, WriteOperations, WriteTool,
    WriteToolDetails, WriteToolOptions, apply_edits_to_normalized_content, detect_line_ending,
    generate_diff_string, generate_unified_patch, normalize_to_lf, restore_line_endings, strip_bom,
    truncate_head, truncate_tail,
};
