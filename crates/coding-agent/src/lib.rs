pub mod core;
pub mod extensions;
pub mod models;
pub mod settings;
pub mod stream_bridge;
pub mod tools;

// End-to-end session entry point.
pub use core::provider::{Auth, ProviderBuild, build_provider};
pub use core::runtime::{AgentSessionRuntime, AgentSessionRuntimeOptions};
pub use core::sdk::{
    AgentSessionHandle, CreateAgentSessionFromServicesOptions, CreateAgentSessionOptions,
    create_agent_session, create_agent_session_from_services,
};
pub use core::services::{
    AgentSessionServices, AgentSessionServicesOptions, SessionDiagnostic, SessionDiagnosticKind,
    create_agent_session_services,
};
pub use core::session::{
    ALL_BUILTIN_TOOLS, BuildToolsOptions, BuiltTools, BuiltinTool, CodingAgentSession,
    DEFAULT_ACTIVE_TOOLS, READ_ONLY_TOOLS, SessionBuildError, SessionOptions, ToolName, ToolPreset,
    ToolSelection, ToolSelectionError, build_tools, build_tools_from_names,
    build_tools_with_options,
};
pub use core::session_cwd::{
    SessionCwdIssue, assert_session_cwd_exists, format_missing_session_cwd_error,
    format_missing_session_cwd_prompt, missing_session_cwd_issue,
};
pub use core::session_manager::{
    ForkPosition, ManagedSession, ManagedSessionMetadata, SessionManager,
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
