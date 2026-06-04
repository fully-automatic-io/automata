pub mod auth;
pub mod bus;
pub mod config;
pub mod exec;
pub mod models;
pub mod prompt;
pub mod provider;
pub mod resource;
pub mod runtime;
pub mod sdk;
pub mod services;
pub mod session;
pub mod session_manager;
pub mod slash;

pub use auth::AuthStorage;
pub use bus::EventBus;
pub use config::{resolve_config_value, resolve_config_value_opt};
pub use exec::{
    BashExecutorOptions, BashResult, ExecOptions, ExecResult, ShellConfig,
    cleanup_detached_processes, exec_command, execute_bash, get_shell_config, get_shell_env,
    kill_process_tree, track_detached_child_pid, untrack_detached_child_pid,
};
pub use models::{
    DEFAULT_AGENT_DIR, DEFAULT_COMPACTION_KEEP_RECENT_TOKENS, DEFAULT_COMPACTION_RESERVE_TOKENS,
    DEFAULT_SESSIONS_DIR, DEFAULT_THINKING_LEVEL, ScopedModel, clamp_thinking_level,
    default_model_for_provider,
};
pub use prompt::{
    BuildSystemPromptOptions, ContextFile, ContextFiles, LoadedContextFile, Skill,
    build_system_prompt, discover_extension_paths, load_context_files, load_skills,
};
pub use provider::{Auth, ProviderBuild, build_provider};
pub use resource::{
    DefaultResourceLoader, ResourceDiagnostic, ResourceDiagnosticKind, ResourceLoaderOptions,
    ResourceSet, context_files_for_prompt, load_project_context_files,
};
pub use runtime::{AgentSessionRuntime, AgentSessionRuntimeOptions};
pub use sdk::{
    AgentSessionHandle, CreateAgentSessionFromServicesOptions, CreateAgentSessionOptions,
    create_agent_session, create_agent_session_from_services,
};
pub use services::{
    AgentSessionServices, AgentSessionServicesOptions, SessionDiagnostic, SessionDiagnosticKind,
    create_agent_session_services,
};
pub use session::{
    BuildToolsOptions, CodingAgentSession, DEFAULT_TOOL_NAMES, SessionOptions, build_tools,
    build_tools_with_options,
};
pub use session_manager::{ForkPosition, ManagedSession, ManagedSessionMetadata, SessionManager};
pub use slash::{
    SlashCommandInfo, SlashCommandSource, SourceInfo, SourceOrigin, SourceScope,
    builtin_slash_commands,
};
