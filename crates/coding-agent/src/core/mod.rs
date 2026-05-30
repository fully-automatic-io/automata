pub mod auth;
pub mod bus;
pub mod config;
pub mod exec;
pub mod models;
pub mod prompt;
pub mod provider;
pub mod sdk;
pub mod session;
pub mod slash;

pub use auth::AuthStorage;
pub use bus::EventBus;
pub use config::{resolve_config_value, resolve_config_value_opt};
pub use exec::{
    cleanup_detached_processes, execute_bash, exec_command, get_shell_config, get_shell_env,
    kill_process_tree, track_detached_child_pid, untrack_detached_child_pid,
    BashExecutorOptions, BashResult, ExecOptions, ExecResult, ShellConfig,
};
pub use models::{
    clamp_thinking_level, default_model_for_provider, ScopedModel,
    DEFAULT_AGENT_DIR, DEFAULT_COMPACTION_KEEP_RECENT_TOKENS, DEFAULT_COMPACTION_RESERVE_TOKENS,
    DEFAULT_SESSIONS_DIR, DEFAULT_THINKING_LEVEL,
};
pub use prompt::{
    build_system_prompt, discover_extension_paths, load_context_files, load_skills,
    BuildSystemPromptOptions, ContextFile, ContextFiles, LoadedContextFile, Skill,
};
pub use provider::{build_provider, Auth, ProviderBuild};
pub use sdk::{create_agent_session, AgentSessionHandle, CreateAgentSessionOptions};
pub use session::{
    build_tools, CodingAgentSession, SessionOptions, DEFAULT_TOOL_NAMES,
};
pub use slash::{
    builtin_slash_commands, SlashCommandInfo, SlashCommandSource, SourceInfo, SourceOrigin,
    SourceScope,
};
