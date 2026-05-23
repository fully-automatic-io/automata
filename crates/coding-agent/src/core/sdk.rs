use std::path::PathBuf;
use std::sync::Arc;

use crate::models::ModelRegistry;
use crate::settings::SettingsManager;
use crate::tools::{
    BashTool, BashToolOptions, EditTool, EditToolOptions, FindTool, GrepTool, LsTool, ReadTool,
    ReadToolOptions, WriteTool, WriteToolOptions,
};
use agent_core::harness::session::{JsonlSessionRepo, Session, SessionError};
use agent_core::tool::AgentTool;

use super::auth::AuthStorage;
use super::models::DEFAULT_AGENT_DIR;
use super::prompt::{build_system_prompt, load_context_files, BuildSystemPromptOptions, ContextFile};

pub struct CreateAgentSessionOptions {
    pub cwd: String,
    pub agent_dir: Option<String>,
    pub session_path: Option<String>,
    pub sessions_root: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub thinking_level: Option<String>,
    pub allowed_tools: Option<Vec<String>>,
}

pub struct AgentSessionHandle {
    pub session: Session,
    pub settings: SettingsManager,
    pub models: ModelRegistry,
    pub auth: AuthStorage,
    pub tools: Vec<Arc<dyn AgentTool>>,
    pub cwd: PathBuf,
    pub agent_dir: PathBuf,
}

impl AgentSessionHandle {
    pub fn api_key_for_model(&self, model_id: &str) -> Option<String> {
        let provider = self.models.get_model(model_id)
            .map(|m| m.provider.clone())
            .unwrap_or_default();
        self.auth.get_api_key(&provider)
    }

    pub fn system_prompt(&self) -> String {
        let ctx_files = load_context_files(&self.cwd);
        let context_files: Vec<ContextFile> = ctx_files.files.iter().map(|f| ContextFile {
            path: f.path.to_string_lossy().to_string(),
            content: f.content.clone(),
        }).collect();
        build_system_prompt(BuildSystemPromptOptions {
            custom_prompt: None,
            selected_tools: None,
            tool_snippets: None,
            prompt_guidelines: None,
            append_system_prompt: None,
            cwd: &self.cwd.to_string_lossy(),
            context_files: Some(&context_files),
            skills: None,
        })
    }
}

/// Create or open a coding-agent session, set up settings, models, auth, and tools.
///
/// `sessions_root` is where new sessions are created (defaults to `<agent_dir>/sessions`).
/// `session_path` opens a specific existing JSONL session file.
pub async fn create_agent_session(
    opts: CreateAgentSessionOptions,
) -> Result<AgentSessionHandle, SessionError> {
    let cwd = PathBuf::from(&opts.cwd);
    let agent_dir = opts.agent_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| dirs_next::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(DEFAULT_AGENT_DIR));

    let settings = SettingsManager::with_defaults(&cwd);
    let models = ModelRegistry::new();
    let auth = AuthStorage::load(&agent_dir);

    let sessions_root = opts.sessions_root
        .map(PathBuf::from)
        .unwrap_or_else(|| agent_dir.join("sessions"));
    let repo = JsonlSessionRepo::new(&sessions_root);

    let session = match opts.session_path {
        Some(ref path) => repo.open_by_path(path).await?,
        None => repo.create(&opts.cwd, None, None).await?,
    };

    let tool_names: Vec<&str> = opts.allowed_tools.as_deref()
        .map(|v| v.iter().map(|s| s.as_str()).collect())
        .unwrap_or_else(|| vec!["read", "bash", "edit", "write", "grep", "find", "ls"]);

    let mut tools: Vec<Arc<dyn AgentTool>> = vec![];
    for name in &tool_names {
        match *name {
            "read"  => tools.push(Arc::new(ReadTool::new(opts.cwd.clone(), ReadToolOptions::default()))),
            "bash"  => tools.push(Arc::new(BashTool::new(opts.cwd.clone(), BashToolOptions::default()))),
            "edit"  => tools.push(Arc::new(EditTool::new(opts.cwd.clone(), EditToolOptions::default()))),
            "write" => tools.push(Arc::new(WriteTool::new(opts.cwd.clone(), WriteToolOptions::default()))),
            "grep"  => tools.push(Arc::new(GrepTool::new(opts.cwd.clone()))),
            "find"  => tools.push(Arc::new(FindTool::new(opts.cwd.clone()))),
            "ls"    => tools.push(Arc::new(LsTool::new(opts.cwd.clone()))),
            _ => {}
        }
    }

    Ok(AgentSessionHandle { session, settings, models, auth, tools, cwd, agent_dir })
}
