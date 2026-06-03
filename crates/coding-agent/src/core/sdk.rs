use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::models::ModelRegistry;
use crate::settings::SettingsManager;
use agent_core::harness::session::{JsonlSessionRepo, Session, SessionError};
use agent_core::tool::AgentTool;
use agent_core::types::Model;

use super::auth::AuthStorage;
use super::config::resolve_config_value;
use super::models::{DEFAULT_AGENT_DIR, default_model_for_provider};
use super::prompt::{BuildSystemPromptOptions, build_system_prompt};
use super::resource::{DefaultResourceLoader, ResourceSet, context_files_for_prompt};
use super::session::{BuildToolsOptions, DEFAULT_TOOL_NAMES, build_tools_with_options};

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
    pub selected_model: Option<Model>,
    pub resources: ResourceSet,
    pub tools: Vec<Arc<dyn AgentTool>>,
    pub tool_names: Vec<String>,
    pub cwd: PathBuf,
    pub agent_dir: PathBuf,
}

impl AgentSessionHandle {
    pub fn api_key_for_model(&self, model_id: &str) -> Option<String> {
        let provider =
            self.models.get_model(model_id).map(|m| m.provider.as_str()).or_else(|| {
                self.selected_model
                    .as_ref()
                    .filter(|model| model.id == model_id)
                    .map(|model| model.provider.as_str())
            })?;
        self.auth
            .get_api_key(provider)
            .or_else(|| self.models.resolve_api_key(model_id))
    }

    pub fn system_prompt(&self) -> String {
        let context_files = context_files_for_prompt(&self.resources.context_files);
        let tool_names: Vec<&str> = self.tool_names.iter().map(String::as_str).collect();
        let tool_snippets: HashMap<String, String> = self
            .tools
            .iter()
            .map(|tool| (tool.name().to_string(), tool.description().to_string()))
            .collect();
        let append_system_prompt = (!self.resources.append_system_prompt.is_empty())
            .then(|| self.resources.append_system_prompt.join("\n\n"));
        build_system_prompt(BuildSystemPromptOptions {
            custom_prompt: self.resources.system_prompt.as_deref(),
            selected_tools: Some(&tool_names),
            tool_snippets: Some(&tool_snippets),
            prompt_guidelines: None,
            append_system_prompt: append_system_prompt.as_deref(),
            cwd: &self.cwd.to_string_lossy(),
            context_files: Some(&context_files),
            skills: Some(&self.resources.skills),
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
    let agent_dir = opts.agent_dir.map(PathBuf::from).unwrap_or_else(|| {
        dirs_next::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(DEFAULT_AGENT_DIR)
    });

    let mut settings = SettingsManager::new(
        agent_dir.join("settings.json"),
        Some(cwd.join(".automata").join("settings.json")),
    );
    settings
        .load()
        .await
        .map_err(|err| SessionError::Storage(format!("failed to load settings: {}", err)))?;

    let mut models = ModelRegistry::new();
    load_model_file(&mut models, &agent_dir.join("models.json"))?;
    load_model_file(&mut models, &cwd.join(".automata").join("models.json"))?;

    let selected_model = select_model(&models, &settings, opts.model.as_deref());
    let selected_provider = selected_model
        .as_ref()
        .map(|model| model.provider.as_str())
        .or_else(|| settings.get_provider())
        .unwrap_or("anthropic")
        .to_string();

    let mut auth = AuthStorage::load(&agent_dir);
    if let Some(api_key) = resolved_api_key(settings.get_api_key())? {
        auth.set_runtime_api_key(&selected_provider, api_key);
    }
    if let Some(api_key) = resolved_api_key(opts.api_key.as_deref())? {
        auth.set_runtime_api_key(&selected_provider, api_key);
    }

    let mut resource_loader = DefaultResourceLoader::from_settings(&cwd, &agent_dir, &settings);
    resource_loader.reload();
    let resources = resource_loader.into_resources();

    let sessions_root = opts
        .sessions_root
        .map(PathBuf::from)
        .unwrap_or_else(|| agent_dir.join("sessions"));
    let repo = JsonlSessionRepo::new(&sessions_root);

    let session = match opts.session_path {
        Some(ref path) => repo.open_by_path(path).await?,
        None => repo.create(&opts.cwd, None, None).await?,
    };

    let tool_names = opts
        .allowed_tools
        .unwrap_or_else(|| DEFAULT_TOOL_NAMES.iter().map(|name| (*name).to_string()).collect());
    let tool_refs: Vec<&str> = tool_names.iter().map(String::as_str).collect();
    let tools = build_tools_with_options(
        &opts.cwd,
        &tool_refs,
        BuildToolsOptions {
            shell_path: settings.get_shell_path().map(ToOwned::to_owned),
            shell_command_prefix: settings.get_shell_command_prefix().map(ToOwned::to_owned),
        },
    );

    Ok(AgentSessionHandle {
        session,
        settings,
        models,
        auth,
        selected_model,
        resources,
        tools,
        tool_names,
        cwd,
        agent_dir,
    })
}

fn load_model_file(models: &mut ModelRegistry, path: &Path) -> Result<(), SessionError> {
    if !path.exists() {
        return Ok(());
    }
    models.load_from_file(path).map_err(|err| {
        SessionError::Storage(format!("failed to load models from {}: {}", path.display(), err))
    })
}

fn select_model(
    models: &ModelRegistry,
    settings: &SettingsManager,
    explicit_model: Option<&str>,
) -> Option<Model> {
    if let Some(model_id) = explicit_model.or_else(|| settings.get_model())
        && let Some(model) = find_model(models, settings.get_provider(), model_id)
    {
        return Some(model);
    }

    let provider = settings.get_provider().unwrap_or("anthropic");
    default_model_for_provider(provider)
        .and_then(|model_id| find_model(models, Some(provider), model_id))
        .or_else(|| models.list_models().into_iter().next().cloned())
}

fn find_model(models: &ModelRegistry, provider: Option<&str>, model_id: &str) -> Option<Model> {
    provider
        .and_then(|provider| models.find(provider, model_id))
        .or_else(|| models.get_model(model_id))
        .cloned()
}

fn resolved_api_key(value: Option<&str>) -> Result<Option<String>, SessionError> {
    match value {
        None | Some("") => Ok(None),
        Some(value) => resolve_config_value(value)
            .map(Some)
            .map_err(|err| SessionError::Storage(format!("failed to resolve api_key: {}", err))),
    }
}
