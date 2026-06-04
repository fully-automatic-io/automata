use std::path::PathBuf;
use std::sync::Arc;

use crate::models::ModelRegistry;
use crate::settings::SettingsManager;
use agent_core::harness::session::{Session, SessionError};
use agent_core::tool::AgentTool;
use agent_core::types::{Model, ThinkingLevel};

use super::auth::AuthStorage;
use super::config::resolve_config_value;
use super::models::{clamp_thinking_level, default_model_for_provider};
use super::resource::ResourceSet;
use super::services::{
    AgentSessionServices, AgentSessionServicesOptions, SessionDiagnostic,
    build_system_prompt_from_resources, create_agent_session_services, default_agent_dir,
    resolve_path,
};
use super::session::{BuildToolsOptions, DEFAULT_TOOL_NAMES, build_tools_with_options};
use super::session_manager::{ManagedSessionMetadata, SessionManager};

pub struct CreateAgentSessionOptions {
    pub cwd: PathBuf,
    pub agent_dir: Option<PathBuf>,
    pub session: SessionManager,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub thinking_level: Option<ThinkingLevel>,
    pub allowed_tools: Option<Vec<String>>,
}

impl CreateAgentSessionOptions {
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self {
            cwd: cwd.into(),
            agent_dir: None,
            session: SessionManager::default(),
            model: None,
            api_key: None,
            thinking_level: None,
            allowed_tools: None,
        }
    }
}

pub struct CreateAgentSessionFromServicesOptions {
    pub services: AgentSessionServices,
    pub session: SessionManager,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub thinking_level: Option<ThinkingLevel>,
    pub allowed_tools: Option<Vec<String>>,
}

pub struct AgentSessionHandle {
    pub session: Session,
    pub session_metadata: ManagedSessionMetadata,
    pub settings: SettingsManager,
    pub models: ModelRegistry,
    pub auth: AuthStorage,
    pub selected_model: Option<Model>,
    pub selected_thinking_level: ThinkingLevel,
    pub resources: ResourceSet,
    pub tools: Vec<Arc<dyn AgentTool>>,
    pub tool_names: Vec<String>,
    pub diagnostics: Vec<SessionDiagnostic>,
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
        build_system_prompt_from_resources(
            &self.cwd,
            &self.resources,
            &self.tools,
            &self.tool_names,
        )
    }
}

/// Create or open a coding-agent session, set up settings, models, auth, resources, and tools.
pub async fn create_agent_session(
    opts: CreateAgentSessionOptions,
) -> Result<AgentSessionHandle, SessionError> {
    let initial_cwd = resolve_path(&opts.cwd)?;
    let agent_dir = match &opts.agent_dir {
        Some(path) => resolve_path(path)?,
        None => default_agent_dir(),
    };
    let effective_cwd = opts.session.effective_cwd(&initial_cwd).await?;
    let services = create_agent_session_services(AgentSessionServicesOptions {
        cwd: effective_cwd,
        agent_dir: Some(agent_dir),
        auth: None,
        settings: None,
        models: None,
        resource_options: None,
    })
    .await?;
    create_agent_session_from_services(CreateAgentSessionFromServicesOptions {
        services,
        session: opts.session,
        model: opts.model,
        api_key: opts.api_key,
        thinking_level: opts.thinking_level,
        allowed_tools: opts.allowed_tools,
    })
    .await
}

pub async fn create_agent_session_from_services(
    opts: CreateAgentSessionFromServicesOptions,
) -> Result<AgentSessionHandle, SessionError> {
    let CreateAgentSessionFromServicesOptions {
        services,
        session,
        model,
        api_key,
        thinking_level,
        allowed_tools,
    } = opts;

    let AgentSessionServices {
        cwd,
        agent_dir,
        mut auth,
        settings,
        models,
        resources,
        diagnostics,
    } = services;

    let managed_session = session.open_session(&cwd, &agent_dir).await?;
    let session_context = managed_session.session.build_context().await?;
    let selected_model = select_model(
        &models,
        &settings,
        model.as_deref(),
        session_context
            .model
            .as_ref()
            .map(|model| (model.provider.as_str(), model.model_id.as_str())),
    );
    let selected_provider = selected_model
        .as_ref()
        .map(|model| model.provider.as_str())
        .or_else(|| session_context.model.as_ref().map(|model| model.provider.as_str()))
        .or_else(|| settings.get_provider())
        .unwrap_or("anthropic")
        .to_string();

    if let Some(api_key) = resolved_api_key(settings.get_api_key())? {
        auth.set_runtime_api_key(&selected_provider, api_key);
    }
    if let Some(api_key) = resolved_api_key(api_key.as_deref())? {
        auth.set_runtime_api_key(&selected_provider, api_key);
    }

    let selected_thinking_level = selected_model
        .as_ref()
        .map(|model| {
            clamp_thinking_level(
                thinking_level.unwrap_or_else(|| settings.get_thinking_level()),
                model.reasoning,
            )
        })
        .unwrap_or_else(|| thinking_level.unwrap_or_else(|| settings.get_thinking_level()));

    let tool_names = allowed_tools
        .or(session_context.active_tool_names)
        .unwrap_or_else(|| DEFAULT_TOOL_NAMES.iter().map(|name| (*name).to_string()).collect());
    let tool_refs: Vec<&str> = tool_names.iter().map(String::as_str).collect();
    let cwd_string = cwd.to_string_lossy().to_string();
    let tools = build_tools_with_options(
        &cwd_string,
        &tool_refs,
        BuildToolsOptions {
            shell_path: settings.get_shell_path().map(ToOwned::to_owned),
            shell_command_prefix: settings.get_shell_command_prefix().map(ToOwned::to_owned),
        },
    );

    Ok(AgentSessionHandle {
        session: managed_session.session,
        session_metadata: managed_session.metadata,
        settings,
        models,
        auth,
        selected_model,
        selected_thinking_level,
        resources,
        tools,
        tool_names,
        diagnostics,
        cwd,
        agent_dir,
    })
}

fn select_model(
    models: &ModelRegistry,
    settings: &SettingsManager,
    explicit_model: Option<&str>,
    session_model: Option<(&str, &str)>,
) -> Option<Model> {
    if let Some(model_id) = explicit_model.or_else(|| settings.get_model())
        && let Some(model) = find_model(models, settings.get_provider(), model_id)
    {
        return Some(model);
    }

    if let Some((provider, model_id)) = session_model
        && let Some(model) = find_model(models, Some(provider), model_id)
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
