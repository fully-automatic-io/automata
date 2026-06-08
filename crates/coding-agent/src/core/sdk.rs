use std::path::PathBuf;
use std::sync::Arc;

use crate::extensions::{collect_registered_providers, extension_agent_tools};
use crate::models::ModelRegistry;
use crate::settings::SettingsManager;
use agent_core::harness::session::{Session, SessionError, SessionTreeEntry};
use agent_core::tool::AgentTool;
use agent_core::types::{Api, ModelCost};
use agent_core::types::{Model, ThinkingLevel};

use super::auth::AuthStorage;
use super::config::resolve_config_value;
use super::models::{clamp_thinking_level, default_model_for_provider};
use super::provider::Auth;
use super::resource::ResourceSet;
use super::services::{
    AgentSessionServices, AgentSessionServicesOptions, SessionDiagnostic,
    build_system_prompt_from_resources, create_agent_session_services, default_agent_dir,
    resolve_path,
};
use super::session::{
    BuildToolsOptions, CodingAgentSession, SessionBuildError, SessionOptions, ToolSelection,
    build_tools_with_options,
};
use super::session_manager::{ManagedSessionMetadata, SessionManager};

pub struct CreateAgentSessionOptions {
    pub cwd: PathBuf,
    pub agent_dir: Option<PathBuf>,
    pub session: SessionManager,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub thinking_level: Option<ThinkingLevel>,
    pub tools: ToolSelection,
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
            tools: ToolSelection::default(),
        }
    }
}

pub struct CreateAgentSessionFromServicesOptions {
    pub services: AgentSessionServices,
    pub session: SessionManager,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub thinking_level: Option<ThinkingLevel>,
    pub tools: ToolSelection,
}

pub struct AgentSessionHandle {
    pub session: Session,
    pub session_metadata: ManagedSessionMetadata,
    pub settings: SettingsManager,
    pub models: ModelRegistry,
    pub auth: AuthStorage,
    pub selected_model: Option<Model>,
    pub model_fallback_message: Option<String>,
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

    pub async fn into_coding_session(self) -> Result<CodingAgentSession, SessionBuildError> {
        let model = self.selected_model.clone().ok_or(SessionBuildError::NoModelSelected)?;
        let api_key = self
            .api_key_for_model(&model.id)
            .ok_or_else(|| SessionBuildError::NoApiKey(model.provider.clone()))?;
        let system_prompt = self.system_prompt();
        let tools = ToolSelection::from_names(self.tool_names.iter().map(String::as_str))?;
        let base_url = (!model.base_url.is_empty()).then(|| model.base_url.clone());
        let shell_path = self.settings.get_shell_path().map(ToOwned::to_owned);
        let shell_command_prefix = self.settings.get_shell_command_prefix().map(ToOwned::to_owned);
        let extensions = self.resources.extensions;

        let mut options =
            SessionOptions::new(self.cwd.to_string_lossy().to_string(), model, api_key);
        options.base_url = base_url;
        options.auth = Auth::Native;
        options.system_prompt = system_prompt;
        options.thinking_level = self.selected_thinking_level;
        options.tools = tools;
        options.shell_path = shell_path;
        options.shell_command_prefix = shell_command_prefix;
        options.extensions = extensions;

        CodingAgentSession::with_session(self.session, options).await
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
        tools: opts.tools,
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
        tools: tool_selection,
    } = opts;

    let AgentSessionServices {
        cwd,
        agent_dir,
        mut auth,
        settings,
        mut models,
        resources,
        diagnostics,
    } = services;

    if let Some(extensions) = &resources.extensions {
        register_extension_providers(&mut models, collect_registered_providers(extensions));
    }

    let managed_session = session.open_session(&cwd, &agent_dir).await?;
    let mut session = managed_session.session;
    let session_branch = session.get_branch().await?;
    let session_context = session.build_context().await?;
    let has_existing_session = !session_context.messages.is_empty();
    let has_thinking_entry = session_branch
        .iter()
        .any(|entry| matches!(entry, SessionTreeEntry::ThinkingLevelChange { .. }));
    let has_active_tools_entry = session_branch
        .iter()
        .any(|entry| matches!(entry, SessionTreeEntry::ActiveToolsChange { .. }));

    let (selected_model, model_fallback_message) = select_model(
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

    let selected_thinking_level = match selected_model.as_ref() {
        Some(model) => clamp_thinking_level(
            thinking_level.unwrap_or_else(|| settings.get_thinking_level()),
            model.reasoning,
        ),
        None => ThinkingLevel::Off,
    };

    let cwd_string = cwd.to_string_lossy().to_string();
    let extension_tools =
        resources.extensions.as_ref().map(extension_agent_tools).unwrap_or_default();
    let built_tools = build_tools_with_options(
        &cwd_string,
        &tool_selection,
        session_context.active_tool_names,
        BuildToolsOptions {
            shell_path: settings.get_shell_path().map(ToOwned::to_owned),
            shell_command_prefix: settings.get_shell_command_prefix().map(ToOwned::to_owned),
            extension_tools,
        },
    )
    .map_err(|err| SessionError::InvalidArgument(err.to_string()))?;

    if !has_existing_session {
        if let Some(model) = selected_model.as_ref() {
            session.append_model_change(&model.provider, &model.id).await?;
        }
    }
    if !has_existing_session || !has_thinking_entry {
        session.append_thinking_level_change(selected_thinking_level).await?;
    }
    if tool_selection.should_write_active_tools(has_active_tools_entry) {
        session.append_active_tools_change(built_tools.names.clone()).await?;
    }

    Ok(AgentSessionHandle {
        session,
        session_metadata: managed_session.metadata,
        settings,
        models,
        auth,
        selected_model,
        model_fallback_message,
        selected_thinking_level,
        resources,
        tools: built_tools.tools,
        tool_names: built_tools.names,
        diagnostics,
        cwd,
        agent_dir,
    })
}

fn register_extension_providers(
    models: &mut ModelRegistry,
    providers: std::collections::HashMap<String, crate::extensions::ProviderConfig>,
) {
    for (provider_name, provider) in providers {
        models.register_provider(
            provider_name.clone(),
            crate::models::ProviderConfig {
                api_key: provider.api_key.clone(),
                base_url: provider.base_url.clone(),
                api: provider.api.as_deref().and_then(Api::from_wire_str),
                headers: provider.headers.clone().unwrap_or_default(),
                auth_header: provider.auth_header,
            },
        );
        for model in provider.models.unwrap_or_default() {
            models.register_model(agent_core::types::Model {
                id: model.id,
                name: model.name,
                api: model
                    .api
                    .as_deref()
                    .and_then(Api::from_wire_str)
                    .or_else(|| provider.api.as_deref().and_then(Api::from_wire_str))
                    .unwrap_or_default(),
                provider: provider_name.clone(),
                base_url: provider.base_url.clone().unwrap_or_default(),
                reasoning: model.reasoning,
                input: model.input,
                cost: ModelCost {
                    input: model.cost.input,
                    output: model.cost.output,
                    cache_read: model.cost.cache_read,
                    cache_write: model.cost.cache_write,
                },
                context_window: model.context_window,
                max_tokens: model.max_tokens,
                compat: Default::default(),
            });
        }
    }
}

fn select_model(
    models: &ModelRegistry,
    settings: &SettingsManager,
    explicit_model: Option<&str>,
    session_model: Option<(&str, &str)>,
) -> (Option<Model>, Option<String>) {
    if let Some(model_id) = explicit_model
        && let Some(model) = find_model(models, settings.get_provider(), model_id)
    {
        return (Some(model), None);
    }

    let mut fallback_message = None;
    if let Some((provider, model_id)) = session_model {
        if let Some(model) = find_model(models, Some(provider), model_id) {
            return (Some(model), None);
        }
        fallback_message = Some(format!("Could not restore model {}/{}", provider, model_id));
    }

    if let Some(model_id) = settings.get_model()
        && let Some(model) = find_model(models, settings.get_provider(), model_id)
    {
        if let Some(message) = &mut fallback_message {
            *message = format!("{}. Using {}/{}", message, model.provider, model.id);
        }
        return (Some(model), fallback_message);
    }

    let provider = settings.get_provider().unwrap_or("anthropic");
    let model = default_model_for_provider(provider)
        .and_then(|model_id| find_model(models, Some(provider), model_id))
        .or_else(|| models.list_models().into_iter().next().cloned());
    if let (Some(message), Some(model)) = (&mut fallback_message, model.as_ref()) {
        *message = format!("{}. Using {}/{}", message, model.provider, model.id);
    }
    (model, fallback_message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BuiltinTool;

    #[tokio::test]
    async fn create_session_persists_initial_runtime_state() {
        let dir = tempfile::TempDir::new().unwrap();
        let cwd = dir.path().join("repo");
        let agent_dir = dir.path().join("agent");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&agent_dir).unwrap();
        std::fs::write(
            agent_dir.join("settings.json"),
            r#"{"provider":"anthropic","model":"claude-sonnet-4-6"}"#,
        )
        .unwrap();

        let handle = create_agent_session(CreateAgentSessionOptions {
            cwd,
            agent_dir: Some(agent_dir),
            session: SessionManager::in_memory(),
            model: None,
            api_key: None,
            thinking_level: None,
            tools: ToolSelection::only([BuiltinTool::Read]),
        })
        .await
        .unwrap();

        let context = handle.session.build_context().await.unwrap();
        let selected_model = handle.selected_model.as_ref().unwrap();
        assert_eq!(
            context
                .model
                .as_ref()
                .map(|model| (model.provider.as_str(), model.model_id.as_str())),
            Some((selected_model.provider.as_str(), selected_model.id.as_str()))
        );
        assert_eq!(context.active_tool_names, Some(vec!["read".to_string()]));
    }

    #[tokio::test]
    async fn handle_converts_into_runnable_coding_session() {
        let dir = tempfile::TempDir::new().unwrap();
        let cwd = dir.path().join("repo");
        let agent_dir = dir.path().join("agent");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&agent_dir).unwrap();
        std::fs::write(
            agent_dir.join("settings.json"),
            r#"{"provider":"anthropic","model":"claude-sonnet-4-6"}"#,
        )
        .unwrap();

        let handle = create_agent_session(CreateAgentSessionOptions {
            cwd,
            agent_dir: Some(agent_dir),
            session: SessionManager::in_memory(),
            model: None,
            api_key: Some("sk-test".into()),
            thinking_level: None,
            tools: ToolSelection::only([BuiltinTool::Read]),
        })
        .await
        .unwrap();

        let coding_session = handle.into_coding_session().await.unwrap();
        let context = coding_session.harness().build_context().await.unwrap();
        assert_eq!(context.active_tool_names, Some(vec!["read".to_string()]));
        assert_eq!(
            context
                .model
                .as_ref()
                .map(|model| (model.provider.as_str(), model.model_id.as_str())),
            Some(("anthropic", "claude-sonnet-4-6"))
        );
    }

    #[tokio::test]
    async fn create_session_uses_agent_dir_sessions_by_default() {
        let dir = tempfile::TempDir::new().unwrap();
        let cwd = dir.path().join("repo");
        let agent_dir = dir.path().join("agent");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&agent_dir).unwrap();
        std::fs::write(
            agent_dir.join("settings.json"),
            r#"{"provider":"anthropic","model":"claude-sonnet-4-6"}"#,
        )
        .unwrap();

        let handle = create_agent_session(CreateAgentSessionOptions {
            cwd: cwd.clone(),
            agent_dir: Some(agent_dir.clone()),
            session: SessionManager::create(),
            model: None,
            api_key: None,
            thinking_level: None,
            tools: ToolSelection::only([BuiltinTool::Read]),
        })
        .await
        .unwrap();

        assert_eq!(handle.session_metadata.cwd, cwd);
        assert_eq!(handle.session_metadata.sessions_root, Some(agent_dir.join("sessions")));
        assert!(handle.session_metadata.path.as_ref().unwrap().exists());
    }

    #[tokio::test]
    async fn create_session_preserves_explicit_session_manager_root() {
        let dir = tempfile::TempDir::new().unwrap();
        let cwd = dir.path().join("repo");
        let agent_dir = dir.path().join("agent");
        let sessions_root = dir.path().join("custom-sessions");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&agent_dir).unwrap();
        std::fs::write(
            agent_dir.join("settings.json"),
            r#"{"provider":"anthropic","model":"claude-sonnet-4-6"}"#,
        )
        .unwrap();

        let handle = create_agent_session(CreateAgentSessionOptions {
            cwd,
            agent_dir: Some(agent_dir),
            session: SessionManager::create_in(&sessions_root),
            model: None,
            api_key: None,
            thinking_level: None,
            tools: ToolSelection::only([BuiltinTool::Read]),
        })
        .await
        .unwrap();

        assert_eq!(handle.session_metadata.sessions_root, Some(sessions_root.clone()));
        assert!(handle.session_metadata.path.as_ref().unwrap().starts_with(&sessions_root));
    }
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
