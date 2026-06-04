use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent_core::harness::session::SessionError;
use agent_core::tool::AgentTool;

use crate::models::ModelRegistry;
use crate::settings::SettingsManager;

use super::auth::AuthStorage;
use super::models::DEFAULT_AGENT_DIR;
use super::prompt::{BuildSystemPromptOptions, build_system_prompt};
use super::resource::{
    DefaultResourceLoader, ResourceDiagnosticKind, ResourceLoaderOptions, ResourceSet,
    context_files_for_prompt,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionDiagnosticKind {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionDiagnostic {
    pub kind: SessionDiagnosticKind,
    pub message: String,
    pub path: Option<PathBuf>,
}

pub struct AgentSessionServicesOptions {
    pub cwd: PathBuf,
    pub agent_dir: Option<PathBuf>,
    pub auth: Option<AuthStorage>,
    pub settings: Option<SettingsManager>,
    pub models: Option<ModelRegistry>,
    pub resource_options: Option<ResourceLoaderOptions>,
}

impl AgentSessionServicesOptions {
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self {
            cwd: cwd.into(),
            agent_dir: None,
            auth: None,
            settings: None,
            models: None,
            resource_options: None,
        }
    }
}

pub struct AgentSessionServices {
    pub cwd: PathBuf,
    pub agent_dir: PathBuf,
    pub auth: AuthStorage,
    pub settings: SettingsManager,
    pub models: ModelRegistry,
    pub resources: ResourceSet,
    pub diagnostics: Vec<SessionDiagnostic>,
}

impl AgentSessionServices {
    pub fn system_prompt(&self, tools: &[Arc<dyn AgentTool>], tool_names: &[String]) -> String {
        build_system_prompt_from_resources(&self.cwd, &self.resources, tools, tool_names)
    }
}

pub(crate) fn build_system_prompt_from_resources(
    cwd: &Path,
    resources: &ResourceSet,
    tools: &[Arc<dyn AgentTool>],
    tool_names: &[String],
) -> String {
    let context_files = context_files_for_prompt(&resources.context_files);
    let tool_name_refs: Vec<&str> = tool_names.iter().map(String::as_str).collect();
    let tool_snippets: HashMap<String, String> = tools
        .iter()
        .map(|tool| (tool.name().to_string(), tool.description().to_string()))
        .collect();
    let append_system_prompt = (!resources.append_system_prompt.is_empty())
        .then(|| resources.append_system_prompt.join("\n\n"));
    build_system_prompt(BuildSystemPromptOptions {
        custom_prompt: resources.system_prompt.as_deref(),
        selected_tools: Some(&tool_name_refs),
        tool_snippets: Some(&tool_snippets),
        prompt_guidelines: None,
        append_system_prompt: append_system_prompt.as_deref(),
        cwd: &cwd.to_string_lossy(),
        context_files: Some(&context_files),
        skills: Some(&resources.skills),
    })
}

pub async fn create_agent_session_services(
    options: AgentSessionServicesOptions,
) -> Result<AgentSessionServices, SessionError> {
    let cwd = resolve_path(&options.cwd)?;
    let agent_dir = match options.agent_dir {
        Some(path) => resolve_path(path)?,
        None => default_agent_dir(),
    };

    let mut diagnostics = Vec::new();

    let mut settings = match options.settings {
        Some(settings) => settings,
        None => {
            let mut settings = SettingsManager::new(
                agent_dir.join("settings.json"),
                Some(cwd.join(".automata").join("settings.json")),
            );
            settings.load().await.map_err(|err| {
                SessionError::Storage(format!("failed to load settings: {}", err))
            })?;
            settings
        }
    };
    diagnostics.extend(settings.drain_errors().into_iter().map(|err| SessionDiagnostic {
        kind: SessionDiagnosticKind::Warning,
        message: err.message,
        path: Some(err.path),
    }));

    let mut auth = options.auth.unwrap_or_else(|| AuthStorage::load(&agent_dir));
    diagnostics.extend(auth.drain_errors().into_iter().map(|message| SessionDiagnostic {
        kind: SessionDiagnosticKind::Warning,
        message,
        path: Some(agent_dir.join("auth.json")),
    }));

    let mut models = options.models.unwrap_or_else(ModelRegistry::new);
    load_model_file(&mut models, &agent_dir.join("models.json"))?;
    load_model_file(&mut models, &cwd.join(".automata").join("models.json"))?;

    let mut resource_loader = match options.resource_options {
        Some(mut resource_options) => {
            resource_options.cwd = cwd.clone();
            resource_options.agent_dir = agent_dir.clone();
            DefaultResourceLoader::new(resource_options)
        }
        None => DefaultResourceLoader::from_settings(&cwd, &agent_dir, &settings),
    };
    resource_loader.reload();
    let resources = resource_loader.into_resources();
    diagnostics.extend(resources.diagnostics.iter().map(|diagnostic| SessionDiagnostic {
        kind: match diagnostic.kind {
            ResourceDiagnosticKind::Error => SessionDiagnosticKind::Error,
            ResourceDiagnosticKind::Warning => SessionDiagnosticKind::Warning,
        },
        message: diagnostic.message.clone(),
        path: diagnostic.path.clone(),
    }));

    Ok(AgentSessionServices {
        cwd,
        agent_dir,
        auth,
        settings,
        models,
        resources,
        diagnostics,
    })
}

pub(crate) fn load_model_file(models: &mut ModelRegistry, path: &Path) -> Result<(), SessionError> {
    if !path.exists() {
        return Ok(());
    }
    models.load_from_file(path).map_err(|err| {
        SessionError::Storage(format!("failed to load models from {}: {}", path.display(), err))
    })
}

pub(crate) fn resolve_path(path: impl AsRef<Path>) -> Result<PathBuf, SessionError> {
    let path = path.as_ref();
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(path))
        .map_err(|err| SessionError::Storage(format!("failed to resolve path: {}", err)))
}

pub(crate) fn default_agent_dir() -> PathBuf {
    dirs_next::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(DEFAULT_AGENT_DIR)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn creates_cwd_bound_services_from_agent_and_project_files() {
        let dir = tempfile::TempDir::new().unwrap();
        let cwd = dir.path().join("repo");
        let agent_dir = dir.path().join("agent");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&agent_dir).unwrap();
        std::fs::write(
            agent_dir.join("settings.json"),
            r#"{"provider":"anthropic","system_prompt":"from settings"}"#,
        )
        .unwrap();
        std::fs::write(cwd.join("AGENTS.md"), "project context").unwrap();

        let services = create_agent_session_services(AgentSessionServicesOptions {
            cwd: cwd.clone(),
            agent_dir: Some(agent_dir.clone()),
            auth: None,
            settings: None,
            models: None,
            resource_options: None,
        })
        .await
        .unwrap();

        assert_eq!(services.cwd, cwd);
        assert_eq!(services.agent_dir, agent_dir);
        assert_eq!(services.settings.get_provider(), Some("anthropic"));
        assert_eq!(services.resources.context_files.len(), 1);
        assert!(services.diagnostics.is_empty());
    }
}
