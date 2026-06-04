use std::path::{Path, PathBuf};

use agent_core::harness::session::{Session, SessionError};

use super::sdk::{AgentSessionHandle, CreateAgentSessionOptions, create_agent_session};
use super::services::SessionDiagnostic;
use super::session_manager::{ForkPosition, SessionManager};

pub struct AgentSessionRuntimeOptions {
    pub cwd: PathBuf,
    pub agent_dir: Option<PathBuf>,
    pub session: SessionManager,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub thinking_level: Option<agent_core::types::ThinkingLevel>,
    pub allowed_tools: Option<Vec<String>>,
}

impl AgentSessionRuntimeOptions {
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

pub struct AgentSessionRuntime {
    handle: AgentSessionHandle,
    options: RuntimeFixedOptions,
}

#[derive(Debug, Clone)]
struct RuntimeFixedOptions {
    agent_dir: Option<PathBuf>,
    model: Option<String>,
    api_key: Option<String>,
    thinking_level: Option<agent_core::types::ThinkingLevel>,
    allowed_tools: Option<Vec<String>>,
}

impl AgentSessionRuntime {
    pub async fn create(options: AgentSessionRuntimeOptions) -> Result<Self, SessionError> {
        let fixed = RuntimeFixedOptions {
            agent_dir: options.agent_dir.clone(),
            model: options.model.clone(),
            api_key: options.api_key.clone(),
            thinking_level: options.thinking_level,
            allowed_tools: options.allowed_tools.clone(),
        };
        let handle = create_agent_session(CreateAgentSessionOptions {
            cwd: options.cwd,
            agent_dir: options.agent_dir,
            session: options.session,
            model: options.model,
            api_key: options.api_key,
            thinking_level: options.thinking_level,
            allowed_tools: options.allowed_tools,
        })
        .await?;
        Ok(Self { handle, options: fixed })
    }

    pub fn handle(&self) -> &AgentSessionHandle {
        &self.handle
    }

    pub fn handle_mut(&mut self) -> &mut AgentSessionHandle {
        &mut self.handle
    }

    pub fn session(&self) -> &Session {
        &self.handle.session
    }

    pub fn session_mut(&mut self) -> &mut Session {
        &mut self.handle.session
    }

    pub fn cwd(&self) -> &Path {
        &self.handle.cwd
    }

    pub fn diagnostics(&self) -> &[SessionDiagnostic] {
        &self.handle.diagnostics
    }

    pub async fn new_session(&mut self) -> Result<(), SessionError> {
        let cwd = self.handle.cwd.clone();
        self.replace(cwd, self.session_manager_for_create()).await
    }

    pub async fn switch_session(&mut self, path: impl Into<PathBuf>) -> Result<(), SessionError> {
        let path = path.into();
        let cwd = SessionManager::open(&path).effective_cwd(&self.handle.cwd).await?;
        self.replace(cwd, SessionManager::open(path)).await
    }

    pub async fn continue_recent(&mut self) -> Result<(), SessionError> {
        let cwd = self.handle.cwd.clone();
        self.replace(cwd, self.session_manager_for_continue()).await
    }

    pub async fn fork(
        &mut self,
        source_path: impl Into<PathBuf>,
        entry_id: Option<String>,
        position: ForkPosition,
    ) -> Result<(), SessionError> {
        let mut manager = match entry_id {
            Some(entry_id) => SessionManager::fork(source_path).fork_at(entry_id, position),
            None => SessionManager::fork(source_path),
        };
        if let Some(root) = self.handle.session_metadata.sessions_root.clone() {
            manager = manager.with_sessions_root(root);
        }
        let cwd = self.handle.cwd.clone();
        self.replace(cwd, manager).await
    }

    fn session_manager_for_create(&self) -> SessionManager {
        match self.handle.session_metadata.sessions_root.clone() {
            Some(root) => SessionManager::create_in(root),
            None => SessionManager::create(),
        }
    }

    fn session_manager_for_continue(&self) -> SessionManager {
        match self.handle.session_metadata.sessions_root.clone() {
            Some(root) => SessionManager::continue_recent_in(root),
            None => SessionManager::continue_recent(),
        }
    }

    async fn replace(&mut self, cwd: PathBuf, session: SessionManager) -> Result<(), SessionError> {
        self.handle = create_agent_session(CreateAgentSessionOptions {
            cwd,
            agent_dir: self.options.agent_dir.clone(),
            session,
            model: self.options.model.clone(),
            api_key: self.options.api_key.clone(),
            thinking_level: self.options.thinking_level,
            allowed_tools: self.options.allowed_tools.clone(),
        })
        .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::types::AgentMessage;

    #[tokio::test]
    async fn switch_session_rebinds_services_to_session_cwd() {
        let dir = tempfile::TempDir::new().unwrap();
        let cwd_a = dir.path().join("a");
        let cwd_b = dir.path().join("b");
        let agent_dir = dir.path().join("agent");
        let sessions_root = dir.path().join("sessions");
        std::fs::create_dir_all(&cwd_a).unwrap();
        std::fs::create_dir_all(&cwd_b).unwrap();
        std::fs::create_dir_all(&agent_dir).unwrap();

        let mut runtime = AgentSessionRuntime::create(AgentSessionRuntimeOptions {
            cwd: cwd_a.clone(),
            agent_dir: Some(agent_dir.clone()),
            session: SessionManager::create_in(&sessions_root),
            model: None,
            api_key: None,
            thinking_level: None,
            allowed_tools: None,
        })
        .await
        .unwrap();
        assert_eq!(runtime.cwd(), cwd_a.as_path());

        let mut other = create_agent_session(CreateAgentSessionOptions {
            cwd: cwd_b.clone(),
            agent_dir: Some(agent_dir),
            session: SessionManager::create_in(&sessions_root),
            model: None,
            api_key: None,
            thinking_level: None,
            allowed_tools: None,
        })
        .await
        .unwrap();
        other.session.append_message(AgentMessage::user_text("from b")).await.unwrap();
        let other_path = other.session_metadata.path.clone().unwrap();

        runtime.switch_session(other_path).await.unwrap();
        assert_eq!(runtime.cwd(), cwd_b.as_path());
        assert_eq!(runtime.session().build_context().await.unwrap().messages.len(), 1);
    }
}
