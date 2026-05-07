// AgentSession — high-level session management tying agent + session + tools together

use std::path::{Path, PathBuf};

use crate::session::manager::{SessionManager, SessionContext};
use crate::settings::SettingsManager;
use crate::models::ModelRegistry;
use crate::compaction::compactor::{CompactionSettings, should_compact, prepare_compaction};

pub struct AgentSession {
    pub session: SessionManager,
    pub settings: SettingsManager,
    pub models: ModelRegistry,
    cwd: PathBuf,
}

impl AgentSession {
    /// Create a new session in the given working directory
    pub fn new(cwd: impl AsRef<Path>) -> Self {
        let cwd = cwd.as_ref().to_path_buf();
        Self {
            session: SessionManager::create(
                &cwd.to_string_lossy(),
                None,
            ),
            settings: SettingsManager::with_defaults(&cwd),
            models: ModelRegistry::new(),
            cwd,
        }
    }

    /// Open an existing session from file
    pub fn open(path: impl AsRef<Path>, cwd: impl AsRef<Path>) -> Self {
        let cwd = cwd.as_ref().to_path_buf();
        Self {
            session: SessionManager::open(
                &path.as_ref().to_string_lossy(),
                None,
                Some(&cwd.to_string_lossy()),
            ),
            settings: SettingsManager::with_defaults(&cwd),
            models: ModelRegistry::new(),
            cwd,
        }
    }

    /// Get the current working directory
    pub fn cwd(&self) -> &Path { &self.cwd }

    /// Build context for LLM call
    pub fn build_context(&self) -> SessionContext {
        self.session.build_context()
    }

    /// Append a user message
    pub fn append_user_message(&mut self, text: &str) -> String {
        self.session.append_message(serde_json::json!({
            "role": "user",
            "content": [{"type": "text", "text": text}],
            "timestamp": chrono::Utc::now().timestamp_millis()
        }))
    }

    /// Append an assistant message
    pub fn append_assistant_message(&mut self, message: serde_json::Value) -> String {
        self.session.append_message(message)
    }

    /// Check if compaction is needed and return preparation if so
    pub fn check_compaction(
        &self,
        context_tokens: u64,
        context_window: u64,
    ) -> Option<crate::compaction::compactor::CompactionPreparation> {
        let settings = CompactionSettings {
            enabled: self.settings.get().compaction.enabled,
            reserve_tokens: 16384,
            keep_recent_tokens: 20000,
        };

        if !should_compact(context_tokens, context_window, &settings) {
            return None;
        }

        let entries = self.session.entries();
        let entries_vec: Vec<_> = entries.into_iter().cloned().collect();
        prepare_compaction(&entries_vec, &settings)
    }

    /// Apply compaction result to session
    pub fn apply_compaction(
        &mut self,
        summary: &str,
        first_kept_entry_id: &str,
        tokens_before: u64,
    ) -> String {
        self.session.append_compaction(summary, first_kept_entry_id, tokens_before, None, None)
    }

    /// Get the current model ID from settings
    pub fn model_id(&self) -> &str {
        self.settings.get_model().unwrap_or("claude-opus-4-7")
    }

    /// Get the API key for the current model
    pub fn api_key(&self) -> Option<&str> {
        self.settings.get_api_key()
            .or_else(|| self.models.get_api_key(self.model_id()))
    }

    /// Switch to a different session file
    pub fn switch_session(&mut self, path: impl AsRef<Path>) {
        self.session = SessionManager::open(
            &path.as_ref().to_string_lossy(),
            None,
            Some(&self.cwd.to_string_lossy()),
        );
    }

    /// Fork the current session
    pub fn fork(&mut self, entry_id: &str) {
        self.session.branch(entry_id);
    }

    /// Get session name
    pub fn session_name(&self) -> Option<&str> {
        self.session.session_name()
    }

    /// Set session name
    pub fn set_session_name(&mut self, name: &str) {
        self.session.append_session_info(name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_session_new() {
        let session = AgentSession::new("/tmp");
        assert_eq!(session.cwd(), Path::new("/tmp"));
        assert_eq!(session.model_id(), "claude-opus-4-7");
    }

    #[test]
    fn test_append_messages() {
        let mut session = AgentSession::new("/tmp");
        let id = session.append_user_message("hello");
        assert!(!id.is_empty());

        let ctx = session.build_context();
        assert_eq!(ctx.messages.len(), 1);
    }
}
