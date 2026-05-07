// Agent session runtime — ties agent + session manager together

use super::manager::{SessionManager, SessionContext};
use std::path::PathBuf;

pub struct AgentSessionRuntime {
    manager: SessionManager,
}

impl AgentSessionRuntime {
    pub fn new(path: PathBuf) -> Self {
        Self {
            manager: SessionManager::create(
                &std::env::current_dir()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default(),
                Some(&path.to_string_lossy()),
            ),
        }
    }

    pub fn manager(&self) -> &SessionManager { &self.manager }
    pub fn manager_mut(&mut self) -> &mut SessionManager { &mut self.manager }

    pub fn build_context(&self) -> SessionContext {
        self.manager.build_context()
    }

    pub fn append_message(&mut self, message: serde_json::Value) -> String {
        self.manager.append_message(message)
    }

    pub fn branch(&mut self, branch_from_id: &str) {
        self.manager.branch(branch_from_id)
    }

    pub fn branch_with_summary(
        &mut self, branch_from_id: Option<&str>, summary: &str,
    ) -> String {
        self.manager.branch_with_summary(branch_from_id, summary, None, None)
    }

    pub fn append_compaction(
        &mut self, summary: &str, first_kept_entry_id: &str, tokens_before: u64,
    ) -> String {
        self.manager.append_compaction(summary, first_kept_entry_id, tokens_before, None, None)
    }
}
