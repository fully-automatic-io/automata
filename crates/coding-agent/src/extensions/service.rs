use super::runner::ExtensionRunner;
use super::types::*;

pub struct ExtensionService {
    runner: ExtensionRunner,
}

impl ExtensionService {
    pub fn new() -> Self {
        Self { runner: ExtensionRunner::new() }
    }

    pub fn load_and_get_result(&mut self, paths: &[String], cwd: &str) -> LoadExtensionsResult {
        let result = super::loader::load_extensions(paths, cwd);
        // Re-load into runner (LoadExtensionsResult is not Clone, so we reload from paths)
        let result2 = super::loader::load_extensions(paths, cwd);
        self.runner.load(result2);
        result
    }

    pub fn discover_and_load(
        &mut self,
        configured_paths: &[String],
        cwd: &str,
        agent_dir: Option<&str>,
    ) -> LoadExtensionsResult {
        let result = super::loader::discover_and_load_extensions(configured_paths, cwd, agent_dir);
        let result2 = super::loader::discover_and_load_extensions(configured_paths, cwd, agent_dir);
        self.runner.load(result2);
        result
    }

    pub fn dispatch(&self, event: &ExtensionEvent) -> Vec<(String, serde_json::Value)> {
        self.runner.dispatch_event(event)
    }

    pub fn invoke_tool(&self, name: &str, args: serde_json::Value) -> Result<serde_json::Value, String> {
        self.runner.invoke_tool(name, args)
    }

    pub fn tools(&self) -> std::collections::HashMap<String, RegisteredTool> {
        self.runner.collect_tools()
    }

    pub fn commands(&self) -> std::collections::HashMap<String, RegisteredCommand> {
        self.runner.collect_commands()
    }

    pub fn flags(&self) -> std::collections::HashMap<String, ExtensionFlag> {
        self.runner.collect_flags()
    }

    pub fn extension_count(&self) -> usize {
        self.runner.extension_count()
    }
}

impl Default for ExtensionService {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_new() {
        let svc = ExtensionService::new();
        assert_eq!(svc.extension_count(), 0);
    }
}
