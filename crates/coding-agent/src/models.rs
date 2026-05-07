use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub context_window: u32,
    pub max_output_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub api_key: String,
    pub base_url: Option<String>,
}

pub struct ModelRegistry {
    models: HashMap<String, Model>,
    providers: HashMap<String, ProviderConfig>,
    overrides: HashMap<String, String>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            models: HashMap::new(),
            providers: HashMap::new(),
            overrides: HashMap::new(),
        };
        registry.load_defaults();
        registry
    }

    fn load_defaults(&mut self) {
        for model in [
            Model { id: "claude-opus-4-7".into(), name: "Claude Opus 4.7".into(), provider: "anthropic".into(), context_window: 1_000_000, max_output_tokens: 32_000 },
            Model { id: "claude-sonnet-4-6".into(), name: "Claude Sonnet 4.6".into(), provider: "anthropic".into(), context_window: 200_000, max_output_tokens: 16_000 },
            Model { id: "claude-haiku-4-5".into(), name: "Claude Haiku 4.5".into(), provider: "anthropic".into(), context_window: 200_000, max_output_tokens: 8_000 },
            Model { id: "gpt-4o".into(), name: "GPT-4o".into(), provider: "openai".into(), context_window: 128_000, max_output_tokens: 16_384 },
        ] {
            self.models.insert(model.id.clone(), model);
        }
    }

    /// Load additional models from a JSON config file
    pub fn load_from_file(&mut self, path: &std::path::Path) -> Result<(), String> {
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let models: Vec<Model> = serde_json::from_str(&content).map_err(|e| e.to_string())?;
        for model in models {
            self.models.insert(model.id.clone(), model);
        }
        Ok(())
    }

    /// Register a model directly
    pub fn register_model(&mut self, model: Model) {
        self.models.insert(model.id.clone(), model);
    }

    pub fn register_provider(&mut self, name: String, config: ProviderConfig) {
        self.providers.insert(name, config);
    }

    pub fn unregister_provider(&mut self, name: &str) {
        self.providers.remove(name);
    }

    pub fn get_model(&self, id: &str) -> Option<&Model> {
        let resolved = self.overrides.get(id).map(|s| s.as_str()).unwrap_or(id);
        self.models.get(resolved)
    }

    pub fn get_provider(&self, name: &str) -> Option<&ProviderConfig> {
        self.providers.get(name)
    }

    pub fn set_override(&mut self, from: String, to: String) {
        self.overrides.insert(from, to);
    }

    pub fn list_models(&self) -> Vec<&Model> {
        self.models.values().collect()
    }

    pub fn get_api_key(&self, model_id: &str) -> Option<&str> {
        let model = self.get_model(model_id)?;
        // Check provider config first, then environment
        if let Some(key) = self.providers.get(&model.provider).map(|p| p.api_key.as_str()) {
            return Some(key);
        }
        None
    }

    /// Resolve API key: provider config → environment variable
    pub fn resolve_api_key(&self, model_id: &str) -> Option<String> {
        let model = self.get_model(model_id)?;
        if let Some(p) = self.providers.get(&model.provider) {
            if !p.api_key.is_empty() {
                return Some(p.api_key.clone());
            }
        }
        // Fall back to environment variables
        match model.provider.as_str() {
            "anthropic" => std::env::var("ANTHROPIC_API_KEY").ok(),
            "openai" => std::env::var("OPENAI_API_KEY").ok(),
            _ => std::env::var(format!("{}_API_KEY", model.provider.to_uppercase())).ok(),
        }
    }
}

impl Default for ModelRegistry {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_models() {
        let r = ModelRegistry::new();
        assert!(r.get_model("claude-opus-4-7").is_some());
        assert!(r.get_model("claude-haiku-4-5").is_some());
        assert!(r.get_model("gpt-4o").is_some());
    }

    #[test]
    fn test_register_model() {
        let mut r = ModelRegistry::new();
        r.register_model(Model { id: "custom-model".into(), name: "Custom".into(), provider: "custom".into(), context_window: 8000, max_output_tokens: 2000 });
        assert!(r.get_model("custom-model").is_some());
    }

    #[test]
    fn test_model_override() {
        let mut r = ModelRegistry::new();
        r.set_override("default".into(), "claude-opus-4-7".into());
        assert!(r.get_model("default").is_some());
    }

    #[test]
    fn test_provider_api_key() {
        let mut r = ModelRegistry::new();
        r.register_provider("anthropic".into(), ProviderConfig { api_key: "sk-test".into(), base_url: None });
        assert_eq!(r.get_api_key("claude-opus-4-7"), Some("sk-test"));
    }
}
