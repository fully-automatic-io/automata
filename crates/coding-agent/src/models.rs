use agent_core::types::{Api, Model, ModelCost};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Provider credentials/endpoint registered for a model's `provider` name.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub api_key: String,
    pub base_url: Option<String>,
}

/// In-memory registry of available models plus per-provider credentials and
/// id aliases. Models are the canonical [`agent_core::types::Model`] so they
/// flow directly into [`crate::SessionOptions`] / `build_provider`.
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
        let anthropic = |id: &str, name: &str, ctx: u64, max: u64| Model {
            id: id.into(),
            name: name.into(),
            api: Api::Anthropic,
            provider: "anthropic".into(),
            base_url: String::new(),
            reasoning: true,
            input: vec!["text".into()],
            cost: ModelCost::default(),
            context_window: ctx,
            max_tokens: max,
        };
        let openai = |id: &str, name: &str, ctx: u64, max: u64| Model {
            id: id.into(),
            name: name.into(),
            api: Api::Openai,
            provider: "openai".into(),
            base_url: String::new(),
            reasoning: false,
            input: vec!["text".into()],
            cost: ModelCost::default(),
            context_window: ctx,
            max_tokens: max,
        };
        for model in [
            anthropic("claude-opus-4-7", "Claude Opus 4.7", 1_000_000, 32_000),
            anthropic("claude-sonnet-4-6", "Claude Sonnet 4.6", 200_000, 16_000),
            anthropic("claude-haiku-4-5", "Claude Haiku 4.5", 200_000, 8_000),
            openai("gpt-4o", "GPT-4o", 128_000, 16_384),
        ] {
            self.models.insert(model.id.clone(), model);
        }
    }

    /// Load additional models from a JSON config file.
    pub fn load_from_file(&mut self, path: &std::path::Path) -> Result<(), String> {
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let models: Vec<Model> = serde_json::from_str(&content).map_err(|e| e.to_string())?;
        for model in models {
            self.models.insert(model.id.clone(), model);
        }
        Ok(())
    }

    /// Register a model directly.
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
        let resolved = self.overrides.get(id).map(String::as_str).unwrap_or(id);
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
        self.providers.get(&model.provider).map(|p| p.api_key.as_str())
    }

    /// Resolve API key: provider config first, then environment variable.
    pub fn resolve_api_key(&self, model_id: &str) -> Option<String> {
        let model = self.get_model(model_id)?;
        if let Some(p) = self.providers.get(&model.provider)
            && !p.api_key.is_empty() {
                return Some(p.api_key.clone());
            }
        match model.provider.as_str() {
            "anthropic" => std::env::var("ANTHROPIC_API_KEY").ok(),
            "openai" => std::env::var("OPENAI_API_KEY").ok(),
            other => std::env::var(format!("{}_API_KEY", other.to_uppercase())).ok(),
        }
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
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
        r.register_model(Model {
            id: "custom-model".into(),
            name: "Custom".into(),
            provider: "custom".into(),
            ..Model::default()
        });
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
        r.register_provider(
            "anthropic".into(),
            ProviderConfig { api_key: "sk-test".into(), base_url: None },
        );
        assert_eq!(r.get_api_key("claude-opus-4-7"), Some("sk-test"));
    }
}
