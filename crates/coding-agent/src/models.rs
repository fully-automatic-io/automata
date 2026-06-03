use agent_core::types::{Api, Model, ModelCompat, ModelCost};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::core::config::{resolve_config_value, resolve_config_value_opt};

// `builtin_models()` is generated at build time from `models.json` by build.rs.
include!(concat!(env!("OUT_DIR"), "/models_generated.rs"));

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderConfig {
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub api: Option<Api>,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    #[serde(default)]
    pub auth_header: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedProviderAuth {
    pub api_key: Option<String>,
    pub headers: HashMap<String, String>,
    pub auth_header: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ModelsConfigFile {
    #[serde(default)]
    providers: HashMap<String, ProviderConfigInput>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ProviderConfigInput {
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    api: Option<Api>,
    #[serde(default)]
    headers: HashMap<String, String>,
    #[serde(default)]
    auth_header: Option<bool>,
    #[serde(default)]
    models: Vec<ModelDefinition>,
    #[serde(default)]
    model_overrides: HashMap<String, ModelOverride>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ModelDefinition {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    api: Option<Api>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    reasoning: Option<bool>,
    #[serde(default)]
    input: Option<Vec<String>>,
    #[serde(default)]
    cost: Option<ModelCost>,
    #[serde(default)]
    context_window: Option<u64>,
    #[serde(default)]
    max_tokens: Option<u64>,
    #[serde(default)]
    headers: HashMap<String, String>,
    #[serde(default)]
    compat: ModelCompat,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ModelOverride {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    reasoning: Option<bool>,
    #[serde(default)]
    input: Option<Vec<String>>,
    #[serde(default)]
    cost: Option<PartialModelCost>,
    #[serde(default)]
    context_window: Option<u64>,
    #[serde(default)]
    max_tokens: Option<u64>,
    #[serde(default)]
    headers: HashMap<String, String>,
    #[serde(default)]
    compat: ModelCompat,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct PartialModelCost {
    input: Option<f64>,
    output: Option<f64>,
    cache_read: Option<f64>,
    cache_write: Option<f64>,
}

pub struct ModelRegistry {
    models: HashMap<String, Model>,
    providers: HashMap<String, ProviderConfig>,
    model_headers: HashMap<String, HashMap<String, String>>,
    overrides: HashMap<String, String>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            models: HashMap::new(),
            providers: HashMap::new(),
            model_headers: HashMap::new(),
            overrides: HashMap::new(),
        };
        registry.load_defaults();
        registry
    }

    fn load_defaults(&mut self) {
        for model in builtin_models() {
            self.models.insert(model_key(&model.provider, &model.id), model.clone());
            self.models.entry(model.id.clone()).or_insert(model);
        }
    }

    pub fn load_from_file(&mut self, path: &std::path::Path) -> Result<(), String> {
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let config: ModelsConfigFile = serde_json::from_str(&content).map_err(|e| e.to_string())?;
        self.load_config(config)
    }

    fn load_config(&mut self, config: ModelsConfigFile) -> Result<(), String> {
        for (provider_name, provider_input) in config.providers {
            let provider_config = ProviderConfig {
                api_key: provider_input.api_key.clone(),
                base_url: provider_input.base_url.clone(),
                api: provider_input.api,
                headers: provider_input.headers.clone(),
                auth_header: provider_input.auth_header,
            };
            self.register_provider(provider_name.clone(), provider_config);

            let provider_defaults = ProviderDefaults {
                api: provider_input.api,
                base_url: provider_input.base_url.clone(),
                headers: provider_input.headers.clone(),
            };
            for model_def in provider_input.models {
                let model_headers = model_def.headers.clone();
                let mut model =
                    model_from_definition(&provider_name, &provider_defaults, model_def)?;
                merge_model_headers(&mut model, &provider_defaults.headers);
                let key = model_key(&model.provider, &model.id);
                if !model_headers.is_empty() {
                    self.model_headers.insert(key, model_headers);
                }
                self.register_model(model);
            }

            for (model_id, model_override) in provider_input.model_overrides {
                let key = model_key(&provider_name, &model_id);
                if self.models.contains_key(&key) {
                    let override_headers = model_override.headers.clone();
                    if let Some(model) = self.models.get_mut(&key) {
                        apply_model_override(model, model_override);
                    }
                    if !override_headers.is_empty() {
                        self.model_headers.entry(key).or_default().extend(override_headers);
                    }
                } else if let Some(model) = self.models.get_mut(&model_id) {
                    let override_headers = model_override.headers.clone();
                    let provider = model.provider.clone();
                    apply_model_override(model, model_override);
                    if !override_headers.is_empty() {
                        self.model_headers
                            .entry(model_key(&provider, &model_id))
                            .or_default()
                            .extend(override_headers);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn register_model(&mut self, model: Model) {
        self.models.insert(model_key(&model.provider, &model.id), model.clone());
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

    pub fn find(&self, provider: &str, id: &str) -> Option<&Model> {
        self.models
            .get(&model_key(provider, id))
            .or_else(|| self.get_model(id).filter(|model| model.provider == provider))
    }

    pub fn get_provider(&self, name: &str) -> Option<&ProviderConfig> {
        self.providers.get(name)
    }

    pub fn set_override(&mut self, from: String, to: String) {
        self.overrides.insert(from, to);
    }

    pub fn list_models(&self) -> Vec<&Model> {
        let mut models: Vec<&Model> = self
            .models
            .iter()
            .filter_map(|(key, model)| (!key.contains('/')).then_some(model))
            .collect();
        models.sort_by(|a, b| a.id.cmp(&b.id));
        models
    }

    pub fn get_api_key(&self, model_id: &str) -> Option<&str> {
        let model = self.get_model(model_id)?;
        self.providers.get(&model.provider).and_then(|p| p.api_key.as_deref())
    }

    pub fn resolve_api_key(&self, model_id: &str) -> Option<String> {
        let model = self.get_model(model_id)?;
        self.resolve_provider_auth(&model.provider)
            .api_key
            .or_else(|| env_api_key(&model.provider))
    }

    pub fn resolve_provider_auth(&self, provider: &str) -> ResolvedProviderAuth {
        self.resolve_provider_auth_for_model(provider, None)
    }

    pub fn resolve_provider_auth_for_model(
        &self,
        provider: &str,
        model_id: Option<&str>,
    ) -> ResolvedProviderAuth {
        let Some(config) = self.providers.get(provider) else {
            return ResolvedProviderAuth {
                api_key: None,
                headers: HashMap::new(),
                auth_header: None,
            };
        };
        let api_key = resolve_config_value_opt(config.api_key.as_deref()).ok().flatten();
        let mut headers: HashMap<String, String> = config
            .headers
            .iter()
            .filter_map(|(key, value)| {
                resolve_config_value(value).ok().map(|value| (key.clone(), value))
            })
            .collect();
        if let Some(model_id) = model_id
            && let Some(model_headers) = self.model_headers.get(&model_key(provider, model_id))
        {
            headers.extend(model_headers.iter().filter_map(|(key, value)| {
                resolve_config_value(value).ok().map(|value| (key.clone(), value))
            }));
        }
        ResolvedProviderAuth {
            api_key,
            headers,
            auth_header: config.auth_header,
        }
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn model_key(provider: &str, id: &str) -> String {
    format!("{}/{}", provider, id)
}

#[derive(Debug, Clone)]
struct ProviderDefaults {
    api: Option<Api>,
    base_url: Option<String>,
    headers: HashMap<String, String>,
}

fn model_from_definition(
    provider_name: &str,
    provider: &ProviderDefaults,
    def: ModelDefinition,
) -> Result<Model, String> {
    if def.id.trim().is_empty() {
        return Err("custom model id must not be empty".into());
    }
    let api = def.api.or(provider.api).unwrap_or(Api::Anthropic);
    let base_url = def.base_url.or_else(|| provider.base_url.clone()).unwrap_or_default();
    Ok(Model {
        id: def.id.clone(),
        name: def.name.unwrap_or(def.id),
        api,
        provider: provider_name.to_string(),
        base_url,
        reasoning: def.reasoning.unwrap_or(false),
        input: def.input.unwrap_or_else(|| vec!["text".into()]),
        cost: def.cost.unwrap_or_default(),
        context_window: def.context_window.unwrap_or(0),
        max_tokens: def.max_tokens.unwrap_or(0),
        compat: def.compat,
    })
}

fn apply_model_override(model: &mut Model, override_: ModelOverride) {
    if let Some(name) = override_.name {
        model.name = name;
    }
    if let Some(reasoning) = override_.reasoning {
        model.reasoning = reasoning;
    }
    if let Some(input) = override_.input {
        model.input = input;
    }
    if let Some(context_window) = override_.context_window {
        model.context_window = context_window;
    }
    if let Some(max_tokens) = override_.max_tokens {
        model.max_tokens = max_tokens;
    }
    if let Some(cost) = override_.cost {
        if let Some(input) = cost.input {
            model.cost.input = input;
        }
        if let Some(output) = cost.output {
            model.cost.output = output;
        }
        if let Some(cache_read) = cost.cache_read {
            model.cost.cache_read = cache_read;
        }
        if let Some(cache_write) = cost.cache_write {
            model.cost.cache_write = cache_write;
        }
    }
    if override_.compat.force_adaptive_thinking.is_some() {
        model.compat.force_adaptive_thinking = override_.compat.force_adaptive_thinking;
    }
    if override_.compat.supports_temperature.is_some() {
        model.compat.supports_temperature = override_.compat.supports_temperature;
    }
    let _ = override_.headers;
}

fn merge_model_headers(_model: &mut Model, _headers: &HashMap<String, String>) {
    // `agent_core::Model` intentionally does not carry request headers; provider
    // request headers are resolved from ProviderConfig at request time.
}

fn env_api_key(provider: &str) -> Option<String> {
    let key = format!("{}_API_KEY", provider.to_uppercase().replace('-', "_"));
    std::env::var(key).ok().filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

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
        assert!(r.find("custom", "custom-model").is_some());
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
            ProviderConfig {
                api_key: Some("sk-test".into()),
                ..Default::default()
            },
        );
        assert_eq!(r.get_api_key("claude-opus-4-7"), Some("sk-test"));
    }

    #[test]
    fn test_load_provider_config_file() {
        let file = NamedTempFile::new().unwrap();
        std::fs::write(file.path(), r#"
        {
          "providers": {
            "local": {
              "api": "openai",
              "base_url": "http://localhost:1234/v1/chat/completions",
              "api_key": "sk-local",
              "headers": { "X-Test": "literal" },
              "models": [{ "id": "local-model", "name": "Local", "context_window": 8192, "max_tokens": 1024 }]
            }
          }
        }
        "#).unwrap();
        let mut r = ModelRegistry::new();
        r.load_from_file(file.path()).unwrap();
        let model = r.find("local", "local-model").unwrap();
        assert_eq!(model.api, Api::Openai);
        assert_eq!(model.base_url, "http://localhost:1234/v1/chat/completions");
        let auth = r.resolve_provider_auth("local");
        assert_eq!(auth.api_key, Some("sk-local".into()));
        assert_eq!(auth.headers.get("X-Test"), Some(&"literal".to_string()));
    }

    #[test]
    fn test_rejects_old_model_array_config_shape() {
        let file = NamedTempFile::new().unwrap();
        std::fs::write(file.path(), r#"[{ "id": "old-shape" }]"#).unwrap();
        let mut r = ModelRegistry::new();
        assert!(r.load_from_file(file.path()).is_err());
    }

    #[test]
    fn test_generated_catalog_metadata() {
        let r = ModelRegistry::new();
        let opus = r.get_model("claude-opus-4-8").expect("opus-4-8 in catalog");
        assert_eq!(opus.cost.input, 5.0);
        assert_eq!(opus.context_window, 1_000_000);
        assert_eq!(opus.compat.supports_temperature, Some(false));
        assert_eq!(opus.compat.force_adaptive_thinking, Some(true));

        let gpt = r.get_model("gpt-5.5").expect("gpt-5.5 in catalog");
        assert_eq!(gpt.api, agent_core::types::Api::OpenaiResponses);
        assert!(gpt.reasoning);

        let haiku = r.get_model("claude-haiku-4-5").expect("haiku in catalog");
        assert_eq!(haiku.compat.supports_temperature, None);
    }
}
