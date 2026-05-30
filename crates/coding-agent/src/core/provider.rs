// Provider factory — builds a concrete `LlmProvider` from a resolved `Model`
// plus credentials. Centralizes the API-family → provider-impl mapping so the
// session layer never has to match on `Api` directly.

use std::sync::Arc;

use agent_core::types::{Api, Model};
use llm_client::provider::{AuthMethod, LlmProvider, ProviderConfig};
use llm_client::providers::anthropic::AnthropicProvider;
use llm_client::providers::openai::OpenAIProvider;
use llm_client::providers::openai_responses::OpenAIResponsesProvider;

/// How a provider should authenticate. Anthropic-compatible relay endpoints
/// (DeepSeek, OpenRouter, ...) often want a bearer token rather than the
/// native `x-api-key` header, so callers can override the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Auth {
    /// Native scheme for the API family (`x-api-key` for Anthropic, bearer for OpenAI).
    #[default]
    Native,
    /// Force bearer-token auth regardless of API family.
    Bearer,
}

/// Inputs for [`build_provider`]. `base_url`, when set, fully overrides the
/// provider's default endpoint (it must be the complete URL, e.g.
/// `https://api.deepseek.com/anthropic/v1/messages`).
pub struct ProviderBuild<'a> {
    pub model: &'a Model,
    pub api_key: String,
    pub base_url: Option<String>,
    pub auth: Auth,
}

/// Construct the concrete provider for a model's API family.
pub fn build_provider(opts: ProviderBuild<'_>) -> Arc<dyn LlmProvider> {
    let mut config = ProviderConfig::new(opts.api_key);
    if let Some(url) = opts.base_url {
        config = config.with_base_url(url);
    }
    config = config.with_auth_method(match opts.auth {
        Auth::Bearer => AuthMethod::Bearer,
        Auth::Native => match opts.model.api {
            Api::Anthropic => AuthMethod::ApiKeyHeader,
            _ => AuthMethod::Bearer,
        },
    });

    match opts.model.api {
        Api::Anthropic => Arc::new(AnthropicProvider::new(config)),
        Api::Openai | Api::Mock => Arc::new(OpenAIProvider::new(config)),
        Api::OpenaiResponses => Arc::new(OpenAIResponsesProvider::new(config)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(api: Api) -> Model {
        Model { api, ..Default::default() }
    }

    #[test]
    fn builds_each_api_family() {
        for api in [Api::Anthropic, Api::Openai, Api::OpenaiResponses] {
            let m = model(api);
            let _p = build_provider(ProviderBuild {
                model: &m,
                api_key: "sk-test".into(),
                base_url: None,
                auth: Auth::Native,
            });
        }
    }
}
