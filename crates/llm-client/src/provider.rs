use crate::streaming::LlmEvent;
use crate::types::{LlmRequest, LlmResponse};
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

pub type LlmStream = Pin<Box<dyn Stream<Item = Result<LlmEvent, LlmError>> + Send>>;

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("HTTP error {status}: {message}")]
    Http { status: u16, message: String },
    #[error("Rate limited, retry after {retry_after_secs}s")]
    RateLimit { retry_after_secs: u64 },
    #[error("Authentication failed")]
    AuthError,
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    #[error("Stream error: {0}")]
    StreamError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error(transparent)]
    Reqwest(#[from] reqwest::Error),
    #[error("{0}")]
    Other(String),
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse, LlmError>;
    async fn stream(&self, request: LlmRequest) -> Result<LlmStream, LlmError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AuthMethod {
    #[default]
    ApiKeyHeader,
    Bearer,
}

#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub api_key: String,
    pub base_url: Option<String>,
    pub timeout_secs: u64,
    pub max_retries: u32,
    pub auth_method: AuthMethod,
}

impl ProviderConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: None,
            timeout_secs: 120,
            max_retries: 3,
            auth_method: AuthMethod::ApiKeyHeader,
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    pub fn with_auth_method(mut self, method: AuthMethod) -> Self {
        self.auth_method = method;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_config() {
        let config = ProviderConfig::new("sk-test").with_base_url("https://api.example.com");
        assert_eq!(config.api_key, "sk-test");
        assert_eq!(config.base_url, Some("https://api.example.com".into()));
    }
}
