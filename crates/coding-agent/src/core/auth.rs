use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::config::resolve_config_value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AuthCredential {
    ApiKey { key: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthSource {
    Stored,
    Runtime,
    Environment(String),
    Fallback,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthStatus {
    pub configured: bool,
    pub source: Option<AuthSource>,
}

pub type FallbackResolver = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

#[derive(Clone, Default)]
pub struct AuthStorage {
    data: HashMap<String, AuthCredential>,
    runtime_overrides: HashMap<String, String>,
    fallback_resolver: Option<FallbackResolver>,
    path: Option<PathBuf>,
    errors: Vec<String>,
}

impl std::fmt::Debug for AuthStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthStorage")
            .field("providers", &self.data.keys().collect::<Vec<_>>())
            .field("runtime_overrides", &self.runtime_overrides.keys().collect::<Vec<_>>())
            .field("path", &self.path)
            .field("errors", &self.errors)
            .finish()
    }
}

impl AuthStorage {
    pub fn new(data: HashMap<String, AuthCredential>) -> Self {
        Self { data, ..Default::default() }
    }

    pub fn load(agent_dir: &Path) -> Self {
        let path = agent_dir.join("auth.json");
        let mut storage = Self {
            path: Some(path.clone()),
            ..Default::default()
        };
        storage.reload_from_path(&path);
        storage
    }

    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let mut storage = Self {
            path: Some(path.clone()),
            ..Default::default()
        };
        storage.reload_from_path(&path);
        storage
    }

    fn reload_from_path(&mut self, path: &Path) {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                match serde_json::from_str::<HashMap<String, AuthCredential>>(&content) {
                    Ok(data) => self.data = data,
                    Err(err) => {
                        self.errors.push(format!("Failed to parse {}: {}", path.display(), err))
                    }
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => self.errors.push(format!("Failed to read {}: {}", path.display(), err)),
        }
    }

    pub fn reload(&mut self) {
        if let Some(path) = self.path.clone() {
            self.reload_from_path(&path);
        }
    }

    pub fn save(&self) -> Result<(), AuthStorageError> {
        let path = self.path.as_ref().ok_or(AuthStorageError::NoPath)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| AuthStorageError::Io(err.to_string()))?;
        }
        let content = serde_json::to_string_pretty(&self.data)
            .map_err(|err| AuthStorageError::Parse(err.to_string()))?;
        std::fs::write(path, content).map_err(|err| AuthStorageError::Io(err.to_string()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
        }
        Ok(())
    }

    pub fn get(&self, provider: &str) -> Option<&AuthCredential> {
        self.data.get(provider)
    }

    pub fn set(&mut self, provider: impl Into<String>, credential: AuthCredential) {
        self.data.insert(provider.into(), credential);
    }

    pub fn remove(&mut self, provider: &str) {
        self.data.remove(provider);
    }

    pub fn list(&self) -> Vec<String> {
        let mut providers: Vec<String> = self.data.keys().cloned().collect();
        providers.sort();
        providers
    }

    pub fn has(&self, provider: &str) -> bool {
        self.data.contains_key(provider)
    }

    pub fn set_runtime_api_key(&mut self, provider: impl Into<String>, api_key: impl Into<String>) {
        self.runtime_overrides.insert(provider.into(), api_key.into());
    }

    pub fn remove_runtime_api_key(&mut self, provider: &str) {
        self.runtime_overrides.remove(provider);
    }

    pub fn set_fallback_resolver<F>(&mut self, resolver: F)
    where
        F: Fn(&str) -> Option<String> + Send + Sync + 'static,
    {
        self.fallback_resolver = Some(Arc::new(resolver));
    }

    pub fn clear_fallback_resolver(&mut self) {
        self.fallback_resolver = None;
    }

    pub fn has_auth(&self, provider: &str) -> bool {
        self.runtime_overrides.contains_key(provider)
            || self.data.contains_key(provider)
            || env_api_key(provider).is_some()
            || self.fallback_resolver.as_ref().and_then(|f| f(provider)).is_some()
    }

    pub fn auth_status(&self, provider: &str) -> AuthStatus {
        if self.runtime_overrides.contains_key(provider) {
            return AuthStatus {
                configured: true,
                source: Some(AuthSource::Runtime),
            };
        }
        if self.data.contains_key(provider) {
            return AuthStatus {
                configured: true,
                source: Some(AuthSource::Stored),
            };
        }
        if let Some((name, _)) = env_api_key_with_name(provider) {
            return AuthStatus {
                configured: true,
                source: Some(AuthSource::Environment(name)),
            };
        }
        if self.fallback_resolver.as_ref().and_then(|f| f(provider)).is_some() {
            return AuthStatus {
                configured: true,
                source: Some(AuthSource::Fallback),
            };
        }
        AuthStatus { configured: false, source: None }
    }

    pub fn get_api_key(&self, provider: &str) -> Option<String> {
        if let Some(key) = self.runtime_overrides.get(provider).filter(|key| !key.is_empty()) {
            return Some(key.clone());
        }
        if let Some(AuthCredential::ApiKey { key }) = self.data.get(provider) {
            return resolve_config_value(key).ok().filter(|key| !key.is_empty());
        }
        env_api_key(provider).or_else(|| self.fallback_resolver.as_ref().and_then(|f| f(provider)))
    }

    pub fn drain_errors(&mut self) -> Vec<String> {
        std::mem::take(&mut self.errors)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AuthStorageError {
    #[error("auth storage has no backing path")]
    NoPath,
    #[error("IO error: {0}")]
    Io(String),
    #[error("Parse error: {0}")]
    Parse(String),
}

fn env_api_key(provider: &str) -> Option<String> {
    env_api_key_with_name(provider).map(|(_, value)| value)
}

fn env_api_key_with_name(provider: &str) -> Option<(String, String)> {
    let canonical = format!("{}_API_KEY", provider.to_uppercase().replace('-', "_"));
    let aliases: &[(&str, &[&str])] =
        &[("anthropic", &["ANTHROPIC_API_KEY"]), ("openai", &["OPENAI_API_KEY"])];
    let mut names = vec![canonical];
    for (known, known_aliases) in aliases {
        if provider == *known {
            names.extend(known_aliases.iter().map(|alias| alias.to_string()));
        }
    }
    names.sort();
    names.dedup();
    for name in names {
        if let Ok(value) = std::env::var(&name)
            && !value.is_empty()
        {
            return Some((name, value));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_provider_keyed_auth_json_shape() {
        let data: HashMap<String, AuthCredential> =
            serde_json::from_str(r#"{"anthropic":{"type":"api_key","key":"sk-test"}}"#).unwrap();
        let storage = AuthStorage::new(data);
        assert_eq!(storage.get_api_key("anthropic"), Some("sk-test".into()));
    }

    #[test]
    fn runtime_key_wins_over_stored_key() {
        let mut storage = AuthStorage::new(HashMap::from([(
            "anthropic".into(),
            AuthCredential::ApiKey { key: "stored".into() },
        )]));
        storage.set_runtime_api_key("anthropic", "runtime");
        assert_eq!(storage.get_api_key("anthropic"), Some("runtime".into()));
        assert_eq!(storage.auth_status("anthropic").source, Some(AuthSource::Runtime));
    }

    #[test]
    fn fallback_resolver_is_used_last() {
        let mut storage = AuthStorage::default();
        storage.set_fallback_resolver(|provider| {
            (provider == "custom").then(|| "fallback".to_string())
        });
        assert_eq!(storage.get_api_key("custom"), Some("fallback".into()));
        assert!(storage.has_auth("custom"));
        assert_eq!(storage.auth_status("custom").source, Some(AuthSource::Fallback));
    }
}
