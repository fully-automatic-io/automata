use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::fs;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Settings {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub compaction: CompactionSettings,
    #[serde(default)]
    pub retry: RetrySettings,
    #[serde(default)]
    pub working_directory: Option<String>,
    #[serde(default)]
    pub thinking_level: Option<String>,
    #[serde(default)]
    pub transport: Option<String>, // "sse" | "json"
    #[serde(default)]
    pub steering_mode: Option<String>, // "all" | "one-at-a-time"
    #[serde(default)]
    pub follow_up_mode: Option<String>,
    #[serde(default)]
    pub shell_path: Option<String>,
    #[serde(default)]
    pub extensions: Vec<String>,
    #[serde(default)]
    pub extra: serde_json::Value,
}

impl Settings {
    /// Merge another settings object into this one (other takes precedence for non-None values)
    pub fn merge(&mut self, other: &Settings) {
        if other.model.is_some() { self.model = other.model.clone(); }
        if other.api_key.is_some() { self.api_key = other.api_key.clone(); }
        if other.max_tokens.is_some() { self.max_tokens = other.max_tokens; }
        if other.temperature.is_some() { self.temperature = other.temperature; }
        if other.working_directory.is_some() { self.working_directory = other.working_directory.clone(); }
        if other.thinking_level.is_some() { self.thinking_level = other.thinking_level.clone(); }
        if other.transport.is_some() { self.transport = other.transport.clone(); }
        if other.steering_mode.is_some() { self.steering_mode = other.steering_mode.clone(); }
        if other.follow_up_mode.is_some() { self.follow_up_mode = other.follow_up_mode.clone(); }
        if other.shell_path.is_some() { self.shell_path = other.shell_path.clone(); }
        if !other.extensions.is_empty() { self.extensions = other.extensions.clone(); }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionSettings {
    pub enabled: bool,
    pub threshold_tokens: u64,
    pub target_tokens: u64,
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self { enabled: true, threshold_tokens: 150_000, target_tokens: 50_000 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrySettings {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
}

impl Default for RetrySettings {
    fn default() -> Self {
        Self { max_retries: 3, initial_delay_ms: 500, max_delay_ms: 30_000 }
    }
}

#[derive(Debug, Clone)]
pub struct SettingsManager {
    global_path: PathBuf,
    project_path: Option<PathBuf>,
    merged: Settings,
}

impl SettingsManager {
    pub fn new(global_path: PathBuf, project_path: Option<PathBuf>) -> Self {
        Self { global_path, project_path, merged: Settings::default() }
    }

    /// Create with default paths (~/.automata/agent/settings.json + ./.automata/settings.json)
    pub fn with_defaults(cwd: &std::path::Path) -> Self {
        let global = dirs_next::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".automata/agent/settings.json");
        let project = cwd.join(".automata/settings.json");
        Self::new(global, Some(project))
    }

    pub async fn load(&mut self) -> Result<(), SettingsError> {
        let mut settings = Settings::default();

        if self.global_path.exists() {
            let content = fs::read_to_string(&self.global_path).await
                .map_err(|e| SettingsError::Io(e.to_string()))?;
            let global: Settings = serde_json::from_str(&content)
                .map_err(|e| SettingsError::Parse(e.to_string()))?;
            settings.merge(&global);
        }

        if let Some(ref project_path) = self.project_path {
            if project_path.exists() {
                let content = fs::read_to_string(project_path).await
                    .map_err(|e| SettingsError::Io(e.to_string()))?;
                let project: Settings = serde_json::from_str(&content)
                    .map_err(|e| SettingsError::Parse(e.to_string()))?;
                settings.merge(&project);
            }
        }

        // Resolve API key from environment if not set
        if settings.api_key.is_none() {
            settings.api_key = std::env::var("ANTHROPIC_API_KEY").ok()
                .or_else(|| std::env::var("OPENAI_API_KEY").ok());
        }

        self.merged = settings;
        Ok(())
    }

    pub async fn save_global(&self, settings: &Settings) -> Result<(), SettingsError> {
        if let Some(parent) = self.global_path.parent() {
            fs::create_dir_all(parent).await
                .map_err(|e| SettingsError::Io(e.to_string()))?;
        }
        let content = serde_json::to_string_pretty(settings)
            .map_err(|e| SettingsError::Parse(e.to_string()))?;
        fs::write(&self.global_path, content).await
            .map_err(|e| SettingsError::Io(e.to_string()))
    }

    pub fn get(&self) -> &Settings { &self.merged }
    pub fn get_model(&self) -> Option<&str> { self.merged.model.as_deref() }
    pub fn get_api_key(&self) -> Option<&str> { self.merged.api_key.as_deref() }
    pub fn get_thinking_level(&self) -> &str {
        self.merged.thinking_level.as_deref().unwrap_or("off")
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    #[error("IO error: {0}")]
    Io(String),
    #[error("Parse error: {0}")]
    Parse(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_settings_merge() {
        let mut base = Settings::default();
        let other = Settings { model: Some("claude-opus-4-7".to_string()), ..Default::default() };
        base.merge(&other);
        assert_eq!(base.model.as_deref(), Some("claude-opus-4-7"));
    }

    #[test]
    fn test_settings_default() {
        let s = Settings::default();
        assert!(s.compaction.enabled);
        assert_eq!(s.retry.max_retries, 3);
    }
}
