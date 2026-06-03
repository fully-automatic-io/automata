use agent_core::auto_retry::RetrySettings;
use agent_core::harness::CompactionSettings;
use agent_core::queue::QueueMode;
use agent_core::types::{ThinkingBudgets, ThinkingLevel, Transport};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub model: Option<String>,
    pub provider: Option<String>,
    pub api_key: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub compaction: CompactionSettings,
    pub retry: RetrySettings,
    pub provider_retry: ProviderRetrySettings,
    pub working_directory: Option<String>,
    pub thinking_level: Option<ThinkingLevel>,
    pub thinking_budgets: ThinkingBudgets,
    pub transport: Transport,
    pub steering_mode: QueueMode,
    pub follow_up_mode: QueueMode,
    pub shell_path: Option<String>,
    pub shell_command_prefix: Option<String>,
    pub system_prompt: Option<String>,
    pub append_system_prompt: Vec<String>,
    pub extensions: Vec<String>,
    pub skills: Vec<String>,
    pub prompts: Vec<String>,
    pub images: ImageSettings,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            model: None,
            provider: None,
            api_key: None,
            max_tokens: None,
            temperature: None,
            compaction: CompactionSettings::default(),
            retry: RetrySettings::default(),
            provider_retry: ProviderRetrySettings::default(),
            working_directory: None,
            thinking_level: Some(ThinkingLevel::Medium),
            thinking_budgets: ThinkingBudgets::default(),
            transport: Transport::Auto,
            steering_mode: QueueMode::OneAtATime,
            follow_up_mode: QueueMode::OneAtATime,
            shell_path: None,
            shell_command_prefix: None,
            system_prompt: None,
            append_system_prompt: Vec::new(),
            extensions: Vec::new(),
            skills: Vec::new(),
            prompts: Vec::new(),
            images: ImageSettings::default(),
            extra: HashMap::new(),
        }
    }
}

impl Settings {
    pub fn merge_partial(&mut self, other: PartialSettings) {
        if other.model.is_some() {
            self.model = other.model;
        }
        if other.provider.is_some() {
            self.provider = other.provider;
        }
        if other.api_key.is_some() {
            self.api_key = other.api_key;
        }
        if other.max_tokens.is_some() {
            self.max_tokens = other.max_tokens;
        }
        if other.temperature.is_some() {
            self.temperature = other.temperature;
        }
        if let Some(compaction) = other.compaction {
            merge_compaction(&mut self.compaction, compaction);
        }
        if let Some(retry) = other.retry {
            merge_retry(&mut self.retry, retry);
        }
        if let Some(provider_retry) = other.provider_retry {
            self.provider_retry.merge(provider_retry);
        }
        if other.working_directory.is_some() {
            self.working_directory = other.working_directory;
        }
        if other.thinking_level.is_some() {
            self.thinking_level = other.thinking_level;
        }
        if let Some(thinking_budgets) = other.thinking_budgets {
            merge_thinking_budgets(&mut self.thinking_budgets, thinking_budgets);
        }
        if let Some(transport) = other.transport {
            self.transport = transport;
        }
        if let Some(mode) = other.steering_mode {
            self.steering_mode = mode;
        }
        if let Some(mode) = other.follow_up_mode {
            self.follow_up_mode = mode;
        }
        if other.shell_path.is_some() {
            self.shell_path = other.shell_path;
        }
        if other.shell_command_prefix.is_some() {
            self.shell_command_prefix = other.shell_command_prefix;
        }
        if other.system_prompt.is_some() {
            self.system_prompt = other.system_prompt;
        }
        if let Some(append_system_prompt) = other.append_system_prompt {
            self.append_system_prompt = append_system_prompt;
        }
        if let Some(extensions) = other.extensions {
            self.extensions = extensions;
        }
        if let Some(skills) = other.skills {
            self.skills = skills;
        }
        if let Some(prompts) = other.prompts {
            self.prompts = prompts;
        }
        if let Some(images) = other.images {
            self.images.merge(images);
        }
        self.extra.extend(other.extra);
    }

    /// Merge another complete settings object. This is mainly useful for tests
    /// and programmatic composition; file loading uses `PartialSettings` so
    /// absent fields do not overwrite earlier scopes.
    pub fn merge(&mut self, other: &Settings) {
        self.merge_partial(PartialSettings::from(other.clone()));
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(default)]
pub struct ProviderRetrySettings {
    pub timeout_ms: Option<u64>,
    pub max_retries: Option<u32>,
    pub max_retry_delay_ms: Option<u64>,
}

impl ProviderRetrySettings {
    fn merge(&mut self, other: PartialProviderRetrySettings) {
        if other.timeout_ms.is_some() {
            self.timeout_ms = other.timeout_ms;
        }
        if other.max_retries.is_some() {
            self.max_retries = other.max_retries;
        }
        if other.max_retry_delay_ms.is_some() {
            self.max_retry_delay_ms = other.max_retry_delay_ms;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ImageSettings {
    pub auto_resize: bool,
    pub block_images: bool,
}

impl Default for ImageSettings {
    fn default() -> Self {
        Self { auto_resize: true, block_images: false }
    }
}

impl ImageSettings {
    fn merge(&mut self, other: PartialImageSettings) {
        if let Some(auto_resize) = other.auto_resize {
            self.auto_resize = auto_resize;
        }
        if let Some(block_images) = other.block_images {
            self.block_images = block_images;
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct PartialSettings {
    pub model: Option<String>,
    pub provider: Option<String>,
    pub api_key: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub compaction: Option<PartialCompactionSettings>,
    pub retry: Option<PartialRetrySettings>,
    pub provider_retry: Option<PartialProviderRetrySettings>,
    pub working_directory: Option<String>,
    pub thinking_level: Option<ThinkingLevel>,
    pub thinking_budgets: Option<PartialThinkingBudgets>,
    pub transport: Option<Transport>,
    pub steering_mode: Option<QueueMode>,
    pub follow_up_mode: Option<QueueMode>,
    pub shell_path: Option<String>,
    pub shell_command_prefix: Option<String>,
    pub system_prompt: Option<String>,
    pub append_system_prompt: Option<Vec<String>>,
    pub extensions: Option<Vec<String>>,
    pub skills: Option<Vec<String>>,
    pub prompts: Option<Vec<String>>,
    pub images: Option<PartialImageSettings>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl From<Settings> for PartialSettings {
    fn from(settings: Settings) -> Self {
        Self {
            model: settings.model,
            provider: settings.provider,
            api_key: settings.api_key,
            max_tokens: settings.max_tokens,
            temperature: settings.temperature,
            compaction: Some(PartialCompactionSettings {
                enabled: Some(settings.compaction.enabled),
                reserve_tokens: Some(settings.compaction.reserve_tokens),
                keep_recent_tokens: Some(settings.compaction.keep_recent_tokens),
            }),
            retry: Some(PartialRetrySettings {
                enabled: Some(settings.retry.enabled),
                max_retries: Some(settings.retry.max_retries),
                base_delay_ms: Some(settings.retry.base_delay_ms),
            }),
            provider_retry: Some(PartialProviderRetrySettings {
                timeout_ms: settings.provider_retry.timeout_ms,
                max_retries: settings.provider_retry.max_retries,
                max_retry_delay_ms: settings.provider_retry.max_retry_delay_ms,
            }),
            working_directory: settings.working_directory,
            thinking_level: settings.thinking_level,
            thinking_budgets: Some(PartialThinkingBudgets {
                minimal: settings.thinking_budgets.minimal,
                low: settings.thinking_budgets.low,
                medium: settings.thinking_budgets.medium,
                high: settings.thinking_budgets.high,
                xhigh: settings.thinking_budgets.xhigh,
            }),
            transport: Some(settings.transport),
            steering_mode: Some(settings.steering_mode),
            follow_up_mode: Some(settings.follow_up_mode),
            shell_path: settings.shell_path,
            shell_command_prefix: settings.shell_command_prefix,
            system_prompt: settings.system_prompt,
            append_system_prompt: Some(settings.append_system_prompt),
            extensions: Some(settings.extensions),
            skills: Some(settings.skills),
            prompts: Some(settings.prompts),
            images: Some(PartialImageSettings {
                auto_resize: Some(settings.images.auto_resize),
                block_images: Some(settings.images.block_images),
            }),
            extra: settings.extra,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct PartialCompactionSettings {
    pub enabled: Option<bool>,
    pub reserve_tokens: Option<usize>,
    pub keep_recent_tokens: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct PartialRetrySettings {
    pub enabled: Option<bool>,
    pub max_retries: Option<u32>,
    pub base_delay_ms: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct PartialProviderRetrySettings {
    pub timeout_ms: Option<u64>,
    pub max_retries: Option<u32>,
    pub max_retry_delay_ms: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct PartialImageSettings {
    pub auto_resize: Option<bool>,
    pub block_images: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct PartialThinkingBudgets {
    pub minimal: Option<u32>,
    pub low: Option<u32>,
    pub medium: Option<u32>,
    pub high: Option<u32>,
    pub xhigh: Option<u32>,
}

fn merge_compaction(settings: &mut CompactionSettings, partial: PartialCompactionSettings) {
    if let Some(enabled) = partial.enabled {
        settings.enabled = enabled;
    }
    if let Some(reserve_tokens) = partial.reserve_tokens {
        settings.reserve_tokens = reserve_tokens;
    }
    if let Some(keep_recent_tokens) = partial.keep_recent_tokens {
        settings.keep_recent_tokens = keep_recent_tokens;
    }
}

fn merge_retry(settings: &mut RetrySettings, partial: PartialRetrySettings) {
    if let Some(enabled) = partial.enabled {
        settings.enabled = enabled;
    }
    if let Some(max_retries) = partial.max_retries {
        settings.max_retries = max_retries;
    }
    if let Some(base_delay_ms) = partial.base_delay_ms {
        settings.base_delay_ms = base_delay_ms;
    }
}

fn merge_thinking_budgets(settings: &mut ThinkingBudgets, partial: PartialThinkingBudgets) {
    if partial.minimal.is_some() {
        settings.minimal = partial.minimal;
    }
    if partial.low.is_some() {
        settings.low = partial.low;
    }
    if partial.medium.is_some() {
        settings.medium = partial.medium;
    }
    if partial.high.is_some() {
        settings.high = partial.high;
    }
    if partial.xhigh.is_some() {
        settings.xhigh = partial.xhigh;
    }
}

#[derive(Debug, Clone)]
pub struct SettingsManager {
    global_path: PathBuf,
    project_path: Option<PathBuf>,
    merged: Settings,
    errors: Vec<SettingsLoadError>,
}

impl SettingsManager {
    pub fn new(global_path: PathBuf, project_path: Option<PathBuf>) -> Self {
        Self {
            global_path,
            project_path,
            merged: Settings::default(),
            errors: Vec::new(),
        }
    }

    pub fn with_defaults(cwd: &Path) -> Self {
        let global = dirs_next::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".automata/agent/settings.json");
        let project = cwd.join(".automata/settings.json");
        Self::new(global, Some(project))
    }

    pub fn from_settings(settings: Settings) -> Self {
        Self {
            global_path: PathBuf::new(),
            project_path: None,
            merged: settings,
            errors: Vec::new(),
        }
    }

    pub async fn load(&mut self) -> Result<(), SettingsError> {
        self.errors.clear();
        let mut settings = Settings::default();
        let global_path = self.global_path.clone();
        if let Some(partial) = self.read_partial(&global_path, SettingsScope::Global).await? {
            settings.merge_partial(partial);
        }
        if let Some(project_path) = self.project_path.clone()
            && let Some(partial) = self.read_partial(&project_path, SettingsScope::Project).await?
        {
            settings.merge_partial(partial);
        }
        self.merged = settings;
        Ok(())
    }

    async fn read_partial(
        &mut self,
        path: &Path,
        scope: SettingsScope,
    ) -> Result<Option<PartialSettings>, SettingsError> {
        match fs::read_to_string(path).await {
            Ok(content) => match serde_json::from_str::<PartialSettings>(&content) {
                Ok(settings) => Ok(Some(settings)),
                Err(err) => {
                    let err = SettingsLoadError {
                        scope,
                        path: path.to_path_buf(),
                        message: err.to_string(),
                    };
                    self.errors.push(err.clone());
                    Err(SettingsError::Parse(err.message))
                }
            },
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => {
                let err = SettingsLoadError {
                    scope,
                    path: path.to_path_buf(),
                    message: err.to_string(),
                };
                self.errors.push(err.clone());
                Err(SettingsError::Io(err.message))
            }
        }
    }

    pub async fn save_global(&self, settings: &Settings) -> Result<(), SettingsError> {
        write_settings(&self.global_path, settings).await
    }

    pub async fn save_project(&self, settings: &Settings) -> Result<(), SettingsError> {
        let path = self.project_path.as_ref().ok_or(SettingsError::NoProjectPath)?;
        write_settings(path, settings).await
    }

    pub fn get(&self) -> &Settings {
        &self.merged
    }
    pub fn get_model(&self) -> Option<&str> {
        self.merged.model.as_deref()
    }
    pub fn get_provider(&self) -> Option<&str> {
        self.merged.provider.as_deref()
    }
    pub fn get_api_key(&self) -> Option<&str> {
        self.merged.api_key.as_deref()
    }
    pub fn get_thinking_level(&self) -> ThinkingLevel {
        self.merged.thinking_level.unwrap_or(ThinkingLevel::Off)
    }
    pub fn get_transport(&self) -> Transport {
        self.merged.transport
    }
    pub fn get_steering_mode(&self) -> QueueMode {
        self.merged.steering_mode
    }
    pub fn get_follow_up_mode(&self) -> QueueMode {
        self.merged.follow_up_mode
    }
    pub fn get_retry_settings(&self) -> RetrySettings {
        self.merged.retry
    }
    pub fn get_provider_retry_settings(&self) -> &ProviderRetrySettings {
        &self.merged.provider_retry
    }
    pub fn get_compaction_settings(&self) -> &CompactionSettings {
        &self.merged.compaction
    }
    pub fn get_shell_path(&self) -> Option<&str> {
        self.merged.shell_path.as_deref()
    }
    pub fn get_shell_command_prefix(&self) -> Option<&str> {
        self.merged.shell_command_prefix.as_deref()
    }
    pub fn get_system_prompt(&self) -> Option<&str> {
        self.merged.system_prompt.as_deref()
    }
    pub fn get_append_system_prompt(&self) -> &[String] {
        &self.merged.append_system_prompt
    }
    pub fn get_image_auto_resize(&self) -> bool {
        self.merged.images.auto_resize
    }
    pub fn get_block_images(&self) -> bool {
        self.merged.images.block_images
    }
    pub fn get_thinking_budgets(&self) -> &ThinkingBudgets {
        &self.merged.thinking_budgets
    }

    pub fn drain_errors(&mut self) -> Vec<SettingsLoadError> {
        std::mem::take(&mut self.errors)
    }
}

async fn write_settings(path: &Path, settings: &Settings) -> Result<(), SettingsError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .await
            .map_err(|err| SettingsError::Io(err.to_string()))?;
    }
    let content = serde_json::to_string_pretty(settings)
        .map_err(|err| SettingsError::Parse(err.to_string()))?;
    fs::write(path, content).await.map_err(|err| SettingsError::Io(err.to_string()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsScope {
    Global,
    Project,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingsLoadError {
    pub scope: SettingsScope,
    pub path: PathBuf,
    pub message: String,
}

#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    #[error("IO error: {0}")]
    Io(String),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("project settings path is not configured")]
    NoProjectPath,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_settings_merge() {
        let mut base = Settings::default();
        base.compaction.reserve_tokens = 100;
        let partial: PartialSettings =
            serde_json::from_str(r#"{"model":"claude-opus-4-7"}"#).unwrap();
        base.merge_partial(partial);
        assert_eq!(base.model.as_deref(), Some("claude-opus-4-7"));
        assert_eq!(base.compaction.reserve_tokens, 100);
    }

    #[test]
    fn test_settings_default() {
        let s = Settings::default();
        assert!(s.compaction.enabled);
        assert_eq!(s.retry.max_retries, 3);
        assert_eq!(s.transport, Transport::Auto);
    }

    #[tokio::test]
    async fn test_settings_manager_deep_merges_nested_scopes() {
        let dir = tempfile::TempDir::new().unwrap();
        let global = dir.path().join("global.json");
        let project = dir.path().join("project.json");
        std::fs::write(
            &global,
            r#"{"compaction":{"enabled":false,"reserve_tokens":123},"extensions":["a"]}"#,
        )
        .unwrap();
        std::fs::write(&project, r#"{"compaction":{"keep_recent_tokens":456},"model":"m"}"#)
            .unwrap();
        let mut manager = SettingsManager::new(global, Some(project));
        manager.load().await.unwrap();
        let settings = manager.get();
        assert_eq!(settings.model.as_deref(), Some("m"));
        assert!(!settings.compaction.enabled);
        assert_eq!(settings.compaction.reserve_tokens, 123);
        assert_eq!(settings.compaction.keep_recent_tokens, 456);
        assert_eq!(settings.extensions, vec!["a".to_string()]);
    }
}
