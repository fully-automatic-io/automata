use agent_core::types::ThinkingLevel;
use serde::{Deserialize, Serialize};

// ── Defaults ──────────────────────────────────────────────────────────────────

pub const DEFAULT_THINKING_LEVEL: ThinkingLevel = ThinkingLevel::Medium;
pub const DEFAULT_AGENT_DIR: &str = ".automata";
pub const DEFAULT_SESSIONS_DIR: &str = ".automata/sessions";
pub const DEFAULT_COMPACTION_RESERVE_TOKENS: u64 = 16384;
pub const DEFAULT_COMPACTION_KEEP_RECENT_TOKENS: u64 = 20000;

// ── Model resolver ────────────────────────────────────────────────────────────

pub fn default_model_for_provider(provider: &str) -> Option<&'static str> {
    match provider {
        "anthropic" => Some("claude-sonnet-4-6"),
        "openai" => Some("gpt-4o"),
        _ => None,
    }
}

/// Clamp a thinking level to what the model can do: models without reasoning
/// always collapse to [`ThinkingLevel::Off`].
pub fn clamp_thinking_level(level: ThinkingLevel, model_supports_reasoning: bool) -> ThinkingLevel {
    if model_supports_reasoning {
        level
    } else {
        ThinkingLevel::Off
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopedModel {
    pub provider: String,
    pub model_id: String,
    pub api_key_source: Option<String>,
    pub base_url: Option<String>,
}

impl ScopedModel {
    pub fn new(provider: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model_id: model_id.into(),
            api_key_source: None,
            base_url: None,
        }
    }
}
