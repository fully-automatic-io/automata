use serde::{Deserialize, Serialize};

// ── Defaults ──────────────────────────────────────────────────────────────────

pub const DEFAULT_THINKING_LEVEL: &str = "medium";
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

pub fn clamp_thinking_level(level: &str, model_supports_reasoning: bool) -> &str {
    if !model_supports_reasoning { return "off"; }
    match level {
        "off" | "minimal" | "low" | "medium" | "high" | "xhigh" => level,
        _ => "medium",
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
