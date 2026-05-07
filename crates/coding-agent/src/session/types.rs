use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::messages::AgentMessage;

/// Session file format version
pub const CURRENT_SESSION_VERSION: u32 = 3;

/// Session header - first line in JSONL file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionHeader {
    #[serde(rename = "type")]
    pub entry_type: String, // Always "session"
    pub version: u32,
    pub id: String,
    pub timestamp: String,
    pub cwd: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session: Option<String>,
}

impl SessionHeader {
    pub fn new(id: String, cwd: String, parent_session: Option<String>) -> Self {
        Self {
            entry_type: "session".to_string(),
            version: CURRENT_SESSION_VERSION,
            id,
            timestamp: Utc::now().to_rfc3339(),
            cwd,
            parent_session,
        }
    }
}

/// Base fields for all session entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEntryBase {
    #[serde(rename = "type")]
    pub entry_type: String,
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    pub timestamp: String,
}

/// All possible session entry types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SessionEntry {
    #[serde(rename = "message")]
    Message {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        timestamp: String,
        message: AgentMessage,
    },

    #[serde(rename = "thinking_level_change")]
    ThinkingLevelChange {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        timestamp: String,
        thinking_level: String,
    },

    #[serde(rename = "model_change")]
    ModelChange {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        timestamp: String,
        provider: String,
        model_id: String,
    },

    #[serde(rename = "compaction")]
    Compaction {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        timestamp: String,
        summary: String,
        first_kept_entry_id: String,
        tokens_before: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        from_hook: Option<bool>,
    },

    #[serde(rename = "branch_summary")]
    BranchSummary {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        timestamp: String,
        from_id: String,
        summary: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        from_hook: Option<bool>,
    },

    #[serde(rename = "custom")]
    Custom {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        timestamp: String,
        custom_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<serde_json::Value>,
    },

    #[serde(rename = "custom_message")]
    CustomMessage {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        timestamp: String,
        custom_type: String,
        content: serde_json::Value, // String or array of content blocks
        display: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<serde_json::Value>,
    },

    #[serde(rename = "label")]
    Label {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        timestamp: String,
        target_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },

    #[serde(rename = "session_info")]
    SessionInfo {
        id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
}

impl SessionEntry {
    pub fn id(&self) -> &str {
        match self {
            Self::Message { id, .. } => id,
            Self::ThinkingLevelChange { id, .. } => id,
            Self::ModelChange { id, .. } => id,
            Self::Compaction { id, .. } => id,
            Self::BranchSummary { id, .. } => id,
            Self::Custom { id, .. } => id,
            Self::CustomMessage { id, .. } => id,
            Self::Label { id, .. } => id,
            Self::SessionInfo { id, .. } => id,
        }
    }

    pub fn parent_id(&self) -> Option<&str> {
        match self {
            Self::Message { parent_id, .. } => parent_id.as_deref(),
            Self::ThinkingLevelChange { parent_id, .. } => parent_id.as_deref(),
            Self::ModelChange { parent_id, .. } => parent_id.as_deref(),
            Self::Compaction { parent_id, .. } => parent_id.as_deref(),
            Self::BranchSummary { parent_id, .. } => parent_id.as_deref(),
            Self::Custom { parent_id, .. } => parent_id.as_deref(),
            Self::CustomMessage { parent_id, .. } => parent_id.as_deref(),
            Self::Label { parent_id, .. } => parent_id.as_deref(),
            Self::SessionInfo { parent_id, .. } => parent_id.as_deref(),
        }
    }

    pub fn timestamp(&self) -> &str {
        match self {
            Self::Message { timestamp, .. } => timestamp,
            Self::ThinkingLevelChange { timestamp, .. } => timestamp,
            Self::ModelChange { timestamp, .. } => timestamp,
            Self::Compaction { timestamp, .. } => timestamp,
            Self::BranchSummary { timestamp, .. } => timestamp,
            Self::Custom { timestamp, .. } => timestamp,
            Self::CustomMessage { timestamp, .. } => timestamp,
            Self::Label { timestamp, .. } => timestamp,
            Self::SessionInfo { timestamp, .. } => timestamp,
        }
    }

    pub fn entry_type(&self) -> &str {
        match self {
            Self::Message { .. } => "message",
            Self::ThinkingLevelChange { .. } => "thinking_level_change",
            Self::ModelChange { .. } => "model_change",
            Self::Compaction { .. } => "compaction",
            Self::BranchSummary { .. } => "branch_summary",
            Self::Custom { .. } => "custom",
            Self::CustomMessage { .. } => "custom_message",
            Self::Label { .. } => "label",
            Self::SessionInfo { .. } => "session_info",
        }
    }
}

/// Session context for LLM
#[derive(Debug, Clone)]
pub struct SessionContext {
    pub messages: Vec<AgentMessage>,
    pub thinking_level: String,
    pub model: Option<ModelInfo>,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub provider: String,
    pub model_id: String,
}

/// Tree node for visualization
#[derive(Debug, Clone, Serialize)]
pub struct SessionTreeNode {
    pub entry: SessionEntry,
    pub label: Option<String>,
    pub children: Vec<SessionTreeNode>,
}

/// Session metadata for listing
#[derive(Debug, Clone, Serialize)]
pub struct SessionInfo {
    pub path: String,
    pub id: String,
    pub cwd: String,
    pub name: Option<String>,
    pub parent_session_path: Option<String>,
    pub created: DateTime<Utc>,
    pub modified: DateTime<Utc>,
    pub message_count: usize,
    pub first_message: Option<String>,
}

/// CWD validation issue
#[derive(Debug, Clone)]
pub enum SessionCwdIssue {
    Missing { stored_cwd: String },
    Inaccessible { stored_cwd: String, error: String },
}

/// File entry wrapper (for internal use)
#[derive(Debug, Clone)]
pub(crate) struct FileEntry {
    pub entry: SessionEntry,
    pub line_number: usize,
}
