use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::uuid::{now_iso, uuidv7};
use crate::types::{AgentMessage, ThinkingLevel};

// ============================================================================
// Session tree entry types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SessionTreeEntry {
    #[serde(rename = "message")]
    Message {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        message: AgentMessage,
    },
    #[serde(rename = "thinking_level_change")]
    ThinkingLevelChange {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(rename = "thinkingLevel")]
        thinking_level: ThinkingLevel,
    },
    #[serde(rename = "model_change")]
    ModelChange {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        provider: String,
        #[serde(rename = "modelId")]
        model_id: String,
    },
    #[serde(rename = "compaction")]
    Compaction {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        summary: String,
        #[serde(rename = "firstKeptEntryId")]
        first_kept_entry_id: String,
        #[serde(rename = "tokensBefore")]
        tokens_before: u64,
        /// Compaction-extension-defined extras (e.g. read/modified file lists).
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<serde_json::Value>,
        #[serde(rename = "fromHook", skip_serializing_if = "Option::is_none")]
        from_hook: Option<bool>,
    },
    #[serde(rename = "branch_summary")]
    BranchSummary {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(rename = "fromId")]
        from_id: String,
        summary: String,
        /// Branch-summary extension extras (e.g. read/modified file lists).
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<serde_json::Value>,
        #[serde(rename = "fromHook", skip_serializing_if = "Option::is_none")]
        from_hook: Option<bool>,
    },
    #[serde(rename = "custom")]
    Custom {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        /// Plugin-defined entry kind (extensible).
        #[serde(rename = "customType")]
        custom_type: String,
        /// Plugin-defined payload; shape is set by `custom_type`.
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<serde_json::Value>,
    },
    #[serde(rename = "custom_message")]
    CustomMessage {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        /// Plugin-defined message kind (extensible).
        #[serde(rename = "customType")]
        custom_type: String,
        /// Plugin-defined payload; shape is set by `custom_type`.
        content: serde_json::Value,
        display: bool,
        /// Plugin-defined extra metadata.
        #[serde(skip_serializing_if = "Option::is_none")]
        details: Option<serde_json::Value>,
    },
    #[serde(rename = "label")]
    Label {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(rename = "targetId")]
        target_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    #[serde(rename = "session_info")]
    SessionInfo {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    #[serde(rename = "leaf")]
    Leaf {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(rename = "targetId")]
        target_id: Option<String>,
    },
}

impl SessionTreeEntry {
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
            Self::Leaf { id, .. } => id,
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
            Self::Leaf { parent_id, .. } => parent_id.as_deref(),
        }
    }

    pub fn timestamp(&self) -> &str {
        match self {
            Self::Message { timestamp, .. }
            | Self::ThinkingLevelChange { timestamp, .. }
            | Self::ModelChange { timestamp, .. }
            | Self::Compaction { timestamp, .. }
            | Self::BranchSummary { timestamp, .. }
            | Self::Custom { timestamp, .. }
            | Self::CustomMessage { timestamp, .. }
            | Self::Label { timestamp, .. }
            | Self::SessionInfo { timestamp, .. }
            | Self::Leaf { timestamp, .. } => timestamp,
        }
    }

    fn leaf_id_after(&self) -> Option<Option<String>> {
        match self {
            Self::Leaf { target_id, .. } => Some(target_id.clone()),
            _ => None,
        }
    }
}

// ============================================================================
// Session metadata
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub id: String,
    #[serde(rename = "createdAt")]
    pub created_at: String,
}

// ============================================================================
// Session context (built from entries)
// ============================================================================

#[derive(Debug, Clone)]
pub struct SessionContext {
    pub messages: Vec<AgentMessage>,
    pub thinking_level: ThinkingLevel,
    pub model: Option<ModelRef>,
}

#[derive(Debug, Clone)]
pub struct ModelRef {
    pub provider: String,
    pub model_id: String,
}

pub fn build_session_context(path_entries: &[SessionTreeEntry]) -> SessionContext {
    let mut thinking_level = ThinkingLevel::Off;
    let mut model: Option<ModelRef> = None;
    let mut compaction_idx: Option<usize> = None;
    let mut compaction_summary = String::new();
    let mut compaction_tokens_before = 0u64;
    let mut compaction_first_kept_id = String::new();
    let mut compaction_timestamp = String::new();

    for (i, entry) in path_entries.iter().enumerate() {
        match entry {
            SessionTreeEntry::ThinkingLevelChange { thinking_level: tl, .. } => {
                thinking_level = *tl;
            }
            SessionTreeEntry::ModelChange { provider, model_id, .. } => {
                model = Some(ModelRef { provider: provider.clone(), model_id: model_id.clone() });
            }
            SessionTreeEntry::Message { message, .. } => {
                if let AgentMessage::Assistant { provider, model: model_id, .. } = message {
                    model = Some(ModelRef { provider: provider.clone(), model_id: model_id.clone() });
                }
            }
            SessionTreeEntry::Compaction { summary, tokens_before, first_kept_entry_id, timestamp, .. } => {
                compaction_idx = Some(i);
                compaction_summary = summary.clone();
                compaction_tokens_before = *tokens_before;
                compaction_first_kept_id = first_kept_entry_id.clone();
                compaction_timestamp = timestamp.clone();
            }
            _ => {}
        }
    }

    let mut messages: Vec<AgentMessage> = vec![];

    if let Some(cidx) = compaction_idx {
        messages.push(AgentMessage::CompactionSummary {
            summary: compaction_summary,
            tokens_before: compaction_tokens_before,
            timestamp: parse_timestamp_millis(&compaction_timestamp),
        });

        let first_kept_pos = path_entries[..cidx]
            .iter()
            .position(|e| e.id() == compaction_first_kept_id)
            .unwrap_or(cidx);

        for entry in &path_entries[first_kept_pos..cidx] {
            append_message_from_entry(entry, &mut messages);
        }
        for entry in &path_entries[cidx + 1..] {
            append_message_from_entry(entry, &mut messages);
        }
    } else {
        for entry in path_entries {
            append_message_from_entry(entry, &mut messages);
        }
    }

    SessionContext { messages, thinking_level, model }
}

fn parse_timestamp_millis(s: &str) -> u64 {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.timestamp_millis().max(0) as u64)
        .unwrap_or(0)
}

fn append_message_from_entry(entry: &SessionTreeEntry, messages: &mut Vec<AgentMessage>) {
    match entry {
        SessionTreeEntry::Message { message, .. } => messages.push(message.clone()),
        SessionTreeEntry::CustomMessage { custom_type, content, display, details, timestamp, .. } => {
            messages.push(AgentMessage::Custom {
                custom_type: custom_type.clone(),
                content: content.clone(),
                display: *display,
                details: details.clone(),
                timestamp: parse_timestamp_millis(timestamp),
            });
        }
        SessionTreeEntry::BranchSummary { summary, from_id, timestamp, .. } if !summary.is_empty() => {
            messages.push(AgentMessage::BranchSummary {
                summary: summary.clone(),
                from_id: from_id.clone(),
                timestamp: parse_timestamp_millis(timestamp),
            });
        }
        _ => {}
    }
}

// ============================================================================
// SessionStorage trait
// ============================================================================

#[async_trait::async_trait]
pub trait SessionStorage: Send + Sync {
    async fn get_metadata(&self) -> SessionMetadata;
    async fn get_leaf_id(&self) -> Option<String>;
    async fn set_leaf_id(&mut self, leaf_id: Option<String>) -> Result<(), SessionError>;
    async fn create_entry_id(&self) -> String;
    async fn append_entry(&mut self, entry: SessionTreeEntry) -> Result<(), SessionError>;
    async fn get_entry(&self, id: &str) -> Option<SessionTreeEntry>;
    async fn find_entries_by_type(&self, entry_type: &str) -> Vec<SessionTreeEntry>;
    async fn get_label(&self, id: &str) -> Option<String>;
    async fn get_path_to_root(&self, leaf_id: Option<&str>) -> Result<Vec<SessionTreeEntry>, SessionError>;
    async fn get_entries(&self) -> Vec<SessionTreeEntry>;
}

// ============================================================================
// SessionError
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("Session not found: {0}")]
    NotFound(String),
    #[error("Invalid session: {0}")]
    InvalidSession(String),
    #[error("Invalid entry: {0}")]
    InvalidEntry(String),
    #[error("Invalid fork target: {0}")]
    InvalidForkTarget(String),
    #[error("Storage error: {0}")]
    Storage(String),
    #[error("Unknown error: {0}")]
    Unknown(String),
}

// ============================================================================
// Session wrapper
// ============================================================================

pub struct Session {
    storage: Box<dyn SessionStorage>,
}

impl Session {
    pub fn new(storage: Box<dyn SessionStorage>) -> Self {
        Self { storage }
    }

    pub fn storage(&self) -> &dyn SessionStorage {
        self.storage.as_ref()
    }

    pub fn storage_mut(&mut self) -> &mut dyn SessionStorage {
        self.storage.as_mut()
    }

    pub async fn get_metadata(&self) -> SessionMetadata {
        self.storage.get_metadata().await
    }

    pub async fn get_leaf_id(&self) -> Option<String> {
        self.storage.get_leaf_id().await
    }

    pub async fn get_branch(&self) -> Result<Vec<SessionTreeEntry>, SessionError> {
        let leaf_id = self.storage.get_leaf_id().await;
        self.storage.get_path_to_root(leaf_id.as_deref()).await
    }

    pub async fn get_branch_from(&self, entry_id: &str) -> Result<Vec<SessionTreeEntry>, SessionError> {
        self.storage.get_path_to_root(Some(entry_id)).await
    }

    pub async fn build_context(&self) -> Result<SessionContext, SessionError> {
        let branch = self.get_branch().await?;
        Ok(build_session_context(&branch))
    }

    pub async fn get_session_name(&self) -> Option<String> {
        let entries = self.storage.find_entries_by_type("session_info").await;
        entries.last().and_then(|e| {
            if let SessionTreeEntry::SessionInfo { name, .. } = e {
                name.as_ref().map(|n| n.trim().to_string()).filter(|n| !n.is_empty())
            } else {
                None
            }
        })
    }

    pub async fn append_message(&mut self, message: AgentMessage) -> Result<String, SessionError> {
        let id = self.storage.create_entry_id().await;
        let parent_id = self.storage.get_leaf_id().await;
        self.storage.append_entry(SessionTreeEntry::Message {
            id: id.clone(),
            parent_id,
            timestamp: now_iso(),
            message,
        }).await?;
        Ok(id)
    }

    pub async fn append_thinking_level_change(&mut self, level: ThinkingLevel) -> Result<String, SessionError> {
        let id = self.storage.create_entry_id().await;
        let parent_id = self.storage.get_leaf_id().await;
        self.storage.append_entry(SessionTreeEntry::ThinkingLevelChange {
            id: id.clone(),
            parent_id,
            timestamp: now_iso(),
            thinking_level: level,
        }).await?;
        Ok(id)
    }

    pub async fn append_model_change(&mut self, provider: &str, model_id: &str) -> Result<String, SessionError> {
        let id = self.storage.create_entry_id().await;
        let parent_id = self.storage.get_leaf_id().await;
        self.storage.append_entry(SessionTreeEntry::ModelChange {
            id: id.clone(),
            parent_id,
            timestamp: now_iso(),
            provider: provider.to_string(),
            model_id: model_id.to_string(),
        }).await?;
        Ok(id)
    }

    pub async fn append_compaction(
        &mut self,
        summary: &str,
        first_kept_entry_id: &str,
        tokens_before: u64,
        details: Option<serde_json::Value>,
        from_hook: Option<bool>,
    ) -> Result<String, SessionError> {
        let id = self.storage.create_entry_id().await;
        let parent_id = self.storage.get_leaf_id().await;
        self.storage.append_entry(SessionTreeEntry::Compaction {
            id: id.clone(),
            parent_id,
            timestamp: now_iso(),
            summary: summary.to_string(),
            first_kept_entry_id: first_kept_entry_id.to_string(),
            tokens_before,
            details,
            from_hook,
        }).await?;
        Ok(id)
    }

    pub async fn append_branch_summary(
        &mut self,
        from_id: &str,
        summary: &str,
        details: Option<serde_json::Value>,
        from_hook: Option<bool>,
    ) -> Result<String, SessionError> {
        let id = self.storage.create_entry_id().await;
        let parent_id = self.storage.get_leaf_id().await;
        self.storage.append_entry(SessionTreeEntry::BranchSummary {
            id: id.clone(),
            parent_id,
            timestamp: now_iso(),
            from_id: from_id.to_string(),
            summary: summary.to_string(),
            details,
            from_hook,
        }).await?;
        Ok(id)
    }

    pub async fn append_session_name(&mut self, name: &str) -> Result<String, SessionError> {
        let id = self.storage.create_entry_id().await;
        let parent_id = self.storage.get_leaf_id().await;
        self.storage.append_entry(SessionTreeEntry::SessionInfo {
            id: id.clone(),
            parent_id,
            timestamp: now_iso(),
            name: Some(name.trim().to_string()),
        }).await?;
        Ok(id)
    }

    pub async fn move_to(
        &mut self,
        entry_id: Option<&str>,
        summary: Option<BranchSummaryOptions>,
    ) -> Result<Option<String>, SessionError> {
        if let Some(id) = entry_id {
            if self.storage.get_entry(id).await.is_none() {
                return Err(SessionError::NotFound(format!("Entry {} not found", id)));
            }
        }
        self.storage.set_leaf_id(entry_id.map(|s| s.to_string())).await?;
        if let Some(opts) = summary {
            let sid = self.storage.create_entry_id().await;
            self.storage.append_entry(SessionTreeEntry::BranchSummary {
                id: sid.clone(),
                parent_id: entry_id.map(|s| s.to_string()),
                timestamp: now_iso(),
                from_id: entry_id.unwrap_or("root").to_string(),
                summary: opts.summary,
                details: opts.details,
                from_hook: opts.from_hook,
            }).await?;
            return Ok(Some(sid));
        }
        Ok(None)
    }
}

pub struct BranchSummaryOptions {
    pub summary: String,
    /// Branch-summary extension extras (e.g. read/modified file lists).
    pub details: Option<serde_json::Value>,
    pub from_hook: Option<bool>,
}

// ============================================================================
// InMemorySessionStorage
// ============================================================================

pub struct InMemorySessionStorage {
    metadata: SessionMetadata,
    entries: Vec<SessionTreeEntry>,
    by_id: HashMap<String, usize>,
    labels_by_id: HashMap<String, String>,
    leaf_id: Option<String>,
}

impl InMemorySessionStorage {
    pub fn new(metadata: Option<SessionMetadata>) -> Self {
        let metadata = metadata.unwrap_or_else(|| SessionMetadata {
            id: uuidv7(),
            created_at: now_iso(),
        });
        Self {
            metadata,
            entries: vec![],
            by_id: HashMap::new(),
            labels_by_id: HashMap::new(),
            leaf_id: None,
        }
    }

    pub fn with_entries(metadata: Option<SessionMetadata>, entries: Vec<SessionTreeEntry>) -> Self {
        let mut s = Self::new(metadata);
        let mut leaf_id = None;
        for entry in entries {
            if let SessionTreeEntry::Leaf { target_id, .. } = &entry {
                leaf_id = target_id.clone();
            } else {
                leaf_id = Some(entry.id().to_string());
            }
            let idx = s.entries.len();
            s.by_id.insert(entry.id().to_string(), idx);
            update_label_cache(&mut s.labels_by_id, &entry);
            s.entries.push(entry);
        }
        s.leaf_id = leaf_id;
        s
    }

    pub fn snapshot_entries(&self) -> Vec<SessionTreeEntry> {
        self.entries.clone()
    }
}

fn update_label_cache(labels: &mut HashMap<String, String>, entry: &SessionTreeEntry) {
    if let SessionTreeEntry::Label { target_id, label, .. } = entry {
        match label.as_ref().map(|l| l.trim()) {
            Some(l) if !l.is_empty() => { labels.insert(target_id.clone(), l.to_string()); }
            _ => { labels.remove(target_id); }
        }
    }
}

fn generate_entry_id(by_id: &HashMap<String, usize>) -> String {
    for _ in 0..100 {
        let id = uuidv7()[..8].to_string();
        if !by_id.contains_key(&id) {
            return id;
        }
    }
    uuidv7()
}

#[async_trait::async_trait]
impl SessionStorage for InMemorySessionStorage {
    async fn get_metadata(&self) -> SessionMetadata {
        self.metadata.clone()
    }

    async fn get_leaf_id(&self) -> Option<String> {
        self.leaf_id.clone()
    }

    async fn set_leaf_id(&mut self, leaf_id: Option<String>) -> Result<(), SessionError> {
        if let Some(ref id) = leaf_id {
            if !self.by_id.contains_key(id) {
                return Err(SessionError::NotFound(format!("Entry {} not found", id)));
            }
        }
        let entry_id = generate_entry_id(&self.by_id);
        let entry = SessionTreeEntry::Leaf {
            id: entry_id.clone(),
            parent_id: self.leaf_id.clone(),
            timestamp: now_iso(),
            target_id: leaf_id.clone(),
        };
        let idx = self.entries.len();
        self.by_id.insert(entry_id, idx);
        self.entries.push(entry);
        self.leaf_id = leaf_id;
        Ok(())
    }

    async fn create_entry_id(&self) -> String {
        generate_entry_id(&self.by_id)
    }

    async fn append_entry(&mut self, entry: SessionTreeEntry) -> Result<(), SessionError> {
        let leaf_after = entry.leaf_id_after();
        let idx = self.entries.len();
        self.by_id.insert(entry.id().to_string(), idx);
        update_label_cache(&mut self.labels_by_id, &entry);
        if let Some(new_leaf) = leaf_after {
            self.leaf_id = new_leaf;
        } else {
            self.leaf_id = Some(entry.id().to_string());
        }
        self.entries.push(entry);
        Ok(())
    }

    async fn get_entry(&self, id: &str) -> Option<SessionTreeEntry> {
        self.by_id.get(id).map(|&idx| self.entries[idx].clone())
    }

    async fn find_entries_by_type(&self, entry_type: &str) -> Vec<SessionTreeEntry> {
        self.entries.iter()
            .filter(|e| match (entry_type, e) {
                ("message", SessionTreeEntry::Message { .. }) => true,
                ("compaction", SessionTreeEntry::Compaction { .. }) => true,
                ("branch_summary", SessionTreeEntry::BranchSummary { .. }) => true,
                ("session_info", SessionTreeEntry::SessionInfo { .. }) => true,
                ("label", SessionTreeEntry::Label { .. }) => true,
                ("leaf", SessionTreeEntry::Leaf { .. }) => true,
                ("custom", SessionTreeEntry::Custom { .. }) => true,
                ("custom_message", SessionTreeEntry::CustomMessage { .. }) => true,
                ("thinking_level_change", SessionTreeEntry::ThinkingLevelChange { .. }) => true,
                ("model_change", SessionTreeEntry::ModelChange { .. }) => true,
                _ => false,
            })
            .cloned()
            .collect()
    }

    async fn get_label(&self, id: &str) -> Option<String> {
        self.labels_by_id.get(id).cloned()
    }

    async fn get_path_to_root(&self, leaf_id: Option<&str>) -> Result<Vec<SessionTreeEntry>, SessionError> {
        let Some(leaf_id) = leaf_id else { return Ok(vec![]); };
        let mut path = vec![];
        let mut current_id = leaf_id.to_string();
        loop {
            let idx = self.by_id.get(&current_id)
                .ok_or_else(|| SessionError::NotFound(format!("Entry {} not found", current_id)))?;
            let entry = &self.entries[*idx];
            path.push(entry.clone());
            match entry.parent_id() {
                Some(pid) => current_id = pid.to_string(),
                None => break,
            }
        }
        path.reverse();
        Ok(path)
    }

    async fn get_entries(&self) -> Vec<SessionTreeEntry> {
        self.entries.clone()
    }
}

// ============================================================================
// InMemorySessionRepo
// ============================================================================

pub struct InMemorySessionRepo {
    sessions: HashMap<String, Vec<SessionTreeEntry>>,
    metadata: HashMap<String, SessionMetadata>,
}

impl InMemorySessionRepo {
    pub fn new() -> Self {
        Self { sessions: HashMap::new(), metadata: HashMap::new() }
    }

    pub fn create_session(&mut self, id: Option<String>) -> Session {
        let meta = SessionMetadata {
            id: id.unwrap_or_else(uuidv7),
            created_at: now_iso(),
        };
        self.sessions.insert(meta.id.clone(), vec![]);
        self.metadata.insert(meta.id.clone(), meta.clone());
        Session::new(Box::new(InMemorySessionStorage::new(Some(meta))))
    }

    pub fn open_session(&self, id: &str) -> Option<Session> {
        let meta = self.metadata.get(id)?.clone();
        let entries = self.sessions.get(id)?.clone();
        Some(Session::new(Box::new(InMemorySessionStorage::with_entries(Some(meta), entries))))
    }
}

impl Default for InMemorySessionRepo {
    fn default() -> Self { Self::new() }
}
