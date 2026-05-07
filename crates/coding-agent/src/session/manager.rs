//
// Manages conversation sessions as append-only trees stored in JSONL files.
// Each entry has an id and parentId forming a tree structure.
// The "leaf" pointer tracks the current position.

use crate::messages::{
    create_branch_summary_message, create_compaction_summary_message, create_custom_message,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ============================================================================
// Constants
// ============================================================================

pub const CURRENT_SESSION_VERSION: u32 = 3;

// ============================================================================
// Session Header
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionHeader {
    #[serde(rename = "type")]
    pub entry_type: String, // "session"
    #[serde(default)]
    pub version: Option<u32>,
    pub id: String,
    pub timestamp: String,
    pub cwd: String,
    #[serde(rename = "parentSession", skip_serializing_if = "Option::is_none")]
    pub parent_session: Option<String>,
}

// ============================================================================
// Session Entries
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEntryBase {
    #[serde(rename = "type")]
    pub entry_type: String,
    pub id: String,
    #[serde(rename = "parentId")]
    pub parent_id: Option<String>,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessageEntry {
    #[serde(rename = "type")]
    pub entry_type: String, // "message"
    pub id: String,
    #[serde(rename = "parentId")]
    pub parent_id: Option<String>,
    pub timestamp: String,
    pub message: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingLevelChangeEntry {
    #[serde(rename = "type")]
    pub entry_type: String, // "thinking_level_change"
    pub id: String,
    #[serde(rename = "parentId")]
    pub parent_id: Option<String>,
    pub timestamp: String,
    #[serde(rename = "thinkingLevel")]
    pub thinking_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelChangeEntry {
    #[serde(rename = "type")]
    pub entry_type: String, // "model_change"
    pub id: String,
    #[serde(rename = "parentId")]
    pub parent_id: Option<String>,
    pub timestamp: String,
    pub provider: String,
    #[serde(rename = "modelId")]
    pub model_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionEntry {
    #[serde(rename = "type")]
    pub entry_type: String, // "compaction"
    pub id: String,
    #[serde(rename = "parentId")]
    pub parent_id: Option<String>,
    pub timestamp: String,
    pub summary: String,
    #[serde(rename = "firstKeptEntryId")]
    pub first_kept_entry_id: String,
    #[serde(rename = "tokensBefore")]
    pub tokens_before: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    #[serde(rename = "fromHook", skip_serializing_if = "Option::is_none")]
    pub from_hook: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchSummaryEntry {
    #[serde(rename = "type")]
    pub entry_type: String, // "branch_summary"
    pub id: String,
    #[serde(rename = "parentId")]
    pub parent_id: Option<String>,
    pub timestamp: String,
    #[serde(rename = "fromId")]
    pub from_id: String,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    #[serde(rename = "fromHook", skip_serializing_if = "Option::is_none")]
    pub from_hook: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomEntry {
    #[serde(rename = "type")]
    pub entry_type: String, // "custom"
    pub id: String,
    #[serde(rename = "parentId")]
    pub parent_id: Option<String>,
    pub timestamp: String,
    #[serde(rename = "customType")]
    pub custom_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelEntry {
    #[serde(rename = "type")]
    pub entry_type: String, // "label"
    pub id: String,
    #[serde(rename = "parentId")]
    pub parent_id: Option<String>,
    pub timestamp: String,
    #[serde(rename = "targetId")]
    pub target_id: String,
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfoEntry {
    #[serde(rename = "type")]
    pub entry_type: String, // "session_info"
    pub id: String,
    #[serde(rename = "parentId")]
    pub parent_id: Option<String>,
    pub timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMessageEntry {
    #[serde(rename = "type")]
    pub entry_type: String, // "custom_message"
    pub id: String,
    #[serde(rename = "parentId")]
    pub parent_id: Option<String>,
    pub timestamp: String,
    #[serde(rename = "customType")]
    pub custom_type: String,
    pub content: serde_json::Value,
    pub display: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

// ============================================================================
// SessionEntry enum — union of all entry types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEntry {
    #[serde(rename = "message")]
    Message {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        message: serde_json::Value,
    },
    #[serde(rename = "thinking_level_change")]
    ThinkingLevelChange {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(rename = "thinkingLevel")]
        thinking_level: String,
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
        #[serde(rename = "customType")]
        custom_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<serde_json::Value>,
    },
    #[serde(rename = "custom_message")]
    CustomMessage {
        id: String,
        #[serde(rename = "parentId")]
        parent_id: Option<String>,
        timestamp: String,
        #[serde(rename = "customType")]
        custom_type: String,
        content: serde_json::Value,
        display: bool,
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
}

impl SessionEntry {
    pub fn id(&self) -> &str {
        match self {
            Self::Message { id, .. }
            | Self::ThinkingLevelChange { id, .. }
            | Self::ModelChange { id, .. }
            | Self::Compaction { id, .. }
            | Self::BranchSummary { id, .. }
            | Self::Custom { id, .. }
            | Self::CustomMessage { id, .. }
            | Self::Label { id, .. }
            | Self::SessionInfo { id, .. } => id,
        }
    }

    pub fn parent_id(&self) -> Option<&str> {
        match self {
            Self::Message { parent_id, .. }
            | Self::ThinkingLevelChange { parent_id, .. }
            | Self::ModelChange { parent_id, .. }
            | Self::Compaction { parent_id, .. }
            | Self::BranchSummary { parent_id, .. }
            | Self::Custom { parent_id, .. }
            | Self::CustomMessage { parent_id, .. }
            | Self::Label { parent_id, .. }
            | Self::SessionInfo { parent_id, .. } => parent_id.as_deref(),
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
            | Self::SessionInfo { timestamp, .. } => timestamp,
        }
    }
}

// ============================================================================
// FileEntry — header or session entry
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FileEntry {
    Header(SessionHeader),
    Entry(SessionEntry),
}

// ============================================================================
// Session Context
// ============================================================================

#[derive(Debug, Clone)]
pub struct SessionContext {
    pub messages: Vec<serde_json::Value>,
    pub thinking_level: String,
    pub model: Option<ModelInfo>,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub provider: String,
    pub model_id: String,
}

// ============================================================================
// Session Info (for listing)
// ============================================================================

#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub path: String,
    pub id: String,
    pub cwd: String,
    pub name: Option<String>,
    pub parent_session_path: Option<String>,
    pub created: String,
    pub modified: i64,
    pub message_count: usize,
    pub first_message: String,
    pub all_messages_text: String,
}

// ============================================================================
// Session Tree Node
// ============================================================================

#[derive(Debug, Clone)]
pub struct SessionTreeNode {
    pub entry: SessionEntry,
    pub children: Vec<SessionTreeNode>,
    pub label: Option<String>,
    pub label_timestamp: Option<String>,
}

// ============================================================================
// Progress callback type
// ============================================================================

pub type SessionListProgress = Arc<dyn Fn(usize, usize) + Send + Sync>;

// ============================================================================
// buildSessionContext — pure function
// ============================================================================

/// Build the session context from entries using tree traversal.
pub fn build_session_context(
    entries: &[SessionEntry],
    leaf_id: Option<&str>,
    by_id: Option<&HashMap<String, usize>>,
) -> SessionContext {
    // Build id → index map if not provided
    let id_to_idx: HashMap<String, usize> = if let Some(map) = by_id {
        map.clone()
    } else {
        entries.iter().enumerate()
            .map(|(i, e)| (e.id().to_string(), i))
            .collect()
    };

    // Find leaf
    let leaf = if let Some(lid) = leaf_id {
        if lid.is_empty() {
            return SessionContext { messages: vec![], thinking_level: "off".into(), model: None };
        }
        id_to_idx.get(lid).and_then(|&i| entries.get(i))
    } else {
        entries.last()
    };

    let Some(leaf) = leaf else {
        return SessionContext { messages: vec![], thinking_level: "off".into(), model: None };
    };

    // Walk from leaf to root
    let mut path: Vec<&SessionEntry> = vec![];
    let mut current: Option<&SessionEntry> = Some(leaf);
    while let Some(entry) = current {
        path.insert(0, entry);
        current = entry.parent_id()
            .and_then(|pid| id_to_idx.get(pid))
            .and_then(|&i| entries.get(i));
    }

    // Extract settings and find compaction
    let mut thinking_level = "off".to_string();
    let mut model: Option<ModelInfo> = None;
    let mut compaction: Option<&SessionEntry> = None;

    for entry in &path {
        match entry {
            SessionEntry::ThinkingLevelChange { thinking_level: tl, .. } => {
                thinking_level = tl.clone();
            }
            SessionEntry::ModelChange { provider, model_id, .. } => {
                model = Some(ModelInfo { provider: provider.clone(), model_id: model_id.clone() });
            }
            SessionEntry::Message { message, .. } => {
                if message.get("role").and_then(|r| r.as_str()) == Some("assistant") {
                    if let (Some(provider), Some(mid)) = (
                        message.get("provider").and_then(|v| v.as_str()),
                        message.get("model").and_then(|v| v.as_str()),
                    ) {
                        model = Some(ModelInfo { provider: provider.to_string(), model_id: mid.to_string() });
                    }
                }
            }
            SessionEntry::Compaction { .. } => {
                compaction = Some(entry);
            }
            _ => {}
        }
    }

    // Build messages
    let mut messages: Vec<serde_json::Value> = vec![];

    let append_message = |msgs: &mut Vec<serde_json::Value>, entry: &SessionEntry| {
        match entry {
            SessionEntry::Message { message, .. } => {
                msgs.push(message.clone());
            }
            SessionEntry::CustomMessage { custom_type, content, display, details, timestamp, .. } => {
                let custom = create_custom_message(
                    custom_type.clone(),
                    content.clone(),
                    *display,
                    details.clone(),
                    chrono::DateTime::parse_from_rfc3339(timestamp)
                        .map(|dt| dt.timestamp_millis())
                        .unwrap_or(0),
                );
                msgs.push(serde_json::to_value(custom).unwrap_or_default());
            }
            SessionEntry::BranchSummary { summary, from_id, timestamp, .. } => {
                if !summary.is_empty() {
                    let bs = create_branch_summary_message(
                        summary.clone(),
                        from_id.clone(),
                        chrono::DateTime::parse_from_rfc3339(timestamp)
                            .map(|dt| dt.timestamp_millis())
                            .unwrap_or(0),
                    );
                    msgs.push(serde_json::to_value(bs).unwrap_or_default());
                }
            }
            _ => {}
        }
    };

    if let Some(comp) = compaction {
        if let SessionEntry::Compaction { summary, tokens_before, first_kept_entry_id, timestamp, .. } = comp {
            let cs = create_compaction_summary_message(
                summary.clone(),
                *tokens_before,
                chrono::DateTime::parse_from_rfc3339(timestamp)
                    .map(|dt| dt.timestamp_millis())
                    .unwrap_or(0),
            );
            messages.push(serde_json::to_value(cs).unwrap_or_default());

            let comp_idx = path.iter().position(|e| e.id() == comp.id());
            if let Some(comp_idx) = comp_idx {
                let mut found_first_kept = false;
                for (i, entry) in path.iter().enumerate() {
                    if i >= comp_idx { break; }
                    if entry.id() == first_kept_entry_id.as_str() {
                        found_first_kept = true;
                    }
                    if found_first_kept {
                        append_message(&mut messages, entry);
                    }
                }
                for i in (comp_idx + 1)..path.len() {
                    append_message(&mut messages, path[i]);
                }
            }
        }
    } else {
        for entry in &path {
            append_message(&mut messages, entry);
        }
    }

    SessionContext { messages, thinking_level, model }
}

// ============================================================================
// Helper: generate a unique short ID
// ============================================================================

fn generate_id(existing: &HashSet<String>) -> String {
    for _ in 0..100 {
        let id = uuid::Uuid::new_v4().to_string()[..8].to_string();
        if !existing.contains(&id) {
            return id;
        }
    }
    uuid::Uuid::new_v4().to_string()
}

fn create_session_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

// ============================================================================
// Helper: default session directory
// ============================================================================

pub fn get_default_session_dir(cwd: &str, agent_dir: &str) -> PathBuf {
    let safe_path = format!("--{}--", cwd
        .trim_start_matches('/')
        .trim_start_matches('\\')
        .replace(&['/', '\\', ':'][..], "-"));
    let dir = Path::new(agent_dir).join("sessions").join(&safe_path);
    std::fs::create_dir_all(&dir).ok();
    dir
}

// ============================================================================
// Helper: load entries from JSONL file
// ============================================================================

pub fn load_entries_from_file(file_path: &Path) -> Vec<FileEntry> {
    if !file_path.exists() {
        return vec![];
    }
    let content = match std::fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };

    let entries: Vec<FileEntry> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<FileEntry>(line).ok())
        .collect();

    // Validate header
    if entries.is_empty() {
        return entries;
    }
    if !matches!(&entries[0], FileEntry::Header(_)) {
        return vec![];
    }
    entries
}

// ============================================================================
// SessionManager
// ============================================================================

pub struct SessionManager {
    session_id: String,
    session_file: Option<PathBuf>,
    session_dir: PathBuf,
    cwd: String,
    persist: bool,
    flushed: bool,
    file_entries: Vec<FileEntry>,
    by_id: HashMap<String, usize>,
    labels_by_id: HashMap<String, String>,
    label_timestamps_by_id: HashMap<String, String>,
    leaf_id: Option<String>,
}

impl SessionManager {
    fn new_internal(cwd: String, session_dir: PathBuf, session_file: Option<PathBuf>, persist: bool) -> Self {
        let mut mgr = Self {
            session_id: String::new(),
            session_file,
            session_dir,
            cwd,
            persist,
            flushed: false,
            file_entries: vec![],
            by_id: HashMap::new(),
            labels_by_id: HashMap::new(),
            label_timestamps_by_id: HashMap::new(),
            leaf_id: None,
        };

        if persist {
            std::fs::create_dir_all(&mgr.session_dir).ok();
        }

        if let Some(sf) = mgr.session_file.clone() {
            mgr.set_session_file(&sf);
        } else {
            mgr.new_session(None);
        }

        mgr
    }

    // =========================================================================
    // Factory methods
    // =========================================================================

    pub fn create(cwd: &str, session_dir: Option<&str>) -> Self {
        let dir = session_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| get_default_session_dir(cwd, &Self::default_agent_dir()));
        Self::new_internal(cwd.to_string(), dir, None, true)
    }

    pub fn open(path: &str, session_dir: Option<&str>, cwd_override: Option<&str>) -> Self {
        let file_path = PathBuf::from(path);
        let dir = session_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| file_path.parent().unwrap_or(Path::new(".")).to_path_buf());

        let cwd = if let Some(c) = cwd_override {
            c.to_string()
        } else {
            load_entries_from_file(&file_path)
                .first()
                .and_then(|e| match e {
                    FileEntry::Header(h) => Some(h.cwd.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| std::env::current_dir().map(|p| p.to_string_lossy().to_string()).unwrap_or_default())
        };

        Self::new_internal(cwd, dir, Some(file_path), true)
    }

    pub fn continue_recent(cwd: &str, session_dir: Option<&str>) -> Self {
        let dir = session_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| get_default_session_dir(cwd, &Self::default_agent_dir()));

        let recent = find_most_recent_session(&dir);
        if let Some(path) = recent {
            Self::new_internal(cwd.to_string(), dir.clone(), Some(path), true)
        } else {
            Self::new_internal(cwd.to_string(), dir, None, true)
        }
    }

    pub fn in_memory(cwd: &str) -> Self {
        Self::new_internal(cwd.to_string(), PathBuf::new(), None, false)
    }

    fn default_agent_dir() -> String {
        dirs_next::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".pi")
            .join("agent")
            .to_string_lossy()
            .to_string()
    }

    // =========================================================================
    // Session file management
    // =========================================================================

    fn set_session_file(&mut self, session_file: &Path) {
        self.session_file = Some(session_file.to_path_buf());

        if session_file.exists() {
            self.file_entries = load_entries_from_file(session_file);

            if self.file_entries.is_empty() {
                let explicit = session_file.to_path_buf();
                self.new_session(None);
                self.session_file = Some(explicit);
                self.rewrite_file();
                self.flushed = true;
                return;
            }

            let header = self.file_entries.iter()
                .find_map(|e| if let FileEntry::Header(h) = e { Some(h) } else { None });

            self.session_id = header.map(|h| h.id.clone()).unwrap_or_else(create_session_id);

            if migrate_to_current_version(&mut self.file_entries) {
                self.rewrite_file();
            }

            self.build_index();
            self.flushed = true;
        } else {
            let explicit = session_file.to_path_buf();
            self.new_session(None);
            self.session_file = Some(explicit);
        }
    }

    fn new_session(&mut self, id: Option<String>) -> Option<String> {
        self.session_id = id.unwrap_or_else(create_session_id);
        let timestamp = chrono::Utc::now().to_rfc3339();
        let header = SessionHeader {
            entry_type: "session".to_string(),
            version: Some(CURRENT_SESSION_VERSION),
            id: self.session_id.clone(),
            timestamp: timestamp.clone(),
            cwd: self.cwd.clone(),
            parent_session: None,
        };
        self.file_entries = vec![FileEntry::Header(header)];
        self.by_id.clear();
        self.labels_by_id.clear();
        self.label_timestamps_by_id.clear();
        self.leaf_id = None;
        self.flushed = false;

        if self.persist {
            let file_ts = timestamp.replace(&[':', '.'][..], "-");
            let path = self.session_dir.join(format!("{}_{}.jsonl", file_ts, self.session_id));
            self.session_file = Some(path.clone());
            return Some(path.to_string_lossy().to_string());
        }
        None
    }

    fn build_index(&mut self) {
        self.by_id.clear();
        self.labels_by_id.clear();
        self.label_timestamps_by_id.clear();
        self.leaf_id = None;

        for (i, entry) in self.file_entries.iter().enumerate() {
            if let FileEntry::Entry(e) = entry {
                self.by_id.insert(e.id().to_string(), i);
                self.leaf_id = Some(e.id().to_string());
                if let SessionEntry::Label { target_id, label, timestamp, .. } = e {
                    if let Some(l) = label {
                        self.labels_by_id.insert(target_id.clone(), l.clone());
                        self.label_timestamps_by_id.insert(target_id.clone(), timestamp.clone());
                    } else {
                        self.labels_by_id.remove(target_id);
                        self.label_timestamps_by_id.remove(target_id);
                    }
                }
            }
        }
    }

    fn rewrite_file(&self) {
        if !self.persist { return; }
        if let Some(path) = &self.session_file {
            let content: String = self.file_entries.iter()
                .map(|e| serde_json::to_string(e).unwrap_or_default())
                .collect::<Vec<_>>()
                .join("\n");
            std::fs::write(path, format!("{}\n", content)).ok();
        }
    }

    // =========================================================================
    // Persistence
    // =========================================================================

    fn persist_entry(&mut self, entry: &SessionEntry) {
        if !self.persist { return; }
        let sf = match self.session_file.clone() {
            Some(f) => f,
            None => return,
        };

        let has_assistant = self.file_entries.iter().any(|e| {
            matches!(e, FileEntry::Entry(SessionEntry::Message { message, .. }) if message.get("role").and_then(|r| r.as_str()) == Some("assistant"))
        });

        if !has_assistant {
            self.flushed = false;
            return;
        }

        use std::io::Write;
        if !self.flushed {
            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&sf) {
                for e in &self.file_entries {
                    let _ = writeln!(file, "{}", serde_json::to_string(e).unwrap_or_default());
                }
            }
            self.flushed = true;
        } else {
            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&sf) {
                let _ = writeln!(file, "{}", serde_json::to_string(entry).unwrap_or_default());
            }
        }
    }

    fn append_entry(&mut self, entry: SessionEntry) {
        let id = entry.id().to_string();
        self.file_entries.push(FileEntry::Entry(entry.clone()));
        let idx = self.file_entries.len() - 1;
        self.by_id.insert(id.clone(), idx);
        self.leaf_id = Some(id);
        self.persist_entry(&entry);
    }

    // =========================================================================
    // Append methods
    // =========================================================================

    pub fn append_message(&mut self, message: serde_json::Value) -> String {
        let existing: HashSet<String> = self.by_id.keys().cloned().collect();
        let id = generate_id(&existing);
        let entry = SessionEntry::Message {
            id: id.clone(),
            parent_id: self.leaf_id.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            message,
        };
        self.append_entry(entry);
        id
    }

    pub fn append_thinking_level_change(&mut self, thinking_level: &str) -> String {
        let existing: HashSet<String> = self.by_id.keys().cloned().collect();
        let id = generate_id(&existing);
        let entry = SessionEntry::ThinkingLevelChange {
            id: id.clone(),
            parent_id: self.leaf_id.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            thinking_level: thinking_level.to_string(),
        };
        self.append_entry(entry);
        id
    }

    pub fn append_model_change(&mut self, provider: &str, model_id: &str) -> String {
        let existing: HashSet<String> = self.by_id.keys().cloned().collect();
        let id = generate_id(&existing);
        let entry = SessionEntry::ModelChange {
            id: id.clone(),
            parent_id: self.leaf_id.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            provider: provider.to_string(),
            model_id: model_id.to_string(),
        };
        self.append_entry(entry);
        id
    }

    pub fn append_compaction(
        &mut self, summary: &str, first_kept_entry_id: &str, tokens_before: u64,
        details: Option<serde_json::Value>, from_hook: Option<bool>,
    ) -> String {
        let existing: HashSet<String> = self.by_id.keys().cloned().collect();
        let id = generate_id(&existing);
        let entry = SessionEntry::Compaction {
            id: id.clone(),
            parent_id: self.leaf_id.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            summary: summary.to_string(),
            first_kept_entry_id: first_kept_entry_id.to_string(),
            tokens_before,
            details,
            from_hook,
        };
        self.append_entry(entry);
        id
    }

    pub fn append_custom_entry(&mut self, custom_type: &str, data: Option<serde_json::Value>) -> String {
        let existing: HashSet<String> = self.by_id.keys().cloned().collect();
        let id = generate_id(&existing);
        let entry = SessionEntry::Custom {
            id: id.clone(),
            parent_id: self.leaf_id.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            custom_type: custom_type.to_string(),
            data,
        };
        self.append_entry(entry);
        id
    }

    pub fn append_session_info(&mut self, name: &str) -> String {
        let existing: HashSet<String> = self.by_id.keys().cloned().collect();
        let id = generate_id(&existing);
        let entry = SessionEntry::SessionInfo {
            id: id.clone(),
            parent_id: self.leaf_id.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            name: Some(name.trim().to_string()),
        };
        self.append_entry(entry);
        id
    }

    pub fn append_custom_message_entry(
        &mut self, custom_type: &str, content: serde_json::Value,
        display: bool, details: Option<serde_json::Value>,
    ) -> String {
        let existing: HashSet<String> = self.by_id.keys().cloned().collect();
        let id = generate_id(&existing);
        let entry = SessionEntry::CustomMessage {
            id: id.clone(),
            parent_id: self.leaf_id.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            custom_type: custom_type.to_string(),
            content,
            display,
            details,
        };
        self.append_entry(entry);
        id
    }

    // =========================================================================
    // Labels
    // =========================================================================

    pub fn get_label(&self, id: &str) -> Option<&str> {
        self.labels_by_id.get(id).map(|s| s.as_str())
    }

    pub fn append_label_change(&mut self, target_id: &str, label: Option<&str>) -> String {
        if !self.by_id.contains_key(target_id) {
            panic!("Entry {} not found", target_id);
        }
        let existing: HashSet<String> = self.by_id.keys().cloned().collect();
        let id = generate_id(&existing);
        let ts = chrono::Utc::now().to_rfc3339();
        let entry = SessionEntry::Label {
            id: id.clone(),
            parent_id: self.leaf_id.clone(),
            timestamp: ts.clone(),
            target_id: target_id.to_string(),
            label: label.map(|l| l.to_string()),
        };
        self.append_entry(entry);
        if let Some(l) = label {
            self.labels_by_id.insert(target_id.to_string(), l.to_string());
            self.label_timestamps_by_id.insert(target_id.to_string(), ts);
        } else {
            self.labels_by_id.remove(target_id);
            self.label_timestamps_by_id.remove(target_id);
        }
        id
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    pub fn is_persisted(&self) -> bool { self.persist }
    pub fn cwd(&self) -> &str { &self.cwd }
    pub fn session_dir(&self) -> &Path { &self.session_dir }
    pub fn session_id(&self) -> &str { &self.session_id }
    pub fn session_file(&self) -> Option<&Path> { self.session_file.as_deref() }
    pub fn leaf_id(&self) -> Option<&str> { self.leaf_id.as_deref() }

    pub fn leaf_entry(&self) -> Option<&SessionEntry> {
        self.leaf_id.as_ref()
            .and_then(|lid| self.by_id.get(lid))
            .and_then(|&i| self.file_entries.get(i))
            .and_then(|e| match e { FileEntry::Entry(entry) => Some(entry), _ => None })
    }

    pub fn entry(&self, id: &str) -> Option<&SessionEntry> {
        self.by_id.get(id)
            .and_then(|&i| self.file_entries.get(i))
            .and_then(|e| match e { FileEntry::Entry(entry) => Some(entry), _ => None })
    }

    pub fn header(&self) -> Option<&SessionHeader> {
        self.file_entries.first()
            .and_then(|e| match e { FileEntry::Header(h) => Some(h), _ => None })
    }

    pub fn entries(&self) -> Vec<&SessionEntry> {
        self.file_entries.iter()
            .filter_map(|e| match e { FileEntry::Entry(entry) => Some(entry), _ => None })
            .collect()
    }

    pub fn session_name(&self) -> Option<&str> {
        self.entries().iter().rev()
            .find_map(|e| match e {
                SessionEntry::SessionInfo { name, .. } => name.as_deref(),
                _ => None,
            })
    }

    // =========================================================================
    // Branch
    // =========================================================================

    pub fn branch(&mut self, branch_from_id: &str) {
        if !self.by_id.contains_key(branch_from_id) {
            panic!("Entry {} not found", branch_from_id);
        }
        self.leaf_id = Some(branch_from_id.to_string());
    }

    pub fn reset_leaf(&mut self) {
        self.leaf_id = None;
    }

    pub fn branch_with_summary(
        &mut self, branch_from_id: Option<&str>, summary: &str,
        details: Option<serde_json::Value>, from_hook: Option<bool>,
    ) -> String {
        if let Some(bid) = branch_from_id {
            if !self.by_id.contains_key(bid) {
                panic!("Entry {} not found", bid);
            }
        }
        self.leaf_id = branch_from_id.map(|s| s.to_string());

        let existing: HashSet<String> = self.by_id.keys().cloned().collect();
        let id = generate_id(&existing);
        let entry = SessionEntry::BranchSummary {
            id: id.clone(),
            parent_id: self.leaf_id.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            from_id: branch_from_id.unwrap_or("root").to_string(),
            summary: summary.to_string(),
            details,
            from_hook,
        };
        self.append_entry(entry);
        id
    }

    pub fn branch_path(&self, from_id: Option<&str>) -> Vec<&SessionEntry> {
        let start = from_id.or(self.leaf_id.as_deref());
        let mut path: Vec<&SessionEntry> = vec![];
        let mut current = start.and_then(|sid| self.entry(sid));
        while let Some(entry) = current {
            path.push(entry);
            current = entry.parent_id().and_then(|pid| self.entry(pid));
        }
        path.reverse();
        path
    }

    /// Get direct children of an entry
    pub fn get_children(&self, parent_id: &str) -> Vec<&SessionEntry> {
        self.entries()
            .into_iter()
            .filter(|e| e.parent_id() == Some(parent_id))
            .collect()
    }

    /// Fork this session into a new file at a different cwd
    pub fn fork_from(source_path: &str, target_cwd: &str, session_dir: Option<&str>) -> Self {
        let source = PathBuf::from(source_path);
        let dir = session_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| get_default_session_dir(target_cwd, &Self::default_agent_dir()));

        let source_entries = load_entries_from_file(&source);
        let _source_header = source_entries.iter()
            .find_map(|e| if let FileEntry::Header(h) = e { Some(h.clone()) } else { None });

        let mut mgr = Self::new_internal(target_cwd.to_string(), dir, None, true);

        // Copy entries from source, preserving tree structure
        for entry in &source_entries {
            if let FileEntry::Entry(e) = entry {
                mgr.file_entries.push(FileEntry::Entry(e.clone()));
                mgr.by_id.insert(e.id().to_string(), mgr.file_entries.len() - 1);
                mgr.leaf_id = Some(e.id().to_string());
            }
        }

        // Set parent session reference
        if let Some(FileEntry::Header(h)) = mgr.file_entries.first_mut() {
            h.parent_session = Some(source_path.to_string());
        }

        mgr
    }

    /// Validate that a file is a valid session file
    pub fn is_valid_session_file(path: &Path) -> bool {
        let Ok(content) = std::fs::read_to_string(path) else { return false; };
        let first_line = content.lines().next().unwrap_or("");
        let Ok(val) = serde_json::from_str::<serde_json::Value>(first_line) else { return false; };
        val.get("type").and_then(|t| t.as_str()) == Some("session")
            && val.get("id").and_then(|i| i.as_str()).is_some()
    }

    pub fn build_context(&self) -> SessionContext {
        let entries = self.entries();
        let id_map: HashMap<String, usize> = entries.iter().enumerate()
            .map(|(i, e)| (e.id().to_string(), i))
            .collect();
        let entries_vec: Vec<SessionEntry> = entries.into_iter().cloned().collect();
        build_session_context(&entries_vec, self.leaf_id.as_deref(), Some(&id_map))
    }

    // =========================================================================
    // Tree
    // =========================================================================

    pub fn tree(&self) -> Vec<SessionTreeNode> {
        let entries = self.entries();
        let mut node_map: HashMap<String, SessionTreeNode> = HashMap::new();
        let mut roots: Vec<SessionTreeNode> = vec![];

        for entry in &entries {
            let id = entry.id().to_string();
            let label = self.labels_by_id.get(&id).cloned();
            let label_ts = self.label_timestamps_by_id.get(&id).cloned();
            node_map.insert(id.clone(), SessionTreeNode {
                entry: (*entry).clone(),
                children: vec![],
                label,
                label_timestamp: label_ts,
            });
        }

        for entry in &entries {
            let id = entry.id();
            let node = node_map.remove(id).unwrap();
            match entry.parent_id() {
                None | Some("") => roots.push(node),
                Some(pid) if pid == entry.id() => roots.push(node),
                Some(pid) => {
                    if let Some(parent) = node_map.get_mut(pid) {
                        parent.children.push(node);
                    } else {
                        // Orphaned entry - parent not found, treat as root
                        roots.push(node);
                    }
                }
            }
        }

        // Sort children by timestamp
        fn sort_children(roots: &mut [SessionTreeNode]) {
            for root in roots.iter_mut() {
                root.children.sort_by_key(|c| c.entry.timestamp().to_string());
                sort_children(&mut root.children);
            }
        }
        sort_children(&mut roots);

        roots
    }
}

// ============================================================================
// Migration
// ============================================================================

fn migrate_to_current_version(entries: &mut Vec<FileEntry>) -> bool {
    let version = entries.first()
        .and_then(|e| match e { FileEntry::Header(h) => h.version, _ => None })
        .unwrap_or(1);

    if version >= CURRENT_SESSION_VERSION { return false; }

    if version < 2 { migrate_v1_to_v2(entries); }
    if version < 3 { migrate_v2_to_v3(entries); }

    true
}

fn migrate_v1_to_v2(entries: &mut Vec<FileEntry>) {
    let mut ids = HashSet::new();
    let mut prev_id: Option<String> = None;

    for entry in entries.iter_mut() {
        if let FileEntry::Header(h) = entry {
            h.version = Some(2);
            continue;
        }
        if let FileEntry::Entry(e) = entry {
            let new_id = generate_id(&ids);
            ids.insert(new_id.clone());
            match e {
                SessionEntry::Message { id, parent_id, .. }
                | SessionEntry::ThinkingLevelChange { id, parent_id, .. }
                | SessionEntry::ModelChange { id, parent_id, .. }
                | SessionEntry::Compaction { id, parent_id, .. }
                | SessionEntry::BranchSummary { id, parent_id, .. }
                | SessionEntry::Custom { id, parent_id, .. }
                | SessionEntry::CustomMessage { id, parent_id, .. }
                | SessionEntry::Label { id, parent_id, .. }
                | SessionEntry::SessionInfo { id, parent_id, .. } => {
                    *id = new_id;
                    *parent_id = prev_id.clone();
                    prev_id = Some(id.clone());
                }
            }
        }
    }
}

fn migrate_v2_to_v3(entries: &mut Vec<FileEntry>) {
    for entry in entries.iter_mut() {
        if let FileEntry::Header(h) = entry {
            h.version = Some(3);
            continue;
        }
        if let FileEntry::Entry(SessionEntry::Message { message, .. }) = entry {
            if message.get("role").and_then(|r| r.as_str()) == Some("hookMessage") {
                if let Some(obj) = message.as_object_mut() {
                    obj.insert("role".to_string(), serde_json::json!("custom"));
                }
            }
        }
    }
}

// ============================================================================
// Session listing
// ============================================================================

fn find_most_recent_session(session_dir: &Path) -> Option<PathBuf> {
    let mut files: Vec<(PathBuf, std::time::SystemTime)> = std::fs::read_dir(session_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".jsonl"))
        .filter_map(|e| {
            let path = e.path();
            let meta = std::fs::metadata(&path).ok()?;
            Some((path, meta.modified().ok()?))
        })
        .collect();

    files.sort_by_key(|(_, t)| std::time::Duration::from_secs(0).saturating_sub(t.elapsed().unwrap_or_default()));
    files.first().map(|(p, _)| p.clone())
}

pub async fn list_sessions(
    cwd: &str, session_dir: Option<&str>, on_progress: Option<SessionListProgress>,
) -> Vec<SessionInfo> {
    let dir = session_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| get_default_session_dir(cwd, &SessionManager::default_agent_dir()));

    list_from_dir(&dir, on_progress).await
}

async fn list_from_dir(dir: &Path, on_progress: Option<SessionListProgress>) -> Vec<SessionInfo> {
    let mut sessions = vec![];
    if !dir.exists() { return sessions; }

    let Ok(entries) = std::fs::read_dir(dir) else { return sessions; };
    let files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".jsonl"))
        .map(|e| e.path())
        .collect();

    let total = files.len();
    for (loaded, file) in files.iter().enumerate() {
        if let Some(progress) = &on_progress {
            progress(loaded + 1, total);
        }
        if let Some(info) = build_session_info(file).await {
            sessions.push(info);
        }
    }

    sessions.sort_by_key(|s| -s.modified);
    sessions
}

async fn build_session_info(file_path: &Path) -> Option<SessionInfo> {
    let content = tokio::fs::read_to_string(file_path).await.ok()?;
    let file_entries: Vec<FileEntry> = content.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();

    if file_entries.is_empty() { return None; }

    let header = match &file_entries[0] {
        FileEntry::Header(h) => h,
        _ => return None,
    };

    let mut message_count = 0;
    let mut first_message = String::new();
    let mut all_messages: Vec<String> = vec![];
    let mut name: Option<String> = None;

    for entry in &file_entries {
        if let FileEntry::Entry(SessionEntry::SessionInfo { name: n, .. }) = entry {
            name = n.as_ref().and_then(|n| if n.trim().is_empty() { None } else { Some(n.clone()) });
        }
        if let FileEntry::Entry(SessionEntry::Message { message, .. }) = entry {
            message_count += 1;
            let role = message.get("role").and_then(|r| r.as_str()).unwrap_or("");
            if role != "user" && role != "assistant" { continue; }

            let text = extract_text(message);
            if !text.is_empty() {
                all_messages.push(text.clone());
                if first_message.is_empty() && role == "user" {
                    first_message = text;
                }
            }
        }
    }

    let cwd = header.cwd.clone();
    let parent = header.parent_session.clone();

    Some(SessionInfo {
        path: file_path.to_string_lossy().to_string(),
        id: header.id.clone(),
        cwd,
        name,
        parent_session_path: parent,
        created: header.timestamp.clone(),
        modified: chrono::Utc::now().timestamp(),
        message_count,
        first_message: if first_message.is_empty() { "(no messages)".to_string() } else { first_message },
        all_messages_text: all_messages.join(" "),
    })
}

fn extract_text(msg: &serde_json::Value) -> String {
    msg.get("content")
        .map(|c| match c {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(arr) => arr.iter()
                .filter_map(|b| {
                    if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                        b.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
                    } else { None }
                })
                .collect::<Vec<_>>()
                .join(" "),
            _ => c.to_string(),
        })
        .unwrap_or_default()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_create() {
        let mgr = SessionManager::in_memory("/tmp/test");
        assert!(!mgr.session_id().is_empty());
        assert_eq!(mgr.cwd(), "/tmp/test");
    }

    #[test]
    fn test_append_and_build_context() {
        let mut mgr = SessionManager::in_memory("/tmp/test");
        mgr.append_message(serde_json::json!({
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
            "timestamp": 1000
        }));
        mgr.append_message(serde_json::json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "api": "test", "provider": "test", "model": "claude",
            "usage": {"input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 2, "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
            "stopReason": "stop",
            "timestamp": 2000
        }));

        let ctx = mgr.build_context();
        assert_eq!(ctx.messages.len(), 2);
        assert!(ctx.model.is_some());
        assert_eq!(ctx.model.unwrap().model_id, "claude");
    }

    #[test]
    fn test_branching() {
        let mut mgr = SessionManager::in_memory("/tmp/test");
        let id1 = mgr.append_message(serde_json::json!({
            "role": "user", "content": "msg1", "timestamp": 1000
        }));
        let id2 = mgr.append_message(serde_json::json!({
            "role": "assistant", "content": "msg2", "api": "t", "provider": "t", "model": "m",
            "usage": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},
            "stopReason": "stop", "timestamp": 2000
        }));

        // Branch back to first message
        mgr.branch(&id1);
        assert_eq!(mgr.leaf_id(), Some(id1.as_str()));

        // Append new branch
        let id3 = mgr.append_message(serde_json::json!({
            "role": "assistant", "content": "alt response", "api": "t", "provider": "t", "model": "m",
            "usage": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},
            "stopReason": "stop", "timestamp": 3000
        }));

        let ctx = mgr.build_context();
        assert_eq!(ctx.messages.len(), 2);
        assert!(ctx.messages[1]["content"].to_string().contains("alt response"));
    }

    #[test]
    fn test_labels() {
        let mut mgr = SessionManager::in_memory("/tmp/test");
        let id = mgr.append_message(serde_json::json!({
            "role": "user", "content": "key message", "timestamp": 1000
        }));
        mgr.append_label_change(&id, Some("important"));

        assert_eq!(mgr.get_label(&id), Some("important"));

        mgr.append_label_change(&id, None);
        assert_eq!(mgr.get_label(&id), None);
    }

    #[test]
    fn test_compaction_context() {
        let mut mgr = SessionManager::in_memory("/tmp/test");

        let id1 = mgr.append_message(serde_json::json!({
            "role": "user", "content": "old message", "timestamp": 1000
        }));
        let id2 = mgr.append_message(serde_json::json!({
            "role": "assistant", "content": "old response", "api": "t", "provider": "t", "model": "m",
            "usage": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},
            "stopReason": "stop", "timestamp": 2000
        }));

        mgr.append_compaction("Summary of old", &id2, 500, None, None);

        let id3 = mgr.append_message(serde_json::json!({
            "role": "user", "content": "new message", "timestamp": 3000
        }));

        let ctx = mgr.build_context();
        // Should have: compaction summary + new message
        assert!(ctx.messages.len() >= 1);
        assert!(ctx.messages[0]["role"].as_str().unwrap().contains("compaction"));
    }
}
