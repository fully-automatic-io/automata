use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

use super::types::{
    InMemorySessionStorage, Session, SessionError, SessionMetadata, SessionStorage, SessionTreeEntry,
};
use super::uuid::{now_iso, uuidv7};

// ============================================================================
// JSONL session file format (v3)
// ============================================================================

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JsonlSessionMetadata {
    pub id: String,
    #[serde(rename = "createdAt")]
    pub created_at: String,
    pub cwd: String,
    pub path: String,
    #[serde(rename = "parentSessionPath", skip_serializing_if = "Option::is_none")]
    pub parent_session_path: Option<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SessionHeader {
    #[serde(rename = "type")]
    entry_type: String,
    version: u32,
    id: String,
    timestamp: String,
    cwd: String,
    #[serde(rename = "parentSession", skip_serializing_if = "Option::is_none")]
    parent_session: Option<String>,
}

// ============================================================================
// JsonlSessionStorage
// ============================================================================

pub struct JsonlSessionStorage {
    metadata: JsonlSessionMetadata,
    inner: InMemorySessionStorage,
    file_path: PathBuf,
}

impl JsonlSessionStorage {
    /// Create a new JSONL session file.
    pub async fn create(
        file_path: impl AsRef<Path>,
        cwd: &str,
        session_id: &str,
        parent_session_path: Option<&str>,
    ) -> Result<Self, SessionError> {
        let file_path = file_path.as_ref().to_path_buf();
        let timestamp = now_iso();
        let header = SessionHeader {
            entry_type: "session".into(),
            version: 3,
            id: session_id.to_string(),
            timestamp: timestamp.clone(),
            cwd: cwd.to_string(),
            parent_session: parent_session_path.map(|s| s.to_string()),
        };
        let header_line = serde_json::to_string(&header)
            .map_err(|e| SessionError::Storage(e.to_string()))?;
        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| SessionError::Storage(e.to_string()))?;
        }
        tokio::fs::write(&file_path, format!("{}\n", header_line)).await
            .map_err(|e| SessionError::Storage(e.to_string()))?;

        let metadata = JsonlSessionMetadata {
            id: session_id.to_string(),
            created_at: timestamp,
            cwd: cwd.to_string(),
            path: file_path.to_string_lossy().to_string(),
            parent_session_path: parent_session_path.map(|s| s.to_string()),
        };
        Ok(Self {
            metadata,
            inner: InMemorySessionStorage::new(None),
            file_path,
        })
    }

    /// Open an existing JSONL session file.
    pub async fn open(file_path: impl AsRef<Path>) -> Result<Self, SessionError> {
        let file_path = file_path.as_ref().to_path_buf();
        let content = tokio::fs::read_to_string(&file_path).await
            .map_err(|e| SessionError::Storage(e.to_string()))?;
        let mut lines = content.lines().filter(|l| !l.trim().is_empty());

        let header_line = lines.next()
            .ok_or_else(|| SessionError::InvalidSession("missing header".into()))?;
        let header: SessionHeader = serde_json::from_str(header_line)
            .map_err(|e| SessionError::InvalidSession(e.to_string()))?;
        if header.entry_type != "session" || header.version != 3 {
            return Err(SessionError::InvalidSession("unsupported version".into()));
        }

        let mut entries = vec![];
        for line in lines {
            let entry: SessionTreeEntry = serde_json::from_str(line)
                .map_err(|e| SessionError::InvalidEntry(e.to_string()))?;
            entries.push(entry);
        }

        let metadata = JsonlSessionMetadata {
            id: header.id.clone(),
            created_at: header.timestamp.clone(),
            cwd: header.cwd.clone(),
            path: file_path.to_string_lossy().to_string(),
            parent_session_path: header.parent_session.clone(),
        };

        let inner = InMemorySessionStorage::with_entries(None, entries);
        // Patch metadata id into inner
        let inner = {
            let meta = super::types::SessionMetadata {
                id: header.id,
                created_at: header.timestamp,
            };
            InMemorySessionStorage::with_entries(Some(meta), inner.snapshot_entries())
        };

        Ok(Self { metadata, inner, file_path })
    }

    async fn append_line(&self, line: &str) -> Result<(), SessionError> {
        let mut file = tokio::fs::OpenOptions::new()
            .append(true)
            .open(&self.file_path)
            .await
            .map_err(|e| SessionError::Storage(e.to_string()))?;
        file.write_all(format!("{}\n", line).as_bytes()).await
            .map_err(|e| SessionError::Storage(e.to_string()))?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl SessionStorage for JsonlSessionStorage {
    async fn get_metadata(&self) -> SessionMetadata {
        SessionMetadata {
            id: self.metadata.id.clone(),
            created_at: self.metadata.created_at.clone(),
        }
    }

    async fn get_leaf_id(&self) -> Option<String> {
        self.inner.get_leaf_id().await
    }

    async fn set_leaf_id(&mut self, leaf_id: Option<String>) -> Result<(), SessionError> {
        // Write a leaf entry to the file before updating in-memory state.
        let entry_id = self.inner.create_entry_id().await;
        let parent_id = self.inner.get_leaf_id().await;
        let entry = SessionTreeEntry::Leaf {
            id: entry_id,
            parent_id,
            timestamp: now_iso(),
            target_id: leaf_id.clone(),
        };
        let line = serde_json::to_string(&entry)
            .map_err(|e| SessionError::Storage(e.to_string()))?;
        self.append_line(&line).await?;
        self.inner.set_leaf_id(leaf_id).await
    }

    async fn create_entry_id(&self) -> String {
        self.inner.create_entry_id().await
    }

    async fn append_entry(&mut self, entry: SessionTreeEntry) -> Result<(), SessionError> {
        let line = serde_json::to_string(&entry)
            .map_err(|e| SessionError::Storage(e.to_string()))?;
        self.append_line(&line).await?;
        self.inner.append_entry(entry).await
    }

    async fn get_entry(&self, id: &str) -> Option<SessionTreeEntry> {
        self.inner.get_entry(id).await
    }

    async fn find_entries_by_type(&self, entry_type: &str) -> Vec<SessionTreeEntry> {
        self.inner.find_entries_by_type(entry_type).await
    }

    async fn get_label(&self, id: &str) -> Option<String> {
        self.inner.get_label(id).await
    }

    async fn get_path_to_root(&self, leaf_id: Option<&str>) -> Result<Vec<SessionTreeEntry>, SessionError> {
        self.inner.get_path_to_root(leaf_id).await
    }

    async fn get_entries(&self) -> Vec<SessionTreeEntry> {
        self.inner.get_entries().await
    }
}

// ============================================================================
// JsonlSessionRepo
// ============================================================================

pub struct JsonlSessionRepo {
    sessions_root: PathBuf,
}

impl JsonlSessionRepo {
    pub fn new(sessions_root: impl AsRef<Path>) -> Self {
        Self { sessions_root: sessions_root.as_ref().to_path_buf() }
    }

    fn encode_cwd(cwd: &str) -> String {
        format!("--{}--",
            cwd.trim_start_matches(['/', '\\'])
               .replace(['/', '\\', ':'], "-"))
    }

    async fn session_dir(&self, cwd: &str) -> PathBuf {
        self.sessions_root.join(Self::encode_cwd(cwd))
    }

    async fn session_file_path(&self, cwd: &str, id: &str, timestamp: &str) -> PathBuf {
        let ts = timestamp.replace([':', '.'], "-");
        self.session_dir(cwd).await.join(format!("{}_{}.jsonl", ts, id))
    }

    pub async fn create(
        &self,
        cwd: &str,
        id: Option<String>,
        parent_session_path: Option<&str>,
    ) -> Result<Session, SessionError> {
        let id = id.unwrap_or_else(uuidv7);
        let timestamp = now_iso();
        let dir = self.session_dir(cwd).await;
        tokio::fs::create_dir_all(&dir).await
            .map_err(|e| SessionError::Storage(e.to_string()))?;
        let path = self.session_file_path(cwd, &id, &timestamp).await;
        let storage = JsonlSessionStorage::create(&path, cwd, &id, parent_session_path).await?;
        Ok(Session::new(Box::new(storage)))
    }

    pub async fn open_by_path(&self, path: impl AsRef<Path>) -> Result<Session, SessionError> {
        let storage = JsonlSessionStorage::open(path).await?;
        Ok(Session::new(Box::new(storage)))
    }

    pub async fn list(&self, cwd: Option<&str>) -> Result<Vec<JsonlSessionMetadata>, SessionError> {
        let mut sessions = vec![];
        let dirs: Vec<PathBuf> = if let Some(cwd) = cwd {
            vec![self.session_dir(cwd).await]
        } else {
            let mut d = vec![];
            if let Ok(mut rd) = tokio::fs::read_dir(&self.sessions_root).await {
                while let Ok(Some(entry)) = rd.next_entry().await {
                    if entry.path().is_dir() {
                        d.push(entry.path());
                    }
                }
            }
            d
        };

        for dir in dirs {
            if let Ok(mut rd) = tokio::fs::read_dir(&dir).await {
                while let Ok(Some(entry)) = rd.next_entry().await {
                    let p = entry.path();
                    if p.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                        if let Ok(storage) = JsonlSessionStorage::open(&p).await {
                            sessions.push(storage.metadata.clone());
                        }
                    }
                }
            }
        }
        sessions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(sessions)
    }

    pub async fn fork(
        &self,
        source_path: impl AsRef<Path>,
        cwd: &str,
        fork_at_entry_id: Option<&str>,
        position: Option<&str>,
        new_id: Option<String>,
        parent_session_path: Option<&str>,
    ) -> Result<Session, SessionError> {
        let source_storage = JsonlSessionStorage::open(&source_path).await?;
        let entries_to_fork = get_entries_to_fork(&source_storage, fork_at_entry_id, position).await?;

        let new_id = new_id.unwrap_or_else(uuidv7);
        let timestamp = now_iso();
        let dir = self.session_dir(cwd).await;
        tokio::fs::create_dir_all(&dir).await
            .map_err(|e| SessionError::Storage(e.to_string()))?;
        let path = self.session_file_path(cwd, &new_id, &timestamp).await;
        let parent = parent_session_path
            .or_else(|| Some(source_path.as_ref().to_str().unwrap_or("")));
        let mut storage = JsonlSessionStorage::create(&path, cwd, &new_id, parent).await?;
        for entry in entries_to_fork {
            storage.append_entry(entry).await?;
        }
        Ok(Session::new(Box::new(storage)))
    }
}

async fn get_entries_to_fork(
    storage: &JsonlSessionStorage,
    entry_id: Option<&str>,
    position: Option<&str>,
) -> Result<Vec<SessionTreeEntry>, SessionError> {
    let Some(entry_id) = entry_id else {
        return Ok(storage.get_entries().await);
    };
    let target = storage.get_entry(entry_id).await
        .ok_or_else(|| SessionError::InvalidForkTarget(format!("Entry {} not found", entry_id)))?;
    let effective_leaf_id = if position == Some("at") {
        Some(target.id().to_string())
    } else {
        // "before" — use parent of the target user message
        if !matches!(&target, SessionTreeEntry::Message { message, .. }
            if matches!(message, crate::types::AgentMessage::User { .. }))
        {
            return Err(SessionError::InvalidForkTarget(
                format!("Entry {} is not a user message", entry_id)
            ));
        }
        target.parent_id().map(|s| s.to_string())
    };
    storage.get_path_to_root(effective_leaf_id.as_deref()).await
}
