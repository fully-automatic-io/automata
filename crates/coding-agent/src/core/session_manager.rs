use std::path::{Path, PathBuf};

use agent_core::harness::session::{
    InMemorySessionStorage, JsonlSessionMetadata, JsonlSessionRepo, JsonlSessionStorage, Session,
    SessionError,
};

use super::session_cwd::{SessionCwdIssue, assert_session_cwd_exists, missing_session_cwd_issue};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForkPosition {
    Before,
    At,
}

impl ForkPosition {
    fn as_str(self) -> &'static str {
        match self {
            Self::Before => "before",
            Self::At => "at",
        }
    }
}

#[derive(Debug, Clone)]
pub enum SessionManager {
    InMemory,
    Create {
        sessions_root: Option<PathBuf>,
        id: Option<String>,
        parent_session_path: Option<PathBuf>,
    },
    Open {
        path: PathBuf,
        cwd_override: Option<PathBuf>,
    },
    ContinueRecent {
        sessions_root: Option<PathBuf>,
    },
    Fork {
        source_path: PathBuf,
        sessions_root: Option<PathBuf>,
        entry_id: Option<String>,
        position: ForkPosition,
        id: Option<String>,
        parent_session_path: Option<PathBuf>,
    },
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::Create {
            sessions_root: None,
            id: None,
            parent_session_path: None,
        }
    }
}

impl SessionManager {
    pub fn in_memory() -> Self {
        Self::InMemory
    }

    pub fn create() -> Self {
        Self::default()
    }

    pub fn create_in(sessions_root: impl Into<PathBuf>) -> Self {
        Self::Create {
            sessions_root: Some(sessions_root.into()),
            id: None,
            parent_session_path: None,
        }
    }

    pub fn open(path: impl Into<PathBuf>) -> Self {
        Self::Open { path: path.into(), cwd_override: None }
    }

    pub fn continue_recent() -> Self {
        Self::ContinueRecent { sessions_root: None }
    }

    pub fn continue_recent_in(sessions_root: impl Into<PathBuf>) -> Self {
        Self::ContinueRecent {
            sessions_root: Some(sessions_root.into()),
        }
    }

    pub fn fork(source_path: impl Into<PathBuf>) -> Self {
        Self::Fork {
            source_path: source_path.into(),
            sessions_root: None,
            entry_id: None,
            position: ForkPosition::Before,
            id: None,
            parent_session_path: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        match &mut self {
            Self::Create { id: target, .. } | Self::Fork { id: target, .. } => {
                *target = Some(id.into());
            }
            _ => {}
        }
        self
    }

    pub fn with_sessions_root(mut self, sessions_root: impl Into<PathBuf>) -> Self {
        match &mut self {
            Self::Create { sessions_root: target, .. }
            | Self::ContinueRecent { sessions_root: target }
            | Self::Fork { sessions_root: target, .. } => {
                *target = Some(sessions_root.into());
            }
            _ => {}
        }
        self
    }

    pub fn with_parent_session_path(mut self, parent_session_path: impl Into<PathBuf>) -> Self {
        match &mut self {
            Self::Create { parent_session_path: target, .. }
            | Self::Fork { parent_session_path: target, .. } => {
                *target = Some(parent_session_path.into());
            }
            _ => {}
        }
        self
    }

    pub fn with_cwd_override(mut self, cwd: impl Into<PathBuf>) -> Self {
        if let Self::Open { cwd_override, .. } = &mut self {
            *cwd_override = Some(cwd.into());
        }
        self
    }

    pub fn fork_at(mut self, entry_id: impl Into<String>, position: ForkPosition) -> Self {
        if let Self::Fork {
            entry_id: target,
            position: target_position,
            ..
        } = &mut self
        {
            *target = Some(entry_id.into());
            *target_position = position;
        }
        self
    }

    pub async fn effective_cwd(&self, fallback_cwd: &Path) -> Result<PathBuf, SessionError> {
        match self {
            Self::Open { path, cwd_override } => {
                let storage = JsonlSessionStorage::open(path).await?;
                let cwd =
                    open_session_cwd(storage.metadata(), cwd_override.as_deref(), fallback_cwd);
                assert_session_cwd_exists(Some(path), &cwd, fallback_cwd)?;
                Ok(cwd)
            }
            _ => Ok(fallback_cwd.to_path_buf()),
        }
    }

    pub async fn missing_cwd_issue(
        &self,
        fallback_cwd: &Path,
    ) -> Result<Option<SessionCwdIssue>, SessionError> {
        match self {
            Self::Open { path, cwd_override } => {
                let storage = JsonlSessionStorage::open(path).await?;
                let cwd =
                    open_session_cwd(storage.metadata(), cwd_override.as_deref(), fallback_cwd);
                Ok(missing_session_cwd_issue(Some(path), &cwd, fallback_cwd))
            }
            _ => Ok(None),
        }
    }

    pub async fn open_session(
        &self,
        cwd: &Path,
        agent_dir: &Path,
    ) -> Result<ManagedSession, SessionError> {
        match self {
            Self::InMemory => {
                let session = Session::new(Box::new(InMemorySessionStorage::new(None)));
                let metadata = session.get_metadata().await;
                Ok(ManagedSession {
                    session,
                    metadata: ManagedSessionMetadata {
                        id: metadata.id,
                        created_at: metadata.created_at,
                        cwd: cwd.to_path_buf(),
                        path: None,
                        sessions_root: None,
                        parent_session_path: None,
                    },
                })
            }
            Self::Create { sessions_root, id, parent_session_path } => {
                let root = resolve_sessions_root(agent_dir, sessions_root.as_deref());
                let repo = JsonlSessionRepo::new(&root);
                let parent = parent_session_path.as_ref().map(|path| path_to_string(path));
                let session =
                    repo.create(&path_to_string(cwd), id.clone(), parent.as_deref()).await?;
                let metadata = metadata_for_created(&repo, cwd, &session, Some(root)).await?;
                Ok(ManagedSession { session, metadata })
            }
            Self::Open { path, cwd_override } => {
                let storage = JsonlSessionStorage::open(path).await?;
                let jsonl_metadata = storage.metadata().clone();
                let file_path = storage.file_path().to_path_buf();
                let session = Session::new(Box::new(storage));
                let cwd_override = cwd_override
                    .clone()
                    .or_else(|| jsonl_metadata.cwd.is_empty().then(|| cwd.to_path_buf()));
                let metadata =
                    metadata_from_jsonl(jsonl_metadata, cwd_override, Some(file_path), None);
                Ok(ManagedSession { session, metadata })
            }
            Self::ContinueRecent { sessions_root } => {
                let root = resolve_sessions_root(agent_dir, sessions_root.as_deref());
                let repo = JsonlSessionRepo::new(&root);
                let cwd_string = path_to_string(cwd);
                let listed = repo.list(Some(&cwd_string)).await?;
                if let Some(latest) = listed.first() {
                    let session = repo.open_by_path(&latest.path).await?;
                    return Ok(ManagedSession {
                        session,
                        metadata: metadata_from_jsonl(
                            latest.clone(),
                            None,
                            Some(PathBuf::from(&latest.path)),
                            Some(root),
                        ),
                    });
                }
                let session = repo.create(&cwd_string, None, None).await?;
                let metadata = metadata_for_created(&repo, cwd, &session, Some(root)).await?;
                Ok(ManagedSession { session, metadata })
            }
            Self::Fork {
                source_path,
                sessions_root,
                entry_id,
                position,
                id,
                parent_session_path,
            } => {
                let root = resolve_sessions_root(agent_dir, sessions_root.as_deref());
                let repo = JsonlSessionRepo::new(&root);
                let parent = parent_session_path.as_ref().map(|path| path_to_string(path));
                let session = repo
                    .fork(
                        source_path,
                        &path_to_string(cwd),
                        entry_id.as_deref(),
                        Some(position.as_str()),
                        id.clone(),
                        parent.as_deref(),
                    )
                    .await?;
                let metadata = metadata_for_created(&repo, cwd, &session, Some(root)).await?;
                Ok(ManagedSession { session, metadata })
            }
        }
    }
}

pub struct ManagedSession {
    pub session: Session,
    pub metadata: ManagedSessionMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedSessionMetadata {
    pub id: String,
    pub created_at: String,
    pub cwd: PathBuf,
    pub path: Option<PathBuf>,
    pub sessions_root: Option<PathBuf>,
    pub parent_session_path: Option<PathBuf>,
}

impl ManagedSessionMetadata {
    pub fn is_persisted(&self) -> bool {
        self.path.is_some()
    }
}

fn resolve_sessions_root(agent_dir: &Path, configured: Option<&Path>) -> PathBuf {
    configured.map(Path::to_path_buf).unwrap_or_else(|| agent_dir.join("sessions"))
}

fn open_session_cwd(
    metadata: &JsonlSessionMetadata,
    cwd_override: Option<&Path>,
    fallback_cwd: &Path,
) -> PathBuf {
    cwd_override.map(Path::to_path_buf).unwrap_or_else(|| {
        if metadata.cwd.is_empty() {
            fallback_cwd.to_path_buf()
        } else {
            PathBuf::from(&metadata.cwd)
        }
    })
}

async fn metadata_for_created(
    repo: &JsonlSessionRepo,
    cwd: &Path,
    session: &Session,
    sessions_root: Option<PathBuf>,
) -> Result<ManagedSessionMetadata, SessionError> {
    let session_metadata = session.get_metadata().await;
    let cwd_string = path_to_string(cwd);
    let jsonl_metadata = repo
        .list(Some(&cwd_string))
        .await?
        .into_iter()
        .find(|metadata| metadata.id == session_metadata.id);

    Ok(match jsonl_metadata {
        Some(metadata) => metadata_from_jsonl(
            metadata.clone(),
            None,
            Some(PathBuf::from(&metadata.path)),
            sessions_root,
        ),
        None => ManagedSessionMetadata {
            id: session_metadata.id,
            created_at: session_metadata.created_at,
            cwd: cwd.to_path_buf(),
            path: None,
            sessions_root,
            parent_session_path: None,
        },
    })
}

fn metadata_from_jsonl(
    metadata: JsonlSessionMetadata,
    cwd_override: Option<PathBuf>,
    path_override: Option<PathBuf>,
    sessions_root: Option<PathBuf>,
) -> ManagedSessionMetadata {
    let JsonlSessionMetadata {
        id,
        created_at,
        cwd,
        path,
        parent_session_path,
    } = metadata;
    ManagedSessionMetadata {
        id,
        created_at,
        cwd: cwd_override.unwrap_or_else(|| PathBuf::from(cwd)),
        path: path_override.or_else(|| Some(PathBuf::from(path))),
        sessions_root,
        parent_session_path: parent_session_path.map(PathBuf::from),
    }
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::types::{AgentMessage, ContentBlock};

    fn assistant_text(text: &str) -> AgentMessage {
        AgentMessage::Assistant {
            content: vec![ContentBlock::Text { text: text.into() }],
            api: agent_core::types::Api::Openai,
            provider: "deepseek".into(),
            model: "deepseek-chat".into(),
            usage: agent_core::types::Usage::default(),
            stop_reason: agent_core::types::StopReason::EndTurn,
            error_message: None,
            timestamp: 0,
        }
    }

    #[tokio::test]
    async fn continues_latest_session_for_cwd() {
        let dir = tempfile::TempDir::new().unwrap();
        let cwd = dir.path().join("repo");
        let agent_dir = dir.path().join("agent");
        let sessions_root = dir.path().join("sessions");
        std::fs::create_dir_all(&cwd).unwrap();

        let mut created = SessionManager::create_in(&sessions_root)
            .open_session(&cwd, &agent_dir)
            .await
            .unwrap();
        created.session.append_message(AgentMessage::user_text("hello")).await.unwrap();
        let created_id = created.metadata.id.clone();

        let continued = SessionManager::continue_recent_in(&sessions_root)
            .open_session(&cwd, &agent_dir)
            .await
            .unwrap();
        assert_eq!(continued.metadata.id, created_id);
        assert!(continued.metadata.is_persisted());
        assert_eq!(continued.session.build_context().await.unwrap().messages.len(), 1);
    }

    #[tokio::test]
    async fn continue_recent_filters_by_cwd() {
        let dir = tempfile::TempDir::new().unwrap();
        let cwd_a = dir.path().join("repo-a");
        let cwd_b = dir.path().join("repo-b");
        let agent_dir = dir.path().join("agent");
        let sessions_root = dir.path().join("sessions");
        std::fs::create_dir_all(&cwd_a).unwrap();
        std::fs::create_dir_all(&cwd_b).unwrap();

        let mut session_a = SessionManager::create_in(&sessions_root)
            .open_session(&cwd_a, &agent_dir)
            .await
            .unwrap();
        session_a
            .session
            .append_message(AgentMessage::user_text("from a"))
            .await
            .unwrap();

        let mut session_b = SessionManager::create_in(&sessions_root)
            .open_session(&cwd_b, &agent_dir)
            .await
            .unwrap();
        session_b
            .session
            .append_message(AgentMessage::user_text("from b"))
            .await
            .unwrap();

        let continued = SessionManager::continue_recent_in(&sessions_root)
            .open_session(&cwd_a, &agent_dir)
            .await
            .unwrap();
        let context = continued.session.build_context().await.unwrap();

        assert_eq!(continued.metadata.cwd, cwd_a);
        assert!(matches!(
            context.messages.first(),
            Some(AgentMessage::User { content, .. })
                if matches!(content, agent_core::types::MessageContent::Blocks(blocks)
                    if matches!(blocks.first(), Some(ContentBlock::Text { text }) if text == "from a"))
        ));
    }

    #[tokio::test]
    async fn create_uses_custom_session_id_and_parent_path() {
        let dir = tempfile::TempDir::new().unwrap();
        let cwd = dir.path().join("repo");
        let agent_dir = dir.path().join("agent");
        let sessions_root = dir.path().join("sessions");
        let parent = dir.path().join("parent.jsonl");
        std::fs::create_dir_all(&cwd).unwrap();

        let managed = SessionManager::create_in(&sessions_root)
            .with_id("custom-session")
            .with_parent_session_path(&parent)
            .open_session(&cwd, &agent_dir)
            .await
            .unwrap();

        assert_eq!(managed.metadata.id, "custom-session");
        assert_eq!(managed.metadata.parent_session_path, Some(parent));
        assert!(managed.metadata.path.as_ref().unwrap().exists());
    }

    #[tokio::test]
    async fn fork_before_user_message_excludes_that_turn() {
        let dir = tempfile::TempDir::new().unwrap();
        let cwd = dir.path().join("repo");
        let agent_dir = dir.path().join("agent");
        let sessions_root = dir.path().join("sessions");
        std::fs::create_dir_all(&cwd).unwrap();

        let mut source = SessionManager::create_in(&sessions_root)
            .open_session(&cwd, &agent_dir)
            .await
            .unwrap();
        source.session.append_message(AgentMessage::user_text("u1")).await.unwrap();
        source.session.append_message(assistant_text("a1")).await.unwrap();
        let second_user_id =
            source.session.append_message(AgentMessage::user_text("u2")).await.unwrap();
        source.session.append_message(assistant_text("a2")).await.unwrap();
        let source_path = source.metadata.path.clone().unwrap();

        let forked = SessionManager::fork(&source_path)
            .with_sessions_root(&sessions_root)
            .fork_at(&second_user_id, ForkPosition::Before)
            .open_session(&cwd, &agent_dir)
            .await
            .unwrap();
        let context = forked.session.build_context().await.unwrap();

        assert_eq!(context.messages.len(), 2);
        assert!(matches!(
            context.messages.last(),
            Some(AgentMessage::Assistant { content, .. })
                if matches!(content.first(), Some(ContentBlock::Text { text }) if text == "a1")
        ));
    }

    #[tokio::test]
    async fn fork_at_entry_includes_target_turn() {
        let dir = tempfile::TempDir::new().unwrap();
        let cwd = dir.path().join("repo");
        let agent_dir = dir.path().join("agent");
        let sessions_root = dir.path().join("sessions");
        std::fs::create_dir_all(&cwd).unwrap();

        let mut source = SessionManager::create_in(&sessions_root)
            .open_session(&cwd, &agent_dir)
            .await
            .unwrap();
        source.session.append_message(AgentMessage::user_text("u1")).await.unwrap();
        source.session.append_message(assistant_text("a1")).await.unwrap();
        let second_user_id =
            source.session.append_message(AgentMessage::user_text("u2")).await.unwrap();
        source.session.append_message(assistant_text("a2")).await.unwrap();
        let source_path = source.metadata.path.clone().unwrap();

        let forked = SessionManager::fork(&source_path)
            .with_sessions_root(&sessions_root)
            .fork_at(&second_user_id, ForkPosition::At)
            .open_session(&cwd, &agent_dir)
            .await
            .unwrap();
        let context = forked.session.build_context().await.unwrap();

        assert_eq!(context.messages.len(), 3);
        assert!(matches!(
            context.messages.last(),
            Some(AgentMessage::User { content, .. })
                if matches!(content, agent_core::types::MessageContent::Blocks(blocks)
                    if matches!(blocks.first(), Some(ContentBlock::Text { text }) if text == "u2"))
        ));
    }

    #[tokio::test]
    async fn rejects_non_v3_jsonl_without_migration() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("old.jsonl");
        std::fs::write(
            &path,
            r#"{"type":"session","version":2,"id":"old","timestamp":"2026-01-01T00:00:00Z","cwd":"/tmp"}"#,
        )
        .unwrap();

        let result = SessionManager::open(&path)
            .open_session(Path::new("/tmp"), Path::new("/tmp/agent"))
            .await;
        let err = match result {
            Ok(_) => panic!("expected old JSONL shape to be rejected"),
            Err(err) => err,
        };
        assert!(matches!(err, SessionError::InvalidSession(_)));
    }

    #[tokio::test]
    async fn rejects_missing_session_cwd_without_override() {
        let dir = tempfile::TempDir::new().unwrap();
        let fallback_cwd = dir.path().join("repo");
        let missing_cwd = dir.path().join("missing");
        let path = dir.path().join("session.jsonl");
        std::fs::create_dir_all(&fallback_cwd).unwrap();
        std::fs::write(
            &path,
            format!(
                r#"{{"type":"session","version":3,"id":"session","timestamp":"2026-01-01T00:00:00Z","cwd":"{}"}}"#,
                missing_cwd.display()
            ),
        )
        .unwrap();

        let result = SessionManager::open(&path).effective_cwd(&fallback_cwd).await;
        let err = match result {
            Ok(_) => panic!("expected missing cwd to be rejected"),
            Err(err) => err,
        };

        assert!(
            matches!(err, SessionError::InvalidArgument(message) if message.contains("Stored session working directory does not exist"))
        );
    }

    #[tokio::test]
    async fn open_with_cwd_override_uses_override_metadata() {
        let dir = tempfile::TempDir::new().unwrap();
        let fallback_cwd = dir.path().join("repo");
        let missing_cwd = dir.path().join("missing");
        let agent_dir = dir.path().join("agent");
        let path = dir.path().join("session.jsonl");
        std::fs::create_dir_all(&fallback_cwd).unwrap();
        std::fs::create_dir_all(&agent_dir).unwrap();
        std::fs::write(
            &path,
            format!(
                r#"{{"type":"session","version":3,"id":"session","timestamp":"2026-01-01T00:00:00Z","cwd":"{}"}}"#,
                missing_cwd.display()
            ),
        )
        .unwrap();

        let manager = SessionManager::open(&path).with_cwd_override(&fallback_cwd);
        let effective_cwd = manager.effective_cwd(&fallback_cwd).await.unwrap();
        let managed = manager.open_session(&effective_cwd, &agent_dir).await.unwrap();

        assert_eq!(effective_cwd, fallback_cwd);
        assert_eq!(managed.metadata.cwd, fallback_cwd);
    }
}
