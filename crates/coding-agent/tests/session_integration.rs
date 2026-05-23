// Integration tests for session management — uses agent-core's canonical async Session API.

use agent_core::harness::session::{
    InMemorySessionStorage, JsonlSessionRepo, Session, SessionTreeEntry,
};
use agent_core::types::{AgentMessage, ContentBlock, MessageContent, StopReason, Usage};

fn user(text: &str, ts: u64) -> AgentMessage {
    AgentMessage::User {
        content: MessageContent::Blocks(vec![ContentBlock::Text { text: text.into() }]),
        timestamp: ts,
        metadata: None,
    }
}

fn assistant(text: &str, ts: u64) -> AgentMessage {
    AgentMessage::Assistant {
        content: vec![ContentBlock::Text { text: text.into() }],
        api: agent_core::types::Api::Anthropic, provider: "t".into(), model: "m".into(),
        usage: Usage::default(), stop_reason: StopReason::EndTurn,
        error_message: None, timestamp: ts,
    }
}

#[tokio::test]
async fn test_session_persist_and_reload() {
    let dir = tempfile::TempDir::new().unwrap();
    let cwd = "/tmp/test";

    let repo = JsonlSessionRepo::new(dir.path());
    let mut session = repo.create(cwd, None, None).await.unwrap();

    session.append_message(user("hello", 1000)).await.unwrap();
    session.append_message(AgentMessage::Assistant {
        content: vec![ContentBlock::Text { text: "hi there".into() }],
        api: agent_core::types::Api::Anthropic, provider: "anthropic".into(), model: "claude-opus-4-7".into(),
        usage: Usage { input: 5, output: 5, total_tokens: 10, ..Default::default() },
        stop_reason: StopReason::EndTurn,
        error_message: None,
        timestamp: 2000,
    }).await.unwrap();

    let path = session.get_metadata().await.id;
    drop(session);

    let sessions = repo.list(Some(cwd)).await.unwrap();
    assert!(sessions.iter().any(|m| m.id == path));

    let entry_path = &sessions.iter().find(|m| m.id == path).unwrap().path;
    let reloaded = repo.open_by_path(entry_path).await.unwrap();
    let ctx = reloaded.build_context().await.unwrap();
    assert_eq!(ctx.messages.len(), 2);
    assert!(ctx.model.is_some());
}

#[tokio::test]
async fn test_session_branching_and_context() {
    let mut session = Session::new(Box::new(InMemorySessionStorage::new(None)));

    let id1 = session.append_message(user("q1", 1)).await.unwrap();
    let _id2 = session.append_message(assistant("a1", 2)).await.unwrap();

    session.move_to(Some(&id1), None).await.unwrap();
    let _id3 = session.append_message(assistant("alt answer", 3)).await.unwrap();

    let ctx = session.build_context().await.unwrap();
    assert_eq!(ctx.messages.len(), 2);
    let alt = match &ctx.messages[1] {
        AgentMessage::Assistant { content, .. } => match &content[0] {
            ContentBlock::Text { text } => text.clone(),
            _ => String::new(),
        },
        _ => String::new(),
    };
    assert_eq!(alt, "alt answer");
}

#[tokio::test]
async fn test_session_compaction_context() {
    let mut session = Session::new(Box::new(InMemorySessionStorage::new(None)));

    let _u1 = session.append_message(user("old q", 1)).await.unwrap();
    let a1 = session.append_message(assistant("old a", 2)).await.unwrap();

    session.append_compaction("Summary of old conversation", &a1, 1000, None, None).await.unwrap();
    let _u2 = session.append_message(user("new q", 3)).await.unwrap();

    let ctx = session.build_context().await.unwrap();
    assert!(ctx.messages.len() >= 2);
    assert!(matches!(ctx.messages[0], AgentMessage::CompactionSummary { .. } | AgentMessage::User { .. }));
}

#[tokio::test]
async fn test_session_tree_records_branch() {
    let mut session = Session::new(Box::new(InMemorySessionStorage::new(None)));

    let id1 = session.append_message(user("root", 1)).await.unwrap();
    let _id2 = session.append_message(assistant("branch a", 2)).await.unwrap();

    session.move_to(Some(&id1), None).await.unwrap();
    let _id3 = session.append_message(assistant("branch b", 3)).await.unwrap();

    let entries = session.storage().get_entries().await;
    let messages = entries.iter().filter(|e| matches!(e, SessionTreeEntry::Message { .. })).count();
    assert_eq!(messages, 3);
}

#[tokio::test]
async fn test_session_labels() {
    let mut session = Session::new(Box::new(InMemorySessionStorage::new(None)));
    let id = session.append_message(user("important", 1)).await.unwrap();

    let storage = session.storage_mut();
    let lid = storage.create_entry_id().await;
    storage.append_entry(SessionTreeEntry::Label {
        id: lid,
        parent_id: None,
        timestamp: agent_core::harness::session::now_iso(),
        target_id: id.clone(),
        label: Some("key point".into()),
    }).await.unwrap();
    assert_eq!(session.storage().get_label(&id).await.as_deref(), Some("key point"));

    let storage = session.storage_mut();
    let lid2 = storage.create_entry_id().await;
    storage.append_entry(SessionTreeEntry::Label {
        id: lid2,
        parent_id: None,
        timestamp: agent_core::harness::session::now_iso(),
        target_id: id.clone(),
        label: None,
    }).await.unwrap();
    assert_eq!(session.storage().get_label(&id).await, None);
}
