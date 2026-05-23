use agent_core::agent_loop::{StreamFn, StreamFnInput};
use agent_core::event::EventStream;
use agent_core::harness::{
    AgentHarness, HarnessConfig, HarnessPhase, JsonlSessionRepo, Session,
    InMemorySessionStorage,
};
use agent_core::harness::agent_harness::AgentHarnessOptions;
use agent_core::types::{AgentMessage, ThinkingLevel};
use std::sync::Arc;
use tempfile::TempDir;

fn dummy_stream_fn() -> StreamFn {
    Arc::new(|_input: StreamFnInput| {
        Box::pin(async move {
            let stream = EventStream::new();
            stream.end(AgentMessage::assistant_text("ok"));
            Ok(stream)
        })
    })
}

fn make_options() -> AgentHarnessOptions {
    AgentHarnessOptions {
        stream_fn: dummy_stream_fn(),
        convert_to_llm: None,
        transform_context: None,
        before_tool_call: None,
        after_tool_call: None,
        should_stop_after_turn: None,
        prepare_next_turn: None,
        on_payload: None,
        on_response: None,
    }
}

#[tokio::test]
async fn test_jsonl_session_create_and_reopen() {
    let dir = TempDir::new().unwrap();
    let repo = JsonlSessionRepo::new(dir.path());

    let mut session = repo.create("/tmp/cwd", None, None).await.unwrap();
    let _id = session.append_message(AgentMessage::user_text("hello")).await.unwrap();

    let _meta = session.get_metadata().await;
    let sessions = repo.list(Some("/tmp/cwd")).await.unwrap();
    assert_eq!(sessions.len(), 1);
    let path = sessions[0].path.clone();

    let session2 = repo.open_by_path(&path).await.unwrap();
    let ctx = session2.build_context().await.unwrap();
    assert_eq!(ctx.messages.len(), 1);
    assert_eq!(ctx.messages[0].role(), "user");
}

#[tokio::test]
async fn test_harness_append_and_context() {
    let storage = InMemorySessionStorage::new(None);
    let session = Session::new(Box::new(storage));
    let harness = AgentHarness::new(
        session,
        HarnessConfig {
            system_prompt: "You are helpful.".into(),
            thinking_level: ThinkingLevel::Medium,
            model_provider: "anthropic".into(),
            model_id: "claude-sonnet-4-6".into(),
        },
        make_options(),
    );

    assert_eq!(harness.phase().await, HarnessPhase::Idle);
    harness.append_user_message("hello").await.unwrap();
    let ctx = harness.build_context().await.unwrap();
    assert_eq!(ctx.messages.len(), 1);
    assert_eq!(ctx.messages[0].role(), "user");
}

#[tokio::test]
async fn test_harness_steer_and_follow_up_queues() {
    let storage = InMemorySessionStorage::new(None);
    let session = Session::new(Box::new(storage));
    let harness = AgentHarness::new(
        session,
        HarnessConfig {
            system_prompt: "".into(),
            thinking_level: ThinkingLevel::Off,
            model_provider: "anthropic".into(),
            model_id: "claude-sonnet-4-6".into(),
        },
        make_options(),
    );

    harness.steer(AgentMessage::user_text("steer1")).await;
    harness.steer(AgentMessage::user_text("steer2")).await;
    harness.follow_up(AgentMessage::user_text("followup")).await;

    let steered = harness.drain_steer().await;
    assert_eq!(steered.len(), 2);
    assert!(harness.drain_steer().await.is_empty());

    let fu = harness.drain_follow_up().await;
    assert_eq!(fu.len(), 1);
}

#[tokio::test]
async fn test_harness_set_model_persists_to_session() {
    let storage = InMemorySessionStorage::new(None);
    let session = Session::new(Box::new(storage));
    let harness = AgentHarness::new(
        session,
        HarnessConfig {
            system_prompt: "".into(),
            thinking_level: ThinkingLevel::Off,
            model_provider: "anthropic".into(),
            model_id: "claude-sonnet-4-6".into(),
        },
        make_options(),
    );

    harness.set_model("openai".into(), "gpt-4o".into()).await;
    // Trigger flush by appending a user message
    harness.append_user_message("test").await.unwrap();

    let ctx = harness.build_context().await.unwrap();
    // model_change entry should have been flushed; context model should reflect it
    assert!(ctx.model.is_some());
    let m = ctx.model.unwrap();
    assert_eq!(m.provider, "openai");
    assert_eq!(m.model_id, "gpt-4o");
}
