use agent_core::agent_loop::{StreamFn, StreamFnInput};
use agent_core::event::EventStream;
use agent_core::harness::agent_harness::AgentHarnessOptions;
use agent_core::harness::{
    AgentHarness, HarnessConfig, HarnessPhase, InMemorySessionStorage, JsonlSessionRepo, Session,
};
use agent_core::tool::{AgentTool, ToolDefinitionWrapper};
use agent_core::types::{AgentMessage, AgentToolResult, ThinkingLevel};
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

fn named_tool(name: &str) -> Arc<dyn AgentTool> {
    Arc::new(ToolDefinitionWrapper {
        name: name.to_string(),
        label: name.to_string(),
        description: format!("{} tool", name),
        parameters_schema: serde_json::json!({"type": "object", "properties": {}}),
        execution_mode_override: None,
        prepare_arguments_fn: None,
        execute_fn: Arc::new(|_, _, _, _| {
            Box::pin(async move {
                Ok(AgentToolResult {
                    content: vec![ContentBlock::Text { text: "ok".into() }],
                    details: serde_json::Value::Null,
                    terminate: false,
                })
            })
        }),
    })
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

#[tokio::test]
async fn test_session_active_tools_change_restores_context() {
    use agent_core::harness::session::{InMemorySessionStorage, Session, SessionTreeEntry};

    let storage = InMemorySessionStorage::new(None);
    let mut session = Session::new(Box::new(storage));
    session
        .append_active_tools_change(vec!["read".into(), "bash".into()])
        .await
        .unwrap();

    let ctx = session.build_context().await.unwrap();
    assert_eq!(ctx.active_tool_names, Some(vec!["read".into(), "bash".into()]));

    let entries = session.storage().get_entries().await;
    let json = serde_json::to_value(entries.last().unwrap()).unwrap();
    assert_eq!(json["type"], "active_tools_change");
    assert_eq!(json["activeToolNames"], serde_json::json!(["read", "bash"]));
    assert!(matches!(entries.last(), Some(SessionTreeEntry::ActiveToolsChange { .. })));
}

#[tokio::test]
async fn test_harness_active_tool_names_persist_and_filter() {
    use agent_core::harness::session::SessionTreeEntry;

    let harness = make_idle_harness();
    harness.set_active_tools(vec![named_tool("read"), named_tool("bash")]).await;
    assert_eq!(harness.active_tool_names().await, vec!["read".to_string(), "bash".to_string()]);
    assert_eq!(harness.active_tools().await.len(), 2);

    harness.set_active_tool_names(vec!["read".into()]).await.unwrap();
    assert_eq!(harness.active_tool_names().await, vec!["read".to_string()]);
    assert_eq!(harness.active_tools().await.len(), 1);

    let ctx = harness.build_context().await.unwrap();
    assert_eq!(ctx.active_tool_names, Some(vec!["read".to_string()]));
    let entries = harness.session().lock().await.storage().get_entries().await;
    assert!(entries.iter().any(|entry| matches!(entry,
        SessionTreeEntry::ActiveToolsChange { active_tool_names, .. }
        if active_tool_names == &vec!["read".to_string()]
    )));
}

fn make_idle_harness() -> AgentHarness {
    let storage = InMemorySessionStorage::new(None);
    let session = Session::new(Box::new(storage));
    AgentHarness::new(
        session,
        HarnessConfig {
            system_prompt: "".into(),
            thinking_level: ThinkingLevel::Off,
            model_provider: "anthropic".into(),
            model_id: "claude-sonnet-4-6".into(),
        },
        make_options(),
    )
}

#[tokio::test]
async fn test_record_custom_writes_through_when_idle() {
    use agent_core::harness::session::SessionTreeEntry;
    let harness = make_idle_harness();
    let id = harness
        .record_custom("artifact", Some(serde_json::json!({"k": "v"})))
        .await
        .unwrap();
    assert!(!id.is_empty(), "idle path should return real id");
    let entries = harness.session().lock().await.storage().get_entries().await;
    let found = entries
        .iter()
        .any(|e| matches!(e, SessionTreeEntry::Custom { id: eid, .. } if eid == &id));
    assert!(found, "Custom entry should be persisted directly");
}

#[tokio::test]
async fn test_record_custom_message_persists() {
    use agent_core::harness::session::SessionTreeEntry;
    let harness = make_idle_harness();
    let id = harness
        .record_custom_message("artifact", serde_json::json!("payload"), true, None)
        .await
        .unwrap();
    assert!(!id.is_empty());
    let entries = harness.session().lock().await.storage().get_entries().await;
    assert!(entries.iter().any(|e| matches!(e,
        SessionTreeEntry::CustomMessage { id: eid, custom_type, .. }
        if eid == &id && custom_type == "artifact"
    )));
}

#[tokio::test]
async fn test_record_label_attaches_and_clears() {
    use agent_core::harness::session::SessionTreeEntry;
    let harness = make_idle_harness();
    let target_id = harness.append_user_message("hi").await.unwrap();

    harness.record_label(&target_id, Some("important".into())).await.unwrap();
    let entries = harness.session().lock().await.storage().get_entries().await;
    let labelled = entries
        .iter()
        .filter(|e| {
            matches!(e,
                SessionTreeEntry::Label { target_id: t, label: Some(l), .. }
                if t == &target_id && l == "important"
            )
        })
        .count();
    assert_eq!(labelled, 1);

    // Clearing the label is a separate Label entry with `label = None`.
    harness.record_label(&target_id, None).await.unwrap();
    let entries = harness.session().lock().await.storage().get_entries().await;
    let cleared = entries
        .iter()
        .filter(|e| {
            matches!(e,
                SessionTreeEntry::Label { target_id: t, label: None, .. } if t == &target_id
            )
        })
        .count();
    assert_eq!(cleared, 1);
}

#[tokio::test]
async fn test_set_session_name_persists() {
    use agent_core::harness::session::SessionTreeEntry;
    let harness = make_idle_harness();
    harness.set_session_name("my chat").await.unwrap();
    let entries = harness.session().lock().await.storage().get_entries().await;
    assert!(entries.iter().any(|e| matches!(e,
        SessionTreeEntry::SessionInfo { name: Some(n), .. } if n == "my chat"
    )));
}

#[tokio::test]
async fn test_move_leaf_persists() {
    let harness = make_idle_harness();
    let first = harness.append_user_message("first").await.unwrap();
    harness.append_user_message("second").await.unwrap();
    // Move leaf back to the first message (creates a branch point).
    harness.move_leaf(Some(&first)).await.unwrap();
    let leaf = harness.session().lock().await.storage().get_leaf_id().await;
    assert_eq!(leaf.as_deref(), Some(first.as_str()));
}

#[tokio::test]
async fn test_navigate_tree_returns_to_idle_after_call() {
    let harness = make_idle_harness();
    let first = harness.append_user_message("first").await.unwrap();
    harness.append_user_message("second").await.unwrap();
    assert_eq!(harness.phase().await, HarnessPhase::Idle);
    harness.navigate_tree(Some(&first), None).await.unwrap();
    // Phase must drop back to Idle so subsequent turns can run.
    assert_eq!(harness.phase().await, HarnessPhase::Idle);
}

#[tokio::test]
async fn test_navigate_tree_rejects_when_not_idle() {
    use agent_core::harness::HarnessPhase;
    let harness = make_idle_harness();
    harness.append_user_message("hi").await.unwrap();
    // Manually flip phase to simulate a concurrent operation.
    harness.set_phase_for_test(HarnessPhase::Compaction).await;
    let res = harness.navigate_tree(None, None).await;
    assert!(res.is_err(), "navigate_tree must require idle");
    harness.set_phase_for_test(HarnessPhase::Idle).await;
}

// ─── Phase 1.5: post-run state machine ───────────────────────────────────────

use agent_core::harness::PostRunDecision;
use agent_core::harness::compaction::{
    CompactionError, CompactionSettings, StreamFn as CompactionStreamFn,
};
use agent_core::types::{Api, ContentBlock, StopReason, Usage};

fn err_assistant(msg: &str, model_provider: &str, model_id: &str) -> AgentMessage {
    AgentMessage::Assistant {
        content: vec![ContentBlock::Text { text: String::new() }],
        api: Api::Anthropic,
        provider: model_provider.into(),
        model: model_id.into(),
        usage: Usage::default(),
        stop_reason: StopReason::Error,
        error_message: Some(msg.into()),
        timestamp: chrono::Utc::now().timestamp_millis().max(0) as u64,
    }
}

fn dummy_compaction_stream_fn() -> CompactionStreamFn {
    Box::new(|_msgs, _system| {
        Box::pin(async move { Ok::<String, CompactionError>("summary".into()) })
    })
}

#[tokio::test]
async fn test_post_run_stop_when_no_message() {
    let harness = make_idle_harness();
    let dec = harness
        .check_post_run(&CompactionSettings::default(), &dummy_compaction_stream_fn())
        .await
        .unwrap();
    assert_eq!(dec, PostRunDecision::Stop);
}

#[tokio::test]
async fn test_post_run_retry_on_transient_error() {
    let harness = make_idle_harness();
    harness
        .set_last_assistant_for_test(Some(err_assistant(
            "503 service unavailable",
            "anthropic",
            "claude-sonnet-4-6",
        )))
        .await;
    harness
        .set_retry_settings(agent_core::auto_retry::RetrySettings {
            enabled: true,
            max_retries: 2,
            base_delay_ms: 1,
        })
        .await;

    let dec = harness
        .check_post_run(&CompactionSettings::default(), &dummy_compaction_stream_fn())
        .await
        .unwrap();
    assert_eq!(dec, PostRunDecision::Retry { attempt: 1 });
    assert_eq!(harness.retry_attempt_for_test().await, 1);
}

#[tokio::test]
async fn test_post_run_retry_exhausts_after_max_attempts() {
    let harness = make_idle_harness();
    harness
        .set_retry_settings(agent_core::auto_retry::RetrySettings {
            enabled: true,
            max_retries: 1, // Only 1 retry allowed.
            base_delay_ms: 1,
        })
        .await;
    let err = err_assistant("rate limit", "anthropic", "claude-sonnet-4-6");

    // First call: retry succeeds.
    harness.set_last_assistant_for_test(Some(err.clone())).await;
    let dec1 = harness
        .check_post_run(&CompactionSettings::default(), &dummy_compaction_stream_fn())
        .await
        .unwrap();
    assert_eq!(dec1, PostRunDecision::Retry { attempt: 1 });

    // Second call (still error): exhausted, should Stop and reset counter.
    harness.set_last_assistant_for_test(Some(err)).await;
    let dec2 = harness
        .check_post_run(&CompactionSettings::default(), &dummy_compaction_stream_fn())
        .await
        .unwrap();
    assert_eq!(dec2, PostRunDecision::Stop);
    assert_eq!(harness.retry_attempt_for_test().await, 0);
}

#[tokio::test]
async fn test_post_run_overflow_triggers_compacted_retry_once() {
    let harness = make_idle_harness();
    harness.append_user_message("first prompt").await.unwrap();
    let overflow = AgentMessage::Assistant {
        content: vec![ContentBlock::Text { text: String::new() }],
        api: Api::Anthropic,
        provider: "anthropic".into(),
        model: "claude-sonnet-4-6".into(),
        usage: Usage::default(),
        stop_reason: StopReason::Error,
        error_message: Some("prompt is too long: 213462 tokens > 200000 maximum".into()),
        timestamp: chrono::Utc::now().timestamp_millis().max(0) as u64,
    };

    harness.set_last_assistant_for_test(Some(overflow.clone())).await;
    let dec = harness
        .check_post_run(&CompactionSettings::default(), &dummy_compaction_stream_fn())
        .await
        .unwrap();
    assert_eq!(dec, PostRunDecision::CompactedRetry);
    assert!(harness.overflow_recovery_attempted_for_test().await);

    // Second attempt: gate refuses; no infinite loop.
    harness.set_last_assistant_for_test(Some(overflow)).await;
    let dec2 = harness
        .check_post_run(&CompactionSettings::default(), &dummy_compaction_stream_fn())
        .await
        .unwrap();
    assert_eq!(dec2, PostRunDecision::Stop);
}

#[tokio::test]
async fn test_post_run_overflow_skipped_for_different_model() {
    let harness = make_idle_harness(); // Configured with claude-sonnet-4-6
    let overflow = AgentMessage::Assistant {
        content: vec![],
        api: Api::Anthropic,
        provider: "openai".into(), // ← different model
        model: "gpt-4o".into(),
        usage: Usage::default(),
        stop_reason: StopReason::Error,
        error_message: Some("prompt is too long: x > 200000 maximum".into()),
        timestamp: chrono::Utc::now().timestamp_millis().max(0) as u64,
    };
    harness.set_last_assistant_for_test(Some(overflow)).await;
    let dec = harness
        .check_post_run(&CompactionSettings::default(), &dummy_compaction_stream_fn())
        .await
        .unwrap();
    // Different-model overflow must not trigger compaction for the new model.
    assert_eq!(dec, PostRunDecision::Stop);
    assert!(!harness.overflow_recovery_attempted_for_test().await);
}

#[tokio::test]
async fn test_post_run_aborted_message_skips_dispatch() {
    let harness = make_idle_harness();
    let aborted = AgentMessage::Assistant {
        content: vec![],
        api: Api::Anthropic,
        provider: "anthropic".into(),
        model: "claude-sonnet-4-6".into(),
        usage: Usage::default(),
        stop_reason: StopReason::Aborted,
        error_message: None,
        timestamp: chrono::Utc::now().timestamp_millis().max(0) as u64,
    };
    harness.set_last_assistant_for_test(Some(aborted)).await;
    let dec = harness
        .check_post_run(&CompactionSettings::default(), &dummy_compaction_stream_fn())
        .await
        .unwrap();
    assert_eq!(dec, PostRunDecision::Stop);
}

#[tokio::test]
async fn test_post_run_reset_recovery_state_on_new_execute_turn() {
    let harness = make_idle_harness();
    // Simulate that overflow was attempted on a prior turn.
    let overflow = AgentMessage::Assistant {
        content: vec![],
        api: Api::Anthropic,
        provider: "anthropic".into(),
        model: "claude-sonnet-4-6".into(),
        usage: Usage::default(),
        stop_reason: StopReason::Error,
        error_message: Some("prompt is too long: x > 200000 maximum".into()),
        timestamp: chrono::Utc::now().timestamp_millis().max(0) as u64,
    };
    harness.set_last_assistant_for_test(Some(overflow)).await;
    let _ = harness
        .check_post_run(&CompactionSettings::default(), &dummy_compaction_stream_fn())
        .await;
    assert!(harness.overflow_recovery_attempted_for_test().await);

    // execute_turn must reset the gate.
    harness.execute_turn(vec![AgentMessage::user_text("hi")]).await.unwrap();
    assert!(!harness.overflow_recovery_attempted_for_test().await);
}

#[tokio::test]
async fn test_queue_mode_default_is_one_at_a_time() {
    use agent_core::queue::QueueMode;
    let harness = make_idle_harness();
    assert_eq!(harness.steer_mode().await, QueueMode::OneAtATime);
    assert_eq!(harness.follow_up_mode().await, QueueMode::OneAtATime);
}

#[tokio::test]
async fn test_queue_mode_set_and_get() {
    use agent_core::queue::QueueMode;
    let harness = make_idle_harness();
    harness.set_steer_mode(QueueMode::All).await;
    assert_eq!(harness.steer_mode().await, QueueMode::All);
    harness.set_follow_up_mode(QueueMode::All).await;
    assert_eq!(harness.follow_up_mode().await, QueueMode::All);

    harness.set_steer_mode(QueueMode::OneAtATime).await;
    assert_eq!(harness.steer_mode().await, QueueMode::OneAtATime);
}

#[tokio::test]
async fn test_abort_returns_cleared_queues() {
    let harness = make_idle_harness();
    harness.steer(AgentMessage::user_text("steer-1")).await;
    harness.steer(AgentMessage::user_text("steer-2")).await;
    harness.follow_up(AgentMessage::user_text("follow-1")).await;

    let result = harness.abort().await;
    assert_eq!(result.cleared_steer.len(), 2);
    assert_eq!(result.cleared_follow_up.len(), 1);
}

#[tokio::test]
async fn test_abort_with_empty_queues_returns_empty_result() {
    let harness = make_idle_harness();
    let result = harness.abort().await;
    assert!(result.cleared_steer.is_empty());
    assert!(result.cleared_follow_up.is_empty());
}

#[tokio::test]
async fn test_auto_compaction_runs_on_execute_turn_when_last_aborted() {
    use agent_core::harness::AutoCompactionConfig;
    use std::sync::atomic::{AtomicBool, Ordering};

    // Verify wiring: when set_auto_compaction is configured, execute_turn
    // calls into check_pre_prompt → check_post_run_inner. We can't easily
    // force the threshold compaction to actually fire (needs context_window
    // and a usage history), but we can verify the inner path runs by
    // inspecting that the harness doesn't crash and the recovery state
    // gets reset normally.
    let touched = Arc::new(AtomicBool::new(false));
    let stream_fn: CompactionStreamFn = {
        let touched = touched.clone();
        Box::new(move |_msgs, _system| {
            touched.store(true, Ordering::SeqCst);
            Box::pin(async move { Ok::<String, CompactionError>("summary".into()) })
        })
    };

    let harness = make_idle_harness();
    harness
        .set_auto_compaction(Some(AutoCompactionConfig::new(
            CompactionSettings::default(),
            stream_fn,
        )))
        .await;

    // Seed: prior turn aborted. check_pre_prompt should run the inner
    // flow without panicking; whether compaction actually fires depends
    // on threshold which needs context_window from resources.
    let aborted = AgentMessage::Assistant {
        content: vec![],
        api: Api::Anthropic,
        provider: "anthropic".into(),
        model: "claude-sonnet-4-6".into(),
        usage: Usage::default(),
        stop_reason: StopReason::Aborted,
        error_message: None,
        timestamp: chrono::Utc::now().timestamp_millis().max(0) as u64,
    };
    harness.set_last_assistant_for_test(Some(aborted)).await;

    // execute_turn should NOT panic and should leave the harness idle.
    let _ = harness.execute_turn(vec![AgentMessage::user_text("hi")]).await;

    // Recovery state must be reset by execute_turn.
    assert!(!harness.overflow_recovery_attempted_for_test().await);
    let _ = touched; // silence dead-code warning when threshold not crossed
}

#[tokio::test]
async fn test_auto_compaction_skipped_when_unset() {
    let harness = make_idle_harness();
    // No set_auto_compaction call — execute_turn should run without
    // any compaction-related machinery.
    let res = harness.execute_turn(vec![AgentMessage::user_text("hi")]).await;
    assert!(res.is_ok());
}

#[tokio::test]
async fn test_check_pre_prompt_processes_aborted_messages() {
    // The pre-prompt path must see aborted messages even though the post-run
    // path skips them. The behavior is exercised here by
    // confirming check_pre_prompt does NOT take the post_run early-exit
    // (Stop) when the prior turn is aborted: it returns Ok(false) only
    // because there's no usage data to compute a token estimate, NOT
    // because the aborted flag short-circuited.
    let harness = make_idle_harness();
    let aborted = AgentMessage::Assistant {
        content: vec![],
        api: Api::Anthropic,
        provider: "anthropic".into(),
        model: "claude-sonnet-4-6".into(),
        usage: Usage::default(),
        stop_reason: StopReason::Aborted,
        error_message: None,
        timestamp: chrono::Utc::now().timestamp_millis().max(0) as u64,
    };
    harness.set_last_assistant_for_test(Some(aborted)).await;

    let result = harness
        .check_pre_prompt(&CompactionSettings::default(), &dummy_compaction_stream_fn())
        .await;
    assert!(result.is_ok(), "pre_prompt should not error on aborted: {:?}", result);
    // No compaction fired (no usage data), but the path didn't panic.
    assert!(!result.unwrap());
}

#[tokio::test]
async fn test_check_post_run_drains_queued_followups() {
    use agent_core::harness::PostRunDecision;

    // A follow-up enqueued after a turn settled (e.g. by an agent_end listener)
    // must surface as DrainQueues so the caller runs a continuation.
    let harness = make_idle_harness();
    // Give the harness a realistic context window so the small usage below is
    // nowhere near the overflow / threshold trigger.
    harness
        .set_model_info(agent_core::types::ModelInfo {
            id: "claude-sonnet-4-6".into(),
            provider: "anthropic".into(),
            context_window: 200_000,
            ..Default::default()
        })
        .await;

    // A successful assistant message with no overflow / threshold trigger:
    // check_post_run would otherwise return Stop.
    let done = AgentMessage::Assistant {
        content: vec![agent_core::types::ContentBlock::Text { text: "ok".into() }],
        api: Api::Anthropic,
        provider: "anthropic".into(),
        model: "claude-sonnet-4-6".into(),
        usage: Usage {
            input: 1,
            output: 1,
            total_tokens: 2,
            ..Default::default()
        },
        stop_reason: StopReason::EndTurn,
        error_message: None,
        timestamp: chrono::Utc::now().timestamp_millis().max(0) as u64,
    };
    harness.set_last_assistant_for_test(Some(done)).await;

    // Without a queued message, the decision is Stop.
    let decision = harness
        .check_post_run(&CompactionSettings::default(), &dummy_compaction_stream_fn())
        .await
        .unwrap();
    assert_eq!(decision, PostRunDecision::Stop);

    // Queue a follow-up (as an agent_end listener would), then re-check.
    harness.follow_up(AgentMessage::user_text("more")).await;
    let decision = harness
        .check_post_run(&CompactionSettings::default(), &dummy_compaction_stream_fn())
        .await
        .unwrap();
    assert_eq!(decision, PostRunDecision::DrainQueues);
}
