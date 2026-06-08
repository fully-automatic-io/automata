// Tests for the agent loop's per-turn hook ordering and prepare_next_turn
// config-snapshot semantics.

use agent_core::agent_loop::{
    AgentEventSink, AgentLoop, AssistantMessageEventStream, StreamFn, StreamFnInput,
};
use agent_core::event::{AgentEvent, EventStream};
use agent_core::types::{
    AgentContext, AgentLoopConfig, AgentMessage, ContentBlock, ModelInfo, PrepareNextTurnContext,
    StopReason, ThinkingLevel, TurnUpdate, Usage,
};
use std::sync::{Arc, Mutex};

/// Stream fn that records which model id it was invoked with on each turn,
/// and emits a tool call on turn 0 then a plain answer on turn 1.
fn recording_stream_fn(seen_models: Arc<Mutex<Vec<String>>>) -> StreamFn {
    let turn = Arc::new(Mutex::new(0usize));
    Arc::new(move |input: StreamFnInput| {
        seen_models.lock().unwrap().push(input.model.id.clone());
        let n = {
            let mut t = turn.lock().unwrap();
            let v = *t;
            *t += 1;
            v
        };
        Box::pin(async move {
            let stream: AssistantMessageEventStream = EventStream::new();
            let msg = if n == 0 {
                AgentMessage::Assistant {
                    content: vec![ContentBlock::ToolCall {
                        id: "c1".into(),
                        name: "noop".into(),
                        arguments: serde_json::json!({}),
                    }],
                    api: input.model.api,
                    provider: input.model.provider.clone(),
                    model: input.model.id.clone(),
                    usage: Usage::default(),
                    stop_reason: StopReason::ToolUse,
                    error_message: None,
                    timestamp: 0,
                }
            } else {
                AgentMessage::assistant_text("done")
            };
            stream.end(msg);
            Ok(stream)
        })
    })
}

fn model(id: &str) -> ModelInfo {
    ModelInfo {
        id: id.into(),
        provider: "p".into(),
        ..Default::default()
    }
}

#[tokio::test]
async fn prepare_next_turn_switches_model_for_next_turn() {
    use agent_core::tool::AgentTool;
    // A trivial tool so the first turn's tool call resolves and the loop continues.
    struct Noop;
    #[async_trait::async_trait]
    impl AgentTool for Noop {
        fn name(&self) -> &str {
            "noop"
        }
        fn label(&self) -> &str {
            "noop"
        }
        fn description(&self) -> &str {
            "noop"
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type":"object"})
        }
        async fn execute(
            &self,
            _id: String,
            _args: serde_json::Value,
            _signal: Option<tokio_util::sync::CancellationToken>,
            _on_update: Option<agent_core::types::AgentToolUpdateCallback>,
        ) -> Result<agent_core::types::AgentToolResult, Box<dyn std::error::Error + Send + Sync>>
        {
            Ok(agent_core::types::AgentToolResult {
                content: vec![ContentBlock::Text { text: "ok".into() }],
                details: serde_json::Value::Null,
                terminate: false,
            })
        }
    }

    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let stream_fn = recording_stream_fn(seen.clone());

    // prepare_next_turn switches the model to "model-b" after the first turn.
    let prepare: agent_core::types::PrepareNextTurnFn =
        Arc::new(|_ctx: PrepareNextTurnContext, _sig| {
            Box::pin(async move {
                Some(TurnUpdate {
                    model: Some(ModelInfo {
                        id: "model-b".into(),
                        provider: "p".into(),
                        ..Default::default()
                    }),
                    thinking_level: Some(ThinkingLevel::High),
                    ..Default::default()
                })
            })
        });

    let config = AgentLoopConfig {
        prepare_next_turn: Some(prepare),
        ..AgentLoopConfig::new(model("model-a"), Arc::new(|m| Box::pin(async move { m })))
    };

    let emit: AgentEventSink = Arc::new(|_e: AgentEvent| Box::pin(async {}));
    let stream_fn_ref = stream_fn;
    let context = AgentContext {
        system_prompt: String::new(),
        messages: vec![],
        tools: vec![Arc::new(Noop)],
    };

    let _ = AgentLoop::new(&config, &emit, &stream_fn_ref)
        .run(vec![AgentMessage::user_text("hi")], context, None)
        .await;

    let models = seen.lock().unwrap().clone();
    assert_eq!(models.len(), 2, "expected two LLM turns");
    assert_eq!(models[0], "model-a", "first turn uses the configured model");
    assert_eq!(models[1], "model-b", "prepare_next_turn switched the model for turn 2");
}

#[tokio::test]
async fn should_stop_after_turn_exits_immediately() {
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let stream_fn = recording_stream_fn(seen.clone());

    // Stop right after the first turn — even though that turn has a tool call,
    // the loop must not run a second LLM turn.
    let stop: agent_core::types::ShouldStopAfterTurnFn =
        Arc::new(|_ctx, _sig| Box::pin(async { true }));

    use agent_core::tool::AgentTool;
    struct Noop;
    #[async_trait::async_trait]
    impl AgentTool for Noop {
        fn name(&self) -> &str {
            "noop"
        }
        fn label(&self) -> &str {
            "noop"
        }
        fn description(&self) -> &str {
            "noop"
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type":"object"})
        }
        async fn execute(
            &self,
            _id: String,
            _args: serde_json::Value,
            _signal: Option<tokio_util::sync::CancellationToken>,
            _on_update: Option<agent_core::types::AgentToolUpdateCallback>,
        ) -> Result<agent_core::types::AgentToolResult, Box<dyn std::error::Error + Send + Sync>>
        {
            Ok(agent_core::types::AgentToolResult {
                content: vec![ContentBlock::Text { text: "ok".into() }],
                details: serde_json::Value::Null,
                terminate: false,
            })
        }
    }

    let config = AgentLoopConfig {
        should_stop_after_turn: Some(stop),
        ..AgentLoopConfig::new(model("model-a"), Arc::new(|m| Box::pin(async move { m })))
    };
    let emit: AgentEventSink = Arc::new(|_e| Box::pin(async {}));
    let context = AgentContext {
        system_prompt: String::new(),
        messages: vec![],
        tools: vec![Arc::new(Noop)],
    };

    let _ = AgentLoop::new(&config, &emit, &stream_fn)
        .run(vec![AgentMessage::user_text("hi")], context, None)
        .await;

    assert_eq!(
        seen.lock().unwrap().len(),
        1,
        "should_stop_after_turn must end the run after one turn"
    );
}
