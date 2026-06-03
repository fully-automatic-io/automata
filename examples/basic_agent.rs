// Basic agent example — demonstrates agent-core with a mock LLM provider.

use agent_core::{
    agent_loop::{AgentEventSink, AgentLoop, AssistantMessageEventStream, StreamFn, StreamFnInput},
    event::{AgentEvent, AssistantMessageEvent, EventStream},
    harness::messages::default_convert_to_llm,
    tool::AgentTool,
    types::{
        AgentContext, AgentLoopConfig, AgentMessage, AgentToolResult, AgentToolUpdateCallback,
        ContentBlock, ModelInfo, ToolExecutionMode,
    },
};
use async_trait::async_trait;
use std::{pin::Pin, sync::Arc};
use tokio_util::sync::CancellationToken;

struct EchoTool;

#[async_trait]
impl AgentTool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn label(&self) -> &str {
        "echo"
    }
    fn description(&self) -> &str {
        "Echo the input back"
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        })
    }
    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        None
    }
    async fn execute(
        &self,
        _id: String,
        params: serde_json::Value,
        _signal: Option<CancellationToken>,
        _on_update: Option<AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>> {
        let text = params["text"].as_str().unwrap_or("").to_string();
        Ok(AgentToolResult {
            content: vec![ContentBlock::Text { text }],
            details: serde_json::Value::Null,
            terminate: false,
        })
    }
}

fn make_mock_stream_fn() -> StreamFn {
    Arc::new(|_: StreamFnInput| {
        Box::pin(async move {
            let stream: AssistantMessageEventStream = EventStream::new();
            let stream2 = stream.clone();
            tokio::spawn(async move {
                use agent_core::event::{PartialAssistantMessage, PartialContentBlock};
                let mut partial =
                    PartialAssistantMessage::new(agent_core::types::Api::Mock, "mock", "mock-1");
                stream2.push(AssistantMessageEvent::Start { partial: partial.clone() });
                partial.content.push(PartialContentBlock::Text {
                    text: String::new(),
                    text_signature: None,
                });
                stream2.push(AssistantMessageEvent::TextStart {
                    content_index: 0,
                    partial: partial.clone(),
                });
                if let Some(PartialContentBlock::Text { text, .. }) = partial.content.get_mut(0) {
                    text.push_str("Hello from the mock LLM!");
                }
                stream2.push(AssistantMessageEvent::TextDelta {
                    content_index: 0,
                    delta: "Hello from the mock LLM!".to_string(),
                    partial: partial.clone(),
                });
                stream2.push(AssistantMessageEvent::TextEnd {
                    content_index: 0,
                    content: "Hello from the mock LLM!".to_string(),
                    partial: partial.clone(),
                });
                let final_msg = partial.into_finalized();
                stream2.push(AssistantMessageEvent::Done {
                    reason: agent_core::types::StopReason::EndTurn,
                    message: final_msg.clone(),
                });
                stream2.end(final_msg);
            });
            Ok(stream)
        })
            as Pin<
                Box<
                    dyn std::future::Future<Output = Result<AssistantMessageEventStream, String>>
                        + Send,
                >,
            >
    })
}

#[tokio::main]
async fn main() {
    println!("=== Automata Basic Agent Example ===\n");

    let log: Arc<tokio::sync::Mutex<Vec<String>>> = Arc::new(tokio::sync::Mutex::new(vec![]));
    let log2 = log.clone();
    let emit: AgentEventSink = Arc::new(move |event: AgentEvent| {
        let log = log2.clone();
        Box::pin(async move {
            let label = match &event {
                AgentEvent::AgentStart => "agent_start".to_string(),
                AgentEvent::AgentEnd { messages } => format!("agent_end ({} msgs)", messages.len()),
                AgentEvent::TurnStart => "turn_start".to_string(),
                AgentEvent::TurnEnd { .. } => "turn_end".to_string(),
                AgentEvent::MessageStart { message } => {
                    format!("message_start role={}", message.role())
                }
                AgentEvent::MessageEnd { message } => {
                    format!("message_end role={}", message.role())
                }
                _ => return,
            };
            log.lock().await.push(label);
        })
    });

    let model = ModelInfo {
        id: "mock-1".into(),
        name: "Mock".into(),
        api: agent_core::types::Api::Mock,
        provider: "mock".into(),
        base_url: String::new(),
        reasoning: false,
        input: vec!["text".into()],
        context_window: 128_000,
        max_tokens: 4096,
    };

    let config =
        AgentLoopConfig::new(model, Arc::new(|msgs| Box::pin(default_convert_to_llm(msgs))));

    let tools: Vec<Arc<dyn AgentTool>> = vec![Arc::new(EchoTool)];
    let context = AgentContext {
        system_prompt: "You are a helpful assistant.".into(),
        messages: vec![],
        tools,
    };

    let prompt = AgentMessage::user_text("Hello!");
    let stream_fn = make_mock_stream_fn();

    let messages = AgentLoop::new(&config, &emit, &stream_fn)
        .run(vec![prompt], context, None)
        .await;

    println!("Events:");
    for e in log.lock().await.iter() {
        println!("  {}", e);
    }
    println!("\nMessages produced: {}", messages.len());
    for m in &messages {
        println!("  [{}]", m.role());
    }
}
