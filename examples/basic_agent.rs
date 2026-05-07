// Basic agent example — demonstrates agent-core with a mock LLM provider

use agent_core::{
    agent_loop::{run_agent_loop, AgentEventSink, AssistantMessageEventStream, StreamFn, StreamFnInput},
    event::{AgentEvent, AssistantMessageEvent, EventStream},
    tool::AgentTool,
    types::{
        AgentContext, AgentLoopConfig, AgentMessage, AgentToolResult, AgentToolUpdateCallback,
        ContentBlock, Message, MessageContent, ModelInfo, ToolExecutionMode, Transport,
    },
};
use async_trait::async_trait;
use std::{pin::Pin, sync::Arc};
use tokio_util::sync::CancellationToken;

struct EchoTool;

#[async_trait]
impl AgentTool for EchoTool {
    fn name(&self) -> &str { "echo" }
    fn label(&self) -> &str { "echo" }
    fn description(&self) -> &str { "Echo the input back" }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        })
    }
    fn execution_mode(&self) -> Option<ToolExecutionMode> { None }
    async fn execute(
        &self, _id: String, params: serde_json::Value,
        _signal: Option<CancellationToken>, _on_update: Option<AgentToolUpdateCallback>,
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
                let msg: AgentMessage = serde_json::json!({
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello from the mock LLM!"}],
                    "api": "mock", "provider": "mock", "model": "mock-1",
                    "usage": {"input": 10, "output": 5, "cacheRead": 0, "cacheWrite": 0,
                              "totalTokens": 15, "cost": {"input": 0, "output": 0,
                              "cacheRead": 0, "cacheWrite": 0, "total": 0}},
                    "stopReason": "end_turn",
                    "timestamp": chrono::Utc::now().timestamp_millis()
                });
                stream2.push(AssistantMessageEvent::Start { partial: msg.clone() });
                stream2.push(AssistantMessageEvent::TextStart { content_index: 0, partial: msg.clone() });
                stream2.push(AssistantMessageEvent::TextDelta {
                    content_index: 0,
                    delta: "Hello from the mock LLM!".to_string(),
                    partial: msg.clone(),
                });
                stream2.push(AssistantMessageEvent::TextEnd {
                    content_index: 0,
                    content: "Hello from the mock LLM!".to_string(),
                    partial: msg.clone(),
                });
                stream2.push(AssistantMessageEvent::Done {
                    reason: "end_turn".to_string(),
                    message: msg.clone(),
                });
                stream2.end(msg);
            });
            Ok(stream)
        }) as Pin<Box<dyn std::future::Future<Output = Result<AssistantMessageEventStream, String>> + Send>>
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
                AgentEvent::MessageStart { message } =>
                    format!("message_start role={}", message["role"].as_str().unwrap_or("?")),
                AgentEvent::MessageEnd { message } =>
                    format!("message_end role={}", message["role"].as_str().unwrap_or("?")),
                _ => return,
            };
            log.lock().await.push(label);
        })
    });

    let model = ModelInfo {
        id: "mock-1".to_string(),
        name: "Mock".to_string(),
        api: "mock".to_string(),
        provider: "mock".to_string(),
        base_url: String::new(),
        reasoning: false,
        input: vec!["text".to_string()],
        context_window: 128_000,
        max_tokens: 4096,
    };

    let config = AgentLoopConfig {
        model,
        api_key: None,
        tool_execution: ToolExecutionMode::Sequential,
        session_id: None,
        thinking_budgets: None,
        transport: Transport::Sse,
        max_retry_delay_ms: None,
        reasoning: None,
        temperature: None,
        max_tokens: None,
        before_tool_call: None,
        after_tool_call: None,
        transform_context: None,
        get_steering_messages: None,
        get_follow_up_messages: None,
        get_api_key: None,
        convert_to_llm: Arc::new(|msgs| Box::pin(async move {
            msgs.into_iter().filter_map(|m| {
                let role = m["role"].as_str()?;
                match role {
                    "user" => Some(Message::User {
                        content: MessageContent::String(
                            m["content"][0]["text"].as_str().unwrap_or("").to_string()
                        ),
                        timestamp: m["timestamp"].as_u64().unwrap_or(0),
                        metadata: None,
                    }),
                    _ => None,
                }
            }).collect()
        })),
    };

    let context = AgentContext {
        system_prompt: "You are a helpful assistant.".to_string(),
        messages: vec![],
        tools: vec![],
    };

    let prompt: AgentMessage = serde_json::json!({
        "role": "user",
        "content": [{"type": "text", "text": "Hello!"}],
        "timestamp": chrono::Utc::now().timestamp_millis()
    });

    let tools: Vec<Arc<dyn AgentTool>> = vec![Arc::new(EchoTool)];
    let stream_fn = make_mock_stream_fn();

    let messages = run_agent_loop(
        vec![prompt], context, &config, &tools, &emit, None, &stream_fn,
    ).await;

    println!("Events:");
    for e in log.lock().await.iter() {
        println!("  {}", e);
    }
    println!("\nMessages produced: {}", messages.len());
    for m in &messages {
        println!("  [{}]", m["role"].as_str().unwrap_or("?"));
    }
}
