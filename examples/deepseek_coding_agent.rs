// DeepSeek coding-agent example — uses the real agent loop with bash/write/read/ls tools
// driven by DeepSeek's Anthropic-compatible endpoint to scaffold and run a hello-world Rust project.
//
// Run with:
//   ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic \
//   ANTHROPIC_AUTH_TOKEN=sk-... \
//   cargo run --example deepseek_coding_agent

use agent_core::agent_loop::{
    AgentEventSink, AssistantMessageEventStream, StreamFn, StreamFnInput, run_agent_loop,
};
use agent_core::event::{AgentEvent, AssistantMessageEvent, EventStream};
use agent_core::tool::AgentTool;
use agent_core::types::{
    AgentContext, AgentLoopConfig, AgentMessage, Message, ModelInfo, ToolExecutionMode, Transport,
};
use coding_agent::tools::{
    BashTool, BashToolOptions, LsTool, ReadTool, ReadToolOptions, WriteTool, WriteToolOptions,
};
use llm_client::{
    AnthropicProvider, AuthMethod, LlmMessage, LlmProvider, LlmRequest, ProviderConfig,
    StopReason as LlmStopReason, ToolDefinition,
};
use std::pin::Pin;
use std::sync::Arc;
use tempfile::TempDir;

fn build_stream_fn(provider: Arc<AnthropicProvider>) -> StreamFn {
    Arc::new(move |input: StreamFnInput| {
        let provider = provider.clone();
        Box::pin(async move {
            let stream: AssistantMessageEventStream = EventStream::new();
            let stream2 = stream.clone();

            tokio::spawn(async move {
                let llm_tools: Vec<ToolDefinition> = input
                    .tools
                    .iter()
                    .map(|t| ToolDefinition {
                        name: t.name().into(),
                        description: t.description().into(),
                        input_schema: t.parameters(),
                    })
                    .collect();

                // agent-core Message and llm-client LlmMessage share the same JSON shape
                // (same tag, same field renames) — round-trip via serde_json.
                let llm_messages: Vec<LlmMessage> = input
                    .messages
                    .iter()
                    .filter_map(|m| {
                        let v = serde_json::to_value(m).ok()?;
                        serde_json::from_value::<LlmMessage>(v).ok()
                    })
                    .collect();

                let request = LlmRequest {
                    model: input.model.id.clone(),
                    messages: llm_messages,
                    tools: llm_tools,
                    system: Some(input.system_prompt),
                    max_tokens: input.max_tokens.or(Some(4096)),
                    temperature: input.temperature,
                    stop_sequences: vec![],
                    extra: Default::default(),
                };

                match provider.complete(request).await {
                    Ok(resp) => {
                        let content: Vec<serde_json::Value> = resp
                            .content
                            .iter()
                            .map(|p| serde_json::to_value(p).unwrap_or(serde_json::Value::Null))
                            .collect();
                        let stop_reason = match resp.stop_reason {
                            LlmStopReason::ToolUse => "toolUse",
                            LlmStopReason::MaxTokens => "length",
                            LlmStopReason::StopSequence => "stop_sequence",
                            _ => "stop",
                        };
                        let msg = serde_json::json!({
                            "role": "assistant",
                            "content": content,
                            "api": input.model.api,
                            "provider": input.model.provider,
                            "model": resp.model,
                            "usage": {
                                "input": resp.usage.input,
                                "output": resp.usage.output,
                                "cacheRead": resp.usage.cache_read,
                                "cacheWrite": resp.usage.cache_write,
                                "totalTokens": resp.usage.total_tokens,
                                "cost": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}
                            },
                            "stopReason": stop_reason,
                            "timestamp": chrono::Utc::now().timestamp_millis()
                        });
                        stream2.push(AssistantMessageEvent::Start { partial: msg.clone() });
                        stream2.push(AssistantMessageEvent::Done {
                            reason: stop_reason.into(),
                            message: msg.clone(),
                        });
                        stream2.end(msg);
                    }
                    Err(e) => {
                        let err_text = format!("LLM request failed: {e}");
                        let msg = serde_json::json!({
                            "role": "assistant",
                            "content": [{"type": "text", "text": err_text.clone()}],
                            "api": input.model.api,
                            "provider": input.model.provider,
                            "model": input.model.id,
                            "usage": {
                                "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
                                "totalTokens": 0,
                                "cost": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}
                            },
                            "stopReason": "error",
                            "errorMessage": err_text,
                            "timestamp": chrono::Utc::now().timestamp_millis()
                        });
                        stream2.push(AssistantMessageEvent::Error {
                            reason: e.to_string(),
                            error: msg.clone(),
                        });
                        stream2.end(msg);
                    }
                }
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

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let end: String = s.chars().take(n).collect();
        format!("{end}…")
    }
}

#[tokio::main]
async fn main() {
    let base = std::env::var("ANTHROPIC_BASE_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com/anthropic".into());
    let token = std::env::var("ANTHROPIC_AUTH_TOKEN")
        .or_else(|_| std::env::var("DEEPSEEK_API_KEY"))
        .expect("Set ANTHROPIC_AUTH_TOKEN (or DEEPSEEK_API_KEY)");
    let model_id = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-opus-4-7".into());

    let endpoint = format!("{}/v1/messages", base.trim_end_matches('/'));
    let provider = Arc::new(AnthropicProvider::new(
        ProviderConfig::new(token)
            .with_base_url(endpoint.clone())
            .with_auth_method(AuthMethod::Bearer),
    ));

    //let dir = TempDir::new().expect("tempdir");
    //let cwd = dir.path().to_string_lossy().to_string();
    let cwd = "/tmp/prog/".to_string();

    println!("=== DeepSeek Coding Agent — Rust hello world ===");
    println!("model    : {model_id}");
    println!("endpoint : {endpoint}");
    println!("workdir  : {cwd}\n");

    let bash = Arc::new(BashTool::new(cwd.clone(), BashToolOptions::default()));
    let write = Arc::new(WriteTool::new(cwd.clone(), WriteToolOptions::default()));
    let read = Arc::new(ReadTool::new(cwd.clone(), ReadToolOptions::default()));
    let ls = Arc::new(LsTool::new(cwd.clone()));
    let tools: Vec<Arc<dyn AgentTool>> = vec![bash, write, read, ls];

    let emit: AgentEventSink = Arc::new(|event: AgentEvent| {
        Box::pin(async move {
            match &event {
                AgentEvent::TurnStart => println!("\n--- turn ---"),
                AgentEvent::MessageStart { message } => {
                    let role = message["role"].as_str().unwrap_or("?");
                    if role == "assistant" {
                        if let Some(arr) = message["content"].as_array() {
                            for c in arr {
                                match c["type"].as_str() {
                                    Some("text") => {
                                        if let Some(t) = c["text"].as_str() {
                                            if !t.is_empty() {
                                                println!("[assistant] {}", truncate(t, 800));
                                            }
                                        }
                                    }
                                    Some("toolCall") => {
                                        let name = c["name"].as_str().unwrap_or("?");
                                        let args = serde_json::to_string(&c["arguments"])
                                            .unwrap_or_default();
                                        println!("[tool_call] {} {}", name, truncate(&args, 200));
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                AgentEvent::ToolExecutionEnd { tool_name, result, is_error, .. } => {
                    let summary = result["content"][0]["text"].as_str().unwrap_or("(no text)");
                    let prefix = if *is_error {
                        "[tool_error]"
                    } else {
                        "[tool_result]"
                    };
                    println!("{} {}: {}", prefix, tool_name, truncate(summary, 400));
                }
                _ => {}
            }
        })
    });

    let model = ModelInfo {
        id: model_id.clone(),
        name: model_id.clone(),
        api: "anthropic".into(),
        provider: "deepseek".into(),
        base_url: endpoint,
        reasoning: false,
        input: vec!["text".into()],
        context_window: 128_000,
        max_tokens: 8192,
    };

    let config = AgentLoopConfig {
        model: model.clone(),
        api_key: None,
        tool_execution: ToolExecutionMode::Sequential,
        session_id: None,
        thinking_budgets: None,
        transport: Transport::Sse,
        max_retry_delay_ms: None,
        reasoning: None,
        temperature: Some(0.0),
        max_tokens: Some(4096),
        before_tool_call: None,
        after_tool_call: None,
        transform_context: None,
        get_steering_messages: None,
        get_follow_up_messages: None,
        get_api_key: None,
        convert_to_llm: Arc::new(|msgs| {
            Box::pin(async move {
                msgs.into_iter()
                    .filter_map(|m| serde_json::from_value::<Message>(m).ok())
                    .collect()
            })
        }),
    };

    let context = AgentContext {
        system_prompt: format!(
            "你是一个编程助手，工作目录已经设置为 {cwd}。\n\
             你可以使用以下工具：\n\
             - bash: 在工作目录里执行 shell 命令（cargo、ls、cat 等都可以用）\n\
             - write: 写文件（路径相对工作目录或绝对都可）\n\
             - read: 读文件\n\
             - ls: 列目录\n\
             注意：bash 工具会把 stderr 与 stdout 合并，cargo 的进度输出在 stderr。\n\
             请按用户要求一步步完成，每一步只做一件事。完成后用一句话总结。"
        ),
        messages: vec![],
        tools: vec!["bash".into(), "write".into(), "read".into(), "ls".into()],
    };

    let user_text = format!(
        "请在 {cwd} 目录里完成下面的事：\n\
         1) 用 `cargo new --bin hello` 创建一个二进制项目\n\
         2) 修改 src/main.rs，让它打印 \"Hello, world from automata!\"\n\
         3) 用 `cargo run --quiet` 编译并运行，把运行输出贴出来\n\
         做完之后用一句话告诉我结果。"
    );
    let prompt: AgentMessage = serde_json::json!({
        "role": "user",
        "content": [{"type": "text", "text": user_text}],
        "timestamp": chrono::Utc::now().timestamp_millis()
    });

    let stream_fn = build_stream_fn(provider);
    let messages =
        run_agent_loop(vec![prompt], context, &config, &tools, &emit, None, &stream_fn).await;

    println!(
        "\n=== finished — {} messages, {} tool turns ===",
        messages.len(),
        messages.iter().filter(|m| m["role"] == "toolResult").count(),
    );
}
