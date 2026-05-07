// Multi-turn chat example — keeps a single AgentContext across turns and feeds
// each new user prompt back into run_agent_loop together with the accumulated
// message history (assistant turns, tool calls, tool results all preserved).
//
// Run with:
//   ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic \
//   ANTHROPIC_AUTH_TOKEN=sk-... \
//   cargo run --example multi_turn_chat
//
// Type your message and press Enter. Type `/exit` (or Ctrl-D) to quit.
// Type `/history` to dump the message log, `/reset` to start over.

use agent_core::agent_loop::{
    AgentEventSink, AssistantMessageEventStream, StreamFn, StreamFnInput, run_agent_loop,
};
use agent_core::event::{AgentEvent, AssistantMessageEvent, EventStream};
use agent_core::tool::AgentTool;
use agent_core::types::{
    AgentContext, AgentLoopConfig, AgentMessage, Message, ModelInfo, ToolExecutionMode, Transport,
};
use coding_agent::tools::{BashTool, BashToolOptions, LsTool};
use llm_client::{
    AnthropicProvider, AuthMethod, LlmMessage, LlmProvider, LlmRequest, ProviderConfig,
    StopReason as LlmStopReason, ToolDefinition,
};
use std::pin::Pin;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};

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

    let cwd = std::env::var("AGENT_CWD").unwrap_or_else(|_| "/tmp".into());

    println!("=== Multi-turn chat ===");
    println!("model    : {model_id}");
    println!("endpoint : {endpoint}");
    println!("workdir  : {cwd}");
    println!("commands : /exit, /reset, /history\n");

    let bash = Arc::new(BashTool::new(cwd.clone(), BashToolOptions::default()));
    let ls = Arc::new(LsTool::new(cwd.clone()));
    let tools: Vec<Arc<dyn AgentTool>> = vec![bash, ls];

    let emit: AgentEventSink = Arc::new(|event: AgentEvent| {
        Box::pin(async move {
            match &event {
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

    let system_prompt = format!(
        "你是一个友好的中文助手，工作目录是 {cwd}。\n\
         可以用 bash 在该目录下执行命令，用 ls 列目录。\n\
         请保持简洁，必要时才调用工具。"
    );

    let make_context = || AgentContext {
        system_prompt: system_prompt.clone(),
        messages: vec![],
        tools: vec!["bash".into(), "ls".into()],
    };

    let mut context = make_context();
    let stream_fn = build_stream_fn(provider);

    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();

    let mut turn: u32 = 0;
    loop {
        print!("\n> ");
        // 立即把提示符刷出来
        use std::io::Write as _;
        let _ = std::io::stdout().flush();

        let line = match lines.next_line().await {
            Ok(Some(l)) => l,
            Ok(None) => {
                println!("\n(EOF — bye)");
                break;
            }
            Err(e) => {
                eprintln!("stdin error: {e}");
                break;
            }
        };
        let user_text = line.trim();
        if user_text.is_empty() {
            continue;
        }

        match user_text {
            "/exit" | "/quit" => {
                println!("bye.");
                break;
            }
            "/reset" => {
                context = make_context();
                turn = 0;
                println!("(history cleared)");
                continue;
            }
            "/history" => {
                println!("--- history ({} messages) ---", context.messages.len());
                for (i, m) in context.messages.iter().enumerate() {
                    let role = m["role"].as_str().unwrap_or("?");
                    let preview = match role {
                        "user" => m["content"][0]["text"].as_str().unwrap_or("").to_string(),
                        "assistant" => m["content"]
                            .as_array()
                            .map(|arr| {
                                arr.iter()
                                    .map(|c| match c["type"].as_str() {
                                        Some("text") => {
                                            c["text"].as_str().unwrap_or("").to_string()
                                        }
                                        Some("toolCall") => {
                                            format!("<tool {}>", c["name"].as_str().unwrap_or("?"))
                                        }
                                        _ => String::new(),
                                    })
                                    .collect::<Vec<_>>()
                                    .join(" ")
                            })
                            .unwrap_or_default(),
                        "toolResult" => format!(
                            "<{}: {}>",
                            m["toolName"].as_str().unwrap_or("?"),
                            m["content"][0]["text"].as_str().unwrap_or("")
                        ),
                        _ => serde_json::to_string(m).unwrap_or_default(),
                    };
                    println!("  [{i}] {role}: {}", truncate(&preview, 200));
                }
                continue;
            }
            _ => {}
        }

        turn += 1;
        println!("\n--- turn {turn} ---");

        let prompt: AgentMessage = serde_json::json!({
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
            "timestamp": chrono::Utc::now().timestamp_millis()
        });

        // run_agent_loop 会把 prompt 拼到 context.messages 后面，跑完返回这一轮新增的所有
        // 消息（user prompt + assistant turns + tool results）。把它们追加回 context，
        // 下一轮就拥有完整历史。
        let new_messages =
            run_agent_loop(vec![prompt], context.clone(), &config, &tools, &emit, None, &stream_fn)
                .await;

        context.messages.extend(new_messages);
    }

    println!("\n=== session ended — {} turns, {} messages ===", turn, context.messages.len());
}
