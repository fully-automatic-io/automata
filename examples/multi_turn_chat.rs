// Multi-turn chat example — uses agent-core Session + AgentHarness directly.
//
// This mirrors pi-mono's layering: agent-core owns the session tree and harness;
// coding-agent only supplies coding tools and the llm-client stream bridge.
//
// Run with:
//   ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic \
//   ANTHROPIC_AUTH_TOKEN=sk-... \
//   cargo run --example multi_turn_chat
//
// Optional persistence:
//   AGENT_SESSIONS_DIR=/tmp/automata-sessions cargo run --example multi_turn_chat
//
// Type your message and press Enter. Type `/exit` (or Ctrl-D) to quit.
// Type `/history` to dump the session log, `/reset` to start over.

use agent_core::event::AgentEvent;
use agent_core::harness::session::{InMemorySessionStorage, JsonlSessionRepo, Session};
use agent_core::harness::{
    AgentHarness, AgentHarnessOptions, HarnessConfig, HarnessEvent, StreamOptions,
};
use agent_core::types::{
    AgentMessage, Api, ContentBlock, Model, ModelCost, ModelInfo, ThinkingLevel, ToolExecutionMode,
    Transport,
};
use coding_agent::stream_bridge::create_stream_fn;
use coding_agent::{Auth, ProviderBuild, build_provider, build_tools};
use tokio::io::{AsyncBufReadExt, BufReader};

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let end: String = s.chars().take(n).collect();
        format!("{end}…")
    }
}

fn model(endpoint: &str, model_id: &str) -> Model {
    Model {
        id: model_id.to_string(),
        name: model_id.to_string(),
        api: Api::Anthropic,
        provider: "deepseek".into(),
        base_url: endpoint.to_string(),
        reasoning: false,
        input: vec!["text".into()],
        cost: ModelCost::default(),
        context_window: 128_000,
        max_tokens: 8192,
        ..Default::default()
    }
}

async fn new_agent_core_session(cwd: &str) -> Session {
    if let Ok(root) = std::env::var("AGENT_SESSIONS_DIR") {
        let repo = JsonlSessionRepo::new(root);
        repo.create(cwd, None, None).await.expect("create JSONL session")
    } else {
        Session::new(Box::new(InMemorySessionStorage::new(None)))
    }
}

async fn build_harness(cwd: &str, endpoint: &str, token: &str, model_id: &str) -> AgentHarness {
    let session = new_agent_core_session(cwd).await;
    let model = model(endpoint, model_id);
    let provider = build_provider(ProviderBuild {
        model: &model,
        api_key: token.to_string(),
        base_url: Some(endpoint.to_string()),
        auth: Auth::Bearer,
    });

    let harness = AgentHarness::new(
        session,
        HarnessConfig {
            system_prompt: format!(
                "你是一个友好的中文助手，工作目录是 {cwd}。\n\
                 可以用 bash 在该目录下执行命令，用 ls 列目录。请保持简洁，必要时才调用工具。"
            ),
            thinking_level: ThinkingLevel::Off,
            model_provider: model.provider.clone(),
            model_id: model.id.clone(),
        },
        AgentHarnessOptions {
            stream_fn: create_stream_fn(provider, model.clone()),
            convert_to_llm: None,
            transform_context: None,
            before_tool_call: None,
            after_tool_call: None,
            should_stop_after_turn: None,
            prepare_next_turn: None,
            on_payload: None,
            on_response: None,
        },
    );

    let tool_names = ["bash", "ls"];
    let tools = build_tools(cwd, &tool_names);
    harness
        .set_tools(tools, Some(tool_names.iter().map(|name| (*name).to_string()).collect()))
        .await
        .expect("set active tools");
    harness.set_model_info(ModelInfo::from(&model)).await;
    harness
        .set_stream_options(StreamOptions {
            max_tokens: Some(4096),
            temperature: Some(0.0),
            transport: Some(Transport::Sse),
            tool_execution: Some(ToolExecutionMode::Sequential),
            ..Default::default()
        })
        .await;

    harness
}

async fn subscribe_harness(harness: &AgentHarness) {
    harness
        .subscribe(|event, _signal| async move {
            match event {
                HarnessEvent::Agent(AgentEvent::MessageEnd { message }) => {
                    if let Some(content) = message.assistant_content() {
                        for block in content {
                            match block {
                                ContentBlock::Text { text } if !text.is_empty() => {
                                    println!("[assistant] {}", truncate(text, 800));
                                }
                                ContentBlock::ToolCall { name, arguments, .. } => {
                                    let args = serde_json::to_string(arguments).unwrap_or_default();
                                    println!("[tool_call] {} {}", name, truncate(&args, 200));
                                }
                                _ => {}
                            }
                        }
                    }
                }
                HarnessEvent::Agent(AgentEvent::ToolExecutionEnd {
                    tool_name,
                    result,
                    is_error,
                    ..
                }) => {
                    let summary = result
                        .content
                        .iter()
                        .find_map(|block| {
                            if let ContentBlock::Text { text } = block {
                                Some(text.as_str())
                            } else {
                                None
                            }
                        })
                        .unwrap_or("(no text)");
                    let prefix = if is_error {
                        "[tool_error]"
                    } else {
                        "[tool_result]"
                    };
                    println!("{} {}: {}", prefix, tool_name, truncate(summary, 400));
                }
                HarnessEvent::SavePoint => {
                    println!("[session] save point");
                }
                _ => {}
            }
        })
        .await;
}

fn preview_message(message: &AgentMessage) -> String {
    match message {
        AgentMessage::User { content, .. } => content
            .as_blocks()
            .iter()
            .filter_map(|block| {
                if let ContentBlock::Text { text } = block {
                    Some(text.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        AgentMessage::Assistant { content, .. } => content
            .iter()
            .map(|block| match block {
                ContentBlock::Text { text } => text.clone(),
                ContentBlock::ToolCall { name, .. } => format!("<tool {name}>"),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join(" "),
        AgentMessage::ToolResult { tool_name, content, .. } => {
            let text = content
                .iter()
                .filter_map(|block| {
                    if let ContentBlock::Text { text } = block {
                        Some(text.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("");
            format!("<{tool_name}: {text}>")
        }
        _ => serde_json::to_string(&message.to_json()).unwrap_or_default(),
    }
}

#[tokio::main]
async fn main() {
    let base = std::env::var("ANTHROPIC_BASE_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com/anthropic".into());
    let token = std::env::var("ANTHROPIC_AUTH_TOKEN")
        .or_else(|_| std::env::var("DEEPSEEK_API_KEY"))
        .expect("Set ANTHROPIC_AUTH_TOKEN (or DEEPSEEK_API_KEY)");
    let model_id = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-opus-4-8".into());

    let endpoint = format!("{}/v1/messages", base.trim_end_matches('/'));
    let cwd = std::env::var("AGENT_CWD").unwrap_or_else(|_| "/tmp".into());
    let session_mode = std::env::var("AGENT_SESSIONS_DIR")
        .map(|root| format!("jsonl at {root}"))
        .unwrap_or_else(|_| "in-memory".into());

    println!("=== Agent-core multi-turn chat ===");
    println!("model    : {model_id}");
    println!("endpoint : {endpoint}");
    println!("workdir  : {cwd}");
    println!("session  : {session_mode}");
    println!("commands : /exit, /reset, /history\n");

    let mut harness = build_harness(&cwd, &endpoint, &token, &model_id).await;
    subscribe_harness(&harness).await;

    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();

    let mut turn: u32 = 0;
    loop {
        print!("\n> ");
        use std::io::Write as _;
        let _ = std::io::stdout().flush();

        let line = match lines.next_line().await {
            Ok(Some(line)) => line,
            Ok(None) => {
                println!("\n(EOF — bye)");
                break;
            }
            Err(err) => {
                eprintln!("stdin error: {err}");
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
                harness = build_harness(&cwd, &endpoint, &token, &model_id).await;
                subscribe_harness(&harness).await;
                turn = 0;
                println!("(new agent-core session)");
                continue;
            }
            "/history" => {
                let context = harness.build_context().await.expect("history");
                println!("--- history ({} messages) ---", context.messages.len());
                println!("active tools: {:?}", context.active_tool_names);
                for (i, message) in context.messages.iter().enumerate() {
                    println!(
                        "  [{i}] {}: {}",
                        message.role(),
                        truncate(&preview_message(message), 200)
                    );
                }
                continue;
            }
            _ => {}
        }

        turn += 1;
        println!("\n--- turn {turn} ---");

        if let Err(err) = harness.execute_turn(vec![AgentMessage::user_text(user_text)]).await {
            eprintln!("turn failed: {err}");
        }
    }

    let messages = harness
        .build_context()
        .await
        .map(|context| context.messages.len())
        .unwrap_or_default();
    println!("\n=== session ended — {} turns, {} messages ===", turn, messages);
}
