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

use agent_core::agent_loop::{AgentEventSink, AgentLoop};
use agent_core::event::AgentEvent;
use agent_core::harness::messages::default_convert_to_llm;
use agent_core::tool::AgentTool;
use agent_core::types::{
    AgentContext, AgentLoopConfig, AgentMessage, ContentBlock, Model, ModelInfo, ToolExecutionMode,
    Transport,
};
use coding_agent::stream_bridge::create_stream_fn;
use coding_agent::tools::{BashTool, BashToolOptions, LsTool};
use llm_client::{AnthropicProvider, AuthMethod, LlmProvider, ProviderConfig};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};

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
    let model_id = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-opus-4-8".into());

    let endpoint = format!("{}/v1/messages", base.trim_end_matches('/'));
    let provider: Arc<dyn LlmProvider> = Arc::new(AnthropicProvider::new(
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
                AgentEvent::MessageEnd { message } => {
                    if let Some(content) = message.assistant_content() {
                        for c in content {
                            match c {
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
                AgentEvent::ToolExecutionEnd { tool_name, result, is_error, .. } => {
                    let summary = result
                        .content
                        .iter()
                        .find_map(|b| {
                            if let ContentBlock::Text { text } = b {
                                Some(text.as_str())
                            } else {
                                None
                            }
                        })
                        .unwrap_or("(no text)");
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

    let llm_model = Model {
        id: model_id.clone(),
        name: model_id.clone(),
        api: agent_core::types::Api::Anthropic,
        provider: "deepseek".into(),
        base_url: endpoint.clone(),
        reasoning: false,
        input: vec!["text".into()],
        cost: Default::default(),
        context_window: 128_000,
        max_tokens: 8192,
        ..Default::default()
    };
    let model_info = ModelInfo::from(&llm_model);

    let config = AgentLoopConfig {
        max_tokens: Some(4096),
        temperature: Some(0.0),
        tool_execution: ToolExecutionMode::Sequential,
        transport: Transport::Sse,
        ..AgentLoopConfig::new(model_info, Arc::new(|msgs| Box::pin(default_convert_to_llm(msgs))))
    };

    let system_prompt = format!(
        "你是一个友好的中文助手，工作目录是 {cwd}。\n\
         可以用 bash 在该目录下执行命令，用 ls 列目录。\n\
         请保持简洁，必要时才调用工具。"
    );

    let make_context = || AgentContext {
        system_prompt: system_prompt.clone(),
        messages: vec![],
        tools: tools.clone(),
    };

    let mut context = make_context();
    let stream_fn = create_stream_fn(provider, llm_model);

    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();

    let mut turn: u32 = 0;
    loop {
        print!("\n> ");
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
                    let preview = match m {
                        AgentMessage::User { content, .. } => content
                            .as_blocks()
                            .iter()
                            .filter_map(|b| {
                                if let ContentBlock::Text { text } = b {
                                    Some(text.clone())
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(""),
                        AgentMessage::Assistant { content, .. } => content
                            .iter()
                            .map(|c| match c {
                                ContentBlock::Text { text } => text.clone(),
                                ContentBlock::ToolCall { name, .. } => format!("<tool {}>", name),
                                _ => String::new(),
                            })
                            .collect::<Vec<_>>()
                            .join(" "),
                        AgentMessage::ToolResult { tool_name, content, .. } => {
                            let text = content
                                .iter()
                                .filter_map(|b| {
                                    if let ContentBlock::Text { text } = b {
                                        Some(text.as_str())
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join("");
                            format!("<{}: {}>", tool_name, text)
                        }
                        _ => serde_json::to_string(&m.to_json()).unwrap_or_default(),
                    };
                    println!("  [{i}] {}: {}", m.role(), truncate(&preview, 200));
                }
                continue;
            }
            _ => {}
        }

        turn += 1;
        println!("\n--- turn {turn} ---");

        let prompt = AgentMessage::user_text(user_text);

        // AgentLoop::run appends the prompt to context.messages and returns the new
        // messages from this turn (user prompt + assistant turns + tool results).
        let new_messages = AgentLoop::new(&config, &emit, &stream_fn)
            .run(vec![prompt], context.clone(), None)
            .await;

        context.messages.extend(new_messages);
    }

    println!("\n=== session ended — {} turns, {} messages ===", turn, context.messages.len());
}
