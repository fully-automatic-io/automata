// CodingAgentSession example — the high-level end-to-end API.
//
// Demonstrates `coding_agent::CodingAgentSession`, which wires the LLM provider,
// the AgentHarness (session + queues + auto-compaction/retry), the built-in
// tools, and the system prompt together. Drive it with `prompt(text)` and
// observe progress by subscribing to harness events.
//
// Run against DeepSeek's Anthropic-compatible endpoint:
//   ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic \
//   ANTHROPIC_AUTH_TOKEN=sk-... \
//   cargo run --example coding_session

use agent_core::event::AgentEvent;
use agent_core::harness::HarnessEvent;
use agent_core::types::{Api, ContentBlock, Model, ModelCost};
use coding_agent::{Auth, BuiltinTool, CodingAgentSession, SessionOptions, ToolSelection};

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
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

    let cwd = std::env::var("AUTOMATA_CWD").unwrap_or_else(|_| "/tmp/prog".into());
    std::fs::create_dir_all(&cwd).ok();

    let model = Model {
        id: model_id.clone(),
        name: model_id.clone(),
        api: Api::Anthropic,
        provider: "deepseek".into(),
        base_url: endpoint.clone(),
        reasoning: false,
        input: vec!["text".into()],
        cost: ModelCost::default(),
        context_window: 128_000,
        max_tokens: 8192,
        ..Default::default()
    };

    println!("=== CodingAgentSession — Rust hello world ===");
    println!("model    : {model_id}");
    println!("endpoint : {endpoint}");
    println!("workdir  : {cwd}\n");

    let mut options = SessionOptions::new(cwd.clone(), model, token);
    options.base_url = Some(endpoint);
    options.auth = Auth::Bearer; // DeepSeek's relay wants a bearer token.
    options.tools = ToolSelection::only([
        BuiltinTool::Bash,
        BuiltinTool::Write,
        BuiltinTool::Read,
        BuiltinTool::Ls,
    ]);
    options.system_prompt = format!(
        "You are a coding assistant. Working directory: {cwd}. \
         Use the bash/write/read/ls tools to complete tasks one step at a time."
    );

    let session = CodingAgentSession::builder(options).await.expect("build coding session");

    session
        .subscribe(|event, _signal| async move {
            if let HarnessEvent::Agent(AgentEvent::MessageEnd { message }) = event
                && let Some(content) = message.assistant_content()
            {
                for block in content {
                    match block {
                        ContentBlock::Text { text } if !text.is_empty() => {
                            println!("[assistant] {}", truncate(text, 600));
                        }
                        ContentBlock::ToolCall { name, arguments, .. } => {
                            let args = serde_json::to_string(arguments).unwrap_or_default();
                            println!("[tool_call] {name} {}", truncate(&args, 200));
                        }
                        _ => {}
                    }
                }
            }
        })
        .await;

    let messages = session
        .prompt(format!(
            "在 {cwd} 目录里新建一个 Rust 项目，打印 \"Hello, world from automata!\"，\
             完成后用一句话总结结果。"
        ))
        .await
        .expect("prompt failed");

    println!(
        "\n=== finished — {} messages, {} tool results ===",
        messages.len(),
        messages
            .iter()
            .filter(|m| matches!(m, agent_core::types::AgentMessage::ToolResult { .. }))
            .count(),
    );
}
