// DeepSeek coding-agent example — high-level CodingAgentSession using the
// Anthropic-compatible DeepSeek endpoint.
//
// Run with:
//   ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic \
//   ANTHROPIC_AUTH_TOKEN=sk-... \
//   cargo run --example deepseek_coding_agent

use agent_core::event::AgentEvent;
use agent_core::harness::HarnessEvent;
use agent_core::types::{Api, ContentBlock, Model, ModelCost};
use coding_agent::{Auth, BuiltinTool, CodingAgentSession, SessionOptions, ToolSelection};

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
    let cwd = std::env::var("AUTOMATA_CWD").unwrap_or_else(|_| "/tmp/prog".into());
    std::fs::create_dir_all(&cwd).ok();

    println!("=== DeepSeek Coding Agent — Rust hello world ===");
    println!("model    : {model_id}");
    println!("endpoint : {endpoint}");
    println!("workdir  : {cwd}\n");

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

    let mut options = SessionOptions::new(cwd.clone(), model, token);
    options.base_url = Some(endpoint);
    options.auth = Auth::Bearer;
    options.tools = ToolSelection::only([
        BuiltinTool::Bash,
        BuiltinTool::Write,
        BuiltinTool::Read,
        BuiltinTool::Ls,
    ]);
    options.system_prompt = format!(
        "你是一个编程助手，工作目录已经设置为 {cwd}。\n\
         可以用 bash/write/read/ls 工具完成任务。每一步只做必要操作，完成后用一句话总结。"
    );

    let session = CodingAgentSession::builder(options)
        .await
        .expect("build DeepSeek coding session");

    session
        .subscribe(|event, _signal| async move {
            match event {
                HarnessEvent::Agent(AgentEvent::TurnStart) => println!("\n--- turn ---"),
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
                _ => {}
            }
        })
        .await;

    let messages = session
        .prompt(format!(
            "请在 {cwd} 目录里新建一个 Rust 项目，打印 \"Hello, world from automata!\"，\
             完成后用一句话告诉我结果。"
        ))
        .await
        .expect("prompt failed");

    println!(
        "\n=== finished — {} messages, {} tool turns ===",
        messages.len(),
        messages
            .iter()
            .filter(|m| matches!(m, agent_core::types::AgentMessage::ToolResult { .. }))
            .count(),
    );
}
