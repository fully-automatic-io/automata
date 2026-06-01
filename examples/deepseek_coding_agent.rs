// DeepSeek coding-agent example — uses the real agent loop with bash/write/read/ls tools
// driven by DeepSeek's Anthropic-compatible endpoint to scaffold and run a hello-world Rust project.
//
// Run with:
//   ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic \
//   ANTHROPIC_AUTH_TOKEN=sk-... \
//   cargo run --example deepseek_coding_agent

use agent_core::agent_loop::{AgentEventSink, AgentLoop};
use agent_core::event::AgentEvent;
use agent_core::harness::messages::default_convert_to_llm;
use agent_core::tool::AgentTool;
use agent_core::types::Model;
use agent_core::types::{
    AgentContext, AgentLoopConfig, AgentMessage, ContentBlock, MessageContent, ModelInfo,
    ToolExecutionMode, Transport,
};
use coding_agent::stream_bridge::create_stream_fn;
use coding_agent::tools::{
    BashTool, BashToolOptions, LsTool, ReadTool, ReadToolOptions, WriteTool, WriteToolOptions,
};
use llm_client::{AnthropicProvider, AuthMethod, LlmProvider, ProviderConfig};
use std::sync::Arc;

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
        tools,
    };

    let user_text = format!(
        "请在 {cwd} 目录里完成下面的事：\n\
         新建一个rust项目打印： \"Hello, world from automata!\"\n\
         做完之后用一句话告诉我结果。"
    );
    let prompt = AgentMessage::User {
        content: MessageContent::Blocks(vec![ContentBlock::Text { text: user_text }]),
        timestamp: chrono::Utc::now().timestamp_millis() as u64,
        metadata: None,
    };

    let stream_fn = create_stream_fn(provider, llm_model);
    let messages = AgentLoop::new(&config, &emit, &stream_fn)
        .run(vec![prompt], context, None)
        .await;

    println!(
        "\n=== finished — {} messages, {} tool turns ===",
        messages.len(),
        messages.iter().filter(|m| matches!(m, AgentMessage::ToolResult { .. })).count(),
    );
}
