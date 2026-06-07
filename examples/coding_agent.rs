// Coding agent SDK example — demonstrates the Rust-style setup path:
// settings + auth + model registry + resource loading + JSONL v3 sessions.

use agent_core::harness::session::JsonlSessionRepo;
use agent_core::tool::AgentTool;
use agent_core::types::{AgentMessage, ContentBlock};
use coding_agent::tools::{
    EditTool, EditToolOptions, ReadTool, ReadToolOptions, WriteTool, WriteToolOptions,
};
use coding_agent::{
    BuiltinTool, CreateAgentSessionOptions, SessionManager, ToolSelection, create_agent_session,
};
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("=== Automata Coding Agent SDK Example ===\n");

    let dir = TempDir::new()?;
    let cwd_path = dir.path().join("repo");
    let agent_dir = dir.path().join("agent");
    std::fs::create_dir_all(cwd_path.join(".automata/skills"))?;
    std::fs::create_dir_all(cwd_path.join(".automata/prompts"))?;
    std::fs::create_dir_all(&agent_dir)?;

    let cwd = cwd_path.to_string_lossy().to_string();

    std::fs::write(
        agent_dir.join("settings.json"),
        serde_json::json!({
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "system_prompt": "You are Automata's coding assistant.",
            "append_system_prompt": ["Keep answers concise and cite touched files."],
            "shell_path": std::env::var("SHELL").ok(),
            "shell_command_prefix": "set -e"
        })
        .to_string(),
    )?;
    std::fs::write(
        cwd_path.join("AGENTS.md"),
        "Project instruction: prefer small, focused Rust changes.",
    )?;
    std::fs::write(
        cwd_path.join(".automata/skills/rust-review.md"),
        "---\nname: rust-review\ndescription: Review Rust code for correctness\n---\n# Rust review\n",
    )?;
    std::fs::write(
        agent_dir.join("auth.json"),
        serde_json::json!({
            "anthropic": { "type": "api_key", "key": "sk-example-from-auth-json" }
        })
        .to_string(),
    )?;

    // ── 1. SDK setup demo ───────────────────────────────────────────────────
    println!("1. SDK Setup Demo");
    println!("-----------------");

    let mut handle = create_agent_session(CreateAgentSessionOptions {
        cwd: cwd_path.clone(),
        agent_dir: Some(agent_dir.clone()),
        session: SessionManager::create(),
        model: None,
        api_key: Some("sk-example-runtime-override".into()),
        thinking_level: None,
        tools: ToolSelection::only([BuiltinTool::Read, BuiltinTool::Write, BuiltinTool::Ls]),
    })
    .await?;

    let model_id = handle
        .selected_model
        .as_ref()
        .map(|model| model.id.as_str())
        .unwrap_or("(none)");
    println!("  ✓ Selected model: {model_id}");
    println!("  ✓ Loaded tools: {}", handle.tool_names.join(", "));
    println!("  ✓ Context files: {}", handle.resources.context_files.len());
    println!("  ✓ Skills: {}", handle.resources.skills.len());
    println!(
        "  ✓ Runtime auth configured: {}",
        handle.auth.auth_status("anthropic").configured
    );

    let prompt = handle.system_prompt();
    assert!(prompt.contains("Project instruction"));
    assert!(prompt.contains("rust-review"));
    println!("  ✓ Built system prompt from settings/resources");

    handle.session.append_message(AgentMessage::user_text("hello sdk")).await?;
    let sdk_context = handle.session.build_context().await?;
    assert_eq!(sdk_context.active_tool_names, Some(handle.tool_names.clone()));
    println!("  ✓ JSONL v3 session restored active tools");

    // ── 2. File tools demo ──────────────────────────────────────────────────
    println!("\n2. File Tools Demo");
    println!("------------------");

    let write = WriteTool::new(cwd.clone(), WriteToolOptions::default());
    let read = ReadTool::new(cwd.clone(), ReadToolOptions::default());
    let edit = EditTool::new(cwd.clone(), EditToolOptions::default());

    let file_path = cwd_path.join("hello.rs").to_string_lossy().to_string();

    write
        .execute(
            "w1".into(),
            serde_json::json!({
                "path": &file_path,
                "content": "fn main() {\n    println!(\"hello\");\n}\n"
            }),
            None,
            None,
        )
        .await?;
    println!("  ✓ Wrote hello.rs");

    let result = read
        .execute("r1".into(), serde_json::json!({"path": &file_path}), None, None)
        .await?;
    if let ContentBlock::Text { text } = &result.content[0] {
        println!("  ✓ Read {} bytes", text.len());
    }

    edit.execute(
        "e1".into(),
        serde_json::json!({
            "path": &file_path,
            "edits": [{"oldText": "hello", "newText": "world"}]
        }),
        None,
        None,
    )
    .await?;
    println!("  ✓ Edited hello → world");

    let result = read
        .execute("r2".into(), serde_json::json!({"path": &file_path}), None, None)
        .await?;
    if let ContentBlock::Text { text } = &result.content[0] {
        assert!(text.contains("world"), "edit should have replaced hello with world");
        println!("  ✓ Verified edit");
    }

    // ── 3. Session persistence demo ─────────────────────────────────────────
    println!("\n3. Session Persistence Demo");
    println!("---------------------------");

    let sessions_root = dir.path().join("manual-sessions");
    let repo = JsonlSessionRepo::new(&sessions_root);
    let mut session = repo.create(&cwd, None, None).await?;

    session.append_active_tools_change(vec!["read".into(), "write".into()]).await?;
    session.append_message(AgentMessage::user_text("hello")).await?;

    let id = session.get_metadata().await.id;
    drop(session);

    let listed = repo.list(Some(&cwd)).await?;
    let entry = listed.iter().find(|m| m.id == id).expect("session listed");
    println!("  ✓ Session persisted to: {}", entry.path);

    let reloaded = repo.open_by_path(&entry.path).await?;
    let ctx = reloaded.build_context().await?;
    assert_eq!(ctx.active_tool_names, Some(vec!["read".into(), "write".into()]));
    println!("  ✓ Reloaded v3 session with active tool state");

    println!("\nAll demos completed successfully.");
    Ok(())
}
