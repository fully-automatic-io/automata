// Coding agent example — demonstrates session management and file tools.

use agent_core::harness::session::{InMemorySessionStorage, JsonlSessionRepo, Session};
use agent_core::tool::AgentTool;
use agent_core::types::{AgentMessage, ContentBlock, MessageContent, StopReason, Usage};
use coding_agent::tools::{EditTool, EditToolOptions, ReadTool, ReadToolOptions, WriteTool, WriteToolOptions};
use tempfile::TempDir;

fn user(text: &str) -> AgentMessage {
    AgentMessage::User {
        content: MessageContent::Blocks(vec![ContentBlock::Text { text: text.into() }]),
        timestamp: chrono::Utc::now().timestamp_millis().max(0) as u64,
        metadata: None,
    }
}

fn assistant(text: &str, api: agent_core::types::Api, provider: &str, model: &str) -> AgentMessage {
    AgentMessage::Assistant {
        content: vec![ContentBlock::Text { text: text.into() }],
        api, provider: provider.into(), model: model.into(),
        usage: Usage { input: 10, output: 20, total_tokens: 30, ..Default::default() },
        stop_reason: StopReason::EndTurn,
        error_message: None,
        timestamp: chrono::Utc::now().timestamp_millis().max(0) as u64,
    }
}

#[tokio::main]
async fn main() {
    println!("=== Automata Coding Agent Example ===\n");

    let dir = TempDir::new().unwrap();
    let cwd = dir.path().to_string_lossy().to_string();

    // ── 1. File tools demo ────────────────────────────────────────────────────
    println!("1. File Tools Demo");
    println!("------------------");

    let write = WriteTool::new(cwd.clone(), WriteToolOptions::default());
    let read  = ReadTool::new(cwd.clone(), ReadToolOptions::default());
    let edit  = EditTool::new(cwd.clone(), EditToolOptions::default());

    let file_path = dir.path().join("hello.rs").to_string_lossy().to_string();

    write.execute("w1".into(), serde_json::json!({
        "path": &file_path,
        "content": "fn main() {\n    println!(\"hello\");\n}\n"
    }), None, None).await.unwrap();
    println!("  ✓ Wrote hello.rs");

    let result = read.execute("r1".into(), serde_json::json!({"path": &file_path}), None, None).await.unwrap();
    if let ContentBlock::Text { text } = &result.content[0] {
        println!("  ✓ Read {} bytes", text.len());
    }

    edit.execute("e1".into(), serde_json::json!({
        "path": &file_path,
        "edits": [{"oldText": "hello", "newText": "world"}]
    }), None, None).await.unwrap();
    println!("  ✓ Edited hello → world");

    let result = read.execute("r2".into(), serde_json::json!({"path": &file_path}), None, None).await.unwrap();
    if let ContentBlock::Text { text } = &result.content[0] {
        assert!(text.contains("world"), "edit should have replaced hello with world");
        println!("  ✓ Verified edit");
    }

    // ── 2. Session management (in-memory) ─────────────────────────────────────
    println!("\n2. Session Management Demo");
    println!("--------------------------");

    let mut session = Session::new(Box::new(InMemorySessionStorage::new(None)));

    let id1 = session.append_message(user("What is Rust?")).await.unwrap();
    println!("  ✓ Appended user message (id={})", &id1[..id1.len().min(8)]);

    session.append_message(assistant(
        "Rust is a systems programming language.",
        agent_core::types::Api::Anthropic, "anthropic", "claude-opus-4-7",
    )).await.unwrap();
    println!("  ✓ Appended assistant message");

    let ctx = session.build_context().await.unwrap();
    println!("  ✓ Context: {} messages, model={:?}",
        ctx.messages.len(),
        ctx.model.as_ref().map(|m| m.model_id.as_str())
    );

    session.move_to(Some(&id1), None).await.unwrap();
    let id3 = session.append_message(user("What is Go?")).await.unwrap();
    println!("  ✓ Branched and added alternative question (id={})", &id3[..id3.len().min(8)]);

    let ctx2 = session.build_context().await.unwrap();
    println!("  ✓ Branch context: {} messages", ctx2.messages.len());

    // ── 3. Session persistence (JSONL on disk) ───────────────────────────────
    println!("\n3. Session Persistence Demo");
    println!("---------------------------");

    let sessions_root = dir.path().join("sessions");
    let repo = JsonlSessionRepo::new(&sessions_root);
    let mut mgr = repo.create(dir.path().to_str().unwrap(), None, None).await.unwrap();

    mgr.append_message(user("hello")).await.unwrap();
    mgr.append_message(assistant("hi", agent_core::types::Api::Anthropic, "t", "m")).await.unwrap();

    let id = mgr.get_metadata().await.id;
    drop(mgr);

    let listed = repo.list(Some(dir.path().to_str().unwrap())).await.unwrap();
    let entry = listed.iter().find(|m| m.id == id).expect("session listed");
    println!("  ✓ Session persisted to: {}", entry.path);

    let reloaded = repo.open_by_path(&entry.path).await.unwrap();
    let ctx = reloaded.build_context().await.unwrap();
    println!("  ✓ Reloaded: {} messages", ctx.messages.len());
    assert_eq!(ctx.messages.len(), 2);

    println!("\n✅ All demos completed successfully!");
}
