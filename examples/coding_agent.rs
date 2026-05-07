// Coding agent example — demonstrates session management and file tools

use coding_agent::{
    session::{AgentSession, SessionManager},
    tools::{EditTool, EditToolOptions, ReadTool, ReadToolOptions, WriteTool, WriteToolOptions},
};
use agent_core::tool::AgentTool;
use std::path::Path;
use tempfile::TempDir;

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

    // Write
    write.execute("w1".into(), serde_json::json!({
        "path": &file_path,
        "content": "fn main() {\n    println!(\"hello\");\n}\n"
    }), None, None).await.unwrap();
    println!("  ✓ Wrote hello.rs");

    // Read
    let result = read.execute("r1".into(), serde_json::json!({"path": &file_path}), None, None).await.unwrap();
    if let agent_core::types::ContentBlock::Text { text } = &result.content[0] {
        println!("  ✓ Read {} bytes", text.len());
    }

    // Edit
    edit.execute("e1".into(), serde_json::json!({
        "path": &file_path,
        "edits": [{"oldText": "hello", "newText": "world"}]
    }), None, None).await.unwrap();
    println!("  ✓ Edited hello → world");

    // Verify
    let result = read.execute("r2".into(), serde_json::json!({"path": &file_path}), None, None).await.unwrap();
    if let agent_core::types::ContentBlock::Text { text } = &result.content[0] {
        assert!(text.contains("world"), "edit should have replaced hello with world");
        println!("  ✓ Verified edit");
    }

    // ── 2. Session management demo ────────────────────────────────────────────
    println!("\n2. Session Management Demo");
    println!("--------------------------");

    let mut session = AgentSession::new(dir.path());

    let id1 = session.append_user_message("What is Rust?");
    println!("  ✓ Appended user message (id={})", &id1[..8]);

    // Simulate assistant response
    session.append_assistant_message(serde_json::json!({
        "role": "assistant",
        "content": [{"type": "text", "text": "Rust is a systems programming language."}],
        "api": "anthropic", "provider": "anthropic", "model": "claude-opus-4-7",
        "usage": {"input": 10, "output": 20, "cacheRead": 0, "cacheWrite": 0,
                  "totalTokens": 30, "cost": {"input": 0, "output": 0,
                  "cacheRead": 0, "cacheWrite": 0, "total": 0}},
        "stopReason": "end_turn",
        "timestamp": chrono::Utc::now().timestamp_millis()
    }));
    println!("  ✓ Appended assistant message");

    let ctx = session.build_context();
    println!("  ✓ Context: {} messages, model={:?}",
        ctx.messages.len(),
        ctx.model.as_ref().map(|m| m.model_id.as_str())
    );

    // Branch
    session.fork(&id1);
    let id3 = session.append_user_message("What is Go?");
    println!("  ✓ Branched and added alternative question (id={})", &id3[..8]);

    let ctx2 = session.build_context();
    println!("  ✓ Branch context: {} messages", ctx2.messages.len());

    // ── 3. Session persistence demo ───────────────────────────────────────────
    println!("\n3. Session Persistence Demo");
    println!("---------------------------");

    let session_dir = dir.path().join("sessions").to_string_lossy().to_string();
    let mut mgr = SessionManager::create(dir.path().to_str().unwrap(), Some(&session_dir));

    mgr.append_message(serde_json::json!({
        "role": "user", "content": "hello", "timestamp": 1000
    }));
    mgr.append_message(serde_json::json!({
        "role": "assistant", "content": "hi",
        "api": "t", "provider": "t", "model": "m",
        "usage": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,
                  "cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},
        "stopReason": "stop", "timestamp": 2000
    }));

    if let Some(path) = mgr.session_file() {
        println!("  ✓ Session persisted to: {}", path.file_name().unwrap().to_string_lossy());

        // Reload
        let reloaded = SessionManager::open(&path.to_string_lossy(), Some(&session_dir), None);
        let ctx = reloaded.build_context();
        println!("  ✓ Reloaded: {} messages", ctx.messages.len());
        assert_eq!(ctx.messages.len(), 2);
    } else {
        println!("  ℹ Session not yet persisted (no assistant message written)");
    }

    println!("\n✅ All demos completed successfully!");
}
