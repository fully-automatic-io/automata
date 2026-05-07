// Integration tests for coding-agent tools

use coding_agent::tools::{
    EditTool, EditToolOptions, FindTool, GrepTool, LsTool, ReadTool, ReadToolOptions, WriteTool,
    WriteToolOptions,
};
use agent_core::tool::AgentTool;
use std::fs;
use tempfile::TempDir;

#[tokio::test]
async fn test_write_then_read_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.txt");
    let path_str = path.to_string_lossy().to_string();

    let write = WriteTool::new(dir.path().to_string_lossy().to_string(), WriteToolOptions::default());
    write.execute("1".into(), serde_json::json!({"path": &path_str, "content": "hello\nworld\n"}), None, None).await.unwrap();

    let read = ReadTool::new(dir.path().to_string_lossy().to_string(), ReadToolOptions::default());
    let result = read.execute("2".into(), serde_json::json!({"path": &path_str}), None, None).await.unwrap();

    let text = match &result.content[0] {
        agent_core::types::ContentBlock::Text { text } => text.clone(),
        _ => panic!("expected text"),
    };
    assert!(text.contains("hello"));
    assert!(text.contains("world"));
}

#[tokio::test]
async fn test_write_then_edit_roundtrip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("edit_test.rs");
    let path_str = path.to_string_lossy().to_string();

    let write = WriteTool::new(dir.path().to_string_lossy().to_string(), WriteToolOptions::default());
    write.execute("1".into(), serde_json::json!({"path": &path_str, "content": "fn main() {\n    println!(\"hello\");\n}\n"}), None, None).await.unwrap();

    let edit = EditTool::new(dir.path().to_string_lossy().to_string(), EditToolOptions::default());
    let result = edit.execute("2".into(), serde_json::json!({
        "path": &path_str,
        "edits": [{"oldText": "hello", "newText": "world"}]
    }), None, None).await.unwrap();

    assert!(!result.terminate);
    let content = fs::read_to_string(&path).unwrap();
    assert!(content.contains("world"));
    assert!(!content.contains("\"hello\""));
}

#[tokio::test]
async fn test_ls_lists_written_files() {
    let dir = TempDir::new().unwrap();
    let cwd = dir.path().to_string_lossy().to_string();

    let write = WriteTool::new(cwd.clone(), WriteToolOptions::default());
    write.execute("1".into(), serde_json::json!({"path": dir.path().join("a.txt").to_string_lossy().to_string(), "content": "a"}), None, None).await.unwrap();
    write.execute("2".into(), serde_json::json!({"path": dir.path().join("b.txt").to_string_lossy().to_string(), "content": "b"}), None, None).await.unwrap();

    let ls = LsTool::new(cwd.clone());
    let result = ls.execute("3".into(), serde_json::json!({"path": &cwd}), None, None).await.unwrap();

    let text = match &result.content[0] {
        agent_core::types::ContentBlock::Text { text } => text.clone(),
        _ => panic!("expected text"),
    };
    assert!(text.contains("a.txt"));
    assert!(text.contains("b.txt"));
}

#[tokio::test]
async fn test_find_locates_files() {
    let dir = TempDir::new().unwrap();
    let cwd = dir.path().to_string_lossy().to_string();

    fs::write(dir.path().join("foo.rs"), "fn main() {}").unwrap();
    fs::write(dir.path().join("bar.txt"), "text").unwrap();

    let find = FindTool::new(cwd.clone());
    let result = find.execute("1".into(), serde_json::json!({"pattern": "*.rs", "path": &cwd}), None, None).await.unwrap();

    let text = match &result.content[0] {
        agent_core::types::ContentBlock::Text { text } => text.clone(),
        _ => panic!("expected text"),
    };
    assert!(text.contains("foo.rs"));
    assert!(!text.contains("bar.txt"));
}

#[tokio::test]
async fn test_grep_finds_pattern() {
    // Skip if rg not available
    if std::process::Command::new("rg").arg("--version").output().is_err() {
        return;
    }
    let dir = TempDir::new().unwrap();
    let cwd = dir.path().to_string_lossy().to_string();

    fs::write(dir.path().join("code.rs"), "fn hello_world() {\n    println!(\"hello\");\n}\n").unwrap();

    let grep = GrepTool::new(cwd.clone());
    let result = grep.execute("1".into(), serde_json::json!({"pattern": "hello_world", "path": &cwd}), None, None).await.unwrap();

    let text = match &result.content[0] {
        agent_core::types::ContentBlock::Text { text } => text.clone(),
        _ => panic!("expected text"),
    };
    assert!(text.contains("hello_world"));
}

#[tokio::test]
async fn test_edit_uniqueness_check() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("dup.txt");
    let path_str = path.to_string_lossy().to_string();

    fs::write(&path, "foo\nfoo\nbar\n").unwrap();

    let edit = EditTool::new(dir.path().to_string_lossy().to_string(), EditToolOptions::default());
    let result = edit.execute("1".into(), serde_json::json!({
        "path": &path_str,
        "edits": [{"oldText": "foo", "newText": "baz"}]
    }), None, None).await;

    // Should fail because "foo" appears twice
    // Should fail because "foo" appears twice
    assert!(result.is_err());
}

#[tokio::test]
async fn test_read_offset_limit() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("lines.txt");
    let path_str = path.to_string_lossy().to_string();

    let content: String = (1..=10).map(|i| format!("line {}\n", i)).collect();
    fs::write(&path, &content).unwrap();

    let read = ReadTool::new(dir.path().to_string_lossy().to_string(), ReadToolOptions::default());
    let result = read.execute("1".into(), serde_json::json!({"path": &path_str, "offset": 3, "limit": 3}), None, None).await.unwrap();

    let text = match &result.content[0] {
        agent_core::types::ContentBlock::Text { text } => text.clone(),
        _ => panic!("expected text"),
    };
    assert!(text.contains("line 3"));
    assert!(text.contains("line 5"));
    assert!(!text.contains("line 1"));
    assert!(!text.contains("line 7"));
}
