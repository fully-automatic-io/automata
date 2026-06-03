use agent_core::tool::AgentTool;
use coding_agent::tools::{EditTool, EditToolOptions};
use tempfile::TempDir;
use tokio::fs;

#[tokio::test]
async fn test_fuzzy_match_typographic_quotes() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.txt");
    // File contains typographic (curly) double quotes
    fs::write(&path, "He said \u{201C}hello world\u{201D} to her.").await.unwrap();

    let tool = EditTool::new(dir.path().to_string_lossy().to_string(), EditToolOptions::default());
    let params = serde_json::json!({
        "path": path.to_string_lossy(),
        "edits": [{
            // oldText uses ASCII straight quotes — should match via NFKC fuzzy normalization
            "oldText": "He said \"hello world\" to her.",
            "newText": "He said \"goodbye world\" to her."
        }]
    });

    let result = tool.execute("tc1".into(), params, None, None).await.unwrap();
    assert!(!result.content.is_empty(), "edit should succeed");

    let content = fs::read_to_string(&path).await.unwrap();
    assert!(content.contains("goodbye world"), "replacement should have been applied");
}

#[tokio::test]
async fn test_fuzzy_match_trailing_whitespace() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.txt");
    // File has trailing spaces on a line
    fs::write(&path, "fn foo()   \nfn bar()\n").await.unwrap();

    let tool = EditTool::new(dir.path().to_string_lossy().to_string(), EditToolOptions::default());
    let params = serde_json::json!({
        "path": path.to_string_lossy(),
        "edits": [{
            // oldText without trailing spaces — should match via trim_end normalization
            "oldText": "fn foo()",
            "newText": "fn baz()"
        }]
    });

    let result = tool.execute("tc2".into(), params, None, None).await.unwrap();
    assert!(!result.content.is_empty());
    let content = fs::read_to_string(&path).await.unwrap();
    assert!(content.contains("fn baz()"));
}

#[tokio::test]
async fn test_exact_match_still_works() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("test.txt");
    fs::write(&path, "hello world").await.unwrap();

    let tool = EditTool::new(dir.path().to_string_lossy().to_string(), EditToolOptions::default());
    let params = serde_json::json!({
        "path": path.to_string_lossy(),
        "edits": [{"oldText": "hello world", "newText": "goodbye world"}]
    });

    let result = tool.execute("tc3".into(), params, None, None).await.unwrap();
    assert!(!result.content.is_empty());
    let content = fs::read_to_string(&path).await.unwrap();
    assert_eq!(content, "goodbye world");
}
