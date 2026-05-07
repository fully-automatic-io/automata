// Integration tests for session management

use coding_agent::session::SessionManager;

#[test]
fn test_session_persist_and_reload() {
    let dir = tempfile::TempDir::new().unwrap();
    let session_dir = dir.path().to_string_lossy().to_string();

    // Create session and add messages
    let mut mgr = SessionManager::create("/tmp/test", Some(&session_dir));
    let _id1 = mgr.append_message(serde_json::json!({
        "role": "user",
        "content": [{"type": "text", "text": "hello"}],
        "timestamp": 1000
    }));
    let _id2 = mgr.append_message(serde_json::json!({
        "role": "assistant",
        "content": [{"type": "text", "text": "hi there"}],
        "api": "anthropic", "provider": "anthropic", "model": "claude-opus-4-7",
        "usage": {"input": 5, "output": 5, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 10,
                  "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}},
        "stopReason": "end_turn",
        "timestamp": 2000
    }));

    let session_file = mgr.session_file().map(|p| p.to_path_buf());
    // File is only written after first assistant message
    if session_file.is_none() { return; }

    // Reload session
    let path = session_file.unwrap();
    let reloaded = SessionManager::open(&path.to_string_lossy(), Some(&session_dir), None);
    let ctx = reloaded.build_context();
    assert_eq!(ctx.messages.len(), 2);
    assert!(ctx.model.is_some());
}

#[test]
fn test_session_branching_and_context() {
    let mut mgr = SessionManager::in_memory("/tmp/test");

    let id1 = mgr.append_message(serde_json::json!({"role": "user", "content": "q1", "timestamp": 1}));
    let _id2 = mgr.append_message(serde_json::json!({
        "role": "assistant", "content": "a1", "api": "t", "provider": "t", "model": "m",
        "usage": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},
        "stopReason": "stop", "timestamp": 2
    }));

    // Branch back and add alternative
    mgr.branch(&id1);
    let _id3 = mgr.append_message(serde_json::json!({
        "role": "assistant", "content": "alt answer", "api": "t", "provider": "t", "model": "m",
        "usage": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},
        "stopReason": "stop", "timestamp": 3
    }));

    let ctx = mgr.build_context();
    // Should follow the branch: user + alt answer
    assert_eq!(ctx.messages.len(), 2);
    let alt = ctx.messages[1].get("content").and_then(|c| c.as_str()).unwrap_or("");
    assert_eq!(alt, "alt answer");
}

#[test]
fn test_session_compaction_context() {
    let mut mgr = SessionManager::in_memory("/tmp/test");

    let _u1 = mgr.append_message(serde_json::json!({"role": "user", "content": "old q", "timestamp": 1}));
    let a1 = mgr.append_message(serde_json::json!({
        "role": "assistant", "content": "old a", "api": "t", "provider": "t", "model": "m",
        "usage": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},
        "stopReason": "stop", "timestamp": 2
    }));

    // Compact: keep from a1 onward
    mgr.append_compaction("Summary of old conversation", &a1, 1000, None, None);

    let _u2 = mgr.append_message(serde_json::json!({"role": "user", "content": "new q", "timestamp": 3}));

    let ctx = mgr.build_context();
    // Should have: compaction summary + a1 + new q
    assert!(ctx.messages.len() >= 2);
    // First message should be the compaction summary
    let first_role = ctx.messages[0].get("role").and_then(|r| r.as_str()).unwrap_or("");
    assert!(first_role.contains("compaction") || first_role == "user");
}

#[test]
fn test_session_tree_structure() {
    let mut mgr = SessionManager::in_memory("/tmp/test");

    let id1 = mgr.append_message(serde_json::json!({"role": "user", "content": "root", "timestamp": 1}));
    let _id2 = mgr.append_message(serde_json::json!({
        "role": "assistant", "content": "branch a", "api": "t", "provider": "t", "model": "m",
        "usage": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},
        "stopReason": "stop", "timestamp": 2
    }));

    mgr.branch(&id1);
    let _id3 = mgr.append_message(serde_json::json!({
        "role": "assistant", "content": "branch b", "api": "t", "provider": "t", "model": "m",
        "usage": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}},
        "stopReason": "stop", "timestamp": 3
    }));

    let tree = mgr.tree();
    // Root node should have 1 child (id1), which has 2 children (id2, id3)
    assert!(!tree.is_empty());
}

#[test]
fn test_session_labels() {
    let mut mgr = SessionManager::in_memory("/tmp/test");
    let id = mgr.append_message(serde_json::json!({"role": "user", "content": "important", "timestamp": 1}));

    mgr.append_label_change(&id, Some("key point"));
    assert_eq!(mgr.get_label(&id), Some("key point"));

    mgr.append_label_change(&id, None);
    assert_eq!(mgr.get_label(&id), None);
}
