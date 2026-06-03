use agent_core::harness::{
    InMemorySessionRepo, Skill, convert_to_llm, format_skills_for_system_prompt,
    parse_command_args, substitute_args, truncate_head, truncate_tail,
};
use agent_core::types::{AgentMessage, ContentBlock, MessageContent, StopReason, Usage};

#[test]
fn test_session_append_and_context() {
    let mut repo = InMemorySessionRepo::new();
    let mut session = repo.create_session(None);

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let id = session.append_message(AgentMessage::user_text("hello")).await.unwrap();

        let branch = session.get_branch().await.unwrap();
        assert_eq!(branch.len(), 1);
        assert_eq!(branch[0].id(), id);

        let ctx = session.build_context().await.unwrap();
        assert_eq!(ctx.messages.len(), 1);
        assert_eq!(ctx.messages[0].role(), "user");
    });
}

#[test]
fn test_session_compaction_context() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut repo = InMemorySessionRepo::new();
        let mut session = repo.create_session(None);

        let u1 = session.append_message(AgentMessage::user_text("msg1")).await.unwrap();
        session
            .append_message(AgentMessage::Assistant {
                content: vec![ContentBlock::Text { text: "resp1".into() }],
                api: agent_core::types::Api::Anthropic,
                provider: "p".into(),
                model: "m".into(),
                usage: Usage::default(),
                stop_reason: StopReason::EndTurn,
                error_message: None,
                timestamp: 0,
            })
            .await
            .unwrap();

        session
            .append_compaction("Summary of history", &u1, 1000, None, None)
            .await
            .unwrap();

        session.append_message(AgentMessage::user_text("msg2")).await.unwrap();

        let ctx = session.build_context().await.unwrap();
        assert!(ctx.messages.iter().any(|m| matches!(m, AgentMessage::CompactionSummary { .. })));
        assert!(ctx.messages.iter().any(|m| matches!(m,
            AgentMessage::User { content: MessageContent::Blocks(b), .. }
            if matches!(b.first(), Some(ContentBlock::Text { text }) if text == "msg2"))));
    });
}

#[test]
fn test_convert_to_llm_bash_execution() {
    let messages = vec![AgentMessage::BashExecution {
        command: "ls".into(),
        output: "file.txt".into(),
        exit_code: Some(0),
        cancelled: false,
        truncated: false,
        full_output_path: None,
        timestamp: 0,
        exclude_from_context: false,
    }];
    let llm = convert_to_llm(&messages);
    assert_eq!(llm.len(), 1);
    assert_eq!(llm[0].role(), "user");
    let text = match &llm[0] {
        AgentMessage::User { content: MessageContent::Blocks(b), .. } => match &b[0] {
            ContentBlock::Text { text } => text.clone(),
            _ => panic!("expected text block"),
        },
        _ => panic!("expected user message"),
    };
    assert!(text.contains("ls"));
    assert!(text.contains("file.txt"));
}

#[test]
fn test_convert_to_llm_excludes_excluded_bash() {
    let messages = vec![AgentMessage::BashExecution {
        command: "ls".into(),
        output: String::new(),
        exit_code: Some(0),
        cancelled: false,
        truncated: false,
        full_output_path: None,
        timestamp: 0,
        exclude_from_context: true,
    }];
    assert!(convert_to_llm(&messages).is_empty());
}

#[test]
fn test_convert_to_llm_compaction_summary() {
    let messages = vec![AgentMessage::CompactionSummary {
        summary: "prior work".into(),
        tokens_before: 1000,
        timestamp: 0,
    }];
    let llm = convert_to_llm(&messages);
    assert_eq!(llm.len(), 1);
    let text = match &llm[0] {
        AgentMessage::User { content: MessageContent::Blocks(b), .. } => match &b[0] {
            ContentBlock::Text { text } => text.clone(),
            _ => panic!("expected text block"),
        },
        _ => panic!("expected user message"),
    };
    assert!(text.contains("prior work"));
    assert!(text.contains("<summary>"));
}

#[test]
fn test_format_skills_for_system_prompt() {
    let skills = vec![Skill {
        name: "my-skill".to_string(),
        description: "Does something useful".to_string(),
        content: "skill content".to_string(),
        file_path: "/path/to/SKILL.md".to_string(),
        disable_model_invocation: false,
    }];
    let result = format_skills_for_system_prompt(&skills);
    assert!(result.contains("<available_skills>"));
    assert!(result.contains("<name>my-skill</name>"));
    assert!(result.contains("Does something useful"));
}

#[test]
fn test_format_skills_excludes_disabled() {
    let skills = vec![Skill {
        name: "hidden".to_string(),
        description: "Hidden skill".to_string(),
        content: "".to_string(),
        file_path: "/path/SKILL.md".to_string(),
        disable_model_invocation: true,
    }];
    let result = format_skills_for_system_prompt(&skills);
    assert!(result.is_empty());
}

#[test]
fn test_parse_command_args() {
    assert_eq!(parse_command_args("foo bar"), vec!["foo", "bar"]);
    assert_eq!(parse_command_args("\"hello world\" baz"), vec!["hello world", "baz"]);
    assert_eq!(parse_command_args("'single quoted'"), vec!["single quoted"]);
}

#[test]
fn test_substitute_args() {
    assert_eq!(substitute_args("Hello $1!", &["world".to_string()]), "Hello world!");
    assert_eq!(substitute_args("$@", &["a".to_string(), "b".to_string()]), "a b");
    assert_eq!(substitute_args("$ARGUMENTS", &["x".to_string()]), "x");
    assert_eq!(
        substitute_args("${@:2}", &["a".to_string(), "b".to_string(), "c".to_string()]),
        "b c"
    );
    assert_eq!(
        substitute_args("${@:1:2}", &["a".to_string(), "b".to_string(), "c".to_string()]),
        "a b"
    );
}

#[test]
fn test_truncate_head_no_truncation() {
    let result = truncate_head("hello\nworld", 2000, 50 * 1024);
    assert!(!result.truncated);
    assert_eq!(result.content, "hello\nworld");
}

#[test]
fn test_truncate_head_line_limit() {
    let lines: Vec<String> = (0..2100).map(|i| format!("line {}", i)).collect();
    let input = lines.join("\n");
    let result = truncate_head(&input, 2000, 50 * 1024);
    assert!(result.truncated);
    assert_eq!(result.output_lines, 2000);
}

#[test]
fn test_truncate_tail_keeps_end() {
    let lines: Vec<String> = (0..2100).map(|i| format!("line {}", i)).collect();
    let input = lines.join("\n");
    let result = truncate_tail(&input, 2000, 50 * 1024);
    assert!(result.truncated);
    assert!(result.content.contains("line 2099"));
    assert!(!result.content.contains("line 0\n"));
}
