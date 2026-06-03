use coding_agent::core::{BuildSystemPromptOptions, ContextFile, build_system_prompt};

#[test]
fn test_xml_boundaries_two_context_files() {
    let files = vec![
        ContextFile {
            path: "AGENTS.md".into(),
            content: "agent instructions".into(),
        },
        ContextFile {
            path: "CLAUDE.md".into(),
            content: "claude instructions".into(),
        },
    ];
    let result = build_system_prompt(BuildSystemPromptOptions {
        custom_prompt: None,
        selected_tools: None,
        tool_snippets: None,
        prompt_guidelines: None,
        append_system_prompt: None,
        cwd: "/tmp",
        context_files: Some(&files),
        skills: None,
    });

    assert!(result.contains("<project_context>"), "missing <project_context>");
    assert!(
        result.contains("<project_instructions path=\"AGENTS.md\">"),
        "missing AGENTS.md tag"
    );
    assert!(
        result.contains("<project_instructions path=\"CLAUDE.md\">"),
        "missing CLAUDE.md tag"
    );
    assert!(result.contains("</project_instructions>"), "missing closing tag");
    assert!(result.contains("</project_context>"), "missing </project_context>");
    // Content should be inside the tags
    assert!(result.contains("agent instructions"));
    assert!(result.contains("claude instructions"));
}

#[test]
fn test_xml_boundaries_no_context_files() {
    let result = build_system_prompt(BuildSystemPromptOptions {
        custom_prompt: None,
        selected_tools: None,
        tool_snippets: None,
        prompt_guidelines: None,
        append_system_prompt: None,
        cwd: "/tmp",
        context_files: Some(&[]),
        skills: None,
    });
    assert!(!result.contains("<project_context>"));
}

#[test]
fn test_custom_prompt_also_gets_xml_boundaries() {
    let files = vec![ContextFile {
        path: "AGENTS.md".into(),
        content: "custom context".into(),
    }];
    let result = build_system_prompt(BuildSystemPromptOptions {
        custom_prompt: Some("My custom prompt."),
        selected_tools: None,
        tool_snippets: None,
        prompt_guidelines: None,
        append_system_prompt: None,
        cwd: "/tmp",
        context_files: Some(&files),
        skills: None,
    });
    assert!(result.starts_with("My custom prompt."));
    assert!(result.contains("<project_context>"));
    assert!(result.contains("<project_instructions path=\"AGENTS.md\">"));
}
