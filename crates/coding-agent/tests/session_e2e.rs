// End-to-end test of `CodingAgentSession`: a stub `LlmProvider` drives a real
// tool-calling turn (write a file via the bash tool, then answer), exercising
// the full stream-bridge -> AgentHarness -> tool-loop wiring.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use agent_core::harness::session::{InMemorySessionStorage, Session};
use agent_core::types::{Api, Model, ModelCost};
use async_trait::async_trait;
use coding_agent::{Auth, CodingAgentSession, SessionOptions};
use llm_client::provider::{LlmError, LlmProvider, LlmStream};
use llm_client::streaming::{Delta, LlmEvent, MessageDelta};
use llm_client::types::{LlmRequest, LlmResponse, StopReason as LlmStopReason, Usage};

/// A scripted provider: turn 0 emits a `bash` tool call, turn 1 emits a final
/// text answer. `complete` (used by compaction) returns a canned summary.
struct ScriptedProvider {
    turn: AtomicUsize,
    marker_path: String,
}

impl ScriptedProvider {
    fn new(marker_path: String) -> Self {
        Self { turn: AtomicUsize::new(0), marker_path }
    }
}

fn sse(events: Vec<LlmEvent>) -> LlmStream {
    Box::pin(futures::stream::iter(events.into_iter().map(Ok::<LlmEvent, LlmError>)))
}

#[async_trait]
impl LlmProvider for ScriptedProvider {
    async fn complete(&self, _request: LlmRequest) -> Result<LlmResponse, LlmError> {
        Ok(LlmResponse {
            id: "resp".into(),
            model: "stub".into(),
            content: vec![agent_core::types::ContentBlock::Text {
                text: "## Goal\nstub summary".into(),
            }],
            stop_reason: LlmStopReason::EndTurn,
            usage: Usage::default(),
        })
    }

    async fn stream(&self, _request: LlmRequest) -> Result<LlmStream, LlmError> {
        let turn = self.turn.fetch_add(1, Ordering::SeqCst);
        if turn == 0 {
            // Assistant decides to run bash to create a marker file.
            let args = serde_json::json!({
                "command": format!("echo hi > {}", self.marker_path)
            })
            .to_string();
            Ok(sse(vec![
                LlmEvent::ContentBlockStart {
                    index: 0,
                    content_block: agent_core::types::ContentPart::ToolCall {
                        id: "call-1".into(),
                        name: "bash".into(),
                        arguments: serde_json::json!({}),
                    },
                },
                LlmEvent::ContentBlockDelta {
                    index: 0,
                    delta: Delta::InputJsonDelta { partial_json: args },
                },
                LlmEvent::ContentBlockStop { index: 0 },
                LlmEvent::MessageDelta {
                    delta: MessageDelta {
                        stop_reason: Some(LlmStopReason::ToolUse),
                        stop_sequence: None,
                    },
                    usage: Some(Usage {
                        input: 10,
                        output: 5,
                        total_tokens: 15,
                        ..Default::default()
                    }),
                },
                LlmEvent::MessageStop,
            ]))
        } else {
            // Final answer.
            Ok(sse(vec![
                LlmEvent::ContentBlockStart {
                    index: 0,
                    content_block: agent_core::types::ContentPart::Text { text: String::new() },
                },
                LlmEvent::ContentBlockDelta {
                    index: 0,
                    delta: Delta::TextDelta { text: "done".into() },
                },
                LlmEvent::ContentBlockStop { index: 0 },
                LlmEvent::MessageDelta {
                    delta: MessageDelta {
                        stop_reason: Some(LlmStopReason::EndTurn),
                        stop_sequence: None,
                    },
                    usage: Some(Usage {
                        input: 12,
                        output: 3,
                        total_tokens: 15,
                        ..Default::default()
                    }),
                },
                LlmEvent::MessageStop,
            ]))
        }
    }
}

fn stub_model() -> Model {
    Model {
        id: "stub".into(),
        name: "Stub".into(),
        api: Api::Anthropic,
        provider: "stub".into(),
        base_url: String::new(),
        reasoning: false,
        input: vec!["text".into()],
        cost: ModelCost::default(),
        context_window: 200_000,
        max_tokens: 8192,
        ..Default::default()
    }
}

#[tokio::test]
async fn prompt_drives_tool_call_then_answers() {
    let dir = tempfile::TempDir::new().unwrap();
    let cwd = dir.path().to_string_lossy().to_string();
    let marker = dir.path().join("marker.txt").to_string_lossy().to_string();

    let provider = Arc::new(ScriptedProvider::new(marker.clone()));
    let mut options = SessionOptions::new(cwd, stub_model(), "unused");
    options.system_prompt = "You are a test agent.".into();
    options.auth = Auth::Native;
    // Disable auto-compaction so the post-run check stays a no-op for this test.
    options.compaction = None;

    let session = CodingAgentSession::with_provider(
        Session::new(Box::new(InMemorySessionStorage::new(None))),
        provider,
        options,
    )
    .await
    .unwrap();

    let messages = session.prompt("create the marker file").await.unwrap();

    // The bash tool must have actually run and created the file.
    assert!(
        std::path::Path::new(&marker).exists(),
        "bash tool should have created the marker file"
    );

    // The final assistant message should carry the "done" answer.
    let has_done = messages.iter().any(|m| {
        match m {
        agent_core::types::AgentMessage::Assistant { content, .. } => content.iter().any(|b| {
            matches!(b, agent_core::types::ContentBlock::Text { text } if text.contains("done"))
        }),
        _ => false,
    }
    });
    assert!(has_done, "final assistant message should contain 'done'");

    // A tool result must be present in the transcript.
    let has_tool_result = messages
        .iter()
        .any(|m| matches!(m, agent_core::types::AgentMessage::ToolResult { .. }));
    assert!(has_tool_result, "transcript should include a tool result");
}

#[cfg(unix)]
#[tokio::test]
async fn prompt_uses_configured_shell_and_prefix() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::TempDir::new().unwrap();
    let cwd = dir.path().to_string_lossy().to_string();
    let marker = dir.path().join("marker.txt").to_string_lossy().to_string();
    let shell_marker = dir.path().join("custom-shell-used.txt");
    let prefix_marker = dir.path().join("prefix-used.txt");
    let shell_path = dir.path().join("custom-shell");

    std::fs::write(
        &shell_path,
        format!(
            "#!/bin/sh\nprintf used > {}\nexec /bin/sh \"$@\"\n",
            shell_quote(&shell_marker.to_string_lossy())
        ),
    )
    .unwrap();
    let mut perms = std::fs::metadata(&shell_path).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&shell_path, perms).unwrap();

    let provider = Arc::new(ScriptedProvider::new(marker.clone()));
    let mut options = SessionOptions::new(cwd, stub_model(), "unused");
    options.system_prompt = "You are a test agent.".into();
    options.auth = Auth::Native;
    options.compaction = None;
    options.shell_path = Some(shell_path.to_string_lossy().to_string());
    options.shell_command_prefix =
        Some(format!("printf prefix > {}", shell_quote(&prefix_marker.to_string_lossy())));

    let session = CodingAgentSession::with_provider(
        Session::new(Box::new(InMemorySessionStorage::new(None))),
        provider,
        options,
    )
    .await
    .unwrap();

    session.prompt("create the marker file").await.unwrap();

    assert!(std::path::Path::new(&marker).exists());
    assert!(shell_marker.exists(), "configured shell should have been invoked");
    assert!(prefix_marker.exists(), "configured command prefix should have run");
}

#[cfg(unix)]
fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}
