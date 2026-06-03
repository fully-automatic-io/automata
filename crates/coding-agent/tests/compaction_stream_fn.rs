use agent_core::harness::compaction::{
    CompactionPreparation, CompactionSettings, prepare_compaction,
};
use agent_core::harness::session::{InMemorySessionStorage, Session};
use agent_core::types::{AgentMessage, ContentBlock, MessageContent, StopReason, Usage};

#[tokio::test]
async fn test_compaction_prepare_returns_preparation_or_none() {
    let mut session = Session::new(Box::new(InMemorySessionStorage::new(None)));
    for i in 0..5u64 {
        let msg = if i % 2 == 0 {
            AgentMessage::User {
                content: MessageContent::Blocks(vec![ContentBlock::Text {
                    text: format!("message {}", i),
                }]),
                timestamp: i * 1000,
                metadata: None,
            }
        } else {
            AgentMessage::Assistant {
                content: vec![ContentBlock::Text { text: format!("message {}", i) }],
                api: agent_core::types::Api::Anthropic,
                provider: "t".into(),
                model: "m".into(),
                usage: Usage::default(),
                stop_reason: StopReason::EndTurn,
                error_message: None,
                timestamp: i * 1000,
            }
        };
        session.append_message(msg).await.unwrap();
    }
    let entries = session.storage().get_entries().await;
    let settings = CompactionSettings::default();
    let _ = prepare_compaction(&entries, &settings).expect("prepare_compaction should not error");
}

#[test]
fn test_compaction_preparation_struct_fields() {
    let prep = CompactionPreparation {
        first_kept_entry_id: "entry-1".into(),
        messages_to_summarize: vec![AgentMessage::user_text("hello")],
        turn_prefix_messages: vec![],
        is_split_turn: false,
        tokens_before: 100,
        previous_summary: None,
        file_ops: agent_core::harness::compaction::create_file_ops(),
        settings: CompactionSettings::default(),
    };
    assert_eq!(prep.first_kept_entry_id, "entry-1");
    assert_eq!(prep.messages_to_summarize.len(), 1);
    assert!(!prep.is_split_turn);
    assert_eq!(prep.tokens_before, 100);
}
