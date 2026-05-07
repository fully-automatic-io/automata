use crate::types::{ContentPart, StopReason, Usage};
use serde::{Deserialize, Serialize};

/// Events emitted during LLM streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LlmEvent {
    MessageStart {
        id: String,
        model: String,
    },
    ContentBlockStart {
        index: usize,
        content_block: ContentPart,
    },
    ContentBlockDelta {
        index: usize,
        delta: Delta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: MessageDelta,
        usage: Option<Usage>,
    },
    MessageStop,
    Ping,
    Error {
        error: StreamError,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Delta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDelta {
    pub stop_reason: Option<StopReason>,
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamError {
    pub error_type: String,
    pub message: String,
}

/// Parse a single SSE line into an LlmEvent
pub fn parse_sse_event(event_type: &str, data: &str) -> Option<LlmEvent> {
    if data == "[DONE]" {
        return Some(LlmEvent::MessageStop);
    }
    match event_type {
        "ping" => Some(LlmEvent::Ping),
        "message_start" => {
            #[derive(Deserialize)]
            struct Wrapper { message: MessageStart }
            #[derive(Deserialize)]
            struct MessageStart { id: String, model: String }
            let w: Wrapper = serde_json::from_str(data).ok()?;
            Some(LlmEvent::MessageStart { id: w.message.id, model: w.message.model })
        }
        "content_block_start" => {
            #[derive(Deserialize)]
            struct Wrapper { index: usize, content_block: ContentPart }
            let w: Wrapper = serde_json::from_str(data).ok()?;
            Some(LlmEvent::ContentBlockStart { index: w.index, content_block: w.content_block })
        }
        "content_block_delta" => {
            #[derive(Deserialize)]
            struct Wrapper { index: usize, delta: Delta }
            let w: Wrapper = serde_json::from_str(data).ok()?;
            Some(LlmEvent::ContentBlockDelta { index: w.index, delta: w.delta })
        }
        "content_block_stop" => {
            #[derive(Deserialize)]
            struct Wrapper { index: usize }
            let w: Wrapper = serde_json::from_str(data).ok()?;
            Some(LlmEvent::ContentBlockStop { index: w.index })
        }
        "message_delta" => {
            #[derive(Deserialize)]
            struct Wrapper { delta: MessageDelta, usage: Option<Usage> }
            let w: Wrapper = serde_json::from_str(data).ok()?;
            Some(LlmEvent::MessageDelta { delta: w.delta, usage: w.usage })
        }
        "message_stop" => Some(LlmEvent::MessageStop),
        "error" => {
            #[derive(Deserialize)]
            struct Wrapper { error: StreamError }
            let w: Wrapper = serde_json::from_str(data).ok()?;
            Some(LlmEvent::Error { error: w.error })
        }
        _ => None,
    }
}
