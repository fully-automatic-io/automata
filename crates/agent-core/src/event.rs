// Agent event types and the streaming primitives used by the agent loop.

use crate::types::{AgentMessage, ContentBlock, StopReason, Usage};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, Notify};

// ============================================================================
// PartialAssistantMessage — typed view of a streaming assistant message.
// Same wire shape as `AgentMessage::Assistant` but with content blocks that
// may be partially streamed (toolCall arguments are progressively filled in).
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialAssistantMessage {
    pub content: Vec<PartialContentBlock>,
    pub api: crate::types::Api,
    pub provider: String,
    pub model: String,
    pub usage: Usage,
    #[serde(rename = "stopReason")]
    pub stop_reason: StopReason,
    #[serde(rename = "errorMessage", default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    #[serde(default)]
    pub timestamp: u64,
}

/// Content block in a streaming assistant message. Wire-compatible with
/// `ContentBlock`, plus a transient `partialJson` for in-flight tool-call
/// arguments (cleared on block-end).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum PartialContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(rename = "textSignature", default, skip_serializing_if = "Option::is_none")]
        text_signature: Option<String>,
    },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(rename = "thinkingSignature", default, skip_serializing_if = "Option::is_none")]
        thinking_signature: Option<String>,
    },
    #[serde(rename = "toolCall")]
    ToolCall {
        id: String,
        name: String,
        /// Tool args are open by design — JSON Schema chosen by the tool.
        #[serde(default)]
        arguments: serde_json::Value,
        /// Raw incremental JSON during streaming; cleared once the block ends.
        #[serde(rename = "partialJson", default, skip_serializing_if = "Option::is_none")]
        partial_json: Option<String>,
    },
    #[serde(rename = "image")]
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
}

impl PartialAssistantMessage {
    pub fn new(api: crate::types::Api, provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            content: Vec::new(),
            api,
            provider: provider.into(),
            model: model.into(),
            usage: Usage::default(),
            stop_reason: StopReason::EndTurn,
            error_message: None,
            timestamp: chrono::Utc::now().timestamp_millis().max(0) as u64,
        }
    }

    /// Pad `content` so `index` is in-bounds, filling gaps with empty text.
    pub fn ensure_block_at(&mut self, index: usize) {
        while self.content.len() <= index {
            self.content.push(PartialContentBlock::Text { text: String::new(), text_signature: None });
        }
    }

    /// Promote into a finalized `AgentMessage::Assistant`. Drops `partialJson`
    /// from any tool-call blocks; pure transformation, never fails.
    pub fn into_finalized(self) -> AgentMessage {
        AgentMessage::Assistant {
            content: self.content.into_iter().map(PartialContentBlock::into_block).collect(),
            api: self.api,
            provider: self.provider,
            model: self.model,
            usage: self.usage,
            stop_reason: self.stop_reason,
            error_message: self.error_message,
            timestamp: self.timestamp,
        }
    }
}

impl PartialContentBlock {
    pub fn into_block(self) -> ContentBlock {
        match self {
            Self::Text { text, .. } => ContentBlock::Text { text },
            Self::Thinking { thinking, .. } => ContentBlock::Thinking { thinking },
            Self::ToolCall { id, name, arguments, .. } => ContentBlock::ToolCall { id, name, arguments },
            Self::Image { data, mime_type } => ContentBlock::Image { data, mime_type },
        }
    }
}

// ============================================================================
// AgentEvent
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    #[serde(rename = "agent_start")]
    AgentStart,
    #[serde(rename = "agent_end")]
    AgentEnd { messages: Vec<AgentMessage> },

    #[serde(rename = "turn_start")]
    TurnStart,
    #[serde(rename = "turn_end")]
    TurnEnd {
        message: AgentMessage,
        #[serde(rename = "toolResults")]
        tool_results: Vec<AgentMessage>,
    },

    #[serde(rename = "message_start")]
    MessageStart { message: AgentMessage },
    #[serde(rename = "message_update")]
    MessageUpdate {
        partial: PartialAssistantMessage,
        #[serde(rename = "assistantMessageEvent")]
        assistant_message_event: AssistantMessageEvent,
    },
    #[serde(rename = "message_end")]
    MessageEnd { message: AgentMessage },

    #[serde(rename = "tool_execution_start")]
    ToolExecutionStart {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        /// Tool args are open by design (any JSON shape valid for the tool).
        args: serde_json::Value,
    },
    #[serde(rename = "tool_execution_update")]
    ToolExecutionUpdate {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        args: serde_json::Value,
        /// Tool partial result is tool-defined; left as Value.
        #[serde(rename = "partialResult")]
        partial_result: serde_json::Value,
    },
    #[serde(rename = "tool_execution_end")]
    ToolExecutionEnd {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        result: crate::types::AgentToolResult,
        #[serde(rename = "isError")]
        is_error: bool,
    },
}

// ============================================================================
// AssistantMessageEvent — streamed assistant-message events. Each variant
// carries the typed `PartialAssistantMessage` snapshot at that point in the
// stream. The terminal event is `Done` (with the finalized `AgentMessage`).
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantMessageEvent {
    Start { partial: PartialAssistantMessage },

    TextStart {
        #[serde(rename = "contentIndex")] content_index: usize,
        partial: PartialAssistantMessage,
    },
    TextDelta {
        #[serde(rename = "contentIndex")] content_index: usize,
        delta: String,
        partial: PartialAssistantMessage,
    },
    TextEnd {
        #[serde(rename = "contentIndex")] content_index: usize,
        content: String,
        partial: PartialAssistantMessage,
    },

    ThinkingStart {
        #[serde(rename = "contentIndex")] content_index: usize,
        partial: PartialAssistantMessage,
    },
    ThinkingDelta {
        #[serde(rename = "contentIndex")] content_index: usize,
        delta: String,
        partial: PartialAssistantMessage,
    },
    ThinkingEnd {
        #[serde(rename = "contentIndex")] content_index: usize,
        content: String,
        partial: PartialAssistantMessage,
    },

    #[serde(rename = "toolcall_start")]
    ToolCallStart {
        #[serde(rename = "contentIndex")] content_index: usize,
        partial: PartialAssistantMessage,
    },
    #[serde(rename = "toolcall_delta")]
    ToolCallDelta {
        #[serde(rename = "contentIndex")] content_index: usize,
        delta: String,
        partial: PartialAssistantMessage,
    },
    #[serde(rename = "toolcall_end")]
    ToolCallEnd {
        #[serde(rename = "contentIndex")] content_index: usize,
        #[serde(rename = "toolCall")] tool_call: ContentBlock,
        partial: PartialAssistantMessage,
    },

    Done { reason: StopReason, message: AgentMessage },
    Error { reason: StopReason, error: PartialAssistantMessage },
}

impl AssistantMessageEvent {
    /// Snapshot of the streaming partial message carried by this event, if any.
    pub fn partial(&self) -> Option<&PartialAssistantMessage> {
        match self {
            Self::Start { partial }
            | Self::TextStart { partial, .. }
            | Self::TextDelta { partial, .. }
            | Self::TextEnd { partial, .. }
            | Self::ThinkingStart { partial, .. }
            | Self::ThinkingDelta { partial, .. }
            | Self::ThinkingEnd { partial, .. }
            | Self::ToolCallStart { partial, .. }
            | Self::ToolCallDelta { partial, .. }
            | Self::ToolCallEnd { partial, .. }
            | Self::Error { error: partial, .. } => Some(partial),
            Self::Done { .. } => None,
        }
    }
}

// ============================================================================
// EventStream — push-based streaming buffer used by the agent loop.
// ============================================================================

pub struct EventStream<T, R> {
    events: Arc<Mutex<Vec<T>>>,
    is_done: Arc<Notify>,
    result: Arc<Mutex<Option<R>>>,
    is_complete: Arc<Mutex<bool>>,
}

impl<T, R> Clone for EventStream<T, R> {
    fn clone(&self) -> Self {
        Self {
            events: Arc::clone(&self.events),
            is_done: self.is_done.clone(),
            result: Arc::clone(&self.result),
            is_complete: Arc::clone(&self.is_complete),
        }
    }
}

impl<T: Clone, R: Clone> EventStream<T, R> {
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            is_done: Arc::new(Notify::new()),
            result: Arc::new(Mutex::new(None)),
            is_complete: Arc::new(Mutex::new(false)),
        }
    }

    pub fn push(&self, event: T) {
        if *self.is_complete.lock().unwrap() { return; }
        self.events.lock().unwrap().push(event);
        self.is_done.notify_one();
    }

    pub fn end(&self, result: R) {
        let mut done = self.is_complete.lock().unwrap();
        if *done { return; }
        *done = true;
        *self.result.lock().unwrap() = Some(result);
        self.is_done.notify_one();
    }

    pub fn is_complete(&self) -> bool { *self.is_complete.lock().unwrap() }
    pub fn take_events(&self) -> Vec<T> { std::mem::take(&mut *self.events.lock().unwrap()) }

    /// Park until either at least one event is available or the stream has ended.
    /// Returns immediately if either condition already holds.
    pub async fn wait_for_more(&self) {
        loop {
            // Register the waker BEFORE re-checking state — `notify_one` stores a
            // permit, so a push that happens between this `notified()` call and
            // the state check still wakes us up.
            let notified = self.is_done.notified();
            if !self.events.lock().unwrap().is_empty()
                || *self.is_complete.lock().unwrap()
            {
                return;
            }
            notified.await;
        }
    }

    pub fn wait_for_result_try(&self) -> Result<Option<R>, ()> {
        if *self.is_complete.lock().unwrap() {
            Ok(self.result.lock().unwrap().clone())
        } else {
            Err(())
        }
    }

    pub async fn wait_for_result(&self) -> R {
        loop {
            if *self.is_complete.lock().unwrap() {
                if let Some(r) = self.result.lock().unwrap().clone() {
                    return r;
                }
            }
            self.is_done.notified().await;
        }
    }
}

impl<T: Clone, R: Clone> Default for EventStream<T, R> {
    fn default() -> Self { Self::new() }
}

// ============================================================================
// AgentEventChannel — fan-out for AgentEvent.
// ============================================================================

pub struct AgentEventReceiver {
    rx: mpsc::UnboundedReceiver<AgentEvent>,
}

impl AgentEventReceiver {
    pub async fn recv(&mut self) -> Option<AgentEvent> { self.rx.recv().await }
}

pub struct AgentEventChannel {
    tx: mpsc::UnboundedSender<AgentEvent>,
    closed: Arc<Mutex<bool>>,
}

impl AgentEventChannel {
    pub fn new(_buffer: usize) -> (Self, AgentEventReceiver) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { tx, closed: Arc::new(Mutex::new(false)) }, AgentEventReceiver { rx })
    }

    pub fn send(&self, event: AgentEvent) {
        if *self.closed.lock().unwrap() { return; }
        let _ = self.tx.send(event);
    }

    pub fn close(&self) { *self.closed.lock().unwrap() = true; }
}

impl Clone for AgentEventChannel {
    fn clone(&self) -> Self {
        Self { tx: self.tx.clone(), closed: Arc::clone(&self.closed) }
    }
}

// ============================================================================
// Listener trait
// ============================================================================

pub trait AgentEventListener: Send + Sync {
    fn on_event(
        &self,
        event: AgentEvent,
        signal: Option<tokio_util::sync::CancellationToken>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;
}

pub struct FnEventListener<F> { f: F }

impl<F, Fut> FnEventListener<F>
where
    F: Fn(AgentEvent, Option<tokio_util::sync::CancellationToken>) -> Fut + Send + Sync,
    Fut: Future<Output = ()> + Send + 'static,
{
    pub fn new(f: F) -> Self { Self { f } }
}

impl<F, Fut> AgentEventListener for FnEventListener<F>
where
    F: Fn(AgentEvent, Option<tokio_util::sync::CancellationToken>) -> Fut + Send + Sync,
    Fut: Future<Output = ()> + Send + 'static,
{
    fn on_event(
        &self,
        event: AgentEvent,
        signal: Option<tokio_util::sync::CancellationToken>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin((self.f)(event, signal))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assistant_event_partial() {
        let ev = AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: "hi".into(),
            partial: PartialAssistantMessage::new(crate::types::Api::Anthropic, "p", "m"),
        };
        assert!(ev.partial().is_some());
    }

    #[test]
    fn test_partial_into_finalized_text() {
        let mut p = PartialAssistantMessage::new(crate::types::Api::Anthropic, "p", "m");
        p.content.push(PartialContentBlock::Text { text: "hi".into(), text_signature: None });
        match p.into_finalized() {
            AgentMessage::Assistant { content, stop_reason, .. } => {
                assert_eq!(stop_reason, StopReason::EndTurn);
                assert!(matches!(content[0], ContentBlock::Text { ref text } if text == "hi"));
            }
            _ => panic!("expected Assistant"),
        }
    }

    #[test]
    fn test_partial_into_finalized_drops_partial_json() {
        let mut p = PartialAssistantMessage::new(crate::types::Api::Anthropic, "p", "m");
        p.content.push(PartialContentBlock::ToolCall {
            id: "tc1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"cmd": "ls"}),
            partial_json: Some("{\"cmd\":\"ls\"".into()),
        });
        match p.into_finalized() {
            AgentMessage::Assistant { content, .. } => {
                match &content[0] {
                    ContentBlock::ToolCall { id, name, arguments } => {
                        assert_eq!(id, "tc1");
                        assert_eq!(name, "bash");
                        assert_eq!(arguments["cmd"], "ls");
                    }
                    _ => panic!("expected ToolCall"),
                }
            }
            _ => panic!("expected Assistant"),
        }
    }

    #[test]
    fn test_partial_default_serializes_with_stop_reason_stop() {
        // Wire shape compat: default serializes with `stopReason: "stop"`.
        let p = PartialAssistantMessage::new(crate::types::Api::Anthropic, "p", "m");
        let v = serde_json::to_value(&p).unwrap();
        assert_eq!(v["stopReason"], "stop");
        assert_eq!(v["api"], "anthropic");
        assert_eq!(v["content"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn test_event_channel_basic() {
        let (ch, mut rx) = AgentEventChannel::new(10);
        ch.send(AgentEvent::AgentStart);
        let event = rx.recv().await.unwrap();
        match event { AgentEvent::AgentStart => {}, _ => panic!("Expected AgentStart") }
    }
}
