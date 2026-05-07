
use crate::types::AgentMessage;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

// ============================================================================
// AgentEvent — discriminated union matching TS AgentEvent exactly
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Agent lifecycle
    #[serde(rename = "agent_start")]
    AgentStart,
    #[serde(rename = "agent_end")]
    AgentEnd {
        messages: Vec<AgentMessage>,
    },

    /// Turn lifecycle
    #[serde(rename = "turn_start")]
    TurnStart,
    #[serde(rename = "turn_end")]
    TurnEnd {
        message: AgentMessage,
        #[serde(rename = "toolResults")]
        tool_results: Vec<AgentMessage>,
    },

    /// Message lifecycle
    #[serde(rename = "message_start")]
    MessageStart {
        message: AgentMessage,
    },
    #[serde(rename = "message_update")]
    MessageUpdate {
        message: AgentMessage,
        #[serde(rename = "assistantMessageEvent")]
        assistant_message_event: AssistantMessageEvent,
    },
    #[serde(rename = "message_end")]
    MessageEnd {
        message: AgentMessage,
    },

    /// Tool execution lifecycle
    #[serde(rename = "tool_execution_start")]
    ToolExecutionStart {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        #[serde(rename = "args")]
        args: serde_json::Value,
    },
    #[serde(rename = "tool_execution_update")]
    ToolExecutionUpdate {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        #[serde(rename = "args")]
        args: serde_json::Value,
        #[serde(rename = "partialResult")]
        partial_result: serde_json::Value,
    },
    #[serde(rename = "tool_execution_end")]
    ToolExecutionEnd {
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
        #[serde(rename = "result")]
        result: serde_json::Value,
        #[serde(rename = "isError")]
        is_error: bool,
    },
}

// ============================================================================
// AssistantMessageEvent — streamed assistant message events
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantMessageEvent {
    #[serde(rename = "start")]
    Start {
        partial: AgentMessage,
    },
    #[serde(rename = "text_start")]
    TextStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        partial: AgentMessage,
    },
    #[serde(rename = "text_delta")]
    TextDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
        partial: AgentMessage,
    },
    #[serde(rename = "text_end")]
    TextEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        content: String,
        partial: AgentMessage,
    },
    #[serde(rename = "thinking_start")]
    ThinkingStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        partial: AgentMessage,
    },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
        partial: AgentMessage,
    },
    #[serde(rename = "thinking_end")]
    ThinkingEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        content: String,
        partial: AgentMessage,
    },
    #[serde(rename = "toolcall_start")]
    ToolCallStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        partial: AgentMessage,
    },
    #[serde(rename = "toolcall_delta")]
    ToolCallDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
        partial: AgentMessage,
    },
    #[serde(rename = "toolcall_end")]
    ToolCallEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        #[serde(rename = "toolCall")]
        tool_call: serde_json::Value,
        partial: AgentMessage,
    },
    #[serde(rename = "done")]
    Done {
        reason: String,
        message: AgentMessage,
    },
    #[serde(rename = "error")]
    Error {
        reason: String,
        error: AgentMessage,
    },
}

// ============================================================================
// Event Stream — push-based streaming with result extraction
// ============================================================================

/// EventStream<T> matches TS EventStream pattern.
/// Push events in, extract results via the completion predicate.
pub struct EventStream<T, R = Vec<AgentMessage>> {
    events: Arc<Mutex<Vec<T>>>,
    is_done: Arc<tokio::sync::Notify>,
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
            is_done: Arc::new(tokio::sync::Notify::new()),
            result: Arc::new(Mutex::new(None)),
            is_complete: Arc::new(Mutex::new(false)),
        }
    }

    pub fn push(&self, event: T) {
        if *self.is_complete.lock().unwrap() {
            return;
        }
        self.events.lock().unwrap().push(event);
    }

    pub fn is_complete(&self) -> bool {
        *self.is_complete.lock().unwrap()
    }

    pub fn end(&self, result: R) {
        let mut done = self.is_complete.lock().unwrap();
        if *done {
            return;
        }
        *done = true;
        *self.result.lock().unwrap() = Some(result);
        self.is_done.notify_one();
    }

    pub fn take_events(&self) -> Vec<T> {
        std::mem::take(&mut *self.events.lock().unwrap())
    }

    /// Non-blocking check: returns Some(result) if the stream has ended.
    pub fn wait_for_result_try(&self) -> Result<Option<R>, ()> {
        if *self.is_complete.lock().unwrap() {
            Ok(self.result.lock().unwrap().clone())
        } else {
            Err(())
        }
    }

    pub async fn wait_for_result(&self) -> R {
        if *self.is_complete.lock().unwrap() {
            return self.result.lock().unwrap().clone().unwrap();
        }
        self.is_done.notified().await;
        self.result.lock().unwrap().clone().unwrap()
    }
}

impl<T: Clone, R: Clone> Default for EventStream<T, R> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Agent Event Sink
// ============================================================================

/// Receiver end of agent events. Receive events one at a time.
pub struct AgentEventReceiver {
    rx: mpsc::UnboundedReceiver<AgentEvent>,
}

impl AgentEventReceiver {
    pub async fn recv(&mut self) -> Option<AgentEvent> {
        self.rx.recv().await
    }
}

/// Sender + Receiver pair for agent events.
/// The sender is used by the agent loop; the receiver is for external observers.
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
        if *self.closed.lock().unwrap() {
            return;
        }
        let _ = self.tx.send(event);
    }

    pub fn close(&self) {
        *self.closed.lock().unwrap() = true;
    }
}

impl Clone for AgentEventChannel {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            closed: Arc::clone(&self.closed),
        }
    }
}

// ============================================================================
// Agent Event Listener
// ============================================================================

/// Trait for listening to agent events.
/// Called with the event and the current abort signal.
pub trait AgentEventListener: Send + Sync {
    fn on_event(
        &self,
        event: AgentEvent,
        signal: Option<tokio_util::sync::CancellationToken>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;
}

/// A simple function-wrapper listener.
pub struct FnEventListener<F> {
    f: F,
}

impl<F, Fut> FnEventListener<F>
where
    F: Fn(AgentEvent, Option<tokio_util::sync::CancellationToken>) -> Fut + Send + Sync,
    Fut: Future<Output = ()> + Send + 'static,
{
    pub fn new(f: F) -> Self {
        Self { f }
    }
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_event_serde() {
        let event = AgentEvent::AgentStart;
        let json = serde_json::to_string(&event).unwrap();
        assert_eq!(json, r#"{"type":"agent_start"}"#);

        let event = AgentEvent::ToolExecutionEnd {
            tool_call_id: "tc1".to_string(),
            tool_name: "bash".to_string(),
            result: serde_json::json!({"output": "hello"}),
            is_error: false,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("tool_execution_end"));
        assert!(json.contains("bash"));
        let deser: AgentEvent = serde_json::from_str(&json).unwrap();
        match deser {
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
                assert_eq!(tool_call_id, "tc1");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_assistant_message_event_serde() {
        let event = AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: "hi".to_string(),
            partial: serde_json::json!({"role": "assistant", "content": [{"type": "text", "text": "hi"}]}),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("text_delta"));
        assert!(json.contains("contentIndex"));
    }

    #[test]
    fn test_event_stream_basic() {
        let stream = EventStream::<AgentEvent, Vec<AgentMessage>>::new();
        stream.push(AgentEvent::AgentStart);
        stream.push(AgentEvent::TurnStart);
        stream.end(vec![serde_json::json!({"role": "user"})]);

        let events = stream.take_events();
        assert_eq!(events.len(), 2);
    }

    #[tokio::test]
    async fn test_event_channel() {
        let (ch, mut rx) = AgentEventChannel::new(10);
        ch.send(AgentEvent::AgentStart);
        let event = rx.recv().await.unwrap();
        match event {
            AgentEvent::AgentStart => {}
            _ => panic!("Expected AgentStart"),
        }
    }
}
