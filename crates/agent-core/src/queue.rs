// Queue module — message queue for steering and follow-up.

use crate::types::AgentMessage;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QueueMode {
    /// Drain all queued messages on each `drain()` call.
    All,
    /// Drain a single message per `drain()` call.
    OneAtATime,
}

impl Default for QueueMode {
    fn default() -> Self { Self::OneAtATime }
}

#[derive(Debug)]
pub struct MessageQueue {
    messages: Vec<AgentMessage>,
    mode: QueueMode,
}

impl MessageQueue {
    pub fn new(mode: QueueMode) -> Self {
        Self { messages: Vec::new(), mode }
    }

    pub fn enqueue(&mut self, message: AgentMessage) { self.messages.push(message); }
    pub fn has_items(&self) -> bool { !self.messages.is_empty() }

    pub fn drain(&mut self) -> Vec<AgentMessage> {
        match self.mode {
            QueueMode::All => std::mem::take(&mut self.messages),
            QueueMode::OneAtATime => {
                if self.messages.is_empty() { Vec::new() }
                else { vec![self.messages.remove(0)] }
            }
        }
    }

    pub fn clear(&mut self) { self.messages.clear(); }
    pub fn mode(&self) -> QueueMode { self.mode }
    pub fn set_mode(&mut self, mode: QueueMode) { self.mode = mode; }
    pub fn len(&self) -> usize { self.messages.len() }
    pub fn is_empty(&self) -> bool { self.messages.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_all_mode() {
        let mut queue = MessageQueue::new(QueueMode::All);
        queue.enqueue(AgentMessage::user_text("a"));
        queue.enqueue(AgentMessage::user_text("b"));
        let drained = queue.drain();
        assert_eq!(drained.len(), 2);
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_queue_one_at_a_time_mode() {
        let mut queue = MessageQueue::new(QueueMode::OneAtATime);
        queue.enqueue(AgentMessage::user_text("a"));
        queue.enqueue(AgentMessage::user_text("b"));
        assert_eq!(queue.drain().len(), 1);
        assert_eq!(queue.len(), 1);
    }
}
