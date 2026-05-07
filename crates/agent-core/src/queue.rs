// Queue module - Message queue management for steering and follow-up

use serde::{Deserialize, Serialize};

/// Queue drain mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QueueMode {
    /// Drain all messages at once
    All,
    /// Drain one message at a time
    OneAtATime,
}

impl Default for QueueMode {
    fn default() -> Self {
        Self::OneAtATime
    }
}

/// Message queue for steering and follow-up messages
#[derive(Debug)]
pub struct MessageQueue {
    messages: Vec<serde_json::Value>,
    mode: QueueMode,
}

impl MessageQueue {
    /// Create a new message queue with the specified mode
    pub fn new(mode: QueueMode) -> Self {
        Self {
            messages: Vec::new(),
            mode,
        }
    }

    /// Enqueue a message
    pub fn enqueue(&mut self, message: serde_json::Value) {
        self.messages.push(message);
    }

    /// Check if queue has items
    pub fn has_items(&self) -> bool {
        !self.messages.is_empty()
    }

    /// Drain messages according to the queue mode
    pub fn drain(&mut self) -> Vec<serde_json::Value> {
        match self.mode {
            QueueMode::All => {
                let drained = self.messages.clone();
                self.messages.clear();
                drained
            }
            QueueMode::OneAtATime => {
                if self.messages.is_empty() {
                    Vec::new()
                } else {
                    vec![self.messages.remove(0)]
                }
            }
        }
    }

    /// Clear all messages
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Get current queue mode
    pub fn mode(&self) -> QueueMode {
        self.mode
    }

    /// Set queue mode
    pub fn set_mode(&mut self, mode: QueueMode) {
        self.mode = mode;
    }

    /// Get number of queued messages
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_queue_all_mode() {
        let mut queue = MessageQueue::new(QueueMode::All);
        queue.enqueue(json!({"role": "user", "content": "msg1"}));
        queue.enqueue(json!({"role": "user", "content": "msg2"}));
        queue.enqueue(json!({"role": "user", "content": "msg3"}));

        assert_eq!(queue.len(), 3);
        assert!(queue.has_items());

        let drained = queue.drain();
        assert_eq!(drained.len(), 3);
        assert_eq!(queue.len(), 0);
        assert!(!queue.has_items());
    }

    #[test]
    fn test_queue_one_at_a_time_mode() {
        let mut queue = MessageQueue::new(QueueMode::OneAtATime);
        queue.enqueue(json!({"role": "user", "content": "msg1"}));
        queue.enqueue(json!({"role": "user", "content": "msg2"}));
        queue.enqueue(json!({"role": "user", "content": "msg3"}));

        assert_eq!(queue.len(), 3);

        let drained1 = queue.drain();
        assert_eq!(drained1.len(), 1);
        assert_eq!(queue.len(), 2);

        let drained2 = queue.drain();
        assert_eq!(drained2.len(), 1);
        assert_eq!(queue.len(), 1);

        let drained3 = queue.drain();
        assert_eq!(drained3.len(), 1);
        assert_eq!(queue.len(), 0);

        let drained4 = queue.drain();
        assert_eq!(drained4.len(), 0);
    }

    #[test]
    fn test_queue_clear() {
        let mut queue = MessageQueue::new(QueueMode::All);
        queue.enqueue(json!({"role": "user", "content": "msg1"}));
        queue.enqueue(json!({"role": "user", "content": "msg2"}));

        assert_eq!(queue.len(), 2);
        queue.clear();
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_queue_mode_change() {
        let mut queue = MessageQueue::new(QueueMode::All);
        assert_eq!(queue.mode(), QueueMode::All);

        queue.set_mode(QueueMode::OneAtATime);
        assert_eq!(queue.mode(), QueueMode::OneAtATime);
    }
}
