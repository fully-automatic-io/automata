// Message module - Message types

use crate::types::ContentBlock;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::any::Any;

/// Trait for agent messages
///
/// This trait allows for custom message types that can be used in the agent loop.
/// Messages can be converted to/from LLM-compatible formats.
#[async_trait]
pub trait AgentMessage: Send + Sync + std::fmt::Debug {
    /// Get the role of the message (user, assistant, system)
    fn role(&self) -> MessageRole;

    /// Get the content blocks of the message
    fn content(&self) -> Vec<ContentBlock>;

    /// Clone the message as a boxed trait object
    fn clone_box(&self) -> Box<dyn AgentMessage>;

    /// Downcast to Any for type checking
    fn as_any(&self) -> &dyn Any;

    /// Downcast to mutable Any for type checking
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Serialize the message to JSON
    fn to_json(&self) -> serde_json::Value;

    /// Get message metadata
    fn metadata(&self) -> Option<&serde_json::Value> {
        None
    }
}

impl Clone for Box<dyn AgentMessage> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Message role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

/// Standard message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StandardMessage {
    /// User message
    User {
        content: Vec<ContentBlock>,
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<serde_json::Value>,
    },
    /// Assistant message
    Assistant {
        content: Vec<ContentBlock>,
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<serde_json::Value>,
    },
    /// System message
    System {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<serde_json::Value>,
    },
    /// Tool result message
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<serde_json::Value>,
    },
}

impl StandardMessage {
    /// Create a user message with text content
    pub fn user_text(text: impl Into<String>) -> Self {
        Self::User {
            content: vec![ContentBlock::Text { text: text.into() }],
            metadata: None,
        }
    }

    /// Create an assistant message with text content
    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::Assistant {
            content: vec![ContentBlock::Text { text: text.into() }],
            metadata: None,
        }
    }

    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
            metadata: None,
        }
    }

    /// Create a user message with content blocks
    pub fn user_with_content(content: Vec<ContentBlock>) -> Self {
        Self::User {
            content,
            metadata: None,
        }
    }

    /// Create an assistant message with content blocks
    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self::Assistant {
            content,
            metadata: None,
        }
    }

    /// Create a tool result message
    pub fn tool_result(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error: false,
            metadata: None,
        }
    }

    /// Create an error tool result message
    pub fn tool_error(tool_use_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: error.into(),
            is_error: true,
            metadata: None,
        }
    }

    /// Add metadata to the message
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        match &mut self {
            Self::User { metadata: m, .. }
            | Self::Assistant { metadata: m, .. }
            | Self::System { metadata: m, .. }
            | Self::ToolResult { metadata: m, .. } => {
                *m = Some(metadata);
            }
        }
        self
    }
}

#[async_trait]
impl AgentMessage for StandardMessage {
    fn role(&self) -> MessageRole {
        match self {
            Self::User { .. } => MessageRole::User,
            Self::Assistant { .. } => MessageRole::Assistant,
            Self::System { .. } => MessageRole::System,
            Self::ToolResult { .. } => MessageRole::User,
        }
    }

    fn content(&self) -> Vec<ContentBlock> {
        match self {
            Self::User { content, .. } | Self::Assistant { content, .. } => content.clone(),
            Self::System { content, .. } => {
                vec![ContentBlock::Text {
                    text: content.clone(),
                }]
            }
            Self::ToolResult {
                tool_use_id,
                content,
                is_error,
                ..
            } => {
                vec![ContentBlock::ToolResult {
                    tool_call_id: tool_use_id.clone(),
                    tool_name: None,
                    content: vec![ContentBlock::Text { text: content.clone() }],
                    is_error: *is_error,
                    details: None,
                }]
            }
        }
    }

    fn clone_box(&self) -> Box<dyn AgentMessage> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    fn metadata(&self) -> Option<&serde_json::Value> {
        match self {
            Self::User { metadata, .. }
            | Self::Assistant { metadata, .. }
            | Self::System { metadata, .. }
            | Self::ToolResult { metadata, .. } => metadata.as_ref(),
        }
    }
}

/// Helper to downcast a message to a specific type
pub fn downcast_message<T: AgentMessage + 'static>(
    message: &dyn AgentMessage,
) -> Option<&T> {
    message.as_any().downcast_ref::<T>()
}

/// Helper to downcast a mutable message to a specific type
pub fn downcast_message_mut<T: AgentMessage + 'static>(
    message: &mut dyn AgentMessage,
) -> Option<&mut T> {
    message.as_any_mut().downcast_mut::<T>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_message() {
        let msg = StandardMessage::user_text("Hello");
        assert_eq!(msg.role(), MessageRole::User);
        assert_eq!(msg.content().len(), 1);
    }

    #[test]
    fn test_assistant_message() {
        let msg = StandardMessage::assistant_text("Hi there");
        assert_eq!(msg.role(), MessageRole::Assistant);
    }

    #[test]
    fn test_tool_result() {
        let msg = StandardMessage::tool_result("tool_1", "result");
        assert_eq!(msg.role(), MessageRole::User);
        match &msg {
            StandardMessage::ToolResult { is_error, .. } => {
                assert!(!is_error);
            }
            _ => panic!("Expected ToolResult"),
        }
    }

    #[test]
    fn test_metadata() {
        let msg = StandardMessage::user_text("Hello")
            .with_metadata(serde_json::json!({"key": "value"}));
        assert!(msg.metadata().is_some());
    }

    #[test]
    fn test_downcast() {
        let msg = StandardMessage::user_text("Hello");
        let boxed: Box<dyn AgentMessage> = Box::new(msg);
        let downcasted = downcast_message::<StandardMessage>(boxed.as_ref());
        assert!(downcasted.is_some());
    }
}
