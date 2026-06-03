//
// Proxy stream function for apps that route LLM calls through a server.
// The server manages auth and proxies requests to LLM providers.

use crate::event::{AssistantMessageEvent, PartialAssistantMessage, PartialContentBlock};
use crate::types::{ContentBlock, StopReason, ThinkingBudgets, ThinkingLevel, Transport};
use serde::{Deserialize, Serialize};

// ============================================================================
// Proxy event types — server sends these with partial field stripped
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProxyAssistantMessageEvent {
    #[serde(rename = "start")]
    Start,
    #[serde(rename = "text_start")]
    TextStart { content_index: usize },
    #[serde(rename = "text_delta")]
    TextDelta { content_index: usize, delta: String },
    #[serde(rename = "text_end")]
    TextEnd {
        content_index: usize,
        #[serde(rename = "contentSignature", skip_serializing_if = "Option::is_none")]
        content_signature: Option<String>,
    },
    #[serde(rename = "thinking_start")]
    ThinkingStart { content_index: usize },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { content_index: usize, delta: String },
    #[serde(rename = "thinking_end")]
    ThinkingEnd {
        content_index: usize,
        #[serde(rename = "contentSignature", skip_serializing_if = "Option::is_none")]
        content_signature: Option<String>,
    },
    #[serde(rename = "toolcall_start")]
    ToolCallStart {
        content_index: usize,
        id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
    },
    #[serde(rename = "toolcall_delta")]
    ToolCallDelta { content_index: usize, delta: String },
    #[serde(rename = "toolcall_end")]
    ToolCallEnd { content_index: usize },
    #[serde(rename = "done")]
    Done {
        reason: StopReason,
        usage: crate::types::Usage,
    },
    #[serde(rename = "error")]
    Error {
        reason: StopReason,
        #[serde(rename = "errorMessage", skip_serializing_if = "Option::is_none")]
        error_message: Option<String>,
        usage: crate::types::Usage,
    },
}

// ============================================================================
// Proxy stream options
// ============================================================================

#[derive(Debug, Clone)]
pub struct ProxyStreamOptions {
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub reasoning: Option<ThinkingLevel>,
    pub session_id: Option<String>,
    pub headers: Option<std::collections::HashMap<String, String>>,
    pub transport: Transport,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub max_retry_delay_ms: Option<u64>,
    pub auth_token: String,
    pub proxy_url: String,
    pub signal: Option<tokio_util::sync::CancellationToken>,
}

// ============================================================================
// Proxy stream function
// ============================================================================

/// Reconstruct an AssistantMessageEvent from a proxy event and the typed
/// streaming `PartialAssistantMessage`.
pub fn process_proxy_event(
    proxy_event: ProxyAssistantMessageEvent,
    partial: &mut PartialAssistantMessage,
) -> Option<AssistantMessageEvent> {
    match proxy_event {
        ProxyAssistantMessageEvent::Start => {
            Some(AssistantMessageEvent::Start { partial: partial.clone() })
        }
        ProxyAssistantMessageEvent::TextStart { content_index } => {
            partial.ensure_block_at(content_index);
            partial.content[content_index] = PartialContentBlock::Text {
                text: String::new(),
                text_signature: None,
            };
            Some(AssistantMessageEvent::TextStart { content_index, partial: partial.clone() })
        }
        ProxyAssistantMessageEvent::TextDelta { content_index, delta } => {
            if let Some(PartialContentBlock::Text { text, .. }) =
                partial.content.get_mut(content_index)
            {
                text.push_str(&delta);
            }
            Some(AssistantMessageEvent::TextDelta {
                content_index,
                delta,
                partial: partial.clone(),
            })
        }
        ProxyAssistantMessageEvent::TextEnd { content_index, content_signature } => {
            let mut text_out = String::new();
            if let Some(PartialContentBlock::Text { text, text_signature }) =
                partial.content.get_mut(content_index)
            {
                text_out = text.clone();
                if content_signature.is_some() {
                    *text_signature = content_signature;
                }
            }
            Some(AssistantMessageEvent::TextEnd {
                content_index,
                content: text_out,
                partial: partial.clone(),
            })
        }
        ProxyAssistantMessageEvent::ThinkingStart { content_index } => {
            partial.ensure_block_at(content_index);
            partial.content[content_index] = PartialContentBlock::Thinking {
                thinking: String::new(),
                thinking_signature: None,
            };
            Some(AssistantMessageEvent::ThinkingStart { content_index, partial: partial.clone() })
        }
        ProxyAssistantMessageEvent::ThinkingDelta { content_index, delta } => {
            if let Some(PartialContentBlock::Thinking { thinking, .. }) =
                partial.content.get_mut(content_index)
            {
                thinking.push_str(&delta);
            }
            Some(AssistantMessageEvent::ThinkingDelta {
                content_index,
                delta,
                partial: partial.clone(),
            })
        }
        ProxyAssistantMessageEvent::ThinkingEnd { content_index, content_signature } => {
            let mut text_out = String::new();
            if let Some(PartialContentBlock::Thinking { thinking, thinking_signature }) =
                partial.content.get_mut(content_index)
            {
                text_out = thinking.clone();
                if content_signature.is_some() {
                    *thinking_signature = content_signature;
                }
            }
            Some(AssistantMessageEvent::ThinkingEnd {
                content_index,
                content: text_out,
                partial: partial.clone(),
            })
        }
        ProxyAssistantMessageEvent::ToolCallStart { content_index, id, tool_name } => {
            partial.ensure_block_at(content_index);
            partial.content[content_index] = PartialContentBlock::ToolCall {
                id,
                name: tool_name,
                arguments: serde_json::json!({}),
                partial_json: None,
            };
            Some(AssistantMessageEvent::ToolCallStart { content_index, partial: partial.clone() })
        }
        ProxyAssistantMessageEvent::ToolCallDelta { content_index, delta } => {
            if let Some(PartialContentBlock::ToolCall { arguments, partial_json, .. }) =
                partial.content.get_mut(content_index)
            {
                let buf = partial_json.get_or_insert_with(String::new);
                buf.push_str(&delta);
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(buf) {
                    *arguments = parsed;
                }
            }
            Some(AssistantMessageEvent::ToolCallDelta {
                content_index,
                delta,
                partial: partial.clone(),
            })
        }
        ProxyAssistantMessageEvent::ToolCallEnd { content_index } => {
            // Drop in-flight `partialJson` and snapshot the finalized block.
            if let Some(PartialContentBlock::ToolCall { partial_json, .. }) =
                partial.content.get_mut(content_index)
            {
                *partial_json = None;
            }
            let tool_call: ContentBlock = partial
                .content
                .get(content_index)
                .cloned()
                .map(PartialContentBlock::into_block)
                .unwrap_or(ContentBlock::Text { text: String::new() });
            Some(AssistantMessageEvent::ToolCallEnd {
                content_index,
                tool_call,
                partial: partial.clone(),
            })
        }
        ProxyAssistantMessageEvent::Done { reason, usage } => {
            partial.stop_reason = reason;
            partial.usage = usage;
            let final_msg = partial.clone().into_finalized();
            Some(AssistantMessageEvent::Done { reason, message: final_msg })
        }
        ProxyAssistantMessageEvent::Error { reason, error_message, usage } => {
            partial.stop_reason = reason;
            partial.usage = usage;
            if let Some(msg) = error_message {
                partial.error_message = Some(msg);
            }
            Some(AssistantMessageEvent::Error { reason, error: partial.clone() })
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_proxy_event_start() {
        let mut partial = PartialAssistantMessage::new(crate::types::Api::Anthropic, "p", "m");
        let event = process_proxy_event(ProxyAssistantMessageEvent::Start, &mut partial);
        assert!(matches!(event, Some(AssistantMessageEvent::Start { .. })));
    }

    #[test]
    fn test_process_proxy_text_delta() {
        let mut partial = PartialAssistantMessage::new(crate::types::Api::Anthropic, "p", "m");
        partial.content.push(PartialContentBlock::Text {
            text: "Hello".to_string(),
            text_signature: None,
        });
        let event = process_proxy_event(
            ProxyAssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: " world".to_string(),
            },
            &mut partial,
        );
        assert!(event.is_some());
        match &partial.content[0] {
            PartialContentBlock::Text { text, .. } => assert_eq!(text, "Hello world"),
            _ => panic!("expected Text block"),
        }
    }

    #[test]
    fn test_process_proxy_done() {
        let mut partial = PartialAssistantMessage::new(crate::types::Api::Anthropic, "p", "m");
        partial.content.push(PartialContentBlock::Text {
            text: "Done".to_string(),
            text_signature: None,
        });
        let event = process_proxy_event(
            ProxyAssistantMessageEvent::Done {
                reason: StopReason::EndTurn,
                usage: crate::types::Usage::default(),
            },
            &mut partial,
        );
        match event.unwrap() {
            AssistantMessageEvent::Done { reason, message, .. } => {
                assert_eq!(reason, StopReason::EndTurn);
                assert_eq!(message.role(), "assistant");
            }
            _ => panic!("Expected Done event"),
        }
    }
}
