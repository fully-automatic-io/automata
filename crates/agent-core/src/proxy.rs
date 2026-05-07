//
// Proxy stream function for apps that route LLM calls through a server.
// The server manages auth and proxies requests to LLM providers.

use crate::event::AssistantMessageEvent;
use crate::types::{AgentMessage, ThinkingBudgets, Transport};
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
        reason: String,
        usage: crate::types::Usage,
    },
    #[serde(rename = "error")]
    Error {
        reason: String,
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
    pub reasoning: Option<String>,
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

/// Reconstruct an AssistantMessageEvent from a proxy event and partial message.
pub fn process_proxy_event(
    proxy_event: ProxyAssistantMessageEvent,
    partial: &mut AgentMessage,
) -> Option<AssistantMessageEvent> {
    match proxy_event {
        ProxyAssistantMessageEvent::Start => {
            Some(AssistantMessageEvent::Start {
                partial: partial.clone(),
            })
        }
        ProxyAssistantMessageEvent::TextStart { content_index } => {
            let content = partial
                .get_mut("content")
                .and_then(|c| c.as_array_mut());
            if let Some(arr) = content {
                while arr.len() <= content_index {
                    arr.push(serde_json::json!({}));
                }
                arr[content_index] = serde_json::json!({"type": "text", "text": ""});
            }
            Some(AssistantMessageEvent::TextStart {
                content_index,
                partial: partial.clone(),
            })
        }
        ProxyAssistantMessageEvent::TextDelta {
            content_index,
            delta,
        } => {
            if let Some(content) = partial
                .get_mut("content")
                .and_then(|c| c.as_array_mut())
                .and_then(|arr| arr.get_mut(content_index))
            {
                if content.get("type").and_then(|t| t.as_str()) == Some("text") {
                    let current = content
                        .get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("");
                    let new_text = format!("{}{}", current, delta);
                    content["text"] = serde_json::json!(new_text);
                }
            }
            Some(AssistantMessageEvent::TextDelta {
                content_index,
                delta,
                partial: partial.clone(),
            })
        }
        ProxyAssistantMessageEvent::TextEnd {
            content_index,
            content_signature,
        } => {
            let text = partial
                .get("content")
                .and_then(|c| c.as_array())
                .and_then(|arr| arr.get(content_index))
                .and_then(|b| b.get("text"))
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();
            if let Some(content) = partial
                .get_mut("content")
                .and_then(|c| c.as_array_mut())
                .and_then(|arr| arr.get_mut(content_index))
            {
                if let Some(sig) = content_signature {
                    content["textSignature"] = serde_json::json!(sig);
                }
            }
            Some(AssistantMessageEvent::TextEnd {
                content_index,
                content: text,
                partial: partial.clone(),
            })
        }
        ProxyAssistantMessageEvent::ThinkingStart { content_index } => {
            let content = partial
                .get_mut("content")
                .and_then(|c| c.as_array_mut());
            if let Some(arr) = content {
                while arr.len() <= content_index {
                    arr.push(serde_json::json!({}));
                }
                arr[content_index] =
                    serde_json::json!({"type": "thinking", "thinking": ""});
            }
            Some(AssistantMessageEvent::ThinkingStart {
                content_index,
                partial: partial.clone(),
            })
        }
        ProxyAssistantMessageEvent::ThinkingDelta {
            content_index,
            delta,
        } => {
            if let Some(content) = partial
                .get_mut("content")
                .and_then(|c| c.as_array_mut())
                .and_then(|arr| arr.get_mut(content_index))
            {
                if content.get("type").and_then(|t| t.as_str()) == Some("thinking") {
                    let current = content
                        .get("thinking")
                        .and_then(|t| t.as_str())
                        .unwrap_or("");
                    content["thinking"] = serde_json::json!(format!("{}{}", current, delta));
                }
            }
            Some(AssistantMessageEvent::ThinkingDelta {
                content_index,
                delta,
                partial: partial.clone(),
            })
        }
        ProxyAssistantMessageEvent::ThinkingEnd {
            content_index,
            content_signature,
        } => {
            let thinking = partial
                .get("content")
                .and_then(|c| c.as_array())
                .and_then(|arr| arr.get(content_index))
                .and_then(|b| b.get("thinking"))
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();
            if let Some(content) = partial
                .get_mut("content")
                .and_then(|c| c.as_array_mut())
                .and_then(|arr| arr.get_mut(content_index))
            {
                if let Some(sig) = content_signature {
                    content["thinkingSignature"] = serde_json::json!(sig);
                }
            }
            Some(AssistantMessageEvent::ThinkingEnd {
                content_index,
                content: thinking,
                partial: partial.clone(),
            })
        }
        ProxyAssistantMessageEvent::ToolCallStart {
            content_index,
            id,
            tool_name,
        } => {
            let content = partial
                .get_mut("content")
                .and_then(|c| c.as_array_mut());
            if let Some(arr) = content {
                while arr.len() <= content_index {
                    arr.push(serde_json::json!({}));
                }
                arr[content_index] = serde_json::json!({
                    "type": "toolCall",
                    "id": id,
                    "name": tool_name,
                    "arguments": {}
                });
            }
            Some(AssistantMessageEvent::ToolCallStart {
                content_index,
                partial: partial.clone(),
            })
        }
        ProxyAssistantMessageEvent::ToolCallDelta {
            content_index,
            delta,
        } => {
            if let Some(content) = partial
                .get_mut("content")
                .and_then(|c| c.as_array_mut())
                .and_then(|arr| arr.get_mut(content_index))
            {
                if content.get("type").and_then(|t| t.as_str()) == Some("toolCall") {
                    let current = content
                        .get("partialJson")
                        .and_then(|t| t.as_str())
                        .unwrap_or("");
                    let new_json = format!("{}{}", current, delta);
                    content["partialJson"] = serde_json::json!(new_json);
                    // Try to parse incremental JSON
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&new_json) {
                        content["arguments"] = parsed;
                    }
                }
            }
            Some(AssistantMessageEvent::ToolCallDelta {
                content_index,
                delta,
                partial: partial.clone(),
            })
        }
        ProxyAssistantMessageEvent::ToolCallEnd { content_index } => {
            let tool_call = partial
                .get("content")
                .and_then(|c| c.as_array())
                .and_then(|arr| arr.get(content_index))
                .cloned()
                .unwrap_or(serde_json::json!({}));
            if let Some(content) = partial
                .get_mut("content")
                .and_then(|c| c.as_array_mut())
                .and_then(|arr| arr.get_mut(content_index))
            {
                content.as_object_mut().map(|o| o.remove("partialJson"));
            }
            Some(AssistantMessageEvent::ToolCallEnd {
                content_index,
                tool_call,
                partial: partial.clone(),
            })
        }
        ProxyAssistantMessageEvent::Done { reason, usage } => {
            if let Some(obj) = partial.as_object_mut() {
                obj.insert("stopReason".to_string(), serde_json::json!(reason));
                obj.insert("usage".to_string(), serde_json::to_value(usage).unwrap_or_default());
            }
            Some(AssistantMessageEvent::Done {
                reason,
                message: partial.clone(),
            })
        }
        ProxyAssistantMessageEvent::Error {
            reason,
            error_message,
            usage: _,
        } => {
            if let Some(obj) = partial.as_object_mut() {
                obj.insert("stopReason".to_string(), serde_json::json!(reason));
                if let Some(msg) = error_message {
                    obj.insert("errorMessage".to_string(), serde_json::json!(msg));
                }
            }
            Some(AssistantMessageEvent::Error {
                reason,
                error: partial.clone(),
            })
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
        let mut partial = serde_json::json!({
            "role": "assistant",
            "content": [],
            "stopReason": "stop"
        });
        let event = process_proxy_event(ProxyAssistantMessageEvent::Start, &mut partial);
        assert!(event.is_some());
        match event.unwrap() {
            AssistantMessageEvent::Start { .. } => {}
            _ => panic!("Expected Start event"),
        }
    }

    #[test]
    fn test_process_proxy_text_delta() {
        let mut partial = serde_json::json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "stopReason": "stop"
        });
        let event = process_proxy_event(
            ProxyAssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: " world".to_string(),
            },
            &mut partial,
        );
        assert!(event.is_some());
        assert_eq!(
            partial["content"][0]["text"].as_str().unwrap(),
            "Hello world"
        );
    }

    #[test]
    fn test_process_proxy_done() {
        let mut partial = serde_json::json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "Done"}],
            "stopReason": "stop"
        });
        let event = process_proxy_event(
            ProxyAssistantMessageEvent::Done {
                reason: "stop".to_string(),
                usage: crate::types::Usage::default(),
            },
            &mut partial,
        );
        assert!(event.is_some());
        match event.unwrap() {
            AssistantMessageEvent::Done { reason, .. } => assert_eq!(reason, "stop"),
            _ => panic!("Expected Done event"),
        }
    }
}
