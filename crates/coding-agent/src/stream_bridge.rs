// Stream bridge — converts llm-client SSE streams to agent-core AssistantMessageEvent streams
// Lives in coding-agent since it depends on both llm-client and agent-core.

use agent_core::agent_loop::{StreamFn, StreamFnInput};
use agent_core::event::{AssistantMessageEvent, EventStream};
use agent_core::types::AgentMessage;
use llm_client::provider::LlmProvider;
use llm_client::streaming::Delta;
use llm_client::types::{LlmMessage, LlmRequest, Model, ToolDefinition};
use std::sync::Arc;

/// Convert an llm-client SSE stream into agent-core's AssistantMessageEvent stream.
pub fn convert_sse_stream(
    sse_stream: llm_client::LlmStream,
    api: String,
    provider: String,
    model_id: String,
) -> EventStream<AssistantMessageEvent, AgentMessage> {
    let stream = EventStream::<AssistantMessageEvent, AgentMessage>::new();
    let stream_clone = stream.clone();

    tokio::spawn(async move {
        use futures::StreamExt;
        let mut sse = sse_stream;

        let mut partial: AgentMessage = serde_json::json!({
            "role": "assistant",
            "stopReason": "stop",
            "content": [],
            "api": api,
            "provider": provider,
            "model": model_id,
            "usage": {
                "input": 0, "output": 0,
                "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0}
            },
            "timestamp": chrono::Utc::now().timestamp_millis()
        });

        stream_clone.push(AssistantMessageEvent::Start { partial: partial.clone() });

        let mut _content_index: usize = 0;

        while let Some(event) = sse.next().await {
            match event {
                Ok(llm_client::streaming::LlmEvent::ContentBlockStart { index, content_block }) => {
                    _content_index = index;
                    ensure_content_index(&mut partial, index);
                    match content_block {
                        llm_client::types::ContentPart::Text { .. } => {
                            partial["content"][index] = serde_json::json!({"type": "text", "text": ""});
                            stream_clone.push(AssistantMessageEvent::TextStart {
                                content_index: index,
                                partial: partial.clone(),
                            });
                        }
                        llm_client::types::ContentPart::Thinking { .. } => {
                            partial["content"][index] = serde_json::json!({"type": "thinking", "thinking": ""});
                            stream_clone.push(AssistantMessageEvent::ThinkingStart {
                                content_index: index,
                                partial: partial.clone(),
                            });
                        }
                        llm_client::types::ContentPart::ToolUse { id, name, .. } => {
                            partial["content"][index] = serde_json::json!({
                                "type": "toolCall", "id": id, "name": name, "arguments": {}
                            });
                            stream_clone.push(AssistantMessageEvent::ToolCallStart {
                                content_index: index,
                                partial: partial.clone(),
                            });
                        }
                        _ => {}
                    }
                }
                Ok(llm_client::streaming::LlmEvent::ContentBlockDelta { index, delta }) => {
                    _content_index = index;
                    match delta {
                        Delta::TextDelta { text } => {
                            if let Some(c) = partial["content"].get_mut(index) {
                                let old = c["text"].as_str().unwrap_or("");
                                c["text"] = serde_json::json!(format!("{}{}", old, text));
                            }
                            stream_clone.push(AssistantMessageEvent::TextDelta {
                                content_index: index, delta: text, partial: partial.clone(),
                            });
                        }
                        Delta::ThinkingDelta { thinking } => {
                            if let Some(c) = partial["content"].get_mut(index) {
                                let old = c["thinking"].as_str().unwrap_or("");
                                c["thinking"] = serde_json::json!(format!("{}{}", old, thinking));
                            }
                            stream_clone.push(AssistantMessageEvent::ThinkingDelta {
                                content_index: index, delta: thinking, partial: partial.clone(),
                            });
                        }
                        Delta::InputJsonDelta { partial_json } => {
                            if let Some(c) = partial["content"].get_mut(index) {
                                if c["type"] == "toolCall" {
                                    let old = c.get("partialJson").and_then(|v| v.as_str()).unwrap_or("");
                                    let new_json = format!("{}{}", old, partial_json);
                                    c["partialJson"] = serde_json::json!(new_json);
                                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&new_json) {
                                        c["arguments"] = parsed;
                                    }
                                }
                            }
                            stream_clone.push(AssistantMessageEvent::ToolCallDelta {
                                content_index: index, delta: partial_json, partial: partial.clone(),
                            });
                        }
                        _ => {}
                    }
                }
                Ok(llm_client::streaming::LlmEvent::ContentBlockStop { index }) => {
                    if let Some(c) = partial["content"].get(index) {
                        match c["type"].as_str() {
                            Some("text") => {
                                let text = c["text"].as_str().unwrap_or("").to_string();
                                stream_clone.push(AssistantMessageEvent::TextEnd {
                                    content_index: index, content: text, partial: partial.clone(),
                                });
                            }
                            Some("thinking") => {
                                let thinking = c["thinking"].as_str().unwrap_or("").to_string();
                                stream_clone.push(AssistantMessageEvent::ThinkingEnd {
                                    content_index: index, content: thinking, partial: partial.clone(),
                                });
                            }
                            Some("toolCall") => {
                                if let Some(obj) = partial["content"][index].as_object_mut() {
                                    obj.remove("partialJson");
                                }
                                let tool_call = partial["content"][index].clone();
                                stream_clone.push(AssistantMessageEvent::ToolCallEnd {
                                    content_index: index, tool_call, partial: partial.clone(),
                                });
                            }
                            _ => {}
                        }
                    }
                }
                Ok(llm_client::streaming::LlmEvent::MessageDelta { delta, usage }) => {
                    if let Some(ref reason) = delta.stop_reason {
                        let reason_str = match reason {
                            llm_client::types::StopReason::EndTurn => "stop",
                            llm_client::types::StopReason::MaxTokens => "length",
                            llm_client::types::StopReason::ToolUse => "toolUse",
                            llm_client::types::StopReason::Error => "error",
                            _ => "stop",
                        };
                        partial["stopReason"] = serde_json::json!(reason_str);
                    }
                    if let Some(ref u) = usage {
                        partial["usage"] = serde_json::json!(u);
                    }
                }
                Ok(llm_client::streaming::LlmEvent::MessageStop) => {
                    stream_clone.push(AssistantMessageEvent::Done {
                        reason: partial["stopReason"].as_str().unwrap_or("stop").to_string(),
                        message: partial.clone(),
                    });
                    stream_clone.end(partial.clone());
                    return;
                }
                Ok(llm_client::streaming::LlmEvent::Error { error }) => {
                    partial["stopReason"] = serde_json::json!("error");
                    partial["errorMessage"] = serde_json::json!(error.message);
                    stream_clone.push(AssistantMessageEvent::Error {
                        reason: "error".to_string(),
                        error: partial.clone(),
                    });
                    stream_clone.end(partial.clone());
                    return;
                }
                Err(e) => {
                    partial["stopReason"] = serde_json::json!("error");
                    partial["errorMessage"] = serde_json::json!(e.to_string());
                    stream_clone.push(AssistantMessageEvent::Error {
                        reason: "error".to_string(),
                        error: partial.clone(),
                    });
                    stream_clone.end(partial.clone());
                    return;
                }
                _ => {}
            }
        }

        stream_clone.end(partial.clone());
    });

    stream
}

fn ensure_content_index(partial: &mut AgentMessage, index: usize) {
    if let Some(arr) = partial["content"].as_array_mut() {
        while arr.len() <= index {
            arr.push(serde_json::json!({}));
        }
    }
}

/// Create a StreamFn compatible with agent-core from an LlmProvider.
pub fn create_stream_fn(provider: Arc<dyn LlmProvider>, model: Model) -> StreamFn {
    Arc::new(move |input: StreamFnInput| {
        let provider = provider.clone();
        let model = model.clone();
        Box::pin(async move {
            let llm_messages: Vec<LlmMessage> = input.messages.iter()
                .filter_map(convert_agent_message_to_llm)
                .collect();

            let tools: Vec<ToolDefinition> = input.tools.iter()
                .map(|t| ToolDefinition {
                    name: t.name().to_string(),
                    description: t.description().to_string(),
                    input_schema: t.parameters(),
                })
                .collect();

            let request = LlmRequest {
                model: model.id.clone(),
                messages: llm_messages,
                tools,
                system: Some(input.system_prompt),
                max_tokens: input.max_tokens,
                temperature: input.temperature,
                stop_sequences: vec![],
                extra: std::collections::HashMap::new(),
            };

            let sse_stream = provider.stream(request).await.map_err(|e| e.to_string())?;

            Ok(convert_sse_stream(sse_stream, model.api.clone(), model.provider.clone(), model.id.clone()))
        })
    })
}

fn convert_agent_message_to_llm(msg: &agent_core::types::Message) -> Option<LlmMessage> {
    match msg {
        agent_core::types::Message::User { content, timestamp, .. } => {
            let parts = match content {
                agent_core::types::MessageContent::String(s) => {
                    vec![llm_client::types::ContentPart::Text { text: s.clone() }]
                }
                agent_core::types::MessageContent::Blocks(blocks) => {
                    blocks.iter().map(convert_block_to_part).collect()
                }
            };
            Some(LlmMessage::User { content: llm_client::types::MessageContent::Blocks(parts), timestamp: *timestamp })
        }
        agent_core::types::Message::Assistant { content, api, provider, model, usage, stop_reason, error_message, timestamp } => {
            let parts: Vec<llm_client::types::ContentPart> = content.iter().map(convert_block_to_part).collect();
            Some(LlmMessage::Assistant {
                content: parts,
                api: api.clone(),
                provider: provider.clone(),
                model: model.clone(),
                usage: llm_client::types::Usage {
                    input: usage.input,
                    output: usage.output,
                    cache_read: usage.cache_read,
                    cache_write: usage.cache_write,
                    total_tokens: usage.total_tokens,
                    cost: llm_client::types::UsageCost {
                        input: usage.cost.input,
                        output: usage.cost.output,
                        cache_read: usage.cost.cache_read,
                        cache_write: usage.cost.cache_write,
                        total: usage.cost.total,
                    },
                },
                stop_reason: stop_reason.clone(),
                error_message: error_message.clone(),
                timestamp: *timestamp,
            })
        }
        agent_core::types::Message::ToolResult { tool_call_id, tool_name, content, details, is_error, timestamp } => {
            let parts: Vec<llm_client::types::ContentPart> = content.iter().map(convert_block_to_part).collect();
            Some(LlmMessage::ToolResult {
                tool_call_id: tool_call_id.clone(),
                tool_name: Some(tool_name.clone()),
                content: parts,
                details: details.clone(),
                is_error: *is_error,
                timestamp: *timestamp,
            })
        }
    }
}

fn convert_block_to_part(block: &agent_core::types::ContentBlock) -> llm_client::types::ContentPart {
    match block {
        agent_core::types::ContentBlock::Text { text } => {
            llm_client::types::ContentPart::Text { text: text.clone() }
        }
        agent_core::types::ContentBlock::Image { data, mime_type } => {
            llm_client::types::ContentPart::Image { data: data.clone(), mime_type: mime_type.clone() }
        }
        agent_core::types::ContentBlock::ToolCall { id, name, arguments } => {
            llm_client::types::ContentPart::ToolUse { id: id.clone(), name: name.clone(), arguments: arguments.clone() }
        }
        agent_core::types::ContentBlock::ToolResult { tool_call_id, content, is_error, .. } => {
            let text = content.iter()
                .filter_map(|b| if let agent_core::types::ContentBlock::Text { text } = b { Some(text.as_str()) } else { None })
                .collect::<Vec<_>>()
                .join("");
            llm_client::types::ContentPart::ToolResult {
                tool_use_id: tool_call_id.clone(),
                content: text,
                is_error: *is_error,
            }
        }
        agent_core::types::ContentBlock::Thinking { thinking } => {
            llm_client::types::ContentPart::Thinking { thinking: thinking.clone() }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensure_content_index() {
        let mut partial = serde_json::json!({"content": []});
        ensure_content_index(&mut partial, 2);
        assert_eq!(partial["content"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_convert_block_text() {
        let block = agent_core::types::ContentBlock::Text { text: "hello".into() };
        let part = convert_block_to_part(&block);
        match part {
            llm_client::types::ContentPart::Text { text } => assert_eq!(text, "hello"),
            _ => panic!("Expected text"),
        }
    }
}

