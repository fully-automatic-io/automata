// Stream bridge — converts llm-client SSE streams into agent-core's
// `AssistantMessageEvent` stream. Lives in coding-agent because it depends on
// both crates.
//
// Maintains a typed `PartialAssistantMessage` reducer over `LlmEvent` deltas;
// the event stream the loop consumes carries the same typed snapshot.

use agent_core::agent_loop::{StreamFn, StreamFnInput};
use agent_core::event::{
    AssistantMessageEvent, EventStream, PartialAssistantMessage, PartialContentBlock,
};
use agent_core::types::{
    AgentMessage, AnthropicOptions, Api, ContentBlock, LlmRequest, Model, ProviderOptions,
    StopReason, ToolDefinition,
};
use llm_client::provider::LlmProvider;
use llm_client::streaming::{Delta, LlmEvent};
use std::sync::Arc;

/// Convert an llm-client SSE stream into agent-core's `AssistantMessageEvent`
/// stream. The reducer maintains a typed `PartialAssistantMessage` and the
/// final reconstructed `AgentMessage::Assistant` is produced via
/// `into_finalized()`.
pub fn convert_sse_stream(
    sse_stream: llm_client::LlmStream,
    api: agent_core::types::Api,
    provider: String,
    model_id: String,
) -> EventStream<AssistantMessageEvent, AgentMessage> {
    let stream = EventStream::<AssistantMessageEvent, AgentMessage>::new();
    let stream_clone = stream.clone();

    tokio::spawn(async move {
        use futures::StreamExt;
        let mut sse = sse_stream;

        let mut partial = PartialAssistantMessage::new(api, provider, model_id);

        stream_clone.push(AssistantMessageEvent::Start { partial: partial.clone() });

        while let Some(event) = sse.next().await {
            match event {
                Ok(LlmEvent::ContentBlockStart { index, content_block }) => {
                    partial.ensure_block_at(index);
                    match content_block {
                        ContentBlock::Text { .. } => {
                            partial.content[index] = PartialContentBlock::Text {
                                text: String::new(),
                                text_signature: None,
                            };
                            stream_clone.push(AssistantMessageEvent::TextStart {
                                content_index: index,
                                partial: partial.clone(),
                            });
                        }
                        ContentBlock::Thinking { .. } => {
                            partial.content[index] = PartialContentBlock::Thinking {
                                thinking: String::new(),
                                thinking_signature: None,
                            };
                            stream_clone.push(AssistantMessageEvent::ThinkingStart {
                                content_index: index,
                                partial: partial.clone(),
                            });
                        }
                        ContentBlock::ToolCall { id, name, .. } => {
                            partial.content[index] = PartialContentBlock::ToolCall {
                                id,
                                name,
                                arguments: serde_json::json!({}),
                                partial_json: None,
                            };
                            stream_clone.push(AssistantMessageEvent::ToolCallStart {
                                content_index: index,
                                partial: partial.clone(),
                            });
                        }
                        _ => {}
                    }
                }
                Ok(LlmEvent::ContentBlockDelta { index, delta }) => match delta {
                    Delta::TextDelta { text } => {
                        if let Some(PartialContentBlock::Text { text: t, .. }) =
                            partial.content.get_mut(index)
                        {
                            t.push_str(&text);
                        }
                        stream_clone.push(AssistantMessageEvent::TextDelta {
                            content_index: index,
                            delta: text,
                            partial: partial.clone(),
                        });
                    }
                    Delta::ThinkingDelta { thinking } => {
                        if let Some(PartialContentBlock::Thinking { thinking: t, .. }) =
                            partial.content.get_mut(index)
                        {
                            t.push_str(&thinking);
                        }
                        stream_clone.push(AssistantMessageEvent::ThinkingDelta {
                            content_index: index,
                            delta: thinking,
                            partial: partial.clone(),
                        });
                    }
                    Delta::InputJsonDelta { partial_json } => {
                        if let Some(PartialContentBlock::ToolCall {
                            arguments,
                            partial_json: pj,
                            ..
                        }) = partial.content.get_mut(index)
                        {
                            let buf = pj.get_or_insert_with(String::new);
                            buf.push_str(&partial_json);
                            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(buf) {
                                *arguments = parsed;
                            }
                        }
                        stream_clone.push(AssistantMessageEvent::ToolCallDelta {
                            content_index: index,
                            delta: partial_json,
                            partial: partial.clone(),
                        });
                    }
                    _ => {}
                },
                Ok(LlmEvent::ContentBlockStop { index }) => {
                    let Some(block) = partial.content.get(index).cloned() else {
                        continue;
                    };
                    match block {
                        PartialContentBlock::Text { text, .. } => {
                            stream_clone.push(AssistantMessageEvent::TextEnd {
                                content_index: index,
                                content: text,
                                partial: partial.clone(),
                            });
                        }
                        PartialContentBlock::Thinking { thinking, .. } => {
                            stream_clone.push(AssistantMessageEvent::ThinkingEnd {
                                content_index: index,
                                content: thinking,
                                partial: partial.clone(),
                            });
                        }
                        PartialContentBlock::ToolCall { .. } => {
                            // Drop in-flight `partialJson` once block is complete.
                            if let Some(PartialContentBlock::ToolCall { partial_json, .. }) =
                                partial.content.get_mut(index)
                            {
                                *partial_json = None;
                            }
                            let tool_call = partial.content[index].clone().into_block();
                            stream_clone.push(AssistantMessageEvent::ToolCallEnd {
                                content_index: index,
                                tool_call,
                                partial: partial.clone(),
                            });
                        }
                        PartialContentBlock::Image { .. } => {}
                    }
                }
                Ok(LlmEvent::MessageDelta { delta, usage }) => {
                    if let Some(reason) = delta.stop_reason {
                        partial.stop_reason = reason;
                    }
                    if let Some(u) = usage {
                        partial.usage = u;
                    }
                }
                Ok(LlmEvent::MessageStop) => {
                    let reason = partial.stop_reason;
                    let final_msg = partial.into_finalized();
                    stream_clone
                        .push(AssistantMessageEvent::Done { reason, message: final_msg.clone() });
                    stream_clone.end(final_msg);
                    return;
                }
                Ok(LlmEvent::Error { error }) => {
                    partial.stop_reason = StopReason::Error;
                    partial.error_message = Some(error.message);
                    let final_msg = partial.clone().into_finalized();
                    stream_clone.push(AssistantMessageEvent::Error {
                        reason: StopReason::Error,
                        error: partial,
                    });
                    stream_clone.end(final_msg);
                    return;
                }
                Err(e) => {
                    partial.stop_reason = StopReason::Error;
                    partial.error_message = Some(e.to_string());
                    let final_msg = partial.clone().into_finalized();
                    stream_clone.push(AssistantMessageEvent::Error {
                        reason: StopReason::Error,
                        error: partial,
                    });
                    stream_clone.end(final_msg);
                    return;
                }
                _ => {}
            }
        }

        let final_msg = partial.into_finalized();
        stream_clone.end(final_msg);
    });

    stream
}

/// Derive `ProviderOptions` from a model's catalog `compat` flags so the
/// provider sees data-driven overrides (adaptive thinking, temperature
/// support) instead of relying on substring matching. Returns `None` when the
/// model carries no relevant flags or its API family has no options type yet.
fn compat_to_provider_options(model: &Model) -> Option<ProviderOptions> {
    match model.api {
        Api::Anthropic => {
            let compat = &model.compat;
            if compat.force_adaptive_thinking.is_none() && compat.supports_temperature.is_none() {
                return None;
            }
            Some(ProviderOptions::Anthropic(AnthropicOptions {
                force_adaptive_thinking: compat.force_adaptive_thinking,
                supports_temperature: compat.supports_temperature,
            }))
        }
        _ => None,
    }
}

/// Create a `StreamFn` compatible with agent-core from an `LlmProvider`.
/// Tools, messages, and system prompt are forwarded directly — the loop and
/// the provider both speak `AgentMessage` now, so no conversion is needed.
pub fn create_stream_fn(provider: Arc<dyn LlmProvider>, model: Model) -> StreamFn {
    Arc::new(move |input: StreamFnInput| {
        let provider = provider.clone();
        let model = model.clone();
        Box::pin(async move {
            let tools: Vec<ToolDefinition> = input
                .tools
                .iter()
                .map(|t| ToolDefinition {
                    name: t.name().to_string(),
                    description: t.description().to_string(),
                    input_schema: t.parameters(),
                })
                .collect();

            let request = LlmRequest {
                model: model.id.clone(),
                messages: input.messages,
                tools,
                system: Some(input.system_prompt),
                max_tokens: input.max_tokens,
                temperature: input.temperature,
                reasoning_effort: input.reasoning,
                thinking_budgets: input.thinking_budgets,
                // Caller-supplied options win; otherwise derive them from the
                // model's catalog `compat` flags so provider behaviour
                // (adaptive thinking, temperature suppression) is data-driven.
                provider_options: input
                    .provider_options
                    .or_else(|| compat_to_provider_options(&model)),
                ..Default::default()
            };

            let sse_stream = provider.stream(request).await.map_err(|e| e.to_string())?;

            Ok(convert_sse_stream(
                sse_stream,
                model.api,
                model.provider.clone(),
                model.id.clone(),
            ))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_ensure_block_at() {
        let mut p = PartialAssistantMessage::new(agent_core::types::Api::Anthropic, "p", "m");
        p.ensure_block_at(2);
        assert_eq!(p.content.len(), 3);
    }

    #[test]
    fn test_compat_to_provider_options_anthropic() {
        let model = Model {
            api: Api::Anthropic,
            compat: agent_core::types::ModelCompat {
                force_adaptive_thinking: Some(true),
                supports_temperature: Some(false),
            },
            ..Default::default()
        };
        let opts = compat_to_provider_options(&model).expect("anthropic compat maps to options");
        match opts {
            ProviderOptions::Anthropic(a) => {
                assert_eq!(a.force_adaptive_thinking, Some(true));
                assert_eq!(a.supports_temperature, Some(false));
            }
            _ => panic!("expected anthropic options"),
        }
    }

    #[test]
    fn test_compat_to_provider_options_none_when_empty() {
        // No compat flags → no derived options (substring fallback applies).
        let model = Model {
            api: Api::Anthropic,
            ..Default::default()
        };
        assert!(compat_to_provider_options(&model).is_none());
        // Non-anthropic API has no options type yet.
        let model = Model {
            api: Api::Openai,
            compat: agent_core::types::ModelCompat {
                supports_temperature: Some(false),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(compat_to_provider_options(&model).is_none());
    }
}
