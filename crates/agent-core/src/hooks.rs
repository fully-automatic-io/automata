
use crate::types::{
    AfterToolCallContext, AfterToolCallResult, AgentMessage, BeforeToolCallContext,
    BeforeToolCallResult, Message,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

// ============================================================================
// Default convertToLlm — filters to user/assistant/toolResult roles
// ============================================================================

/// Default message-to-LLM converter.
/// Filters out messages that don't have user, assistant, or toolResult roles.
pub fn default_convert_to_llm(
    messages: Vec<AgentMessage>,
) -> Pin<Box<dyn Future<Output = Vec<Message>> + Send>> {
    Box::pin(async move {
        messages
            .into_iter()
            .filter(|m| {
                let role = m
                    .get("role")
                    .and_then(|r| r.as_str())
                    .unwrap_or("");
                matches!(role, "user" | "assistant" | "toolResult")
            })
            .filter_map(|m| serde_json::from_value(m).ok())
            .collect()
    })
}

// ============================================================================
// Hook trait definitions
// ============================================================================

/// Hook called before a tool is executed.
/// Return Some(BeforeToolCallResult{ block: true }) to prevent execution.
pub trait BeforeToolCallHook: Send + Sync {
    fn call(
        &self,
        context: BeforeToolCallContext,
        signal: Option<CancellationToken>,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Option<BeforeToolCallResult>, Box<dyn std::error::Error + Send + Sync>>>
                + Send
                + '_,
        >,
    >;
}

/// Hook called after a tool finishes executing.
/// Return Some(AfterToolCallResult) to override parts of the result.
pub trait AfterToolCallHook: Send + Sync {
    fn call(
        &self,
        context: AfterToolCallContext,
        signal: Option<CancellationToken>,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Option<AfterToolCallResult>, Box<dyn std::error::Error + Send + Sync>>>
                + Send
                + '_,
        >,
    >;
}

/// Hook for transforming context (AgentMessage[]) before LLM conversion.
pub trait TransformContextHook: Send + Sync {
    fn call(
        &self,
        messages: Vec<AgentMessage>,
        signal: Option<CancellationToken>,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Vec<AgentMessage>, Box<dyn std::error::Error + Send + Sync>>>
                + Send
                + '_,
        >,
    >;
}

/// Hook for converting AgentMessage[] to LLM Message[].
pub trait ConvertToLlmHook: Send + Sync {
    fn call(
        &self,
        messages: Vec<AgentMessage>,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Vec<Message>, Box<dyn std::error::Error + Send + Sync>>>
                + Send
                + '_,
        >,
    >;
}

// ============================================================================
// Default ConvertToLlmHook implementation
// ============================================================================

/// Default convertToLlm that filters to user/assistant/toolResult roles.
pub struct DefaultConvertToLlmHook;

impl ConvertToLlmHook for DefaultConvertToLlmHook {
    fn call(
        &self,
        messages: Vec<AgentMessage>,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Vec<Message>, Box<dyn std::error::Error + Send + Sync>>>
                + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            Ok(messages
                .into_iter()
                .filter(|m| {
                    let role = m
                        .get("role")
                        .and_then(|r| r.as_str())
                        .unwrap_or("");
                    matches!(role, "user" | "assistant" | "toolResult")
                })
                .filter_map(|m| serde_json::from_value(m).ok())
                .collect())
        })
    }
}

// ============================================================================
// Hook Registry
// ============================================================================

/// Registry holding all hook implementations.
pub struct HookRegistry {
    pub before_tool_call: Option<Arc<dyn BeforeToolCallHook>>,
    pub after_tool_call: Option<Arc<dyn AfterToolCallHook>>,
    pub transform_context: Option<Arc<dyn TransformContextHook>>,
    pub convert_to_llm: Option<Arc<dyn ConvertToLlmHook>>,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self {
            before_tool_call: None,
            after_tool_call: None,
            transform_context: None,
            convert_to_llm: Some(Arc::new(DefaultConvertToLlmHook)),
        }
    }
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Fn-wrapper implementations for ease of use
// ============================================================================

/// Create a BeforeToolCallHook from an async function.
pub fn before_tool_call_fn<F, Fut>(f: F) -> Arc<dyn BeforeToolCallHook>
where
    F: Fn(BeforeToolCallContext, Option<CancellationToken>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Option<BeforeToolCallResult>, Box<dyn std::error::Error + Send + Sync>>>
        + Send
        + 'static,
{
    struct FnHook<F>(F);
    impl<F, Fut> BeforeToolCallHook for FnHook<F>
    where
        F: Fn(BeforeToolCallContext, Option<CancellationToken>) -> Fut + Send + Sync + 'static,
        Fut: Future<
                Output = Result<Option<BeforeToolCallResult>, Box<dyn std::error::Error + Send + Sync>>,
            > + Send
            + 'static,
    {
        fn call(
            &self,
            context: BeforeToolCallContext,
            signal: Option<CancellationToken>,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            Option<BeforeToolCallResult>,
                            Box<dyn std::error::Error + Send + Sync>,
                        >,
                    > + Send
                    + '_,
            >,
        > {
            Box::pin((self.0)(context, signal))
        }
    }
    Arc::new(FnHook(f))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_default_convert_to_llm() {
        let messages = vec![
            serde_json::json!({"role": "user", "content": "hello", "timestamp": 1000}),
            serde_json::json!({"role": "assistant", "content": [{"type": "text", "text": "hi"}], "api": "test", "provider": "test", "model": "test", "usage": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0}}, "stopReason": "stop", "timestamp": 2000}),
            serde_json::json!({"customType": "artifact", "role": "custom"}),
        ];

        let result = default_convert_to_llm(messages).await;
        assert_eq!(result.len(), 2);

        let hook = DefaultConvertToLlmHook;
        let messages2 = vec![
            serde_json::json!({"role": "user", "content": "hi", "timestamp": 1000}),
            serde_json::json!({"role": "notification", "content": "ignored", "timestamp": 2000}),
        ];
        let result2 = hook.call(messages2).await.unwrap();
        assert_eq!(result2.len(), 1);
    }

    #[test]
    fn test_hook_registry_default() {
        let reg = HookRegistry::new();
        assert!(reg.before_tool_call.is_none());
        assert!(reg.after_tool_call.is_none());
        assert!(reg.convert_to_llm.is_some());
    }
}
