#[allow(clippy::module_inception)]
mod extensions;

pub use extensions::{
    CompactionPreparation, CompactionResult, Extension, ExtensionContext, ExtensionEvent,
    ExtensionFlag, ExtensionHostSnapshot, ExtensionManifest, ExtensionRunner, ExtensionService,
    LoadExtensionsResult, ProviderConfig, ProviderModelConfig, RegisteredCommand, RegisteredTool,
    SessionLifecycleReason, ToolDefinition, collect_registered_providers,
    discover_and_load_extensions, dispatch_loaded_extensions, extension_after_tool_call_hook,
    extension_agent_tools, extension_before_tool_call_hook, extension_on_payload_hook,
    extension_on_response_hook, extension_transform_context_hook, load_extensions,
    subscribe_extension_harness_events,
};
