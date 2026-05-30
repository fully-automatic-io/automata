#[allow(clippy::module_inception)]
mod extensions;

pub use extensions::{
    discover_and_load_extensions, load_extensions,
    CompactionPreparation, CompactionResult, Extension, ExtensionContext, ExtensionEvent,
    ExtensionFlag, ExtensionManifest, ExtensionRunner, ExtensionService, LoadExtensionsResult,
    RegisteredCommand, RegisteredTool, SessionLifecycleReason, ToolDefinition, ProviderConfig,
    ProviderModelConfig,
};
