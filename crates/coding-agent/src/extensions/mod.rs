pub mod loader;
pub mod runner;
pub mod service;
pub mod types;

pub use loader::{discover_and_load_extensions, load_extensions};
pub use runner::ExtensionRunner;
pub use service::ExtensionService;
pub use types::{
    Extension, ExtensionContext, ExtensionEvent, ExtensionFlag, ExtensionManifest,
    LoadExtensionsResult, RegisteredCommand, RegisteredTool, ToolDefinition,
    ProviderConfig, ProviderModelConfig, CompactionPreparation, CompactionResult,
};
