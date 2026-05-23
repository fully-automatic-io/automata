// Filesystem and shell trait abstractions used by the harness, tools, and
// extensions. `NativeEnv` is the concrete implementation; tests and remote
// runtimes can supply their own.

use async_trait::async_trait;
use std::collections::HashMap;
use tokio_util::sync::CancellationToken;

use super::native::{EnvError, ExecResult, FileInfo};

// ============================================================================
// FileSystem
// ============================================================================

#[async_trait]
pub trait FileSystem: Send + Sync {
    async fn read_text_file(&self, path: &str) -> Result<String, EnvError>;
    async fn read_binary_file(&self, path: &str) -> Result<Vec<u8>, EnvError>;
    async fn write_file(&self, path: &str, content: &[u8]) -> Result<(), EnvError>;
    async fn append_file(&self, path: &str, content: &[u8]) -> Result<(), EnvError>;
    async fn file_info(&self, path: &str) -> Result<FileInfo, EnvError>;
    async fn list_dir(&self, path: &str) -> Result<Vec<FileInfo>, EnvError>;
    async fn canonical_path(&self, path: &str) -> Result<String, EnvError>;
    async fn exists(&self, path: &str) -> Result<bool, EnvError>;
    async fn create_dir(&self, path: &str, recursive: bool) -> Result<(), EnvError>;
    async fn remove(&self, path: &str, recursive: bool) -> Result<(), EnvError>;
    async fn create_temp_dir(&self, prefix: &str) -> Result<String, EnvError>;
    async fn create_temp_file(&self, prefix: &str, suffix: &str) -> Result<String, EnvError>;
}

// ============================================================================
// Shell
// ============================================================================

#[async_trait]
pub trait Shell: Send + Sync {
    async fn exec(
        &self,
        command: &str,
        cwd: Option<&str>,
        env: Option<&HashMap<String, String>>,
        timeout_secs: Option<u64>,
        cancel: Option<&CancellationToken>,
    ) -> Result<ExecResult, EnvError>;
}

// ============================================================================
// ExecutionEnv — bundles file system + shell behind a single trait
// ============================================================================

pub trait ExecutionEnv: FileSystem + Shell {}
impl<T: FileSystem + Shell + ?Sized> ExecutionEnv for T {}

// ============================================================================
// NativeEnv impls
// ============================================================================

#[async_trait]
impl FileSystem for super::native::NativeEnv {
    async fn read_text_file(&self, path: &str) -> Result<String, EnvError> {
        Self::read_text_file(self, path).await
    }
    async fn read_binary_file(&self, path: &str) -> Result<Vec<u8>, EnvError> {
        Self::read_binary_file(self, path).await
    }
    async fn write_file(&self, path: &str, content: &[u8]) -> Result<(), EnvError> {
        Self::write_file(self, path, content).await
    }
    async fn append_file(&self, path: &str, content: &[u8]) -> Result<(), EnvError> {
        Self::append_file(self, path, content).await
    }
    async fn file_info(&self, path: &str) -> Result<FileInfo, EnvError> {
        Self::file_info(self, path).await
    }
    async fn list_dir(&self, path: &str) -> Result<Vec<FileInfo>, EnvError> {
        Self::list_dir(self, path).await
    }
    async fn canonical_path(&self, path: &str) -> Result<String, EnvError> {
        Self::canonical_path(self, path).await
    }
    async fn exists(&self, path: &str) -> Result<bool, EnvError> {
        Self::exists(self, path).await
    }
    async fn create_dir(&self, path: &str, recursive: bool) -> Result<(), EnvError> {
        Self::create_dir(self, path, recursive).await
    }
    async fn remove(&self, path: &str, recursive: bool) -> Result<(), EnvError> {
        Self::remove(self, path, recursive).await
    }
    async fn create_temp_dir(&self, prefix: &str) -> Result<String, EnvError> {
        Self::create_temp_dir(self, prefix).await
    }
    async fn create_temp_file(&self, prefix: &str, suffix: &str) -> Result<String, EnvError> {
        Self::create_temp_file(self, prefix, suffix).await
    }
}

#[async_trait]
impl Shell for super::native::NativeEnv {
    async fn exec(
        &self,
        command: &str,
        cwd: Option<&str>,
        env: Option<&HashMap<String, String>>,
        timeout_secs: Option<u64>,
        cancel: Option<&CancellationToken>,
    ) -> Result<ExecResult, EnvError> {
        Self::exec(self, command, cwd, env, timeout_secs, cancel).await
    }
}
