use std::path::{Path, PathBuf};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone)]
pub struct FileInfo {
    pub name: String,
    pub path: String,
    pub kind: FileKind,
    pub size: u64,
    pub mtime_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileKind {
    File,
    Directory,
    Symlink,
}

#[derive(Debug, thiserror::Error)]
pub enum EnvError {
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    #[error("Is directory: {0}")]
    IsDirectory(String),
    #[error("Not a directory: {0}")]
    NotDirectory(String),
    #[error("Aborted")]
    Aborted,
    #[error("Shell unavailable: {0}")]
    ShellUnavailable(String),
    #[error("Spawn error: {0}")]
    SpawnError(String),
    #[error("Timeout: {0}s")]
    Timeout(u64),
    #[error("IO error: {0}")]
    Io(String),
}

impl EnvError {
    fn from_io(e: std::io::Error, path: &str) -> Self {
        match e.kind() {
            std::io::ErrorKind::NotFound => Self::NotFound(path.to_string()),
            std::io::ErrorKind::PermissionDenied => Self::PermissionDenied(path.to_string()),
            _ => Self::Io(e.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

pub struct NativeEnv {
    pub cwd: String,
}

impl NativeEnv {
    pub fn new(cwd: impl Into<String>) -> Self {
        Self { cwd: cwd.into() }
    }

    fn resolve(&self, path: &str) -> PathBuf {
        let p = Path::new(path);
        if p.is_absolute() { p.to_path_buf() } else { Path::new(&self.cwd).join(p) }
    }

    pub async fn read_text_file(&self, path: &str) -> Result<String, EnvError> {
        let resolved = self.resolve(path);
        tokio::fs::read_to_string(&resolved).await
            .map_err(|e| EnvError::from_io(e, &resolved.to_string_lossy()))
    }

    pub async fn read_binary_file(&self, path: &str) -> Result<Vec<u8>, EnvError> {
        let resolved = self.resolve(path);
        tokio::fs::read(&resolved).await
            .map_err(|e| EnvError::from_io(e, &resolved.to_string_lossy()))
    }

    pub async fn write_file(&self, path: &str, content: &[u8]) -> Result<(), EnvError> {
        let resolved = self.resolve(path);
        if let Some(parent) = resolved.parent() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| EnvError::from_io(e, &parent.to_string_lossy()))?;
        }
        tokio::fs::write(&resolved, content).await
            .map_err(|e| EnvError::from_io(e, &resolved.to_string_lossy()))
    }

    pub async fn append_file(&self, path: &str, content: &[u8]) -> Result<(), EnvError> {
        let resolved = self.resolve(path);
        if let Some(parent) = resolved.parent() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| EnvError::from_io(e, &parent.to_string_lossy()))?;
        }
        use tokio::io::AsyncWriteExt;
        let mut f = tokio::fs::OpenOptions::new()
            .create(true).append(true).open(&resolved).await
            .map_err(|e| EnvError::from_io(e, &resolved.to_string_lossy()))?;
        f.write_all(content).await
            .map_err(|e| EnvError::from_io(e, &resolved.to_string_lossy()))
    }

    pub async fn file_info(&self, path: &str) -> Result<FileInfo, EnvError> {
        let resolved = self.resolve(path);
        let meta = tokio::fs::symlink_metadata(&resolved).await
            .map_err(|e| EnvError::from_io(e, &resolved.to_string_lossy()))?;
        let kind = if meta.is_file() { FileKind::File }
            else if meta.is_dir() { FileKind::Directory }
            else { FileKind::Symlink };
        let mtime_ms = meta.modified().ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let name = resolved.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        Ok(FileInfo { name, path: resolved.to_string_lossy().to_string(), kind, size: meta.len(), mtime_ms })
    }

    pub async fn list_dir(&self, path: &str) -> Result<Vec<FileInfo>, EnvError> {
        let resolved = self.resolve(path);
        let mut rd = tokio::fs::read_dir(&resolved).await
            .map_err(|e| EnvError::from_io(e, &resolved.to_string_lossy()))?;
        let mut infos = vec![];
        while let Some(entry) = rd.next_entry().await
            .map_err(|e| EnvError::from_io(e, &resolved.to_string_lossy()))? {
            let ep = entry.path();
            if let Ok(info) = self.file_info(&ep.to_string_lossy()).await {
                infos.push(info);
            }
        }
        Ok(infos)
    }

    pub async fn canonical_path(&self, path: &str) -> Result<String, EnvError> {
        let resolved = self.resolve(path);
        tokio::fs::canonicalize(&resolved).await
            .map(|p| p.to_string_lossy().to_string())
            .map_err(|e| EnvError::from_io(e, &resolved.to_string_lossy()))
    }

    pub async fn exists(&self, path: &str) -> Result<bool, EnvError> {
        match self.file_info(path).await {
            Ok(_) => Ok(true),
            Err(EnvError::NotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    pub async fn create_dir(&self, path: &str, recursive: bool) -> Result<(), EnvError> {
        let resolved = self.resolve(path);
        if recursive {
            tokio::fs::create_dir_all(&resolved).await
        } else {
            tokio::fs::create_dir(&resolved).await
        }
        .map_err(|e| EnvError::from_io(e, &resolved.to_string_lossy()))
    }

    pub async fn remove(&self, path: &str, recursive: bool) -> Result<(), EnvError> {
        let resolved = self.resolve(path);
        if recursive {
            tokio::fs::remove_dir_all(&resolved).await
        } else {
            let meta = tokio::fs::symlink_metadata(&resolved).await
                .map_err(|e| EnvError::from_io(e, &resolved.to_string_lossy()))?;
            if meta.is_dir() {
                tokio::fs::remove_dir(&resolved).await
            } else {
                tokio::fs::remove_file(&resolved).await
            }
        }
        .map_err(|e| EnvError::from_io(e, &resolved.to_string_lossy()))
    }

    pub async fn create_temp_dir(&self, prefix: &str) -> Result<String, EnvError> {
        let dir = tempfile::Builder::new()
            .prefix(prefix)
            .tempdir()
            .map_err(|e| EnvError::Io(e.to_string()))?;
        let path = dir.path().to_string_lossy().to_string();
        std::mem::forget(dir); // keep the dir alive (caller owns it)
        Ok(path)
    }

    pub async fn create_temp_file(&self, prefix: &str, suffix: &str) -> Result<String, EnvError> {
        let f = tempfile::Builder::new()
            .prefix(prefix)
            .suffix(suffix)
            .tempfile()
            .map_err(|e| EnvError::Io(e.to_string()))?;
        let path = f.into_temp_path().to_string_lossy().to_string();
        Ok(path)
    }

    pub async fn exec(
        &self,
        command: &str,
        cwd: Option<&str>,
        env: Option<&std::collections::HashMap<String, String>>,
        timeout_secs: Option<u64>,
        cancel: Option<&CancellationToken>,
    ) -> Result<ExecResult, EnvError> {
        let work_dir = cwd.map(|c| self.resolve(c)).unwrap_or_else(|| PathBuf::from(&self.cwd));
        let shell = find_shell().await?;

        let mut cmd = Command::new(&shell);
        cmd.arg("-c").arg(command)
            .current_dir(&work_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .stdin(std::process::Stdio::null());

        #[cfg(unix)]
        unsafe { cmd.pre_exec(|| { libc::setsid(); Ok(()) }); }

        if let Some(env_map) = env {
            for (k, v) in env_map { cmd.env(k, v); }
        }

        let child = cmd.spawn().map_err(|e| EnvError::SpawnError(e.to_string()))?;
        let pid = child.id();

        let timeout_dur = timeout_secs.map(std::time::Duration::from_secs);

        let result = if let Some(dur) = timeout_dur {
            tokio::select! {
                r = child.wait_with_output() => r.map_err(|e| EnvError::Io(e.to_string())),
                _ = tokio::time::sleep(dur) => {
                    kill_pid(pid);
                    return Err(EnvError::Timeout(timeout_secs.unwrap_or(0)));
                }
                _ = async { if let Some(c) = cancel { c.cancelled().await } else { std::future::pending().await } } => {
                    kill_pid(pid);
                    return Err(EnvError::Aborted);
                }
            }
        } else {
            tokio::select! {
                r = child.wait_with_output() => r.map_err(|e| EnvError::Io(e.to_string())),
                _ = async { if let Some(c) = cancel { c.cancelled().await } else { std::future::pending().await } } => {
                    kill_pid(pid);
                    return Err(EnvError::Aborted);
                }
            }
        };

        let output = result?;
        Ok(ExecResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code().unwrap_or(0),
        })
    }
}

fn kill_pid(pid: Option<u32>) {
    let Some(pid) = pid else { return };
    #[cfg(unix)]
    unsafe { libc::kill(-(pid as libc::pid_t), libc::SIGKILL); }
    #[cfg(not(unix))]
    let _ = std::process::Command::new("taskkill").args(["/F", "/T", "/PID", &pid.to_string()]).spawn();
}

async fn find_shell() -> Result<String, EnvError> {
    if cfg!(unix) {
        if tokio::fs::metadata("/bin/bash").await.is_ok() {
            return Ok("/bin/bash".to_string());
        }
        return Ok("sh".to_string());
    }
    // Windows: look for Git bash
    for candidate in &[
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
    ] {
        if tokio::fs::metadata(candidate).await.is_ok() {
            return Ok(candidate.to_string());
        }
    }
    Err(EnvError::ShellUnavailable("No bash shell found".to_string()))
}
