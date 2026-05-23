use std::collections::HashMap;
use std::env;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::tools::truncate_tail;

// ── Simple exec ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ExecOptions {
    pub cwd: Option<String>,
    pub env: Option<HashMap<String, String>>,
    pub timeout_secs: Option<u64>,
}

#[derive(Debug)]
pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
}

pub async fn exec_command(command: &str, opts: ExecOptions) -> Result<ExecResult, String> {
    let mut cmd = Command::new("sh");
    cmd.arg("-c").arg(command)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if let Some(cwd) = &opts.cwd { cmd.current_dir(cwd); }
    if let Some(env) = &opts.env { cmd.envs(env); }

    let run = async {
        let output = cmd.output().await.map_err(|e| e.to_string())?;
        Ok::<ExecResult, String>(ExecResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            exit_code: output.status.code(),
        })
    };

    if let Some(secs) = opts.timeout_secs {
        tokio::time::timeout(std::time::Duration::from_secs(secs), run)
            .await
            .map_err(|_| format!("Command timed out after {}s", secs))?
    } else {
        run.await
    }
}

// ── Bash executor ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BashResult {
    pub output: String,
    pub exit_code: Option<i32>,
    pub cancelled: bool,
    pub truncated: bool,
    pub full_output_path: Option<String>,
}

pub struct BashExecutorOptions {
    pub cwd: String,
    pub shell_path: Option<String>,
    pub env: Option<HashMap<String, String>>,
    pub timeout_secs: Option<u64>,
    pub signal: Option<CancellationToken>,
}

pub async fn execute_bash(command: &str, opts: BashExecutorOptions) -> Result<BashResult, String> {
    let shell = opts.shell_path.as_deref().unwrap_or("/bin/bash");
    let mut cmd = Command::new(shell);
    cmd.arg("-c").arg(command)
        .current_dir(&opts.cwd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if let Some(env) = &opts.env { cmd.envs(env); }

    let mut child = cmd.spawn().map_err(|e| e.to_string())?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    let chunks: Arc<Mutex<Vec<Vec<u8>>>> = Arc::new(Mutex::new(vec![]));

    let chunks_out = chunks.clone();
    let stdout_task = tokio::spawn(async move {
        use tokio::io::AsyncBufReadExt;
        let mut reader = tokio::io::BufReader::new(stdout);
        let mut line = vec![];
        while reader.read_until(b'\n', &mut line).await.unwrap_or(0) > 0 {
            chunks_out.lock().unwrap().push(std::mem::take(&mut line));
        }
    });

    let chunks_err = chunks.clone();
    let stderr_task = tokio::spawn(async move {
        use tokio::io::AsyncBufReadExt;
        let mut reader = tokio::io::BufReader::new(stderr);
        let mut line = vec![];
        while reader.read_until(b'\n', &mut line).await.unwrap_or(0) > 0 {
            chunks_err.lock().unwrap().push(std::mem::take(&mut line));
        }
    });

    let wait_fut = child.wait();
    let status = if let Some(secs) = opts.timeout_secs {
        match tokio::time::timeout(std::time::Duration::from_secs(secs), wait_fut).await {
            Ok(r) => r.map_err(|e| e.to_string())?,
            Err(_) => {
                let _ = child.kill().await;
                let _ = child.wait().await;
                let _ = stdout_task.await;
                let _ = stderr_task.await;
                let raw: Vec<u8> = chunks.lock().unwrap().concat();
                let full = String::from_utf8_lossy(&raw).to_string();
                let trunc = truncate_tail(&full);
                return Ok(BashResult {
                    output: trunc.content,
                    exit_code: None,
                    cancelled: false,
                    truncated: trunc.truncated,
                    full_output_path: None,
                });
            }
        }
    } else {
        wait_fut.await.map_err(|e| e.to_string())?
    };

    let _ = stdout_task.await;
    let _ = stderr_task.await;

    let raw: Vec<u8> = chunks.lock().unwrap().concat();
    let full = String::from_utf8_lossy(&raw).to_string();
    let trunc = truncate_tail(&full);

    Ok(BashResult {
        output: trunc.content,
        exit_code: status.code(),
        cancelled: false,
        truncated: trunc.truncated,
        full_output_path: None,
    })
}

// ── Shell utilities ───────────────────────────────────────────────────────────

static DETACHED_PIDS: OnceLock<Mutex<Vec<u32>>> = OnceLock::new();

fn detached_pids() -> &'static Mutex<Vec<u32>> {
    DETACHED_PIDS.get_or_init(|| Mutex::new(Vec::new()))
}

#[derive(Debug, Clone)]
pub struct ShellConfig {
    pub shell: String,
    pub args: Vec<String>,
}

pub fn get_shell_config(shell_path: Option<&str>) -> ShellConfig {
    if let Some(path) = shell_path {
        return ShellConfig { shell: path.to_string(), args: vec!["-c".to_string()] };
    }
    let shell = env::var("SHELL").unwrap_or_else(|_| "bash".to_string());
    ShellConfig { shell, args: vec!["-c".to_string()] }
}

pub fn get_shell_env() -> HashMap<String, String> {
    env::vars().collect()
}

pub fn track_detached_child_pid(pid: u32) {
    detached_pids().lock().unwrap().push(pid);
}

pub fn untrack_detached_child_pid(pid: u32) {
    detached_pids().lock().unwrap().retain(|&p| p != pid);
}

#[cfg(unix)]
pub fn kill_process_tree(pid: u32) -> Result<(), std::io::Error> {
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;
    let children = get_child_pids(pid);
    for child_pid in &children {
        let _ = kill(Pid::from_raw(*child_pid as i32), Signal::SIGTERM);
    }
    let _ = kill(Pid::from_raw(pid as i32), Signal::SIGTERM);
    std::thread::sleep(std::time::Duration::from_millis(100));
    for child_pid in get_child_pids(pid) {
        let _ = kill(Pid::from_raw(child_pid as i32), Signal::SIGKILL);
    }
    let _ = kill(Pid::from_raw(pid as i32), Signal::SIGKILL);
    Ok(())
}

#[cfg(not(unix))]
pub fn kill_process_tree(pid: u32) -> Result<(), std::io::Error> {
    std::process::Command::new("taskkill")
        .args(&["/F", "/T", "/PID", &pid.to_string()])
        .output()?;
    Ok(())
}

#[cfg(unix)]
fn get_child_pids(parent_pid: u32) -> Vec<u32> {
    std::process::Command::new("pgrep")
        .args(&["-P", &parent_pid.to_string()])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout)
            .lines()
            .filter_map(|l| l.trim().parse().ok())
            .collect())
        .unwrap_or_default()
}

#[cfg(not(unix))]
fn get_child_pids(_parent_pid: u32) -> Vec<u32> { Vec::new() }

pub fn cleanup_detached_processes() {
    let pids = detached_pids().lock().unwrap().clone();
    for pid in pids { let _ = kill_process_tree(pid); }
}

// ── shell_config path helper (used by BashTool) ───────────────────────────────

pub fn resolve_shell_path() -> Option<String> {
    env::var("SHELL").ok().filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_shell_config() {
        let config = get_shell_config(None);
        assert!(!config.shell.is_empty());
        assert!(!config.args.is_empty());
    }

    #[test]
    fn test_get_shell_env() {
        let env = get_shell_env();
        assert!(!env.is_empty());
    }

    #[test]
    fn test_track_untrack_pid() {
        track_detached_child_pid(12345);
        assert!(detached_pids().lock().unwrap().contains(&12345));
        untrack_detached_child_pid(12345);
        assert!(!detached_pids().lock().unwrap().contains(&12345));
    }
}
