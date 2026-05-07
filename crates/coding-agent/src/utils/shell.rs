// Shell utilities - Shell detection, environment, and process management

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref DETACHED_PIDS: Mutex<Vec<u32>> = Mutex::new(Vec::new());
}

/// Shell configuration
#[derive(Debug, Clone)]
pub struct ShellConfig {
    pub shell: String,
    pub args: Vec<String>,
}

/// Detect shell configuration
pub fn get_shell_config(shell_path: Option<&str>) -> ShellConfig {
    if let Some(path) = shell_path {
        return ShellConfig {
            shell: path.to_string(),
            args: vec!["-c".to_string()],
        };
    }

    // Try to detect shell from environment
    if let Ok(shell) = env::var("SHELL") {
        let shell_path = PathBuf::from(&shell);
        let shell_name = shell_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("bash")
            .to_string();

        match shell_name.as_str() {
            "zsh" => ShellConfig {
                shell,
                args: vec!["-c".to_string()],
            },
            "fish" => ShellConfig {
                shell,
                args: vec!["-c".to_string()],
            },
            "bash" | _ => ShellConfig {
                shell,
                args: vec!["-c".to_string()],
            },
        }
    } else {
        // Default to bash
        ShellConfig {
            shell: "bash".to_string(),
            args: vec!["-c".to_string()],
        }
    }
}

/// Get shell environment variables
pub fn get_shell_env() -> HashMap<String, String> {
    env::vars().collect()
}

/// Track a detached child process PID
pub fn track_detached_child_pid(pid: u32) {
    let mut pids = DETACHED_PIDS.lock().unwrap();
    pids.push(pid);
}

/// Untrack a detached child process PID
pub fn untrack_detached_child_pid(pid: u32) {
    let mut pids = DETACHED_PIDS.lock().unwrap();
    pids.retain(|&p| p != pid);
}

/// Kill a process tree (process and all its children)
#[cfg(unix)]
pub fn kill_process_tree(pid: u32) -> Result<(), std::io::Error> {
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;

    // Get all child PIDs
    let children = get_child_pids(pid);

    // Kill children first
    for child_pid in children {
        let _ = kill(Pid::from_raw(child_pid as i32), Signal::SIGTERM);
    }

    // Kill the main process
    let _ = kill(Pid::from_raw(pid as i32), Signal::SIGTERM);

    // Give processes time to terminate gracefully
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Force kill if still alive
    for child_pid in get_child_pids(pid) {
        let _ = kill(Pid::from_raw(child_pid as i32), Signal::SIGKILL);
    }
    let _ = kill(Pid::from_raw(pid as i32), Signal::SIGKILL);

    Ok(())
}

#[cfg(not(unix))]
pub fn kill_process_tree(pid: u32) -> Result<(), std::io::Error> {
    // On Windows, use taskkill
    std::process::Command::new("taskkill")
        .args(&["/F", "/T", "/PID", &pid.to_string()])
        .output()?;
    Ok(())
}

/// Get child PIDs of a process
#[cfg(unix)]
fn get_child_pids(parent_pid: u32) -> Vec<u32> {
    let output = std::process::Command::new("pgrep")
        .args(&["-P", &parent_pid.to_string()])
        .output();

    if let Ok(output) = output {
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| line.trim().parse::<u32>().ok())
            .collect()
    } else {
        Vec::new()
    }
}

#[cfg(not(unix))]
fn get_child_pids(_parent_pid: u32) -> Vec<u32> {
    Vec::new()
}

/// Cleanup all tracked detached processes
pub fn cleanup_detached_processes() {
    let pids = DETACHED_PIDS.lock().unwrap().clone();
    for pid in pids {
        let _ = kill_process_tree(pid);
    }
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
        {
            let pids = DETACHED_PIDS.lock().unwrap();
            assert!(pids.contains(&12345));
        }
        untrack_detached_child_pid(12345);
        {
            let pids = DETACHED_PIDS.lock().unwrap();
            assert!(!pids.contains(&12345));
        }
    }
}
