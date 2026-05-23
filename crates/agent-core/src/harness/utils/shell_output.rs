use crate::harness::env::native::{EnvError, NativeEnv};
use crate::harness::utils::truncate::{truncate_tail, DEFAULT_MAX_BYTES};
use tokio_util::sync::CancellationToken;

pub struct ShellCaptureResult {
    pub output: String,
    pub exit_code: Option<i32>,
    pub cancelled: bool,
    pub truncated: bool,
    pub full_output_path: Option<String>,
}

pub fn sanitize_binary_output(s: &str) -> String {
    s.chars().filter(|&c| {
        let code = c as u32;
        code == 0x09 || code == 0x0a || code == 0x0d || (code > 0x1f && !(0xfff9..=0xfffb).contains(&code))
    }).collect()
}

pub async fn execute_shell_with_capture(
    env: &NativeEnv,
    command: &str,
    cwd: Option<&str>,
    extra_env: Option<&std::collections::HashMap<String, String>>,
    timeout_secs: Option<u64>,
    cancel: Option<&CancellationToken>,
) -> Result<ShellCaptureResult, EnvError> {
    let result = env.exec(command, cwd, extra_env, timeout_secs, cancel).await;

    match result {
        Err(EnvError::Aborted) => {
            return Ok(ShellCaptureResult {
                output: String::new(),
                exit_code: None,
                cancelled: true,
                truncated: false,
                full_output_path: None,
            });
        }
        Err(EnvError::Timeout(_)) => {
            return Ok(ShellCaptureResult {
                output: String::new(),
                exit_code: None,
                cancelled: true,
                truncated: false,
                full_output_path: None,
            });
        }
        Err(e) => return Err(e),
        Ok(exec_result) => {
            let combined = sanitize_binary_output(&format!("{}{}", exec_result.stdout, exec_result.stderr))
                .replace('\r', "");
            let trunc = truncate_tail(&combined, DEFAULT_MAX_BYTES / 4, DEFAULT_MAX_BYTES);

            let full_output_path = if trunc.truncated {
                match env.create_temp_file("bash-", ".log").await {
                    Ok(path) => {
                        let _ = env.append_file(&path, combined.as_bytes()).await;
                        Some(path)
                    }
                    Err(_) => None,
                }
            } else {
                None
            };

            Ok(ShellCaptureResult {
                output: if trunc.truncated { trunc.content } else { combined },
                exit_code: Some(exec_result.exit_code),
                cancelled: false,
                truncated: trunc.truncated,
                full_output_path,
            })
        }
    }
}
