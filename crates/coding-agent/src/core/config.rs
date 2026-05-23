use std::collections::HashMap;
use std::sync::Mutex;

static CACHE: Mutex<Option<HashMap<String, String>>> = Mutex::new(None);

fn cache() -> std::sync::MutexGuard<'static, Option<HashMap<String, String>>> {
    CACHE.lock().unwrap()
}

/// Resolve a config value from env var, shell command (`!cmd`), or literal.
/// Shell command results are cached for the process lifetime.
pub fn resolve_config_value(value: &str) -> Result<String, String> {
    if let Some(cmd) = value.strip_prefix('!') {
        let mut guard = cache();
        let map = guard.get_or_insert_with(HashMap::new);
        if let Some(cached) = map.get(cmd) {
            return Ok(cached.clone());
        }
        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .output()
            .map_err(|e| format!("Failed to run command `{}`: {}", cmd, e))?;
        if !output.status.success() {
            return Err(format!("Command `{}` exited with status {}", cmd, output.status));
        }
        let result = String::from_utf8_lossy(&output.stdout).trim().to_string();
        map.insert(cmd.to_string(), result.clone());
        return Ok(result);
    }

    if let Some(var) = value.strip_prefix('$') {
        return std::env::var(var).map_err(|_| format!("Env var ${} not set", var));
    }

    Ok(value.to_string())
}

pub fn resolve_config_value_opt(value: Option<&str>) -> Result<Option<String>, String> {
    match value {
        None | Some("") => Ok(None),
        Some(v) => resolve_config_value(v).map(Some),
    }
}
