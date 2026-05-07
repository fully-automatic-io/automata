use super::types::*;
use std::path::Path;
use extism::{Manifest, Plugin, Wasm};

pub fn load_extensions(paths: &[String], cwd: &str) -> LoadExtensionsResult {
    let mut extensions = vec![];
    let mut errors = vec![];

    for ext_path in paths {
        let resolved = resolve_path(ext_path, cwd);
        match load_single_extension(ext_path, &resolved) {
            Ok(ext) => extensions.push(ext),
            Err(e) => errors.push(ExtensionLoadError { path: ext_path.clone(), error: e }),
        }
    }

    LoadExtensionsResult { extensions, errors }
}

fn resolve_path(ext_path: &str, cwd: &str) -> String {
    let expanded = if ext_path.starts_with("~/") {
        let home = dirs_next::home_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        format!("{}/{}", home, &ext_path[2..])
    } else {
        ext_path.to_string()
    };

    if Path::new(&expanded).is_absolute() {
        expanded
    } else {
        Path::new(cwd).join(&expanded).to_string_lossy().to_string()
    }
}

fn load_single_extension(original_path: &str, resolved: &str) -> Result<Extension, String> {
    if !Path::new(resolved).exists() {
        return Err(format!("Extension file not found: {}", resolved));
    }

    let manifest = Manifest::new([Wasm::file(resolved)]);
    let mut plugin = Plugin::new(&manifest, [], true)
        .map_err(|e| format!("Failed to load WASM plugin: {}", e))?;

    let manifest_json = plugin.call::<&str, &str>("register", "")
        .map_err(|e| format!("register() failed: {}", e))?;

    let ext_manifest: ExtensionManifest = serde_json::from_str(manifest_json)
        .map_err(|e| format!("Invalid manifest JSON: {}", e))?;

    Ok(Extension::from_manifest(
        original_path.to_string(),
        resolved.to_string(),
        ext_manifest,
        plugin,
    ))
}

pub fn discover_and_load_extensions(
    configured_paths: &[String],
    cwd: &str,
    agent_dir: Option<&str>,
) -> LoadExtensionsResult {
    let mut all_paths = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let default_agent_dir = dirs_next::home_dir()
        .map(|p| p.join(".pi").join("agent").to_string_lossy().to_string())
        .unwrap_or_default();
    let agent_dir = agent_dir.unwrap_or(&default_agent_dir);

    let local_ext_dir = Path::new(cwd).join(".pi").join("extensions");
    discover_in_dir(&local_ext_dir, &mut all_paths, &mut seen);

    let global_ext_dir = Path::new(agent_dir).join("extensions");
    discover_in_dir(&global_ext_dir, &mut all_paths, &mut seen);

    for p in configured_paths {
        let resolved = resolve_path(p, cwd);
        if !seen.contains(&resolved) {
            seen.insert(resolved.clone());
            all_paths.push(resolved);
        }
    }

    load_extensions(&all_paths, cwd)
}

fn discover_in_dir(dir: &Path, paths: &mut Vec<String>, seen: &mut std::collections::HashSet<String>) {
    if !dir.exists() { return; }

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                if path.extension().and_then(|e| e.to_str()) == Some("wasm") {
                    let s = path.to_string_lossy().to_string();
                    if seen.insert(s.clone()) { paths.push(s); }
                }
            } else if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                let index = path.join("extension.wasm");
                if index.exists() {
                    let s = index.to_string_lossy().to_string();
                    if seen.insert(s.clone()) { paths.push(s); }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_path_relative() {
        let cwd = "/home/user/project";
        let resolved = resolve_path("extensions/my-ext.wasm", cwd);
        assert!(resolved.contains("my-ext.wasm"));
        assert!(resolved.starts_with("/home/user/project"));
    }

    #[test]
    fn test_resolve_path_absolute() {
        let resolved = resolve_path("/abs/path/ext.wasm", "/cwd");
        assert_eq!(resolved, "/abs/path/ext.wasm");
    }

    #[test]
    fn test_load_extensions_empty() {
        let result = load_extensions(&[], "/tmp");
        assert!(result.extensions.is_empty());
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_load_extension_not_found() {
        let result = load_extensions(&["/nonexistent/path.wasm".to_string()], "/tmp");
        assert_eq!(result.errors.len(), 1);
        assert!(result.errors[0].error.contains("not found"));
    }
}
