// Resource loader — loads CLAUDE.md, memory files, skills, prompts from filesystem

use std::path::{Path, PathBuf};

/// Context files discovered from cwd and ancestors
#[derive(Debug, Clone)]
pub struct ContextFiles {
    pub files: Vec<ContextFile>,
}

#[derive(Debug, Clone)]
pub struct ContextFile {
    pub path: PathBuf,
    pub content: String,
}

impl ContextFiles {
    /// Concatenate all context file contents
    pub fn combined_content(&self) -> String {
        self.files.iter()
            .map(|f| f.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// Discover and load CLAUDE.md / AGENTS.md from cwd and all ancestor directories
pub fn load_context_files(cwd: &Path) -> ContextFiles {
    let mut files = Vec::new();
    let mut dir = Some(cwd.to_path_buf());

    // Walk up directory tree
    while let Some(current) = dir {
        for name in &["CLAUDE.md", "AGENTS.md", ".claude/CLAUDE.md"] {
            let path = current.join(name);
            if path.exists() {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    files.push(ContextFile { path, content });
                }
            }
        }
        dir = current.parent().map(|p| p.to_path_buf());
    }

    // Reverse so root-level files come first
    files.reverse();
    ContextFiles { files }
}

/// A skill loaded from a markdown file with frontmatter
#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub content: String,
    pub path: PathBuf,
}

/// Load skills from a directory (*.md files with frontmatter)
pub fn load_skills(dir: &Path) -> Vec<Skill> {
    let mut skills = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return skills; };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") { continue; }

        if let Ok(content) = std::fs::read_to_string(&path) {
            let (name, description, body) = parse_frontmatter(&content);
            let name = name.unwrap_or_else(|| {
                path.file_stem().unwrap_or_default().to_string_lossy().to_string()
            });
            skills.push(Skill { name, description: description.unwrap_or_default(), content: body, path });
        }
    }
    skills
}

/// Parse YAML frontmatter from markdown: returns (name, description, body)
fn parse_frontmatter(content: &str) -> (Option<String>, Option<String>, String) {
    if !content.starts_with("---") {
        return (None, None, content.to_string());
    }
    let rest = &content[3..];
    let Some(end) = rest.find("\n---") else {
        return (None, None, content.to_string());
    };
    let frontmatter = &rest[..end];
    let body = rest[end + 4..].trim_start_matches('\n').to_string();

    let mut name = None;
    let mut description = None;
    for line in frontmatter.lines() {
        if let Some(v) = line.strip_prefix("name:") {
            name = Some(v.trim().trim_matches('"').to_string());
        } else if let Some(v) = line.strip_prefix("description:") {
            description = Some(v.trim().trim_matches('"').to_string());
        }
    }
    (name, description, body)
}

/// Discover all extension paths from standard locations
pub fn discover_extension_paths(cwd: &Path, agent_dir: &Path) -> Vec<String> {
    let mut paths = Vec::new();

    // Project-local: .automata/extensions/
    let local = cwd.join(".automata/extensions");
    if local.exists() {
        collect_extension_files(&local, &mut paths);
    }

    // Global: ~/.automata/agent/extensions/
    let global = agent_dir.join("extensions");
    if global.exists() {
        collect_extension_files(&global, &mut paths);
    }

    paths
}

fn collect_extension_files(dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return; };
    for entry in entries.flatten() {
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str());
        if matches!(ext, Some("wasm")) {
            out.push(path.to_string_lossy().to_string());
        } else if path.is_dir() {
            let index = path.join("extension.wasm");
            if index.exists() {
                out.push(index.to_string_lossy().to_string());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_load_context_files_empty() {
        let dir = TempDir::new().unwrap();
        let ctx = load_context_files(dir.path());
        assert!(ctx.files.is_empty());
    }

    #[test]
    fn test_load_context_files_finds_claude_md() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("CLAUDE.md"), "# Context").unwrap();
        let ctx = load_context_files(dir.path());
        assert_eq!(ctx.files.len(), 1);
        assert!(ctx.combined_content().contains("# Context"));
    }

    #[test]
    fn test_parse_frontmatter() {
        let md = "---\nname: my-skill\ndescription: does stuff\n---\n# Body";
        let (name, desc, body) = parse_frontmatter(md);
        assert_eq!(name.as_deref(), Some("my-skill"));
        assert_eq!(desc.as_deref(), Some("does stuff"));
        assert!(body.contains("# Body"));
    }

    #[test]
    fn test_parse_frontmatter_no_frontmatter() {
        let md = "# Just content";
        let (name, desc, body) = parse_frontmatter(md);
        assert!(name.is_none());
        assert!(desc.is_none());
        assert_eq!(body, "# Just content");
    }
}
