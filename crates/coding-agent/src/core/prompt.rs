use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ── Context files ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ContextFiles {
    pub files: Vec<LoadedContextFile>,
}

impl ContextFiles {
    pub fn combined_content(&self) -> String {
        self.files.iter().map(|f| f.content.as_str()).collect::<Vec<_>>().join("\n\n")
    }
}

/// A context file loaded from disk (path is a PathBuf).
#[derive(Debug, Clone)]
pub struct LoadedContextFile {
    pub path: PathBuf,
    pub content: String,
}

/// A context file reference for passing to build_system_prompt (path is a String).
#[derive(Debug, Clone)]
pub struct ContextFile {
    pub path: String,
    pub content: String,
}

pub fn load_context_files(cwd: &Path) -> ContextFiles {
    let mut files = Vec::new();
    let mut dir = Some(cwd.to_path_buf());
    while let Some(current) = dir {
        for name in &["AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"] {
            let path = current.join(name);
            if path.exists() {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    files.push(LoadedContextFile { path, content });
                }
            }
        }
        dir = current.parent().map(|p| p.to_path_buf());
    }
    files.reverse();
    ContextFiles { files }
}

// ── Skills ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub content: String,
    pub path: PathBuf,
}

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

pub fn discover_extension_paths(cwd: &Path, agent_dir: &Path) -> Vec<String> {
    let mut paths = Vec::new();
    let local = cwd.join(".automata/extensions");
    if local.exists() { collect_extension_files(&local, &mut paths); }
    let global = agent_dir.join("extensions");
    if global.exists() { collect_extension_files(&global, &mut paths); }
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
            if index.exists() { out.push(index.to_string_lossy().to_string()); }
        }
    }
}

// ── System prompt ─────────────────────────────────────────────────────────────

pub struct BuildSystemPromptOptions<'a> {
    pub custom_prompt: Option<&'a str>,
    pub selected_tools: Option<&'a [&'a str]>,
    pub tool_snippets: Option<&'a HashMap<String, String>>,
    pub prompt_guidelines: Option<&'a [String]>,
    pub append_system_prompt: Option<&'a str>,
    pub cwd: &'a str,
    pub context_files: Option<&'a [ContextFile]>,
    pub skills: Option<&'a [Skill]>,
}

pub fn build_system_prompt(opts: BuildSystemPromptOptions<'_>) -> String {
    let date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let cwd = opts.cwd.replace('\\', "/");
    let context_files = opts.context_files.unwrap_or(&[]);
    let skills = opts.skills.unwrap_or(&[]);
    let tools = opts.selected_tools.unwrap_or(&["read", "bash", "edit", "write"]);

    let append_section = opts.append_system_prompt
        .map(|s| format!("\n\n{}", s))
        .unwrap_or_default();

    let context_section = if context_files.is_empty() {
        String::new()
    } else {
        let mut s = "\n\n<project_context>\n\nProject-specific instructions and guidelines:\n\n".to_string();
        for f in context_files {
            s.push_str(&format!(
                "<project_instructions path=\"{}\">\n{}\n</project_instructions>\n\n",
                f.path, f.content
            ));
        }
        s.push_str("</project_context>\n");
        s
    };

    let skills_section = if tools.contains(&"read") && !skills.is_empty() {
        let mut s = "\n\n# Available Skills\n\n".to_string();
        for skill in skills {
            s.push_str(&format!("## {}\n{}\n\n", skill.name, skill.description));
        }
        s
    } else {
        String::new()
    };

    let footer = format!("\nCurrent date: {}\nCurrent working directory: {}", date, cwd);

    if let Some(custom) = opts.custom_prompt {
        return format!("{}{}{}{}{}", custom, append_section, context_section, skills_section, footer);
    }

    let tools_list = match opts.tool_snippets {
        Some(snippets) => {
            let visible: Vec<String> = tools.iter()
                .filter_map(|&n| snippets.get(n).map(|s| format!("- {}: {}", n, s)))
                .collect();
            if visible.is_empty() { "(none)".to_string() } else { visible.join("\n") }
        }
        None => "(none)".to_string(),
    };

    let mut guidelines: Vec<String> = vec![];
    let has_bash = tools.contains(&"bash");
    let has_grep = tools.contains(&"grep");
    let has_find = tools.contains(&"find");
    let has_ls = tools.contains(&"ls");
    if has_bash && !has_grep && !has_find && !has_ls {
        guidelines.push("Use bash for file operations like ls, rg, find".to_string());
    } else if has_bash && (has_grep || has_find || has_ls) {
        guidelines.push("Prefer grep/find/ls tools over bash for file exploration".to_string());
    }
    for g in opts.prompt_guidelines.unwrap_or(&[]) {
        let g = g.trim().to_string();
        if !g.is_empty() && !guidelines.contains(&g) { guidelines.push(g); }
    }
    guidelines.push("Be concise in your responses".to_string());
    guidelines.push("Show file paths clearly when working with files".to_string());
    let guidelines_str = guidelines.iter().map(|g| format!("- {}", g)).collect::<Vec<_>>().join("\n");

    let prompt = format!(
        "You are an expert coding assistant. You help users by reading files, executing commands, editing code, and writing new files.\n\nAvailable tools:\n{}\n\nGuidelines:\n{}",
        tools_list, guidelines_str
    );

    format!("{}{}{}{}{}", prompt, append_section, context_section, skills_section, footer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_context_files_empty() {
        let dir = tempfile::TempDir::new().unwrap();
        let ctx = load_context_files(dir.path());
        assert!(ctx.files.is_empty());
    }

    #[test]
    fn test_load_context_files_finds_claude_md() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), "# Context").unwrap();
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
