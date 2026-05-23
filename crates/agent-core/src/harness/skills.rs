use serde_json::Value;

/// Format skills for the system prompt using XML blocks.
pub fn format_skills_for_system_prompt(skills: &[Skill]) -> String {
    let visible: Vec<&Skill> = skills.iter().filter(|s| !s.disable_model_invocation).collect();
    if visible.is_empty() {
        return String::new();
    }
    let mut lines = vec![
        "The following skills provide specialized instructions for specific tasks.".to_string(),
        "Read the full skill file when the task matches its description.".to_string(),
        "When a skill file references a relative path, resolve it against the skill directory and use that absolute path in tool commands.".to_string(),
        String::new(),
        "<available_skills>".to_string(),
    ];
    for skill in &visible {
        lines.push("  <skill>".to_string());
        lines.push(format!("    <name>{}</name>", escape_xml(&skill.name)));
        lines.push(format!("    <description>{}</description>", escape_xml(&skill.description)));
        lines.push(format!("    <location>{}</location>", escape_xml(&skill.file_path)));
        lines.push("  </skill>".to_string());
    }
    lines.push("</available_skills>".to_string());
    lines.join("\n")
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Format a skill invocation prompt.
pub fn format_skill_invocation(skill: &Skill, additional_instructions: Option<&str>) -> String {
    let dir = parent_dir(&skill.file_path);
    let block = format!(
        "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
        skill.name, skill.file_path, dir, skill.content
    );
    match additional_instructions {
        Some(instr) => format!("{}\n\n{}", block, instr),
        None => block,
    }
}

fn parent_dir(path: &str) -> &str {
    let normalized = path.trim_end_matches('/');
    match normalized.rfind('/') {
        Some(0) => "/",
        Some(i) => &normalized[..i],
        None => ".",
    }
}

/// A skill loaded from a SKILL.md file.
#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub content: String,
    pub file_path: String,
    pub disable_model_invocation: bool,
}

/// Load skills from a directory by scanning for *.md files with YAML frontmatter.
pub fn load_skills_from_dir(dir: &std::path::Path) -> Vec<Skill> {
    let mut skills = vec![];
    let Ok(entries) = std::fs::read_dir(dir) else { return skills; };
    let mut paths: Vec<_> = entries.flatten()
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("md"))
        .map(|e| e.path())
        .collect();
    paths.sort();
    for path in paths {
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Some(skill) = parse_skill_file(&content, &path.to_string_lossy()) {
                skills.push(skill);
            }
        }
    }
    skills
}

fn parse_skill_file(content: &str, file_path: &str) -> Option<Skill> {
    let (fm, body) = parse_frontmatter(content);
    let name = fm.get("name")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            let p = std::path::Path::new(file_path);
            p.parent()
                .and_then(|d| d.file_name())
                .map(|n| n.to_string_lossy().to_string())
        })?;
    let description = fm.get("description").and_then(|v| v.as_str())?.to_string();
    if description.trim().is_empty() { return None; }
    let disable = fm.get("disable-model-invocation")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Some(Skill { name, description, content: body.to_string(), file_path: file_path.to_string(), disable_model_invocation: disable })
}

fn parse_frontmatter(content: &str) -> (serde_json::Map<String, Value>, &str) {
    let normalized = content.trim_start_matches('\u{FEFF}');
    if !normalized.starts_with("---") {
        return (serde_json::Map::new(), normalized);
    }
    let rest = &normalized[3..];
    let end = rest.find("\n---").unwrap_or(rest.len());
    let yaml_str = &rest[..end];
    let body = if end + 4 < rest.len() { rest[end + 4..].trim() } else { "" };
    let map = serde_yaml::from_str::<serde_json::Value>(yaml_str)
        .ok()
        .and_then(|v| if let Value::Object(m) = v { Some(m) } else { None })
        .unwrap_or_default();
    (map, body)
}
