/// Parse an argument string using shell-style quoting.
pub fn parse_command_args(args_string: &str) -> Vec<String> {
    let mut args = vec![];
    let mut current = String::new();
    let mut in_quote: Option<char> = None;
    for ch in args_string.chars() {
        if let Some(q) = in_quote {
            if ch == q { in_quote = None; } else { current.push(ch); }
        } else if ch == '"' || ch == '\'' {
            in_quote = Some(ch);
        } else if ch == ' ' || ch == '\t' {
            if !current.is_empty() { args.push(std::mem::take(&mut current)); }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() { args.push(current); }
    args
}

/// Substitute `$1`, `$2`, `$@`, `$ARGUMENTS`, `${@:N}`, `${@:N:L}` in template content.
pub fn substitute_args(content: &str, args: &[String]) -> String {
    let all_args = args.join(" ");
    let mut result = content.to_string();

    // ${@:N:L} and ${@:N}
    let re_slice = regex::Regex::new(r"\$\{@:(\d+)(?::(\d+))?\}").unwrap();
    result = re_slice.replace_all(&result, |caps: &regex::Captures| {
        let start = caps[1].parse::<usize>().unwrap_or(1).saturating_sub(1);
        if let Some(len_str) = caps.get(2) {
            let len = len_str.as_str().parse::<usize>().unwrap_or(0);
            args.get(start..).unwrap_or(&[]).iter().take(len).cloned().collect::<Vec<_>>().join(" ")
        } else {
            args.get(start..).unwrap_or(&[]).join(" ")
        }
    }).to_string();

    // $N positional
    let re_pos = regex::Regex::new(r"\$(\d+)").unwrap();
    result = re_pos.replace_all(&result, |caps: &regex::Captures| {
        let n: usize = caps[1].parse().unwrap_or(0);
        args.get(n.saturating_sub(1)).map(|s| s.as_str()).unwrap_or("").to_string()
    }).to_string();

    result.replace("$ARGUMENTS", &all_args).replace("$@", &all_args)
}

/// A prompt template loaded from a .md file.
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    pub name: String,
    pub description: Option<String>,
    pub content: String,
}

/// Load prompt templates from a directory (direct .md children, non-recursive).
pub fn load_prompt_templates_from_dir(dir: &std::path::Path) -> Vec<PromptTemplate> {
    let mut templates = vec![];
    let Ok(entries) = std::fs::read_dir(dir) else { return templates; };
    let mut paths: Vec<_> = entries.flatten()
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("md"))
        .map(|e| e.path())
        .collect();
    paths.sort();
    for path in paths {
        if let Ok(content) = std::fs::read_to_string(&path) {
            let name = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
            let (fm, body) = parse_frontmatter_str(&content);
            let description = fm.get("description")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| {
                    body.lines().find(|l| !l.trim().is_empty())
                        .map(|l| if l.len() > 60 { format!("{}...", &l[..60]) } else { l.to_string() })
                });
            templates.push(PromptTemplate { name, description, content: body.to_string() });
        }
    }
    templates
}

fn parse_frontmatter_str(content: &str) -> (serde_json::Map<String, serde_json::Value>, &str) {
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
        .and_then(|v| if let serde_json::Value::Object(m) = v { Some(m) } else { None })
        .unwrap_or_default();
    (map, body)
}
