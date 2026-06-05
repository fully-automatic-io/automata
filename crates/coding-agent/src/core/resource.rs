use std::collections::HashSet;
use std::path::{Path, PathBuf};

use agent_core::harness::prompt_templates::{PromptTemplate, load_prompt_templates_from_dir};

use super::prompt::{ContextFile, LoadedContextFile, Skill};
use crate::extensions::{
    ExtensionEvent, LoadExtensionsResult, SessionLifecycleReason, discover_and_load_extensions,
    dispatch_loaded_extensions,
};
use crate::settings::SettingsManager;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceDiagnosticKind {
    Error,
    Warning,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceDiagnostic {
    pub kind: ResourceDiagnosticKind,
    pub message: String,
    pub path: Option<PathBuf>,
}

#[derive(Debug, Default)]
pub struct ResourceSet {
    pub extensions: Option<LoadExtensionsResult>,
    pub skills: Vec<Skill>,
    pub prompts: Vec<PromptTemplate>,
    pub context_files: Vec<LoadedContextFile>,
    pub system_prompt: Option<String>,
    pub append_system_prompt: Vec<String>,
    pub diagnostics: Vec<ResourceDiagnostic>,
}

#[derive(Debug, Clone)]
pub struct ResourceLoaderOptions {
    pub cwd: PathBuf,
    pub agent_dir: PathBuf,
    pub extension_paths: Vec<String>,
    pub skill_paths: Vec<PathBuf>,
    pub prompt_paths: Vec<PathBuf>,
    pub no_extensions: bool,
    pub no_skills: bool,
    pub no_prompts: bool,
    pub no_context_files: bool,
    pub system_prompt: Option<String>,
    pub append_system_prompt: Vec<String>,
}

impl ResourceLoaderOptions {
    pub fn new(cwd: impl Into<PathBuf>, agent_dir: impl Into<PathBuf>) -> Self {
        Self {
            cwd: cwd.into(),
            agent_dir: agent_dir.into(),
            extension_paths: Vec::new(),
            skill_paths: Vec::new(),
            prompt_paths: Vec::new(),
            no_extensions: false,
            no_skills: false,
            no_prompts: false,
            no_context_files: false,
            system_prompt: None,
            append_system_prompt: Vec::new(),
        }
    }
}

pub struct DefaultResourceLoader {
    options: ResourceLoaderOptions,
    resources: ResourceSet,
}

impl DefaultResourceLoader {
    pub fn new(options: ResourceLoaderOptions) -> Self {
        Self {
            options,
            resources: ResourceSet::default(),
        }
    }

    pub fn from_settings(
        cwd: impl Into<PathBuf>,
        agent_dir: impl Into<PathBuf>,
        settings: &SettingsManager,
    ) -> Self {
        let mut options = ResourceLoaderOptions::new(cwd, agent_dir);
        options.extension_paths = settings.get().extensions.clone();
        options.skill_paths = settings.get().skills.iter().map(PathBuf::from).collect();
        options.prompt_paths = settings.get().prompts.iter().map(PathBuf::from).collect();
        options.system_prompt = settings.get_system_prompt().map(ToOwned::to_owned);
        options.append_system_prompt = settings.get_append_system_prompt().to_vec();
        Self::new(options)
    }

    pub fn resources(&self) -> &ResourceSet {
        &self.resources
    }

    pub fn into_resources(self) -> ResourceSet {
        self.resources
    }

    pub fn reload(&mut self) {
        let mut diagnostics = Vec::new();
        let context_files = if self.options.no_context_files {
            Vec::new()
        } else {
            load_project_context_files(&self.options.cwd, &self.options.agent_dir, &mut diagnostics)
        };

        let skills = if self.options.no_skills {
            Vec::new()
        } else {
            load_skill_paths(&self.skill_paths(), &mut diagnostics)
        };

        let prompts = if self.options.no_prompts {
            Vec::new()
        } else {
            load_prompt_paths(&self.prompt_paths(), &mut diagnostics)
        };

        let mut extension_append_system_prompt = Vec::new();
        let extensions = if self.options.no_extensions {
            None
        } else {
            let loaded = discover_and_load_extensions(
                &self.options.extension_paths,
                &self.options.cwd.to_string_lossy(),
                Some(&self.options.agent_dir.to_string_lossy()),
            );
            let resource_updates = dispatch_loaded_extensions(
                &loaded,
                &ExtensionEvent::ResourcesDiscover {
                    cwd: self.options.cwd.to_string_lossy().to_string(),
                    reason: SessionLifecycleReason::Startup,
                },
            );
            for (_, update) in resource_updates {
                collect_extension_prompt_updates(&update, &mut extension_append_system_prompt);
            }
            Some(loaded)
        };

        let system_prompt =
            self.options.system_prompt.as_deref().and_then(|source| {
                resolve_prompt_source(source, "system prompt", &mut diagnostics)
            });
        let mut append_system_prompt: Vec<String> = self
            .options
            .append_system_prompt
            .iter()
            .filter_map(|source| {
                resolve_prompt_source(source, "append system prompt", &mut diagnostics)
            })
            .collect();
        append_system_prompt.extend(extension_append_system_prompt);

        self.resources = ResourceSet {
            extensions,
            skills,
            prompts,
            context_files,
            system_prompt,
            append_system_prompt,
            diagnostics,
        };
    }

    fn skill_paths(&self) -> Vec<PathBuf> {
        let mut paths = vec![
            self.options.agent_dir.join("skills"),
            self.options.cwd.join(".automata").join("skills"),
        ];
        paths
            .extend(self.options.skill_paths.iter().map(|p| resolve_against(&self.options.cwd, p)));
        dedupe_paths(paths)
    }

    fn prompt_paths(&self) -> Vec<PathBuf> {
        let mut paths = vec![
            self.options.agent_dir.join("prompts"),
            self.options.cwd.join(".automata").join("prompts"),
        ];
        paths.extend(
            self.options.prompt_paths.iter().map(|p| resolve_against(&self.options.cwd, p)),
        );
        dedupe_paths(paths)
    }
}

fn collect_extension_prompt_updates(value: &serde_json::Value, output: &mut Vec<String>) {
    if let Some(prompt) = value.get("systemPrompt").and_then(|value| value.as_str()) {
        output.push(prompt.to_string());
    }
    match value.get("appendSystemPrompt") {
        Some(serde_json::Value::String(prompt)) => output.push(prompt.clone()),
        Some(serde_json::Value::Array(prompts)) => {
            output.extend(prompts.iter().filter_map(|value| value.as_str().map(ToOwned::to_owned)));
        }
        _ => {}
    }
}

pub fn load_project_context_files(
    cwd: &Path,
    agent_dir: &Path,
    diagnostics: &mut Vec<ResourceDiagnostic>,
) -> Vec<LoadedContextFile> {
    let mut files = Vec::new();
    let mut seen = HashSet::new();
    if let Some(file) = load_context_file_from_dir(agent_dir, diagnostics)
        && seen.insert(file.path.clone())
    {
        files.push(file);
    }

    let mut ancestor_files = Vec::new();
    let mut current = Some(cwd.to_path_buf());
    while let Some(dir) = current {
        if let Some(file) = load_context_file_from_dir(&dir, diagnostics)
            && seen.insert(file.path.clone())
        {
            ancestor_files.push(file);
        }
        current = dir.parent().map(Path::to_path_buf);
    }
    ancestor_files.reverse();
    files.extend(ancestor_files);
    files
}

pub fn context_files_for_prompt(files: &[LoadedContextFile]) -> Vec<ContextFile> {
    files
        .iter()
        .map(|file| ContextFile {
            path: file.path.to_string_lossy().to_string(),
            content: file.content.clone(),
        })
        .collect()
}

fn load_context_file_from_dir(
    dir: &Path,
    diagnostics: &mut Vec<ResourceDiagnostic>,
) -> Option<LoadedContextFile> {
    for name in ["AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"] {
        let path = dir.join(name);
        if !path.exists() {
            continue;
        }
        match std::fs::read_to_string(&path) {
            Ok(content) => return Some(LoadedContextFile { path, content }),
            Err(err) => diagnostics.push(ResourceDiagnostic {
                kind: ResourceDiagnosticKind::Warning,
                message: format!("failed to read context file: {}", err),
                path: Some(path),
            }),
        }
    }
    None
}

fn load_skill_paths(paths: &[PathBuf], diagnostics: &mut Vec<ResourceDiagnostic>) -> Vec<Skill> {
    let mut skills = Vec::new();
    for path in paths {
        if !path.exists() {
            continue;
        }
        if path.is_dir() {
            skills.extend(agent_core::harness::skills::load_skills_from_dir(path));
        } else {
            diagnostics.push(ResourceDiagnostic {
                kind: ResourceDiagnosticKind::Warning,
                message: "skill path is not a directory".into(),
                path: Some(path.clone()),
            });
        }
    }
    skills.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.file_path.cmp(&b.file_path)));
    skills.dedup_by(|a, b| a.name == b.name);
    skills
}

fn load_prompt_paths(
    paths: &[PathBuf],
    diagnostics: &mut Vec<ResourceDiagnostic>,
) -> Vec<PromptTemplate> {
    let mut prompts = Vec::new();
    for path in paths {
        if !path.exists() {
            continue;
        }
        if path.is_dir() {
            prompts.extend(load_prompt_templates_from_dir(path));
        } else {
            diagnostics.push(ResourceDiagnostic {
                kind: ResourceDiagnosticKind::Warning,
                message: "prompt path is not a directory".into(),
                path: Some(path.clone()),
            });
        }
    }
    prompts.sort_by(|a, b| a.name.cmp(&b.name));
    prompts.dedup_by(|a, b| a.name == b.name);
    prompts
}

fn resolve_prompt_source(
    source: &str,
    description: &str,
    diagnostics: &mut Vec<ResourceDiagnostic>,
) -> Option<String> {
    let path = Path::new(source);
    if path.exists() {
        return match std::fs::read_to_string(path) {
            Ok(content) => Some(content),
            Err(err) => {
                diagnostics.push(ResourceDiagnostic {
                    kind: ResourceDiagnosticKind::Warning,
                    message: format!("failed to read {}: {}", description, err),
                    path: Some(path.to_path_buf()),
                });
                None
            }
        };
    }
    Some(source.to_string())
}

fn resolve_against(cwd: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}

fn dedupe_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for path in paths {
        let key = path.to_string_lossy().to_string();
        if seen.insert(key) {
            out.push(path);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_global_then_project_context_files() {
        let dir = tempfile::TempDir::new().unwrap();
        let agent_dir = dir.path().join("agent");
        let project = dir.path().join("repo");
        std::fs::create_dir_all(&agent_dir).unwrap();
        std::fs::create_dir_all(&project).unwrap();
        std::fs::write(agent_dir.join("AGENTS.md"), "global").unwrap();
        std::fs::write(project.join("AGENTS.md"), "project").unwrap();

        let mut diagnostics = Vec::new();
        let files = load_project_context_files(&project, &agent_dir, &mut diagnostics);
        assert!(diagnostics.is_empty());
        assert_eq!(
            files.iter().map(|f| f.content.as_str()).collect::<Vec<_>>(),
            vec!["global", "project"]
        );
    }

    #[test]
    fn resource_loader_loads_skills_and_prompts() {
        let dir = tempfile::TempDir::new().unwrap();
        let agent_dir = dir.path().join("agent");
        let project = dir.path().join("repo");
        let skills = project.join(".automata/skills");
        let prompts = project.join(".automata/prompts");
        std::fs::create_dir_all(&skills).unwrap();
        std::fs::create_dir_all(&prompts).unwrap();
        std::fs::write(
            skills.join("review.md"),
            "---\nname: review\ndescription: review code\n---\n# Review",
        )
        .unwrap();
        std::fs::write(prompts.join("fix.md"), "---\ndescription: fix bug\n---\nFix $1").unwrap();

        let mut loader =
            DefaultResourceLoader::new(ResourceLoaderOptions::new(&project, &agent_dir));
        loader.reload();
        assert_eq!(loader.resources().skills.len(), 1);
        assert_eq!(loader.resources().prompts.len(), 1);
    }
}
