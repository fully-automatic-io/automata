// ── Slash commands ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SlashCommandInfo {
    pub name: String,
    pub description: String,
    pub source: SlashCommandSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashCommandSource {
    Builtin,
    Skill,
    Prompt,
    Extension,
}

pub fn builtin_slash_commands() -> Vec<SlashCommandInfo> {
    [
        ("settings", "Open settings"),
        ("model", "Select model"),
        ("export", "Export session to HTML"),
        ("compact", "Compact context"),
        ("fork", "Fork current session"),
        ("tree", "Show session tree"),
        ("new", "Start new session"),
        ("resume", "Resume a session"),
        ("name", "Name current session"),
        ("quit", "Quit"),
    ]
    .iter()
    .map(|(name, desc)| SlashCommandInfo {
        name: name.to_string(),
        description: desc.to_string(),
        source: SlashCommandSource::Builtin,
    })
    .collect()
}

// ── Source info ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceOrigin {
    Package,
    TopLevel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceScope {
    User,
    Project,
    Temporary,
}

#[derive(Debug, Clone)]
pub struct SourceInfo {
    pub scope: SourceScope,
    pub origin: SourceOrigin,
    pub path: Option<String>,
}

impl SourceInfo {
    pub fn new(scope: SourceScope, origin: SourceOrigin, path: Option<String>) -> Self {
        Self { scope, origin, path }
    }

    pub fn project_top_level(path: impl Into<String>) -> Self {
        Self::new(SourceScope::Project, SourceOrigin::TopLevel, Some(path.into()))
    }

    pub fn user_top_level(path: impl Into<String>) -> Self {
        Self::new(SourceScope::User, SourceOrigin::TopLevel, Some(path.into()))
    }

    pub fn synthetic() -> Self {
        Self::new(SourceScope::Temporary, SourceOrigin::TopLevel, None)
    }
}
