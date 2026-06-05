use std::path::{Path, PathBuf};

use agent_core::harness::session::SessionError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionCwdIssue {
    pub session_path: Option<PathBuf>,
    pub stored_cwd: PathBuf,
    pub fallback_cwd: PathBuf,
}

pub fn missing_session_cwd_issue(
    session_path: Option<&Path>,
    stored_cwd: &Path,
    fallback_cwd: &Path,
) -> Option<SessionCwdIssue> {
    if session_path.is_none() || stored_cwd.as_os_str().is_empty() || stored_cwd.exists() {
        return None;
    }

    Some(SessionCwdIssue {
        session_path: session_path.map(Path::to_path_buf),
        stored_cwd: stored_cwd.to_path_buf(),
        fallback_cwd: fallback_cwd.to_path_buf(),
    })
}

pub fn format_missing_session_cwd_error(issue: &SessionCwdIssue) -> String {
    let session_path = issue
        .session_path
        .as_ref()
        .map(|path| format!("\nSession file: {}", path.display()))
        .unwrap_or_default();
    format!(
        "Stored session working directory does not exist: {}{}\nCurrent working directory: {}",
        issue.stored_cwd.display(),
        session_path,
        issue.fallback_cwd.display()
    )
}

pub fn format_missing_session_cwd_prompt(issue: &SessionCwdIssue) -> String {
    format!(
        "cwd from session file does not exist\n{}\n\ncontinue in current cwd\n{}",
        issue.stored_cwd.display(),
        issue.fallback_cwd.display()
    )
}

pub fn assert_session_cwd_exists(
    session_path: Option<&Path>,
    stored_cwd: &Path,
    fallback_cwd: &Path,
) -> Result<(), SessionError> {
    match missing_session_cwd_issue(session_path, stored_cwd, fallback_cwd) {
        Some(issue) => Err(SessionError::InvalidArgument(format_missing_session_cwd_error(&issue))),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_missing_persisted_session_cwd() {
        let dir = tempfile::TempDir::new().unwrap();
        let session_path = dir.path().join("session.jsonl");
        let stored_cwd = dir.path().join("missing");
        let fallback_cwd = dir.path().join("fallback");

        let issue =
            missing_session_cwd_issue(Some(&session_path), &stored_cwd, &fallback_cwd).unwrap();

        assert_eq!(issue.session_path, Some(session_path));
        assert_eq!(issue.stored_cwd, stored_cwd);
        assert_eq!(issue.fallback_cwd, fallback_cwd);
    }

    #[test]
    fn ignores_in_memory_and_empty_cwd() {
        let dir = tempfile::TempDir::new().unwrap();
        assert!(missing_session_cwd_issue(None, &dir.path().join("missing"), dir.path()).is_none());
        assert!(missing_session_cwd_issue(Some(dir.path()), Path::new(""), dir.path()).is_none());
    }
}
