//! macOS path variant resolution for the read tool.
//!
//! macOS stores filenames in NFD (decomposed) Unicode form and uses special
//! characters in screenshot names (narrow no-break space before AM/PM,
//! right single quotation mark in French locale). Users typically type the
//! NFC / ASCII equivalents, so we try several variants before giving up.
//!
//! Mirrors pi-mono's `resolveReadPath` / `resolveReadPathAsync`
//! (`path-utils.ts`).

use std::path::{Path, PathBuf};

/// Narrow no-break space (U+202F) used by macOS before AM/PM in screenshot names.
const NARROW_NO_BREAK_SPACE: char = '\u{202F}';

/// Try replacing ASCII space before AM/PM with the narrow no-break space
/// macOS inserts in screenshot filenames.
fn try_macos_screenshot_path(path: &str) -> String {
    // Replace " AM." / " PM." (case-insensitive) with NARROW_NO_BREAK_SPACE + "AM." etc.
    let mut result = String::with_capacity(path.len());
    let bytes = path.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b' ' && i + 3 < bytes.len() {
            let next3 = &path[i + 1..].to_ascii_uppercase();
            if next3.starts_with("AM.") || next3.starts_with("PM.") {
                result.push(NARROW_NO_BREAK_SPACE);
                i += 1;
                continue;
            }
        }
        result.push(path[i..].chars().next().unwrap_or(' '));
        i += path[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
    }
    result
}

/// Try NFD (decomposed) Unicode normalisation — macOS stores filenames in NFD.
fn try_nfd_variant(path: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    path.nfd().collect()
}

/// Try replacing straight apostrophe (U+0027) with right single quotation
/// mark (U+2019) — macOS uses U+2019 in French screenshot names like
/// "Capture d'écran".
fn try_curly_quote_variant(path: &str) -> String {
    path.replace('\'', "\u{2019}")
}

/// Resolve a path for reading, trying macOS-specific filename variants if the
/// canonical path does not exist. Returns the first path that exists, or the
/// canonical path if none do.
///
/// Variants tried (in order):
/// 1. Canonical resolved path
/// 2. AM/PM narrow-no-break-space variant
/// 3. NFD variant
/// 4. Curly-quote variant
/// 5. NFD + curly-quote combined (French macOS screenshots)
pub fn resolve_read_path(path: &str, cwd: &str) -> PathBuf {
    let resolved = if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        Path::new(cwd).join(path)
    };

    if resolved.exists() {
        return resolved;
    }

    let resolved_str = resolved.to_string_lossy();

    let ampm = try_macos_screenshot_path(&resolved_str);
    if ampm != *resolved_str && Path::new(&ampm).exists() {
        return PathBuf::from(ampm);
    }

    let nfd = try_nfd_variant(&resolved_str);
    if nfd != *resolved_str && Path::new(&nfd).exists() {
        return PathBuf::from(nfd);
    }

    let curly = try_curly_quote_variant(&resolved_str);
    if curly != *resolved_str && Path::new(&curly).exists() {
        return PathBuf::from(curly);
    }

    let nfd_curly = try_curly_quote_variant(&nfd);
    if nfd_curly != *resolved_str && Path::new(&nfd_curly).exists() {
        return PathBuf::from(nfd_curly);
    }

    resolved
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absolute_path_returned_unchanged_when_exists() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        let result = resolve_read_path(path, "/tmp");
        assert_eq!(result, tmp.path());
    }

    #[test]
    fn relative_path_resolved_against_cwd() {
        let dir = tempfile::TempDir::new().unwrap();
        let file = dir.path().join("test.txt");
        std::fs::write(&file, b"hi").unwrap();
        let result = resolve_read_path("test.txt", dir.path().to_str().unwrap());
        assert_eq!(result, file);
    }

    #[test]
    fn nonexistent_path_returns_canonical_form() {
        let result = resolve_read_path("no_such_file.txt", "/tmp");
        assert_eq!(result, Path::new("/tmp/no_such_file.txt"));
    }

    #[test]
    fn ampm_variant_tried_when_canonical_missing() {
        let dir = tempfile::TempDir::new().unwrap();
        // Create a file with the narrow no-break space variant.
        let narrow = format!("screenshot\u{202F}AM.png");
        let file = dir.path().join(&narrow);
        std::fs::write(&file, b"img").unwrap();

        // User types with regular space.
        let user_input = "screenshot AM.png";
        let result = resolve_read_path(user_input, dir.path().to_str().unwrap());
        assert_eq!(result, file);
    }

    #[test]
    fn curly_quote_variant_tried() {
        let dir = tempfile::TempDir::new().unwrap();
        // Create file with right single quotation mark.
        let name = "Capture d\u{2019}\u{e9}cran.png";
        let file = dir.path().join(name);
        std::fs::write(&file, b"img").unwrap();

        // User types with straight apostrophe.
        let user_input = "Capture d'\u{e9}cran.png";
        let result = resolve_read_path(user_input, dir.path().to_str().unwrap());
        assert_eq!(result, file);
    }
}
