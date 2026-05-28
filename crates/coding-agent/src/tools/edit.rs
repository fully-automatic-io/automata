
use agent_core::tool::AgentTool;
use agent_core::types::{AgentToolResult, AgentToolUpdateCallback, ContentBlock, ToolExecutionMode};
use async_trait::async_trait;
use similar::{ChangeTag, TextDiff};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use unicode_normalization::UnicodeNormalization;

// ============================================================================
// Edit type
// ============================================================================

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Edit {
    #[serde(rename = "oldText")]
    pub old_text: String,
    #[serde(rename = "newText")]
    pub new_text: String,
}

// ============================================================================
// Edit Tool Details
// ============================================================================

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EditToolDetails {
    /// Display-oriented diff with line numbers
    pub diff: String,
    /// Standard unified patch (git-applyable)
    pub patch: String,
    #[serde(rename = "firstChangedLine", skip_serializing_if = "Option::is_none")]
    pub first_changed_line: Option<usize>,
}

// ============================================================================
// File Mutation Queue — prevents concurrent edits to the same file
// ============================================================================

lazy_static::lazy_static! {
    static ref FILE_MUTATION_QUEUES: Mutex<HashMap<String, Arc<Mutex<()>>>> = Mutex::new(HashMap::new());
}

async fn with_file_mutation_queue<F, T>(path: &str, f: F) -> T
where
    F: std::future::Future<Output = T>,
{
    // Get or create mutex for this file
    let lock = {
        let mut queues = FILE_MUTATION_QUEUES.lock().await;
        queues.entry(path.to_string()).or_insert_with(|| Arc::new(Mutex::new(()))).clone()
    };

    // Acquire lock for this file
    let _guard = lock.lock().await;

    // Execute operation
    f.await
}

// ============================================================================
// EditOperations trait — pluggable file I/O
// ============================================================================

#[async_trait]
pub trait EditOperations: Send + Sync {
    async fn read_file(&self, path: &str) -> Result<Vec<u8>, String>;
    async fn write_file(&self, path: &str, content: &str) -> Result<(), String>;
    async fn access(&self, path: &str) -> Result<(), String>;
}

pub struct LocalEditOperations;

#[async_trait]
impl EditOperations for LocalEditOperations {
    async fn read_file(&self, path: &str) -> Result<Vec<u8>, String> {
        tokio::fs::read(path).await.map_err(|e| e.to_string())
    }

    async fn write_file(&self, path: &str, content: &str) -> Result<(), String> {
        tokio::fs::write(path, content).await.map_err(|e| e.to_string())
    }

    async fn access(&self, path: &str) -> Result<(), String> {
        let meta = tokio::fs::metadata(path).await.map_err(|e| e.to_string())?;
        if !meta.is_file() {
            return Err(format!("Not a file: {}", path));
        }
        Ok(())
    }
}

// ============================================================================
// Edit-diff utilities
// ============================================================================

/// Detect line ending style
pub fn detect_line_ending(content: &str) -> &str {
    if content.contains("\r\n") {
        "\r\n"
    } else {
        "\n"
    }
}

/// Strip UTF-8 BOM if present
pub fn strip_bom(content: &str) -> (&str, &str) {
    if content.starts_with('\u{FEFF}') {
        ("\u{FEFF}", &content[3..])
    } else {
        ("", content)
    }
}

/// Normalize line endings to LF
pub fn normalize_to_lf(content: &str) -> String {
    content.replace("\r\n", "\n")
}

/// Restore original line endings
pub fn restore_line_endings(content: &str, ending: &str) -> String {
    if ending == "\r\n" {
        content.replace('\n', "\r\n")
    } else {
        content.to_string()
    }
}

/// Normalize text for fuzzy matching. Mirrors pi-mono's
/// `normalizeForFuzzyMatch` (edit-diff.ts):
///
/// 1. NFKC compatibility composition
/// 2. Strip trailing whitespace from each line
/// 3. Smart quotes → ASCII (`'` and `"`)
/// 4. Unicode dashes → ASCII `-`
/// 5. Special-width spaces → ASCII space
fn fuzzy_normalize(text: &str) -> String {
    let nfkc: String = text.nfkc().collect();
    let trimmed = nfkc
        .split('\n')
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n");
    trimmed
        .chars()
        .map(|c| match c {
            // Smart single quotes: ‘ ’ ‚ ‛
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
            // Smart double quotes: “ ” „ ‟
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
            // Dashes / hyphens / minus: ‐ ‑ ‒ – — ― −
            '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}'
            | '\u{2212}' => '-',
            // Special-width spaces: NBSP, en/em/thin/hair/etc, narrow NBSP, math, ideographic.
            '\u{00A0}'
            | '\u{2002}'..='\u{200A}'
            | '\u{202F}'
            | '\u{205F}'
            | '\u{3000}' => ' ',
            other => other,
        })
        .collect()
}

/// Outcome of locating `old_text` inside `content`.
///
/// When `used_fuzzy` is `true`, `content_for_replacement` is the
/// fuzzy-normalized version of `content` and the edit must be applied
/// against that string (matching pi-mono semantics: minor formatting
/// differences get normalized into the file).
struct FuzzyMatch {
    range: std::ops::Range<usize>,
    /// `true` if the exact-match path failed and the fuzzy normalization
    /// path produced the hit. Exposed so callers can surface a soft
    /// warning (matches pi-mono `usedFuzzyMatch`); not consumed yet.
    #[allow(dead_code)]
    used_fuzzy: bool,
    content_for_replacement: String,
}

/// Find `old_text` in `content`. Tries exact match first, then falls back
/// to fuzzy matching against [`fuzzy_normalize`]'d versions.
fn fuzzy_find_text(content: &str, old_text: &str) -> Option<FuzzyMatch> {
    if let Some(pos) = content.find(old_text) {
        return Some(FuzzyMatch {
            range: pos..pos + old_text.len(),
            used_fuzzy: false,
            content_for_replacement: content.to_string(),
        });
    }

    let norm_content = fuzzy_normalize(content);
    let norm_old = fuzzy_normalize(old_text);
    let pos = norm_content.find(&norm_old)?;
    Some(FuzzyMatch {
        range: pos..pos + norm_old.len(),
        used_fuzzy: true,
        content_for_replacement: norm_content,
    })
}

/// Count occurrences of `needle` in `haystack` (non-overlapping)
fn count_occurrences(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    let mut count = 0;
    let mut pos = 0;
    while let Some(found) = haystack[pos..].find(needle) {
        count += 1;
        pos += found + needle.len();
    }
    count
}

/// Check if oldText appears exactly once in content (with fuzzy fallback)
fn check_uniqueness(content: &str, old_text: &str) -> Result<(), String> {
    // Try exact match first
    let exact_count = count_occurrences(content, old_text);
    if exact_count == 1 {
        return Ok(());
    }
    if exact_count > 1 {
        return Err(format!(
            "oldText appears {} times in the file. It must be unique. \
            Please provide more context to make it unique.",
            exact_count
        ));
    }

    // Fall back to fuzzy match — handles smart quotes, NFKC variants, etc.
    let norm_content = fuzzy_normalize(content);
    let norm_old = fuzzy_normalize(old_text);
    let fuzzy_count = count_occurrences(&norm_content, &norm_old);

    if fuzzy_count == 0 {
        Err("oldText not found in file".to_string())
    } else if fuzzy_count > 1 {
        Err(format!(
            "oldText appears {} times in the file. It must be unique. \
            Please provide more context to make it unique.",
            fuzzy_count
        ))
    } else {
        Ok(())
    }
}

/// Apply multiple edits to normalized content
pub fn apply_edits_to_normalized_content(
    content: &str,
    edits: &[Edit],
    path: &str,
) -> Result<(String, String), String> {
    let mut result = content.to_string();

    for (idx, edit) in edits.iter().enumerate() {
        // Check for empty oldText
        if edit.old_text.is_empty() {
            return Err("oldText cannot be empty".to_string());
        }

        // Check uniqueness
        check_uniqueness(&result, &edit.old_text)?;

        // Try exact match first, then fuzzy
        if let Some(m) = fuzzy_find_text(&result, &edit.old_text) {
            // When fuzzy matching, work in the normalized space — minor
            // formatting differences (smart quotes, special spaces) are
            // intentionally normalized into the resulting file. Mirrors
            // pi-mono's `contentForReplacement` semantics.
            let base = m.content_for_replacement;
            let before = &base[..m.range.start];
            let after = &base[m.range.end..];
            result = format!("{}{}{}", before, edit.new_text, after);
        } else {
            return Err(format!(
                "Edit #{} failed: oldText not found in {}.\nSearched for: {:?}",
                idx + 1,
                path,
                &edit.old_text[..edit.old_text.len().min(200)]
            ));
        }
    }

    Ok((content.to_string(), result))
}

/// Generate unified diff string with line numbers
pub fn generate_diff_string(old: &str, new: &str) -> (String, Option<usize>) {
    let diff = TextDiff::from_lines(old, new);
    let mut output = String::new();
    let mut first_changed_line: Option<usize> = None;

    // Track line numbers for both old and new
    let mut old_line = 0;
    let mut new_line = 0;

    for change in diff.iter_all_changes() {
        match change.tag() {
            ChangeTag::Equal => {
                old_line += 1;
                new_line += 1;
            }
            ChangeTag::Delete => {
                if first_changed_line.is_none() {
                    first_changed_line = Some(old_line + 1);
                }
                output.push_str(&format!("{:>4} -│{}", old_line + 1, change));
                old_line += 1;
            }
            ChangeTag::Insert => {
                if first_changed_line.is_none() {
                    first_changed_line = Some(new_line + 1);
                }
                output.push_str(&format!("{:>4} +│{}", new_line + 1, change));
                new_line += 1;
            }
        }
    }

    (output, first_changed_line)
}

/// Generate a standard unified patch (git-applyable) using
/// `similar::TextDiff::unified_diff()`.
pub fn generate_unified_patch(path: &str, old: &str, new: &str, context_lines: usize) -> String {
    let diff = TextDiff::from_lines(old, new);
    let header_a = format!("a/{}", path);
    let header_b = format!("b/{}", path);
    diff.unified_diff()
        .context_radius(context_lines)
        .header(&header_a, &header_b)
        .to_string()
}

/// Result of a preview diff computation.
#[derive(Debug, Clone)]
pub struct EditDiffResult {
    pub diff: String,
    pub patch: String,
    pub first_changed_line: Option<usize>,
}

/// Compute the diff for one or more edits **without writing the file**.
/// Used for preview rendering before the tool executes.
///
/// Returns `Err(message)` if the file can't be read or the edits are invalid.
pub async fn compute_edits_diff<O: EditOperations>(
    path: &str,
    edits: &[Edit],
    cwd: &str,
    operations: &O,
) -> Result<EditDiffResult, String> {
    let absolute = if std::path::Path::new(path).is_absolute() {
        path.to_string()
    } else {
        std::path::Path::new(cwd).join(path).to_string_lossy().into_owned()
    };

    let raw = operations.read_file(&absolute).await
        .map_err(|e| format!("Could not read file: {path}. {e}"))?;
    let raw_str = String::from_utf8_lossy(&raw);
    let (_bom, stripped) = strip_bom(&raw_str);
    let normalized = normalize_to_lf(stripped);
    let (base, new) = apply_edits_to_normalized_content(&normalized, edits, path)?;
    let (diff, first_changed_line) = generate_diff_string(&base, &new);
    let patch = generate_unified_patch(path, &base, &new, 4);
    Ok(EditDiffResult { diff, patch, first_changed_line })
}

/// Convenience wrapper for a single-edit preview.
pub async fn compute_edit_diff<O: EditOperations>(
    path: &str,
    old_text: &str,
    new_text: &str,
    cwd: &str,
    operations: &O,
) -> Result<EditDiffResult, String> {
    compute_edits_diff(
        path,
        &[Edit { old_text: old_text.to_string(), new_text: new_text.to_string() }],
        cwd,
        operations,
    ).await
}

// ============================================================================
// Edit Tool
// ============================================================================

pub struct EditToolOptions {
    pub operations: Option<Arc<dyn EditOperations>>,
}

impl Default for EditToolOptions {
    fn default() -> Self {
        Self { operations: None }
    }
}

pub struct EditTool {
    cwd: String,
    operations: Arc<dyn EditOperations>,
}

impl EditTool {
    pub fn new(cwd: String, options: EditToolOptions) -> Self {
        let ops = options
            .operations
            .unwrap_or_else(|| Arc::new(LocalEditOperations));
        Self {
            cwd,
            operations: ops,
        }
    }

    fn resolve_path(&self, path: &str) -> String {
        if Path::new(path).is_absolute() {
            path.to_string()
        } else {
            Path::new(&self.cwd)
                .join(path)
                .to_string_lossy()
                .to_string()
        }
    }
}

#[async_trait]
impl AgentTool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn label(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Edit a file using exact text replacement. Each oldText must match a unique region in the file."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to file to edit"
                },
                "edits": {
                    "type": "array",
                    "description": "Array of edit operations to apply",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {
                                "type": "string",
                                "description": "Exact text to replace (must be unique in file)"
                            },
                            "newText": {
                                "type": "string",
                                "description": "Text to replace with"
                            }
                        },
                        "required": ["oldText", "newText"]
                    }
                }
            },
            "required": ["path", "edits"]
        })
    }

    fn execution_mode(&self) -> Option<ToolExecutionMode> {
        // Edit operations should be sequential to avoid conflicts
        Some(ToolExecutionMode::Sequential)
    }

    fn prepare_arguments(&self, args: serde_json::Value) -> serde_json::Value {
        let mut args = args;

        // Handle legacy format: oldText/newText at top level
        let has_legacy = args.get("oldText").and_then(|v| v.as_str()).is_some()
            && args.get("newText").and_then(|v| v.as_str()).is_some();

        if has_legacy {
            let old_text = args.get("oldText").unwrap().clone();
            let new_text = args.get("newText").unwrap().clone();
            let mut edits = args
                .get("edits")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            edits.push(serde_json::json!({"oldText": old_text, "newText": new_text}));

            if let Some(obj) = args.as_object_mut() {
                obj.remove("oldText");
                obj.remove("newText");
                obj.insert("edits".to_string(), serde_json::Value::Array(edits));
            }
        }

        // Handle edits as JSON string
        if let Some(edits_str) = args.get("edits").and_then(|v| v.as_str()) {
            if let Ok(parsed) = serde_json::from_str::<Vec<serde_json::Value>>(edits_str) {
                if let Some(obj) = args.as_object_mut() {
                    obj.insert("edits".to_string(), serde_json::Value::Array(parsed));
                }
            }
        }

        args
    }

    async fn execute(
        &self,
        _tool_call_id: String,
        params: serde_json::Value,
        signal: Option<CancellationToken>,
        on_update: Option<AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>> {
        let path = params
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' parameter")?;

        let edits: Vec<Edit> = params
            .get("edits")
            .and_then(|v| v.as_array())
            .ok_or("Missing 'edits' parameter")?
            .iter()
            .filter_map(|e| {
                Some(Edit {
                    old_text: e.get("oldText")?.as_str()?.to_string(),
                    new_text: e.get("newText")?.as_str()?.to_string(),
                })
            })
            .collect();

        if edits.is_empty() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "edits must contain at least one replacement",
            )));
        }

        // Check for cancellation
        if signal.as_ref().map_or(false, |s| s.is_cancelled()) {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "Operation cancelled",
            )));
        }

        let absolute_path = self.resolve_path(path);

        let on_update_arc: Option<Arc<AgentToolUpdateCallback>> = on_update.map(Arc::new);

        // Use file mutation queue to prevent concurrent edits
        let result = with_file_mutation_queue(&absolute_path, async {
            // Check file exists and is accessible
            self.operations
                .access(&absolute_path)
                .await
                .map_err(|e| {
                    Box::new(std::io::Error::new(std::io::ErrorKind::NotFound, e))
                        as Box<dyn std::error::Error + Send + Sync>
                })?;

            // Read file
            let buffer = self
                .operations
                .read_file(&absolute_path)
                .await
                .map_err(|e| {
                    Box::new(std::io::Error::new(std::io::ErrorKind::Other, e))
                        as Box<dyn std::error::Error + Send + Sync>
                })?;

            let raw_content = String::from_utf8_lossy(&buffer).to_string();

            // Preserve BOM and line endings
            let (bom, text) = strip_bom(&raw_content);
            let original_ending = detect_line_ending(text);
            let normalized = normalize_to_lf(text);

            // Apply edits
            let (base, new) = apply_edits_to_normalized_content(&normalized, &edits, path)
                .map_err(|e| {
                    Box::new(std::io::Error::new(std::io::ErrorKind::Other, e))
                        as Box<dyn std::error::Error + Send + Sync>
                })?;

            // Check if anything changed
            if base == new {
                return Ok(AgentToolResult {
                    content: vec![ContentBlock::Text {
                        text: format!("No changes made to {} (oldText equals newText)", path),
                    }],
                    details: serde_json::json!({
                        "diff": "",
                        "firstChangedLine": null
                    }),
                    terminate: false,
                });
            }

            // Generate diff (display) + unified patch (git-applyable) early so we
            // can stream a preview before writing.
            let (diff, first_changed_line) = generate_diff_string(&base, &new);
            let patch = generate_unified_patch(&path, &base, &new, 4);

            // Emit streaming partial: diff preview before disk write hits.
            if let Some(cb) = &on_update_arc {
                let partial = AgentToolResult {
                    content: vec![ContentBlock::Text { text: diff.clone() }],
                    details: serde_json::to_value(EditToolDetails {
                        diff: diff.clone(),
                        patch: patch.clone(),
                        first_changed_line,
                    })
                    .unwrap_or_default(),
                    terminate: false,
                };
                cb(partial);
            }

            // Restore BOM and line endings
            let final_content = format!("{}{}", bom, restore_line_endings(&new, original_ending));

            // Write file
            self.operations
                .write_file(&absolute_path, &final_content)
                .await
                .map_err(|e| {
                    Box::new(std::io::Error::new(std::io::ErrorKind::Other, e))
                        as Box<dyn std::error::Error + Send + Sync>
                })?;

            Ok(AgentToolResult {
                content: vec![ContentBlock::Text {
                    text: format!(
                        "Successfully applied {} edit(s) to {}",
                        edits.len(),
                        path
                    ),
                }],
                details: serde_json::to_value(EditToolDetails {
                    diff,
                    patch,
                    first_changed_line,
                })
                .unwrap_or_default(),
                terminate: false,
            })
        })
        .await;

        result
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_line_ending() {
        assert_eq!(detect_line_ending("hello\nworld"), "\n");
        assert_eq!(detect_line_ending("hello\r\nworld"), "\r\n");
    }

    #[test]
    fn test_strip_bom() {
        assert_eq!(strip_bom("\u{FEFF}hello"), ("\u{FEFF}", "hello"));
        assert_eq!(strip_bom("hello"), ("", "hello"));
    }

    #[test]
    fn test_normalize_to_lf() {
        assert_eq!(normalize_to_lf("a\r\nb\r\nc"), "a\nb\nc");
        assert_eq!(normalize_to_lf("a\nb\nc"), "a\nb\nc");
    }

    #[test]
    fn test_restore_line_endings() {
        assert_eq!(restore_line_endings("a\nb\nc", "\r\n"), "a\r\nb\r\nc");
        assert_eq!(restore_line_endings("a\nb\nc", "\n"), "a\nb\nc");
    }

    #[test]
    fn test_fuzzy_normalize() {
        let text = "hello  \nworld  ";
        let normalized = fuzzy_normalize(text);
        assert_eq!(normalized, "hello\nworld");

        let smart_quotes = "\u{2018}hello\u{2019} \u{201C}world\u{201D}";
        let normalized = fuzzy_normalize(smart_quotes);
        assert_eq!(normalized, "'hello' \"world\"");
    }

    #[test]
    fn test_fuzzy_normalize_extended_quote_set() {
        // Single quotes: ‘ ’ ‚ ‛
        for ch in ['\u{2018}', '\u{2019}', '\u{201A}', '\u{201B}'] {
            assert_eq!(fuzzy_normalize(&ch.to_string()), "'");
        }
        // Double quotes: “ ” „ ‟
        for ch in ['\u{201C}', '\u{201D}', '\u{201E}', '\u{201F}'] {
            assert_eq!(fuzzy_normalize(&ch.to_string()), "\"");
        }
    }

    #[test]
    fn test_fuzzy_normalize_extended_dash_set() {
        // ‐ ‑ ‒ – — ― −
        for ch in [
            '\u{2010}', '\u{2011}', '\u{2012}', '\u{2013}', '\u{2014}', '\u{2015}', '\u{2212}',
        ] {
            assert_eq!(fuzzy_normalize(&ch.to_string()), "-");
        }
    }

    #[test]
    fn test_fuzzy_normalize_extended_space_set() {
        // NBSP, en/em/thin/hair/etc, narrow NBSP, math, ideographic.
        // Embed each between letters so trim_end doesn't strip it.
        for ch in [
            '\u{00A0}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}', '\u{2006}', '\u{2007}',
            '\u{2008}', '\u{2009}', '\u{200A}', '\u{202F}', '\u{205F}', '\u{3000}',
        ] {
            let input = format!("a{}b", ch);
            assert_eq!(fuzzy_normalize(&input), "a b", "char U+{:04X}", ch as u32);
        }
    }

    #[test]
    fn test_fuzzy_find_returns_normalized_content_when_used_fuzzy() {
        let content = "fn greet() {\n    println!(\u{201C}hi\u{201D});\n}\n";
        let m = fuzzy_find_text(content, "println!(\"hi\")").expect("fuzzy hit");
        assert!(m.used_fuzzy);
        // The result content has straight quotes, so applying an edit against
        // the normalized base writes ASCII quotes to disk.
        let slice = &m.content_for_replacement[m.range.clone()];
        assert_eq!(slice, "println!(\"hi\")");
    }

    #[test]
    fn test_apply_edits_fuzzy_normalizes_into_file() {
        // Source has smart quotes; oldText uses ASCII. The post-edit content
        // should carry ASCII quotes everywhere — matches pi-mono semantics.
        let src = "let s = \u{201C}hello\u{201D};\nlet t = \u{201C}world\u{201D};\n";
        let edits = vec![Edit {
            old_text: "let s = \"hello\";".into(),
            new_text: "let s = \"HELLO\";".into(),
        }];
        let (_, new) =
            apply_edits_to_normalized_content(src, &edits, "/x.rs").expect("apply");
        assert!(new.contains("let s = \"HELLO\";"));
        // The other line should also have been normalized (fuzzy rewrite of
        // entire file — a deliberate choice mirroring pi-mono).
        assert!(new.contains("let t = \"world\";"));
        assert!(!new.contains('\u{201C}'));
        assert!(!new.contains('\u{201D}'));
    }

    #[test]
    fn test_check_uniqueness() {
        assert!(check_uniqueness("a\nb\nc", "b").is_ok());
        assert!(check_uniqueness("a\nb\nb\nc", "b").is_err());
        assert!(check_uniqueness("a\nc", "b").is_err());
    }

    #[test]
    fn test_apply_edits() {
        let (_, new) = apply_edits_to_normalized_content(
            "a\nb\nc",
            &[Edit {
                old_text: "b".into(),
                new_text: "x".into(),
            }],
            "test.txt",
        )
        .unwrap();
        assert_eq!(new, "a\nx\nc");
    }

    #[test]
    fn test_apply_edits_multiple() {
        let (_, new) = apply_edits_to_normalized_content(
            "a\nb\nc\nd",
            &[
                Edit {
                    old_text: "b".into(),
                    new_text: "x".into(),
                },
                Edit {
                    old_text: "c".into(),
                    new_text: "y".into(),
                },
            ],
            "test.txt",
        )
        .unwrap();
        assert_eq!(new, "a\nx\ny\nd");
    }

    #[test]
    fn test_apply_edits_not_found() {
        assert!(apply_edits_to_normalized_content(
            "a\nb",
            &[Edit {
                old_text: "z".into(),
                new_text: "x".into()
            }],
            "test.txt"
        )
        .is_err());
    }

    #[test]
    fn test_apply_edits_empty_old_text() {
        assert!(apply_edits_to_normalized_content(
            "a\nb",
            &[Edit {
                old_text: "".into(),
                new_text: "x".into()
            }],
            "test.txt"
        )
        .is_err());
    }

    #[test]
    fn test_apply_edits_not_unique() {
        assert!(apply_edits_to_normalized_content(
            "a\nb\nb\nc",
            &[Edit {
                old_text: "b".into(),
                new_text: "x".into()
            }],
            "test.txt"
        )
        .is_err());
    }

    #[test]
    fn test_generate_diff() {
        let (diff, first) = generate_diff_string("a\nb\nc", "a\nx\nc");
        assert!(diff.contains("-│b"));
        assert!(diff.contains("+│x"));
        assert_eq!(first, Some(2));
    }

    #[test]
    fn test_generate_unified_patch_has_git_headers() {
        let patch = generate_unified_patch("src/foo.rs", "a\nb\nc\n", "a\nx\nc\n", 4);
        assert!(patch.contains("--- a/src/foo.rs"), "patch missing -- header: {}", patch);
        assert!(patch.contains("+++ b/src/foo.rs"), "patch missing ++ header: {}", patch);
        assert!(patch.contains("-b"));
        assert!(patch.contains("+x"));
    }

    #[tokio::test]
    async fn test_edit_tool_name() {
        let tool = EditTool::new("/tmp".into(), EditToolOptions::default());
        assert_eq!(tool.name(), "edit");
        assert_eq!(tool.label(), "edit");
        assert_eq!(tool.execution_mode(), Some(ToolExecutionMode::Sequential));
    }

    #[tokio::test]
    async fn test_compute_edits_diff_preview() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.rs");
        std::fs::write(&path, "fn main() {\n    println!(\"hello\");\n}\n").unwrap();

        let ops = LocalEditOperations;
        let result = compute_edits_diff(
            path.to_str().unwrap(),
            &[Edit { old_text: "hello".into(), new_text: "world".into() }],
            dir.path().to_str().unwrap(),
            &ops,
        ).await.unwrap();

        assert!(result.diff.contains("-│") || result.diff.contains("+│"));
        assert!(result.patch.contains("--- a/"));
        assert!(result.patch.contains("+++ b/"));
        assert!(result.patch.contains("-    println!(\"hello\")"));
        assert!(result.patch.contains("+    println!(\"world\")"));
        // File must NOT be modified.
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("hello"), "file must not be modified by preview");
    }

    #[tokio::test]
    async fn test_compute_edit_diff_single() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("f.txt");
        std::fs::write(&path, "a\nb\nc\n").unwrap();

        let ops = LocalEditOperations;
        let result = compute_edit_diff(
            path.to_str().unwrap(),
            "b",
            "x",
            dir.path().to_str().unwrap(),
            &ops,
        ).await.unwrap();

        assert!(result.patch.contains("-b"));
        assert!(result.patch.contains("+x"));
        assert_eq!(result.first_changed_line, Some(2));
    }

    #[tokio::test]
    async fn test_compute_edits_diff_missing_file() {
        let ops = LocalEditOperations;
        let err = compute_edits_diff(
            "/no/such/file.txt",
            &[Edit { old_text: "x".into(), new_text: "y".into() }],
            "/",
            &ops,
        ).await;
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn streaming_callback_receives_diff_preview() {
        use std::sync::Mutex as StdMutex;
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("foo.txt");
        tokio::fs::write(&file, "alpha\nbeta\ngamma\n").await.unwrap();

        let tool = EditTool::new(
            dir.path().to_string_lossy().into_owned(),
            EditToolOptions::default(),
        );

        let updates: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
        let updates_clone = updates.clone();
        let on_update: AgentToolUpdateCallback = Box::new(move |partial: AgentToolResult| {
            if let Some(diff) = partial.details.get("diff").and_then(|v| v.as_str()) {
                updates_clone.lock().unwrap().push(diff.to_string());
            }
        });

        let params = serde_json::json!({
            "path": file.to_string_lossy(),
            "edits": [{ "oldText": "beta", "newText": "BETA" }]
        });
        let result = tool.execute("test".into(), params, None, Some(on_update)).await;
        assert!(result.is_ok(), "edit should succeed: {:?}", result.err());

        let collected = updates.lock().unwrap();
        assert_eq!(collected.len(), 1, "expected one diff preview update");
        assert!(
            collected[0].contains("BETA"),
            "preview diff should contain new text, got: {:?}",
            collected[0]
        );
    }
}
