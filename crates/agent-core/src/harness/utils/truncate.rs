pub const DEFAULT_MAX_LINES: usize = 2000;
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;
pub const GREP_MAX_LINE_LENGTH: usize = 500;

/// Split for line counting that matches user-perceived line count: drops the
/// trailing empty entry when content ends with `\n`.
pub fn split_lines_for_counting(content: &str) -> Vec<&str> {
    if content.is_empty() {
        return vec![];
    }
    let mut lines: Vec<&str> = content.split('\n').collect();
    if content.ends_with('\n') {
        lines.pop();
    }
    lines
}

#[derive(Debug, Clone)]
pub struct TruncationResult {
    pub content: String,
    pub truncated: bool,
    pub truncated_by: Option<&'static str>,
    pub total_lines: usize,
    pub total_bytes: usize,
    pub output_lines: usize,
    pub output_bytes: usize,
    pub last_line_partial: bool,
    pub first_line_exceeds_limit: bool,
    pub max_lines: usize,
    pub max_bytes: usize,
}

/// Truncate from the head (keep first N lines/bytes). Never returns partial lines.
pub fn truncate_head(content: &str, max_lines: usize, max_bytes: usize) -> TruncationResult {
    let total_bytes = content.len();
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return TruncationResult {
            content: content.to_string(),
            truncated: false,
            truncated_by: None,
            total_lines,
            total_bytes,
            output_lines: total_lines,
            output_bytes: total_bytes,
            last_line_partial: false,
            first_line_exceeds_limit: false,
            max_lines,
            max_bytes,
        };
    }

    let first_line_exceeds_limit = lines.first().map(|l| l.len() > max_bytes).unwrap_or(false);
    if first_line_exceeds_limit {
        return TruncationResult {
            content: String::new(),
            truncated: true,
            truncated_by: Some("bytes"),
            total_lines,
            total_bytes,
            output_lines: 0,
            output_bytes: 0,
            last_line_partial: false,
            first_line_exceeds_limit: true,
            max_lines,
            max_bytes,
        };
    }

    let mut out_lines = 0usize;
    let mut out_bytes = 0usize;
    let mut truncated_by = "lines";

    for (i, line) in lines.iter().enumerate() {
        if out_lines >= max_lines {
            break;
        }
        let line_bytes = line.len() + if i > 0 { 1 } else { 0 };
        if out_bytes + line_bytes > max_bytes {
            truncated_by = "bytes";
            break;
        }
        out_lines += 1;
        out_bytes += line_bytes;
    }

    let result_content = lines[..out_lines.min(lines.len())].join("\n");
    TruncationResult {
        content: result_content,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines: out_lines,
        output_bytes: out_bytes,
        last_line_partial: false,
        first_line_exceeds_limit: false,
        max_lines,
        max_bytes,
    }
}

/// Truncate from the tail (keep last N lines/bytes). May return partial first line.
pub fn truncate_tail(content: &str, max_lines: usize, max_bytes: usize) -> TruncationResult {
    let total_bytes = content.len();
    let lines = split_lines_for_counting(content);
    let total_lines = lines.len();

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return TruncationResult {
            content: content.to_string(),
            truncated: false,
            truncated_by: None,
            total_lines,
            total_bytes,
            output_lines: total_lines,
            output_bytes: total_bytes,
            last_line_partial: false,
            first_line_exceeds_limit: false,
            max_lines,
            max_bytes,
        };
    }

    let mut out_lines: Vec<&str> = vec![];
    let mut out_bytes = 0usize;
    let mut truncated_by = "lines";
    let mut last_line_partial = false;

    for (_i, line) in lines.iter().enumerate().rev() {
        if out_lines.len() >= max_lines {
            break;
        }
        let line_bytes = line.len() + if !out_lines.is_empty() { 1 } else { 0 };
        if out_bytes + line_bytes > max_bytes {
            truncated_by = "bytes";
            if out_lines.is_empty() {
                // Take end of this line
                let truncated = truncate_str_to_bytes_from_end(line, max_bytes);
                out_bytes = truncated.len();
                out_lines.insert(0, Box::leak(truncated.into_boxed_str()));
                last_line_partial = true;
            }
            break;
        }
        out_lines.insert(0, line);
        out_bytes += line_bytes;
    }

    let result_content = out_lines.join("\n");
    TruncationResult {
        content: result_content,
        truncated: true,
        truncated_by: Some(truncated_by),
        total_lines,
        total_bytes,
        output_lines: out_lines.len(),
        output_bytes: out_bytes,
        last_line_partial,
        first_line_exceeds_limit: false,
        max_lines,
        max_bytes,
    }
}

fn truncate_str_to_bytes_from_end(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let start = s.len() - max_bytes;
    // Find valid UTF-8 boundary
    let mut i = start;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    s[i..].to_string()
}

/// Truncate a single line to max chars, adding `... [truncated]` suffix.
pub fn truncate_line(line: &str, max_chars: usize) -> (String, bool) {
    if line.chars().count() <= max_chars {
        return (line.to_string(), false);
    }
    let truncated: String = line.chars().take(max_chars).collect();
    (format!("{}... [truncated]", truncated), true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_lines_for_counting_handles_trailing_newline() {
        // "a\nb\n" is 2 lines, not 3.
        assert_eq!(split_lines_for_counting("a\nb\n"), vec!["a", "b"]);
        assert_eq!(split_lines_for_counting("a\nb"), vec!["a", "b"]);
        assert_eq!(split_lines_for_counting(""), Vec::<&str>::new());
        assert_eq!(split_lines_for_counting("\n"), vec![""]);
        assert_eq!(split_lines_for_counting("a"), vec!["a"]);
    }

    #[test]
    fn truncate_head_no_off_by_one_on_trailing_newline() {
        let r = truncate_head("a\nb\n", 10, 1024);
        assert_eq!(r.total_lines, 2);
        assert!(!r.truncated);
    }

    #[test]
    fn truncate_tail_no_off_by_one_on_trailing_newline() {
        let r = truncate_tail("a\nb\n", 10, 1024);
        assert_eq!(r.total_lines, 2);
        assert!(!r.truncated);
    }
}
