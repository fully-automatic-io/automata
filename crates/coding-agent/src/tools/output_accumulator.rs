//! Streaming output accumulator with bounded memory and temp-file fallback.
//!
//! Mirrors pi-mono's `OutputAccumulator` (`output-accumulator.ts`). Appends
//! decoded chunks, keeps a rolling tail for display snapshots, and spills to a
//! temp file when the full output exceeds the configured limits.

use std::io::Write as _;
use std::path::PathBuf;

use crate::tools::bash::{TruncationResult, DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES};
use agent_core::harness::utils::split_lines_for_counting;

/// Snapshot returned by [`OutputAccumulator::snapshot`].
#[derive(Debug, Clone)]
pub struct OutputSnapshot {
    /// Truncated tail content for display.
    pub content: String,
    /// Truncation metadata.
    pub truncation: TruncationResult,
    /// Path to the temp file holding the full output, if one was created.
    pub full_output_path: Option<String>,
}

/// Incrementally tracks streaming bash output with bounded memory.
///
/// - Appends raw bytes as they arrive.
/// - Keeps a rolling decoded tail (≤ `max_bytes * 2`) for display.
/// - Spills to a temp file when total output exceeds limits so the full
///   output is preserved even after truncation.
pub struct OutputAccumulator {
    max_lines: usize,
    max_bytes: usize,
    /// Rolling tail of decoded text (≤ `max_bytes * 2`).
    tail: String,
    tail_bytes: usize,
    tail_starts_at_line_boundary: bool,
    total_raw_bytes: usize,
    total_decoded_bytes: usize,
    completed_lines: usize,
    total_lines: usize,
    current_line_bytes: usize,
    has_open_line: bool,
    /// Temp file path (created lazily when output exceeds limits).
    temp_path: Option<PathBuf>,
    /// Buffered raw chunks before the temp file is opened.
    raw_chunks: Vec<Vec<u8>>,
    finished: bool,
}

impl OutputAccumulator {
    pub fn new(max_lines: usize, max_bytes: usize) -> Self {
        Self {
            max_lines,
            max_bytes,
            tail: String::new(),
            tail_bytes: 0,
            tail_starts_at_line_boundary: true,
            total_raw_bytes: 0,
            total_decoded_bytes: 0,
            completed_lines: 0,
            total_lines: 0,
            current_line_bytes: 0,
            has_open_line: false,
            temp_path: None,
            raw_chunks: Vec::new(),
            finished: false,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES)
    }

    /// Append a raw byte chunk from the process stream.
    pub fn append(&mut self, data: &[u8]) {
        if self.finished { return; }
        self.total_raw_bytes += data.len();
        let text = String::from_utf8_lossy(data).into_owned();
        self.append_decoded_text(&text);

        if self.should_use_temp_file() {
            self.ensure_temp_file();
            if let Some(ref path) = self.temp_path {
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                    let _ = f.write_all(data);
                }
            }
        } else if !data.is_empty() {
            self.raw_chunks.push(data.to_vec());
        }
    }

    /// Signal that no more data will arrive.
    pub fn finish(&mut self) {
        if self.finished { return; }
        self.finished = true;
        if self.should_use_temp_file() {
            self.ensure_temp_file();
        }
    }

    /// Build a display snapshot. If `persist_if_truncated` is true and the
    /// output is truncated, the temp file is created even if limits weren't
    /// exceeded yet (so the full output is always accessible).
    pub fn snapshot(&mut self, persist_if_truncated: bool) -> OutputSnapshot {
        let snapshot_text = self.get_snapshot_text();
        let truncation = self.truncate_tail_for_snapshot(&snapshot_text);

        if persist_if_truncated && truncation.truncated {
            self.ensure_temp_file();
        }

        OutputSnapshot {
            content: truncation.content.clone(),
            truncation,
            full_output_path: self.temp_path.as_ref().map(|p| p.to_string_lossy().into_owned()),
        }
    }

    /// Path to the temp file, if one was created.
    pub fn full_output_path(&self) -> Option<&str> {
        self.temp_path.as_ref().map(|p| p.to_str().unwrap_or(""))
    }

    // ── private ──────────────────────────────────────────────────────────────

    fn append_decoded_text(&mut self, text: &str) {
        if text.is_empty() { return; }
        let bytes = text.len();
        self.total_decoded_bytes += bytes;
        self.tail.push_str(text);
        self.tail_bytes += bytes;

        // Trim rolling tail to 2× max_bytes.
        let max_rolling = self.max_bytes.saturating_mul(2).max(1);
        if self.tail_bytes > max_rolling * 2 {
            self.trim_tail(max_rolling);
        }

        // Update line counters.
        let mut newlines = 0usize;
        let mut last_newline = None;
        for (i, ch) in text.char_indices() {
            if ch == '\n' {
                newlines += 1;
                last_newline = Some(i);
            }
        }
        if newlines == 0 {
            self.current_line_bytes += bytes;
            self.has_open_line = true;
        } else {
            self.completed_lines += newlines;
            let tail_after = last_newline.map(|i| &text[i + 1..]).unwrap_or("");
            self.current_line_bytes = tail_after.len();
            self.has_open_line = !tail_after.is_empty();
        }
        self.total_lines = self.completed_lines + if self.has_open_line { 1 } else { 0 };
    }

    fn trim_tail(&mut self, max_rolling: usize) {
        let bytes = self.tail.as_bytes();
        if bytes.len() <= max_rolling {
            self.tail_bytes = bytes.len();
            return;
        }
        let mut start = bytes.len() - max_rolling;
        // Align to UTF-8 boundary.
        while start < bytes.len() && (bytes[start] & 0xc0) == 0x80 {
            start += 1;
        }
        self.tail_starts_at_line_boundary = if start == 0 {
            self.tail_starts_at_line_boundary
        } else {
            start > 0 && bytes[start - 1] == b'\n'
        };
        self.tail = self.tail[start..].to_string();
        self.tail_bytes = self.tail.len();
    }

    fn get_snapshot_text(&self) -> String {
        if self.tail_starts_at_line_boundary {
            return self.tail.clone();
        }
        match self.tail.find('\n') {
            None => self.tail.clone(),
            Some(i) => self.tail[i + 1..].to_string(),
        }
    }

    fn truncate_tail_for_snapshot(&self, text: &str) -> TruncationResult {
        let lines = split_lines_for_counting(text);
        let total_lines_in_tail = lines.len();
        let total_bytes_in_tail = text.len();

        let truncated = self.total_lines > self.max_lines || self.total_decoded_bytes > self.max_bytes;

        let mut out_lines: Vec<&str> = vec![];
        let mut out_bytes = 0usize;
        for line in lines.iter().rev() {
            let lb = line.len() + 1;
            if out_lines.len() >= self.max_lines || out_bytes + lb > self.max_bytes { break; }
            out_lines.push(line);
            out_bytes += lb;
        }
        out_lines.reverse();
        let output_lines = out_lines.len();
        let content = out_lines.join("\n");

        let truncated_by = if truncated {
            if self.total_decoded_bytes > self.max_bytes { Some("bytes".into()) }
            else { Some("lines".into()) }
        } else { None };

        TruncationResult {
            content,
            truncated,
            total_lines: self.total_lines,
            output_lines,
            output_bytes: out_bytes,
            truncated_by,
            max_bytes: Some(self.max_bytes),
            max_lines: Some(self.max_lines),
            last_line_partial: false,
            first_line_exceeds_limit: total_lines_in_tail > 0
                && total_bytes_in_tail > self.max_bytes
                && output_lines == 0,
        }
    }

    fn should_use_temp_file(&self) -> bool {
        self.total_raw_bytes > self.max_bytes
            || self.total_decoded_bytes > self.max_bytes
            || self.total_lines > self.max_lines
    }

    fn ensure_temp_file(&mut self) {
        if self.temp_path.is_some() { return; }
        let path = std::env::temp_dir().join(format!(
            "automata-output-{}.log",
            uuid::Uuid::new_v4()
        ));
        // Flush buffered raw chunks.
        if let Ok(mut f) = std::fs::File::create(&path) {
            for chunk in self.raw_chunks.drain(..) {
                let _ = f.write_all(&chunk);
            }
        }
        self.temp_path = Some(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_output_no_temp_file() {
        let mut acc = OutputAccumulator::new(2000, 50 * 1024);
        acc.append(b"hello\nworld\n");
        acc.finish();
        let snap = acc.snapshot(false);
        assert!(!snap.truncation.truncated);
        assert!(snap.full_output_path.is_none());
        assert_eq!(snap.truncation.total_lines, 2);
    }

    #[test]
    fn large_output_creates_temp_file() {
        let mut acc = OutputAccumulator::new(2000, 100); // tiny limit
        let big = "x".repeat(200);
        acc.append(big.as_bytes());
        acc.finish();
        let snap = acc.snapshot(true);
        assert!(snap.truncation.truncated);
        assert!(snap.full_output_path.is_some());
        // Temp file should exist and contain the data.
        let path = snap.full_output_path.unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, big);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn line_count_correct_with_trailing_newline() {
        let mut acc = OutputAccumulator::with_defaults();
        acc.append(b"a\nb\n");
        acc.finish();
        let snap = acc.snapshot(false);
        assert_eq!(snap.truncation.total_lines, 2);
    }

    #[test]
    fn persist_if_truncated_creates_temp_file() {
        let mut acc = OutputAccumulator::new(1, 50 * 1024); // 1-line limit
        acc.append(b"line1\nline2\nline3\n");
        acc.finish();
        let snap = acc.snapshot(true);
        assert!(snap.truncation.truncated);
        assert!(snap.full_output_path.is_some());
        let _ = snap.full_output_path.map(std::fs::remove_file);
    }
}
