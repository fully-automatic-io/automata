pub mod shell_output;
pub mod truncate;

pub use shell_output::{execute_shell_with_capture, sanitize_binary_output, ShellCaptureResult};
pub use truncate::{
    split_lines_for_counting, truncate_head, truncate_line, truncate_tail, TruncationResult,
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, GREP_MAX_LINE_LENGTH,
};
