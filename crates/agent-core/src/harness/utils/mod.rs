pub mod shell_output;
pub mod truncate;

pub use shell_output::{ShellCaptureResult, execute_shell_with_capture, sanitize_binary_output};
pub use truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, GREP_MAX_LINE_LENGTH, TruncationResult,
    split_lines_for_counting, truncate_head, truncate_line, truncate_tail,
};
