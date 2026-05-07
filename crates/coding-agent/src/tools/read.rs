
use agent_core::tool::AgentTool;
use agent_core::types::{AgentToolResult, AgentToolUpdateCallback, ContentBlock};
use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

pub const DEFAULT_MAX_BYTES: usize = 100_000;
pub const DEFAULT_MAX_LINES: usize = 2000; // Match TypeScript: 2000 lines

// Image formats supported
const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "gif", "webp"];
const MAX_IMAGE_DIMENSION: u32 = 2000;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TruncationResult {
    pub content: String,
    pub truncated: bool,
    #[serde(rename = "totalLines")]
    pub total_lines: usize,
    #[serde(rename = "outputLines")]
    pub output_lines: usize,
    #[serde(rename = "outputBytes")]
    pub output_bytes: usize,
    #[serde(rename = "truncatedBy", skip_serializing_if = "Option::is_none")]
    pub truncated_by: Option<String>,
    #[serde(rename = "maxBytes", skip_serializing_if = "Option::is_none")]
    pub max_bytes: Option<usize>,
    #[serde(rename = "maxLines", skip_serializing_if = "Option::is_none")]
    pub max_lines: Option<usize>,
    #[serde(rename = "lastLinePartial")]
    pub last_line_partial: bool,
    #[serde(rename = "firstLineExceedsLimit")]
    pub first_line_exceeds_limit: bool,
}

/// Truncate from the head (keep first N lines/bytes)
pub fn truncate_head(full_output: &str, max_lines: usize, max_bytes: usize) -> TruncationResult {
    let lines: Vec<&str> = full_output.split('\n').collect();
    let total_lines = lines.len();

    // Check if first line exceeds byte limit
    let first_line_exceeds_limit = if !lines.is_empty() {
        lines[0].len() > max_bytes
    } else {
        false
    };

    let mut output_lines = 0usize;
    let mut output_bytes = 0usize;
    let mut truncated_by = None;

    for line in &lines {
        let line_bytes = line.len() + 1; // +1 for newline

        // Check line limit
        if output_lines >= max_lines {
            truncated_by = Some("lines".to_string());
            break;
        }

        // Check byte limit
        if output_bytes + line_bytes > max_bytes {
            truncated_by = Some("bytes".to_string());
            break;
        }

        output_lines += 1;
        output_bytes += line_bytes;
    }

    let content = lines[..output_lines].join("\n");
    let truncated = output_lines < total_lines;

    TruncationResult {
        content,
        truncated,
        total_lines,
        output_lines,
        output_bytes,
        truncated_by,
        max_bytes: Some(max_bytes),
        max_lines: Some(max_lines),
        last_line_partial: false,
        first_line_exceeds_limit,
    }
}

#[async_trait]
pub trait ReadOperations: Send + Sync {
    async fn read_file(&self, path: &str) -> Result<Vec<u8>, String>;
    async fn access(&self, path: &str) -> Result<(), String>;
    async fn detect_image_mime_type(&self, path: &str) -> Result<Option<String>, String>;
}

pub struct LocalReadOperations;

#[async_trait]
impl ReadOperations for LocalReadOperations {
    async fn read_file(&self, path: &str) -> Result<Vec<u8>, String> {
        tokio::fs::read(path).await.map_err(|e| e.to_string())
    }

    async fn access(&self, path: &str) -> Result<(), String> {
        let meta = tokio::fs::metadata(path).await.map_err(|e| e.to_string())?;
        if !meta.is_file() {
            return Err(format!("Not a file: {}", path));
        }
        Ok(())
    }

    async fn detect_image_mime_type(&self, path: &str) -> Result<Option<String>, String> {
        let mime = mime_guess::from_path(path).first();
        Ok(mime.map(|m| m.to_string()))
    }
}

pub struct ReadToolOptions {
    pub operations: Option<Arc<dyn ReadOperations>>,
}

impl Default for ReadToolOptions {
    fn default() -> Self {
        Self { operations: None }
    }
}

pub struct ReadTool {
    cwd: String,
    operations: Arc<dyn ReadOperations>,
}

impl ReadTool {
    pub fn new(cwd: String, options: ReadToolOptions) -> Self {
        let ops = options.operations.unwrap_or_else(|| Arc::new(LocalReadOperations));
        Self { cwd, operations: ops }
    }

    /// Check if file is an image based on extension
    fn is_image(path: &str) -> bool {
        if let Some(ext) = Path::new(path).extension() {
            let ext_str = ext.to_string_lossy().to_lowercase();
            IMAGE_EXTENSIONS.contains(&ext_str.as_str())
        } else {
            false
        }
    }

    /// Resize image if it exceeds max dimensions
    async fn resize_image_if_needed(data: &[u8]) -> Result<Vec<u8>, String> {
        let img = image::load_from_memory(data)
            .map_err(|e| format!("Failed to decode image: {}", e))?;

        let (width, height) = (img.width(), img.height());

        // Check if resize is needed
        if width <= MAX_IMAGE_DIMENSION && height <= MAX_IMAGE_DIMENSION {
            return Ok(data.to_vec());
        }

        // Calculate new dimensions maintaining aspect ratio
        let scale = (MAX_IMAGE_DIMENSION as f32 / width.max(height) as f32).min(1.0);
        let new_width = (width as f32 * scale) as u32;
        let new_height = (height as f32 * scale) as u32;

        let resized = img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3);

        // Encode back to PNG
        let mut buffer = Vec::new();
        resized.write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)
            .map_err(|e| format!("Failed to encode resized image: {}", e))?;

        Ok(buffer)
    }

    /// Read image and return as base64-encoded content block
    async fn read_image(&self, _path: &str, absolute_path: &str) -> Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>> {
        // Read image data
        let data = self.operations.read_file(absolute_path).await
            .map_err(|e| format!("Failed to read image: {}", e))?;

        // Detect MIME type
        let mime_type = self.operations.detect_image_mime_type(absolute_path).await?
            .unwrap_or_else(|| "image/png".to_string());

        // Get original dimensions
        let img = image::load_from_memory(&data)
            .map_err(|e| format!("Failed to decode image: {}", e))?;
        let (orig_width, orig_height) = (img.width(), img.height());

        // Resize if needed
        let final_data = Self::resize_image_if_needed(&data).await?;
        let resized = final_data.len() != data.len();

        // Get final dimensions
        let final_img = image::load_from_memory(&final_data)
            .map_err(|e| format!("Failed to decode resized image: {}", e))?;
        let (final_width, final_height) = (final_img.width(), final_img.height());

        // Encode to base64
        use base64::Engine;
        let base64_data = base64::engine::general_purpose::STANDARD.encode(&final_data);

        // Create image content block
        let image_block = ContentBlock::Image {
            data: base64_data,
            mime_type: mime_type.clone(),
        };

        // Add annotation about resizing
        let mut content = vec![image_block];
        if resized {
            content.push(ContentBlock::Text {
                text: format!(
                    "\n<!-- Image resized from {}x{} to {}x{} (max {}x{}) -->",
                    orig_width, orig_height, final_width, final_height,
                    MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION
                ),
            });
        } else {
            content.push(ContentBlock::Text {
                text: format!("\n<!-- Image dimensions: {}x{} -->", final_width, final_height),
            });
        }

        Ok(AgentToolResult {
            content,
            details: serde_json::json!({
                "originalDimensions": { "width": orig_width, "height": orig_height },
                "finalDimensions": { "width": final_width, "height": final_height },
                "resized": resized,
                "mimeType": mime_type
            }),
            terminate: false,
        })
    }

    /// Read text file with offset/limit support
    async fn read_text(&self, absolute_path: &str, offset: usize, limit: Option<u64>) -> Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>> {
        let buffer = self.operations.read_file(absolute_path).await
            .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e)) as Box<_>)?;

        let text = String::from_utf8_lossy(&buffer).to_string();
        let all_lines: Vec<&str> = text.split('\n').collect();
        let total_lines = all_lines.len();

        // Calculate range
        let start = offset.saturating_sub(1).min(all_lines.len());
        let end = if let Some(lim) = limit {
            (start + lim as usize).min(all_lines.len())
        } else {
            all_lines.len()
        };

        let selected = all_lines[start..end].join("\n");

        // Apply truncation
        let truncation = truncate_head(&selected, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);

        // Check if first line exceeds limit
        if truncation.first_line_exceeds_limit {
            let suggestion = format!(
                "The first line of this file is too large to display ({} bytes). \
                Consider using bash tool with: head -c {} {} | tail -c +{}",
                all_lines[start].len(),
                DEFAULT_MAX_BYTES,
                absolute_path,
                start * 100 // rough estimate
            );

            return Ok(AgentToolResult {
                content: vec![ContentBlock::Text { text: suggestion }],
                details: serde_json::json!({
                    "firstLineExceedsLimit": true,
                    "firstLineBytes": all_lines[start].len()
                }),
                terminate: false,
            });
        }

        let mut output = truncation.content;

        // Add continuation hint if truncated
        if truncation.truncated {
            let next_offset = start + truncation.output_lines + 1;
            output.push_str(&format!(
                "\n\n[Showing lines {}-{} of {}. Use offset={} to continue.]",
                start + 1,
                start + truncation.output_lines,
                total_lines,
                next_offset
            ));
        } else if limit.is_none() && end == total_lines {
            // Full file read, no continuation hint
        }

        Ok(AgentToolResult {
            content: vec![ContentBlock::Text { text: output }],
            details: serde_json::json!({
                "truncated": truncation.truncated,
                "totalLines": total_lines,
                "outputLines": truncation.output_lines,
                "startLine": start + 1,
                "endLine": start + truncation.output_lines
            }),
            terminate: false,
        })
    }
}

#[async_trait]
impl AgentTool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn label(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Supports text files, images (jpg, png, gif, webp), and more."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed, for text files)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (for text files)"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(
        &self,
        _tool_call_id: String,
        params: serde_json::Value,
        _signal: Option<CancellationToken>,
        _on_update: Option<AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, Box<dyn std::error::Error + Send + Sync>> {
        let path = params.get("path")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'path' parameter")?;

        let offset = params.get("offset")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;

        let limit = params.get("limit")
            .and_then(|v| v.as_u64());

        // Resolve absolute path
        let absolute_path = if Path::new(path).is_absolute() {
            path.to_string()
        } else {
            Path::new(&self.cwd).join(path).to_string_lossy().to_string()
        };

        // Check file exists and is accessible
        self.operations.access(&absolute_path).await
            .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::NotFound, e)) as Box<_>)?;

        // Check if it's an image
        if Self::is_image(&absolute_path) {
            return self.read_image(path, &absolute_path).await;
        }

        // Otherwise read as text
        self.read_text(&absolute_path, offset, limit).await
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_head_short() {
        let r = truncate_head("hello\nworld", DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
        assert!(!r.truncated);
        assert_eq!(r.total_lines, 2);
        assert_eq!(r.output_lines, 2);
    }

    #[test]
    fn test_truncate_head_long() {
        let long_text = (0..3000).map(|i| format!("line {}", i)).collect::<Vec<_>>().join("\n");
        let r = truncate_head(&long_text, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
        assert!(r.truncated);
        assert_eq!(r.truncated_by, Some("lines".to_string()));
        assert_eq!(r.output_lines, DEFAULT_MAX_LINES);
    }

    #[test]
    fn test_truncate_head_bytes() {
        let long_line = "a".repeat(DEFAULT_MAX_BYTES + 1000);
        let r = truncate_head(&long_line, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
        assert!(r.truncated);
        assert_eq!(r.truncated_by, Some("bytes".to_string()));
    }

    #[test]
    fn test_truncate_head_first_line_exceeds() {
        let huge_line = "x".repeat(DEFAULT_MAX_BYTES + 1);
        let r = truncate_head(&huge_line, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
        assert!(r.first_line_exceeds_limit);
    }

    #[test]
    fn test_is_image() {
        assert!(ReadTool::is_image("test.jpg"));
        assert!(ReadTool::is_image("test.PNG"));
        assert!(ReadTool::is_image("test.webp"));
        assert!(!ReadTool::is_image("test.txt"));
        assert!(!ReadTool::is_image("test"));
    }

    #[tokio::test]
    async fn test_read_tool_name() {
        let tool = ReadTool::new("/tmp".into(), ReadToolOptions::default());
        assert_eq!(tool.name(), "read");
        assert_eq!(tool.label(), "read");
    }

    #[tokio::test]
    async fn test_read_missing_path() {
        let tool = ReadTool::new("/tmp".into(), ReadToolOptions::default());
        let params = serde_json::json!({});
        let result = tool.execute("test".into(), params, None, None).await;
        assert!(result.is_err());
    }
}
