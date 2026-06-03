use llm_client::{
    AnthropicProvider, AuthMethod, ContentPart, LlmMessage, LlmProvider, LlmRequest, ProviderConfig,
};

#[tokio::main]
async fn main() {
    let base = std::env::var("ANTHROPIC_BASE_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com/anthropic".to_string());
    let token = std::env::var("ANTHROPIC_AUTH_TOKEN")
        .or_else(|_| std::env::var("DEEPSEEK_API_KEY"))
        .expect("Set ANTHROPIC_AUTH_TOKEN (or DEEPSEEK_API_KEY)");
    let model = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "deepseek-chat".to_string());

    let endpoint = format!("{}/v1/messages", base.trim_end_matches('/'));
    let config = ProviderConfig::new(token)
        .with_base_url(endpoint)
        .with_auth_method(AuthMethod::Bearer);
    let provider = AnthropicProvider::new(config);

    let request = LlmRequest {
        model: model.clone(),
        messages: vec![LlmMessage::user_text("用一句话自我介绍，然后告诉我 2+2 等于几。")],
        tools: vec![],
        system: Some("You are a concise assistant. Reply in Chinese.".into()),
        max_tokens: Some(256),
        temperature: Some(0.2),
        ..Default::default()
    };

    println!("=== DeepSeek via Anthropic-compatible endpoint ===");
    println!("model    : {model}");
    println!("endpoint : {}/v1/messages\n", base.trim_end_matches('/'));

    match provider.complete(request).await {
        Ok(resp) => {
            println!("stop_reason : {:?}", resp.stop_reason);
            println!(
                "usage       : input={} output={} total={}",
                resp.usage.input, resp.usage.output, resp.usage.total_tokens
            );
            println!("content     :");
            for part in &resp.content {
                if let ContentPart::Text { text } = part {
                    println!("  {text}");
                }
            }
        }
        Err(e) => {
            eprintln!("request failed: {e}");
            std::process::exit(1);
        }
    }
}
