use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Credential {
    ApiKey {
        provider: String,
        key: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuthStorage {
    credentials: Vec<Credential>,
}

impl AuthStorage {
    pub fn load(agent_dir: &std::path::Path) -> Self {
        let path = agent_dir.join("auth.json");
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn get_api_key(&self, provider: &str) -> Option<String> {
        let env_key = format!("{}_API_KEY", provider.to_uppercase().replace('-', "_"));
        if let Ok(v) = std::env::var(&env_key) {
            if !v.is_empty() {
                return Some(v);
            }
        }
        self.credentials.iter().find_map(|c| match c {
            Credential::ApiKey { provider: p, key } if p == provider => Some(key.clone()),
            _ => None,
        })
    }
}
