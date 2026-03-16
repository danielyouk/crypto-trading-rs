use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub exchange: ExchangeConfig,
    pub strategy: StrategyConfig,
    pub risk: RiskConfig,
    pub database: DatabaseConfig,
    pub monitoring: MonitoringConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ExchangeConfig {
    pub name: String,
    pub api_key: String,
    pub api_secret: String,
    pub paper_trading: bool,
    pub base_url: Option<String>,
    pub ws_url: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct StrategyConfig {
    pub name: String,
    pub symbols: Vec<String>,
    pub timeframe: String,
    pub params: std::collections::HashMap<String, f64>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RiskConfig {
    pub max_position_size: f64,
    pub max_drawdown_pct: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    pub max_open_positions: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct DatabaseConfig {
    pub path: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct MonitoringConfig {
    pub enabled: bool,
    pub telegram_token: Option<String>,
    pub telegram_chat_id: Option<String>,
    pub discord_webhook: Option<String>,
}

impl AppConfig {
    pub fn load() -> anyhow::Result<Self> {
        dotenvy::dotenv().ok();

        let config = config::Config::builder()
            .add_source(config::File::with_name("config/default").required(false))
            .add_source(config::File::with_name("config/local").required(false))
            .add_source(config::Environment::with_prefix("CRYPTO"))
            .build()?;

        Ok(config.try_deserialize()?)
    }
}
