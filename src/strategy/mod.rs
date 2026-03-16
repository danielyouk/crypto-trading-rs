pub mod pairs;
pub mod momentum;

use anyhow::Result;
use rust_decimal::Decimal;

/// Every trading strategy implements this trait.
pub trait Strategy: Send + Sync {
    fn name(&self) -> &str;

    /// Evaluate current market state and produce a signal.
    fn evaluate(&mut self, market: &MarketSnapshot) -> Result<Signal>;
}

#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    pub symbol: String,
    pub price: Decimal,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub klines: Vec<crate::data::stream::Kline>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Signal {
    Buy { symbol: String, strength: f64 },
    Sell { symbol: String, strength: f64 },
    Hold,
}
