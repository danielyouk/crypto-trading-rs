pub mod live;
pub mod simulated;

use anyhow::Result;
use rust_decimal::Decimal;

/// Unified interface for order execution (live, paper, simulated).
#[async_trait::async_trait]
pub trait Executor: Send + Sync {
    async fn place_order(&self, order: &Order) -> Result<OrderResult>;
    async fn cancel_order(&self, order_id: &str) -> Result<()>;
    async fn get_position(&self, symbol: &str) -> Result<Option<Position>>;
    async fn get_balance(&self) -> Result<Balance>;
}

#[derive(Debug, Clone)]
pub struct Order {
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: Decimal,
    pub price: Option<Decimal>,
}

#[derive(Debug, Clone)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone)]
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
}

#[derive(Debug, Clone)]
pub struct OrderResult {
    pub order_id: String,
    pub status: String,
    pub filled_price: Option<Decimal>,
    pub filled_quantity: Option<Decimal>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: Decimal,
    pub entry_price: Decimal,
    pub unrealized_pnl: Decimal,
}

#[derive(Debug, Clone)]
pub struct Balance {
    pub total: Decimal,
    pub available: Decimal,
    pub currency: String,
}
