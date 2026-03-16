use rust_decimal::Decimal;
use super::{OrderResult, Position};

/// In-memory simulated exchange for backtesting.
/// No network calls - processes orders against historical data.
pub struct SimulatedExecutor {
    _balance: Decimal,
    _positions: std::collections::HashMap<String, Position>,
    trade_log: Vec<OrderResult>,
}

impl SimulatedExecutor {
    pub fn new(initial_balance: Decimal) -> Self {
        Self {
            _balance: initial_balance,
            _positions: std::collections::HashMap::new(),
            trade_log: Vec::new(),
        }
    }

    pub fn trade_history(&self) -> &[OrderResult] {
        &self.trade_log
    }
}

// TODO: impl Executor for SimulatedExecutor
