use rust_decimal::Decimal;

/// Risk management: position sizing, stop-loss, drawdown limits.
pub struct RiskManager {
    pub max_position_size: Decimal,
    pub max_drawdown_pct: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
    pub max_open_positions: usize,
    current_drawdown: f64,
    peak_equity: Decimal,
}

impl RiskManager {
    pub fn new(
        max_position_size: Decimal,
        max_drawdown_pct: f64,
        stop_loss_pct: f64,
        take_profit_pct: f64,
        max_open_positions: usize,
    ) -> Self {
        Self {
            max_position_size,
            max_drawdown_pct,
            stop_loss_pct,
            take_profit_pct,
            max_open_positions,
            current_drawdown: 0.0,
            peak_equity: Decimal::ZERO,
        }
    }

    /// Check if a new trade is allowed given current risk parameters.
    pub fn allow_trade(&self, open_positions: usize) -> bool {
        if open_positions >= self.max_open_positions {
            tracing::warn!("Max open positions reached: {}", self.max_open_positions);
            return false;
        }
        if self.current_drawdown >= self.max_drawdown_pct {
            tracing::warn!("Max drawdown reached: {:.2}%", self.current_drawdown * 100.0);
            return false;
        }
        true
    }

    pub fn update_equity(&mut self, current_equity: Decimal) {
        if current_equity > self.peak_equity {
            self.peak_equity = current_equity;
        }
        if self.peak_equity > Decimal::ZERO {
            let drawdown = (self.peak_equity - current_equity) / self.peak_equity;
            self.current_drawdown = drawdown
                .try_into()
                .unwrap_or(0.0);
        }
    }
}
