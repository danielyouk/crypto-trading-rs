use anyhow::Result;
use super::{MarketSnapshot, Signal, Strategy};

/// Simple momentum / trend-following strategy.
pub struct MomentumStrategy {
    pub symbol: String,
    pub fast_period: usize,
    pub slow_period: usize,
    price_history: Vec<f64>,
}

impl MomentumStrategy {
    pub fn new(symbol: &str, fast_period: usize, slow_period: usize) -> Self {
        Self {
            symbol: symbol.to_string(),
            fast_period,
            slow_period,
            price_history: Vec::new(),
        }
    }

    fn sma(&self, period: usize) -> Option<f64> {
        if self.price_history.len() < period {
            return None;
        }
        let window = &self.price_history[self.price_history.len() - period..];
        Some(window.iter().sum::<f64>() / period as f64)
    }
}

impl Strategy for MomentumStrategy {
    fn name(&self) -> &str {
        "momentum"
    }

    fn evaluate(&mut self, _market: &MarketSnapshot) -> Result<Signal> {
        // TODO: push new price into price_history from market data
        let (fast, slow) = match (self.sma(self.fast_period), self.sma(self.slow_period)) {
            (Some(f), Some(s)) => (f, s),
            _ => return Ok(Signal::Hold),
        };

        if fast > slow {
            Ok(Signal::Buy {
                symbol: self.symbol.clone(),
                strength: (fast - slow) / slow,
            })
        } else if fast < slow {
            Ok(Signal::Sell {
                symbol: self.symbol.clone(),
                strength: (slow - fast) / slow,
            })
        } else {
            Ok(Signal::Hold)
        }
    }
}
