use anyhow::Result;
use super::{MarketSnapshot, Signal, Strategy};

/// Statistical arbitrage pairs trading strategy.
/// Ported from the PairsTradingAutomation Python project,
/// adapted for crypto markets.
pub struct PairsStrategy {
    pub symbol_a: String,
    pub symbol_b: String,
    pub lookback_period: usize,
    pub entry_z_score: f64,
    pub exit_z_score: f64,
    spread_history: Vec<f64>,
}

impl PairsStrategy {
    pub fn new(
        symbol_a: &str,
        symbol_b: &str,
        lookback_period: usize,
        entry_z_score: f64,
        exit_z_score: f64,
    ) -> Self {
        Self {
            symbol_a: symbol_a.to_string(),
            symbol_b: symbol_b.to_string(),
            lookback_period,
            entry_z_score,
            exit_z_score,
            spread_history: Vec::new(),
        }
    }

    fn calculate_z_score(&self) -> Option<f64> {
        if self.spread_history.len() < self.lookback_period {
            return None;
        }

        let window = &self.spread_history[self.spread_history.len() - self.lookback_period..];
        let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
        let variance: f64 =
            window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return None;
        }

        let current = *self.spread_history.last()?;
        Some((current - mean) / std_dev)
    }
}

impl Strategy for PairsStrategy {
    fn name(&self) -> &str {
        "pairs"
    }

    fn evaluate(&mut self, _market: &MarketSnapshot) -> Result<Signal> {
        // TODO: update spread_history from market data
        match self.calculate_z_score() {
            Some(z) if z > self.entry_z_score => Ok(Signal::Sell {
                symbol: self.symbol_a.clone(),
                strength: z.abs(),
            }),
            Some(z) if z < -self.entry_z_score => Ok(Signal::Buy {
                symbol: self.symbol_a.clone(),
                strength: z.abs(),
            }),
            _ => Ok(Signal::Hold),
        }
    }
}
