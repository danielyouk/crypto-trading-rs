/// Live/Paper execution via exchange API (Bybit demo, Binance testnet, etc.)
pub struct LiveExecutor {
    pub exchange: String,
    pub paper_trading: bool,
}

impl LiveExecutor {
    pub fn new(exchange: &str, paper_trading: bool) -> Self {
        Self {
            exchange: exchange.to_string(),
            paper_trading,
        }
    }
}

// TODO: impl Executor for LiveExecutor using ccxt-rust or barter-execution
