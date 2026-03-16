use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "crypto-trading-rs")]
#[command(about = "Rust-based crypto trading automation system")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// Stream real-time market data
    Stream {
        /// Exchange name (binance, bybit)
        #[arg(short, long, default_value = "binance")]
        exchange: String,
        /// Symbols to stream (e.g. BTCUSDT,ETHUSDT)
        #[arg(short, long, value_delimiter = ',')]
        symbols: Vec<String>,
    },
    /// Run backtesting on historical data
    Backtest {
        /// Strategy name (pairs, momentum)
        #[arg(short = 'S', long)]
        strategy: String,
        /// Start date (YYYY-MM-DD)
        #[arg(long)]
        start: String,
        /// End date (YYYY-MM-DD)
        #[arg(long)]
        end: String,
    },
    /// Run live/paper trading
    Trade {
        /// Strategy name
        #[arg(short = 'S', long)]
        strategy: String,
        /// Paper trading mode
        #[arg(short, long, default_value_t = true)]
        paper: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("crypto_trading_rs=info".parse()?))
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Stream { exchange, symbols } => {
            tracing::info!(%exchange, ?symbols, "Starting market data stream");
            let stream = crypto_trading_rs::data::stream::MarketStream::new(&exchange, symbols);
            stream.connect().await?;
        }
        Commands::Backtest { strategy, start, end } => {
            tracing::info!(%strategy, %start, %end, "Starting backtest");
            // TODO: wire up backtest engine
        }
        Commands::Trade { strategy, paper } => {
            let mode = if paper { "paper" } else { "live" };
            tracing::info!(%strategy, %mode, "Starting trading");
            // TODO: wire up trading engine
        }
    }

    Ok(())
}
