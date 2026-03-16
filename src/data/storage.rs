use anyhow::Result;
use rusqlite::Connection;

/// Local SQLite storage for market data and trade history.
pub struct DataStore {
    conn: Connection,
}

impl DataStore {
    pub fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        let store = Self { conn };
        store.init_tables()?;
        Ok(store)
    }

    fn init_tables(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS klines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                open_time TEXT NOT NULL,
                close_time TEXT NOT NULL,
                UNIQUE(symbol, open_time)
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                timestamp TEXT NOT NULL,
                order_id TEXT,
                strategy TEXT
            );

            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                pnl REAL,
                strategy TEXT
            );
            ",
        )?;
        Ok(())
    }
}
