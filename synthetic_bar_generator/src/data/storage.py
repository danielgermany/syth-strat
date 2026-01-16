"""
Data storage module for historical bars and calibration parameters.

Reference: synthetic_bar_generator_development_plan.md Section 2.1.2
"""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataStorage:
    """
    SQLite-based storage for historical bars and calibration parameters.
    
    Reference: synthetic_bar_generator_development_plan.md Section 2.1.2
    Development environment uses SQLite; production uses TimescaleDB.
    """
    
    def __init__(self, db_path: str = "data/synthetic_bars.db"):
        """
        Initialize data storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Historical bars table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,  -- Unix timestamp (seconds)
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                trade_count INTEGER,
                UNIQUE(symbol, timestamp)
            )
        """)
        
        # Create index
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_bars_symbol_timestamp 
            ON bars(symbol, timestamp)
        """)
        
        # Calibration parameters table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration_params (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                model_type TEXT NOT NULL,  -- 'garch', 'agarch', 'heston', etc.
                param_name TEXT NOT NULL,
                param_value REAL NOT NULL,
                calibration_date INTEGER NOT NULL,
                window_start INTEGER NOT NULL,
                window_end INTEGER NOT NULL,
                UNIQUE(symbol, model_type, param_name, calibration_date)
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Initialized database schema at {self.db_path}")
    
    def store_bars(self, symbol: str, df: pd.DataFrame) -> int:
        """
        Store historical bars in database.
        
        Args:
            symbol: Instrument symbol
            df: DataFrame with OHLCV data indexed by timestamp
            
        Returns:
            Number of bars stored
        """
        conn = sqlite3.connect(self.db_path)
        
        # Convert datetime index to unix timestamp
        df_to_store = df.copy()
        if isinstance(df_to_store.index, pd.DatetimeIndex):
            df_to_store['timestamp'] = df_to_store.index.astype('int64') // 10**9
            df_to_store = df_to_store.reset_index(drop=True)
        else:
            df_to_store['timestamp'] = df_to_store.index
            df_to_store = df_to_store.reset_index(drop=True)
        
        df_to_store['symbol'] = symbol
        
        # Fill NaN values and ensure proper data types
        df_to_store['volume'] = pd.to_numeric(df_to_store['volume'], errors='coerce').fillna(0).astype(int)
        
        # Remove rows with NaN in required OHLC columns
        df_to_store = df_to_store.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Select columns in correct order
        columns = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        if 'trade_count' in df_to_store.columns:
            columns.append('trade_count')
        
        df_to_store = df_to_store[columns]
        
        # Insert with chunking to avoid "too many SQL variables" error
        # SQLite limit is 999 variables, so chunk size should be < 999 / num_columns
        # We have 7-8 columns, so chunk size of 100 is safe
        chunk_size = 100
        n_chunks = (len(df_to_store) + chunk_size - 1) // chunk_size
        
        n_stored_total = 0
        for i in range(n_chunks):
            chunk = df_to_store.iloc[i * chunk_size:(i + 1) * chunk_size]
            # Use INSERT OR IGNORE to handle duplicates gracefully
            try:
                chunk.to_sql(
                    'bars', 
                    conn, 
                    if_exists='append', 
                    index=False,
                    method='multi'
                )
                n_stored_total += len(chunk)
            except sqlite3.IntegrityError:
                # Handle duplicates by using INSERT OR IGNORE for each row
                cursor = conn.cursor()
                for _, row in chunk.iterrows():
                    try:
                        trade_count = int(row['trade_count']) if pd.notna(row.get('trade_count')) else None
                        cursor.execute("""
                            INSERT OR IGNORE INTO bars (symbol, timestamp, open, high, low, close, volume, trade_count)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            str(row['symbol']),
                            int(row['timestamp']),
                            float(row['open']),
                            float(row['high']),
                            float(row['low']),
                            float(row['close']),
                            int(row['volume']),
                            trade_count
                        ))
                        if cursor.rowcount > 0:
                            n_stored_total += 1
                    except Exception as e:
                        logger.debug(f"Error inserting row: {e}")
                        continue
                conn.commit()
        
        conn.close()
        
        logger.info(f"Stored {n_stored_total} bars for {symbol} (skipped {len(df_to_store) - n_stored_total} duplicates)")
        return n_stored_total
    
    def load_bars(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load historical bars from database.
        
        Args:
            symbol: Instrument symbol
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT timestamp, open, high, low, close, volume, trade_count FROM bars WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            start_ts = int(start_date.timestamp())
            query += " AND timestamp >= ?"
            params.append(start_ts)
        
        if end_date:
            end_ts = int(end_date.timestamp())
            query += " AND timestamp <= ?"
            params.append(end_ts)
        
        query += " ORDER BY timestamp"
        
        df = pd.read_sql_query(query, conn, params=params)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df = df.set_index('timestamp')
        
        conn.close()
        
        logger.info(f"Loaded {len(df)} bars for {symbol} from database")
        return df
    
    def store_calibration_params(
        self,
        symbol: str,
        model_type: str,
        params: Dict[str, float],
        calibration_date: datetime,
        window_start: datetime,
        window_end: datetime
    ) -> None:
        """
        Store calibration parameters.
        
        Args:
            symbol: Instrument symbol
            model_type: Model type ('garch', 'agarch', 'heston', etc.)
            params: Dictionary of parameter name -> value
            calibration_date: Date of calibration
            window_start: Start of calibration window
            window_end: End of calibration window
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cal_date_ts = int(calibration_date.timestamp())
        window_start_ts = int(window_start.timestamp())
        window_end_ts = int(window_end.timestamp())
        
        for param_name, param_value in params.items():
            cursor.execute("""
                INSERT OR REPLACE INTO calibration_params
                (symbol, model_type, param_name, param_value, calibration_date, window_start, window_end)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (symbol, model_type, param_name, param_value, cal_date_ts, window_start_ts, window_end_ts))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored calibration parameters for {symbol} {model_type} at {calibration_date}")
    
    def load_calibration_params(
        self,
        symbol: str,
        model_type: str,
        calibration_date: Optional[datetime] = None
    ) -> Optional[Dict[str, float]]:
        """
        Load calibration parameters.
        
        Args:
            symbol: Instrument symbol
            model_type: Model type
            calibration_date: Calibration date (if None, returns most recent)
            
        Returns:
            Dictionary of parameter name -> value, or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        
        if calibration_date:
            cal_date_ts = int(calibration_date.timestamp())
            query = """
                SELECT param_name, param_value 
                FROM calibration_params 
                WHERE symbol = ? AND model_type = ? AND calibration_date = ?
            """
            params = (symbol, model_type, cal_date_ts)
        else:
            # Get most recent
            query = """
                SELECT param_name, param_value 
                FROM calibration_params 
                WHERE symbol = ? AND model_type = ? 
                AND calibration_date = (
                    SELECT MAX(calibration_date) 
                    FROM calibration_params 
                    WHERE symbol = ? AND model_type = ?
                )
            """
            params = (symbol, model_type, symbol, model_type)
        
        df = pd.read_sql_query(query, conn, params=params)
        
        conn.close()
        
        if df.empty:
            return None
        
        params_dict = dict(zip(df['param_name'], df['param_value']))
        return params_dict
