"""
Historical data loader for model calibration.

Reference: monte_carlo_architecture.md Part II and synthetic_bar_generator_development_plan.md Section 2.1.1
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HistoricalDataLoader:
    """
    Loads and validates historical bar data for model calibration.
    
    Reference: synthetic_bar_generator_development_plan.md Section 2.1.1
    """
    
    REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    def __init__(self, data_dir: str):
        """
        Initialize historical data loader.
        
        Args:
            data_dir: Directory containing historical CSV files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
    def load_instrument(
        self, 
        symbol: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load historical data for a single instrument.
        
        Args:
            symbol: Instrument symbol (ES, NQ, GC, SI)
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        # Determine CSV file name (lowercase symbol)
        csv_file = self.data_dir / f"{symbol.lower()}_1min.csv"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"Data file not found: {csv_file}")
        
        # Load CSV file
        logger.info(f"Loading {symbol} data from {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Normalize column names (handle 'Volume' vs 'volume', 'time' vs 'timestamp')
        column_mapping = {}
        if 'time' in df.columns:
            column_mapping['time'] = 'timestamp'
        if 'Volume' in df.columns:
            column_mapping['Volume'] = 'volume'
        
        df = df.rename(columns=column_mapping)
        
        # Validate required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp to datetime
        if df['timestamp'].dtype in ['int64', 'float64']:
            # Unix timestamp (seconds)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Filter by date range if provided
        if start_date:
            start_dt = pd.to_datetime(start_date, utc=True)
            df = df[df.index >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date, utc=True)
            df = df[df.index <= end_dt]
        
        # Validate OHLC relationships
        self._validate_ohlc(df, symbol)
        
        # Remove rows with invalid OHLC
        invalid_mask = (
            (df['high'] < df['open']) | 
            (df['high'] < df['close']) | 
            (df['low'] > df['open']) | 
            (df['low'] > df['close']) |
            (df['high'] < df['low'])
        )
        if invalid_mask.any():
            n_invalid = invalid_mask.sum()
            logger.warning(f"{symbol}: Removing {n_invalid} rows with invalid OHLC relationships")
            df = df[~invalid_mask]
        
        logger.info(f"Loaded {len(df)} bars for {symbol} from {df.index.min()} to {df.index.max()}")
        
        return df
    
    def _validate_ohlc(self, df: pd.DataFrame, symbol: str) -> None:
        """
        Validate OHLC relationships.
        
        Checks:
        - high >= open, close, low
        - low <= open, close, high
        """
        invalid = (
            (df['high'] < df['open']) | 
            (df['high'] < df['close']) | 
            (df['low'] > df['open']) | 
            (df['low'] > df['close']) |
            (df['high'] < df['low'])
        )
        if invalid.any():
            n_invalid = invalid.sum()
            logger.warning(f"{symbol}: Found {n_invalid} bars with invalid OHLC relationships")
    
    def validate_data_quality(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Run data quality checks and return report.
        
        Checks:
        - Missing bar percentage (should be <1% during RTH)
        - OHLC validity (high >= all, low <= all)
        - Volume anomalies (zero volume bars)
        - Price continuity (gaps > 5% flagged)
        - Timestamp continuity
        
        Returns:
            Dictionary with quality metrics
        """
        report = {
            'symbol': symbol,
            'total_bars': len(df),
            'date_range': {
                'start': df.index.min(),
                'end': df.index.max()
            },
            'issues': []
        }
        
        # Check OHLC validity
        invalid_ohlc = (
            (df['high'] < df['open']) | 
            (df['high'] < df['close']) | 
            (df['low'] > df['open']) | 
            (df['low'] > df['close']) |
            (df['high'] < df['low'])
        ).sum()
        report['invalid_ohlc_count'] = invalid_ohlc
        
        # Check for zero volume
        zero_volume = (df['volume'] == 0).sum()
        report['zero_volume_count'] = zero_volume
        if zero_volume > 0:
            report['issues'].append(f"{zero_volume} bars with zero volume")
        
        # Check timestamp continuity
        # Expected: 1-minute intervals
        time_diffs = df.index.to_series().diff().dt.total_seconds() / 60
        expected_interval = 1.0  # 1 minute
        gaps = time_diffs[(time_diffs > expected_interval * 1.5) & (time_diffs.notna())]
        report['missing_bars_count'] = len(gaps)
        if len(gaps) > 0:
            gap_pct = len(gaps) / len(df) * 100
            report['missing_bar_percentage'] = gap_pct
            if gap_pct > 1.0:
                report['issues'].append(f"Missing bar percentage {gap_pct:.2f}% exceeds 1% threshold")
        
        # Check for large price gaps (>5%)
        returns = df['close'].pct_change()
        large_gaps = (returns.abs() > 0.05).sum()
        report['large_gap_count'] = large_gaps
        if large_gaps > 0:
            report['issues'].append(f"{large_gaps} bars with price gaps > 5%")
        
        # Basic statistics
        report['price_stats'] = {
            'min': df['low'].min(),
            'max': df['high'].max(),
            'mean': df['close'].mean()
        }
        report['volume_stats'] = {
            'min': df['volume'].min(),
            'max': df['volume'].max(),
            'mean': df['volume'].mean()
        }
        
        return report
    
    def load_all_instruments(
        self,
        symbols: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for multiple instruments.
        
        Args:
            symbols: List of instrument symbols
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.load_instrument(symbol, start_date, end_date)
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                raise
        
        return data
