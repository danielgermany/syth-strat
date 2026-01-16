"""
Script to verify data integrity by checking timestamp uniqueness.

Usage:
    python scripts/verify_data_integrity.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from data.historical_loader import HistoricalDataLoader
from data.storage import DataStorage
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def check_csv_timestamp_uniqueness(data_dir: Path, symbol: str) -> dict:
    """Check timestamp uniqueness in CSV file."""
    csv_file = data_dir / f"{symbol.lower()}_1min.csv"
    
    if not csv_file.exists():
        return {'error': f'File not found: {csv_file}'}
    
    df = pd.read_csv(csv_file)
    
    # Normalize column name
    time_col = 'time' if 'time' in df.columns else 'timestamp'
    
    if time_col not in df.columns:
        return {'error': f'Timestamp column not found in {symbol}'}
    
    timestamps = df[time_col]
    total_rows = len(df)
    unique_timestamps = timestamps.nunique()
    duplicates = total_rows - unique_timestamps
    
    # Find duplicate timestamps
    duplicate_timestamps = timestamps[timestamps.duplicated(keep=False)].unique()
    
    result = {
        'total_rows': total_rows,
        'unique_timestamps': unique_timestamps,
        'duplicates': duplicates,
        'duplicate_count': len(duplicate_timestamps),
        'is_valid': duplicates == 0
    }
    
    if duplicates > 0:
        result['duplicate_timestamps'] = sorted(duplicate_timestamps.tolist())[:10]  # Show first 10
    
    return result


def check_database_timestamp_uniqueness(storage: DataStorage, symbol: str) -> dict:
    """Check timestamp uniqueness in database."""
    import sqlite3
    
    conn = sqlite3.connect(storage.db_path)
    
    # Count total rows
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM bars WHERE symbol = ?", (symbol,))
    total_rows = cursor.fetchone()[0]
    
    # Count unique timestamps
    cursor.execute("SELECT COUNT(DISTINCT timestamp) FROM bars WHERE symbol = ?", (symbol,))
    unique_timestamps = cursor.fetchone()[0]
    
    # Find duplicate timestamps
    cursor.execute("""
        SELECT timestamp, COUNT(*) as count 
        FROM bars 
        WHERE symbol = ? 
        GROUP BY timestamp 
        HAVING COUNT(*) > 1
        ORDER BY timestamp
        LIMIT 10
    """, (symbol,))
    duplicate_records = cursor.fetchall()
    
    duplicates = total_rows - unique_timestamps
    duplicate_timestamps = [row[0] for row in duplicate_records]
    
    conn.close()
    
    return {
        'total_rows': total_rows,
        'unique_timestamps': unique_timestamps,
        'duplicates': duplicates,
        'duplicate_count': len(duplicate_timestamps),
        'duplicate_timestamps': duplicate_timestamps,
        'is_valid': duplicates == 0
    }


def main():
    """Verify data integrity for all instruments."""
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    db_path = project_root / "data" / "synthetic_bars.db"
    
    # Initialize storage
    storage = DataStorage(str(db_path))
    
    # Instruments to check
    symbols = ['ES', 'NQ', 'GC', 'SI']
    
    logger.info("=" * 70)
    logger.info("Data Integrity Verification - Timestamp Uniqueness")
    logger.info("=" * 70)
    
    all_valid = True
    
    for symbol in symbols:
        logger.info(f"\n{'-' * 70}")
        logger.info(f"Checking {symbol}")
        logger.info(f"{'-' * 70}")
        
        # Check CSV file
        logger.info(f"\n[CSV File] {data_dir / f'{symbol.lower()}_1min.csv'}")
        csv_result = check_csv_timestamp_uniqueness(data_dir, symbol)
        
        if 'error' in csv_result:
            logger.error(f"  ERROR: {csv_result['error']}")
            all_valid = False
            continue
        
        logger.info(f"  Total rows: {csv_result['total_rows']:,}")
        logger.info(f"  Unique timestamps: {csv_result['unique_timestamps']:,}")
        logger.info(f"  Duplicates: {csv_result['duplicates']}")
        
        if csv_result['is_valid']:
            logger.info(f"  ✓ VALID - All timestamps are unique")
        else:
            logger.error(f"  ✗ INVALID - Found {csv_result['duplicates']} duplicate timestamps")
            if 'duplicate_timestamps' in csv_result:
                logger.warning(f"  Sample duplicates: {csv_result['duplicate_timestamps'][:5]}")
            all_valid = False
        
        # Check database
        logger.info(f"\n[Database] {db_path}")
        db_result = check_database_timestamp_uniqueness(storage, symbol)
        
        logger.info(f"  Total rows: {db_result['total_rows']:,}")
        logger.info(f"  Unique timestamps: {db_result['unique_timestamps']:,}")
        logger.info(f"  Duplicates: {db_result['duplicates']}")
        
        if db_result['is_valid']:
            logger.info(f"  ✓ VALID - All timestamps are unique")
        else:
            logger.error(f"  ✗ INVALID - Found {db_result['duplicates']} duplicate timestamps")
            if db_result['duplicate_timestamps']:
                logger.warning(f"  Sample duplicates: {db_result['duplicate_timestamps'][:5]}")
            all_valid = False
        
        # Compare CSV vs Database
        if csv_result['total_rows'] != db_result['total_rows']:
            logger.warning(f"\n  ⚠ WARNING: Row count mismatch!")
            logger.warning(f"     CSV: {csv_result['total_rows']:,} rows")
            logger.warning(f"     DB:  {db_result['total_rows']:,} rows")
            logger.warning(f"     Difference: {abs(csv_result['total_rows'] - db_result['total_rows']):,} rows")
    
    logger.info(f"\n{'=' * 70}")
    if all_valid:
        logger.info("✓ ALL CHECKS PASSED - Data integrity verified!")
    else:
        logger.error("✗ SOME CHECKS FAILED - Data integrity issues found!")
    logger.info(f"{'=' * 70}\n")
    
    return all_valid


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
