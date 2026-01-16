"""
Script to load and validate historical data.

Usage:
    python scripts/load_historical_data.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.historical_loader import HistoricalDataLoader
from data.storage import DataStorage
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Load and validate all historical data."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    db_path = project_root / "synthetic_bar_generator" / "data" / "synthetic_bars.db"
    
    # Initialize loader and storage
    loader = HistoricalDataLoader(str(data_dir))
    storage = DataStorage(str(db_path))
    
    # Instruments to load
    symbols = ['ES', 'NQ', 'GC', 'SI']
    
    logger.info(f"Loading historical data for {symbols}")
    
    # Load and validate each instrument
    for symbol in symbols:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {symbol}")
            logger.info(f"{'='*60}")
            
            # Load data
            df = loader.load_instrument(symbol)
            
            # Validate data quality
            quality_report = loader.validate_data_quality(df, symbol)
            
            # Print quality report
            logger.info(f"\nQuality Report for {symbol}:")
            logger.info(f"  Total bars: {quality_report['total_bars']:,}")
            logger.info(f"  Date range: {quality_report['date_range']['start']} to {quality_report['date_range']['end']}")
            logger.info(f"  Invalid OHLC: {quality_report['invalid_ohlc_count']}")
            logger.info(f"  Zero volume bars: {quality_report['zero_volume_count']}")
            logger.info(f"  Missing bars: {quality_report['missing_bars_count']}")
            if 'missing_bar_percentage' in quality_report:
                logger.info(f"  Missing bar %: {quality_report['missing_bar_percentage']:.2f}%")
            logger.info(f"  Large gaps (>5%): {quality_report['large_gap_count']}")
            
            if quality_report['issues']:
                logger.warning(f"  Issues found: {', '.join(quality_report['issues'])}")
            else:
                logger.info("  ✓ No major issues detected")
            
            # Store in database
            n_stored = storage.store_bars(symbol, df)
            logger.info(f"  ✓ Stored {n_stored:,} bars in database")
            
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}", exc_info=True)
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info("Data loading complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
