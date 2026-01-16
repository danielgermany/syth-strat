"""
Calibration script for regime-switching GARCH models.

This script loads historical data from the database, calibrates regime-switching
GARCH models for all instruments, and saves the calibration results.

Usage:
    python scripts/calibrate_models.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from datetime import datetime
import logging
import sqlite3

from models.regime_config import Instrument, RegimeConfig, TimeframeConfig, Timeframe
from models.regime_calibration import RegimeSwitchingCalibrator
from data.storage import DataStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_historical_data(db_path: str, instruments: list[Instrument]) -> dict[Instrument, pd.DataFrame]:
    """
    Load historical data from database for all instruments.
    
    Args:
        db_path: Path to SQLite database
        instruments: List of instruments to load
        
    Returns:
        Dict mapping Instrument to DataFrame with 'close' and 'volume' columns
    """
    conn = sqlite3.connect(db_path)
    
    data = {}
    for instrument in instruments:
        symbol = instrument.value
        
        query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM bars
            WHERE symbol = ?
            ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol,))
        
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            continue
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('timestamp')
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing required columns for {symbol}")
            continue
        
        data[instrument] = df
        
        logger.info(f"Loaded {len(df):,} bars for {symbol}")
    
    conn.close()
    
    return data


def save_calibration_results(
    calibrator: RegimeSwitchingCalibrator,
    db_path: str
) -> None:
    """
    Save calibration results to database.
    
    Args:
        calibrator: Calibrated RegimeSwitchingCalibrator instance
        db_path: Path to SQLite database
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    calibration_timestamp = int(calibrator.calibration_date.timestamp())
    
    # Save regime-specific GARCH parameters
    # Note: regime is included in param_name (e.g., 'omega_regime_0')
    # since the schema doesn't have a separate regime column
    for instrument, regime_params in calibrator.regime_params.items():
        symbol = instrument.value
        
        # Get window bounds from data
        if instrument in calibrator._last_data:  # Will be set during calibration
            window_start = int(calibrator._last_data[instrument].index[0].timestamp())
            window_end = int(calibrator._last_data[instrument].index[-1].timestamp())
        else:
            # Fallback: use calibration date
            window_start = calibration_timestamp - (90 * 24 * 60 * 60)  # 90 days back
            window_end = calibration_timestamp
        
        for regime_k in range(regime_params.n_regimes):
            params = regime_params.get_regime_params(regime_k)
            
            # Insert or update parameters
            for param_name in ['omega', 'alpha', 'beta', 'gamma', 'mu', 'nu', 'skew']:
                param_value = getattr(params, param_name)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO calibration_params
                    (symbol, model_type, param_name, param_value, calibration_date, window_start, window_end)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    'regime_garch',
                    f'{param_name}_regime_{regime_k}',
                    param_value,
                    calibration_timestamp,
                    window_start,
                    window_end
                ))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Saved calibration results to {db_path}")


def main():
    """Main calibration routine."""
    # Paths
    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "synthetic_bars.db"
    
    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        logger.error("Please run load_historical_data.py first")
        return 1
    
    # Initialize storage (to ensure tables exist)
    storage = DataStorage(str(db_path))
    storage.initialize_db()
    
    # Instruments to calibrate
    instruments = [Instrument.ES, Instrument.NQ, Instrument.GC, Instrument.SI]
    
    logger.info("=" * 70)
    logger.info("Regime-Switching GARCH Calibration")
    logger.info("=" * 70)
    logger.info(f"Database: {db_path}")
    logger.info(f"Instruments: {[inst.value for inst in instruments]}")
    logger.info("")
    
    # Load historical data
    logger.info("Loading historical data...")
    data = load_historical_data(str(db_path), instruments)
    
    if not data:
        logger.error("No data loaded. Exiting.")
        return 1
    
    # Initialize calibrator
    logger.info("Initializing calibrator...")
    calibrator = RegimeSwitchingCalibrator(
        instruments=list(data.keys()),
        regime_configs={inst: RegimeConfig.default_for_instrument(inst) 
                       for inst in data.keys()},
        timeframe_config=TimeframeConfig.default_for_timeframe(Timeframe.M1)
    )
    
    # Run calibration
    logger.info("Running calibration...")
    calibrator.calibrate_all(data, as_of_date=datetime.now())
    
    # Save results
    logger.info("Saving calibration results...")
    save_calibration_results(calibrator, str(db_path))
    
    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("Calibration Summary")
    logger.info("=" * 70)
    
    for instrument in calibrator.regime_params.keys():
        regime_params = calibrator.regime_params[instrument]
        logger.info(f"\n{instrument.value}:")
        logger.info(f"  Regimes: {regime_params.n_regimes}")
        
        for k in range(regime_params.n_regimes):
            params = regime_params.get_regime_params(k)
            logger.info(
                f"  Regime {k}: "
                f"ω={params.omega:.6f}, α={params.alpha:.3f}, "
                f"β={params.beta:.3f}, γ={params.gamma:.3f}"
            )
    
    # Cross-instrument results
    if calibrator.lead_lag_results:
        logger.info("\nCross-Instrument Lead-Lag Results:")
        for (inst1, inst2), result in calibrator.lead_lag_results.items():
            logger.info(
                f"  {inst1.value}-{inst2.value}: "
                f"leader={result.leader}, "
                f"contagion={result.contagion_1_to_2:.2f}/{result.contagion_2_to_1:.2f}"
            )
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("Calibration complete!")
    logger.info("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
