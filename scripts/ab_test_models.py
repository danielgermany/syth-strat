"""
A/B Testing Framework: Compare GARCH vs A-GARCH forecast performance.

This script performs walk-forward validation to compare symmetric GARCH
and asymmetric A-GARCH (GJR-GARCH) models using probabilistic forecast metrics:
- CRPS (Continuous Ranked Probability Score)
- PIT (Probability Integral Transform) uniformity
- Return autocorrelation (should be near zero for both)
- Volatility clustering (squared returns autocorrelation)

Reference: monte_carlo_architecture.md Section 14 (Probabilistic Forecast Metrics)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import sys
from scipy.stats import kstest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.garch import GARCHModel
from src.models.agarch import AGARCHModel
from src.models.regime_config import Instrument
from src.data.storage import DataStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelComparison:
    """Results of model comparison."""
    instrument: str
    garch_crps: float
    agarch_crps: float
    crps_improvement: float  # Percentage improvement (negative = A-GARCH worse)
    pit_uniformity_garch: float  # Kolmogorov-Smirnov test statistic (lower = more uniform)
    pit_uniformity_agarch: float
    garch_autocorr_returns: float
    agarch_autocorr_returns: float
    garch_autocorr_squared: float
    agarch_autocorr_squared: float


def crps(forecast_samples: np.ndarray, observed: float) -> float:
    """
    Continuous Ranked Probability Score (CRPS).
    
    CRPS = E|X - y| - (1/2)E|X - X'|
    Where X, X' are independent samples from forecast distribution,
    and y is the observed value.
    
    Lower CRPS = better forecast.
    
    Args:
        forecast_samples: Array of forecast samples (e.g., final prices from paths)
        observed: Observed value (e.g., realized final price)
        
    Returns:
        CRPS value
    """
    samples = forecast_samples.flatten()
    n = len(samples)
    
    if n == 0:
        return np.inf
    
    # First term: E|X - y|
    term1 = np.mean(np.abs(samples - observed))
    
    # Second term: (1/2)E|X - X'|
    # Approximate using all pairs
    if n > 100:
        # For large n, sample pairs
        indices = np.random.choice(n, size=min(1000, n * 10), replace=True)
        samples1 = samples[np.random.choice(n, size=len(indices))]
        samples2 = samples[np.random.choice(n, size=len(indices))]
        term2 = 0.5 * np.mean(np.abs(samples1 - samples2))
    else:
        # For small n, compute all pairs
        samples1 = samples[:, np.newaxis]
        samples2 = samples[np.newaxis, :]
        term2 = 0.5 * np.mean(np.abs(samples1 - samples2))
    
    return term1 - term2


def pit_uniformity(forecast_samples: np.ndarray, observed: float) -> float:
    """
    Probability Integral Transform (PIT) uniformity test.
    
    Computes PIT value: p = F_forecast(observed_value)
    Tests if p values are uniform (good forecast) vs non-uniform (poor forecast).
    
    Uses Kolmogorov-Smirnov test statistic (distance from uniform CDF).
    Lower = more uniform = better forecast.
    
    Args:
        forecast_samples: Array of forecast samples
        observed: Observed value
        
    Returns:
        KS test statistic (0 = perfectly uniform, 1 = worst)
    """
    samples = forecast_samples.flatten()
    n = len(samples)
    
    if n == 0:
        return 1.0
    
    # Compute empirical CDF at observed value
    pit_value = np.mean(samples <= observed)
    
    # KS statistic: distance from uniform distribution
    # For a single observation, we compute distance from uniform
    # We'll accumulate PIT values across multiple observations in the main function
    # For now, return the PIT value itself (will be aggregated)
    return pit_value


def compute_stylized_facts(returns: np.ndarray) -> Dict[str, float]:
    """
    Compute stylized facts for returns.
    
    Args:
        returns: (n_paths, n_bars) array of returns
        
    Returns:
        Dictionary with stylized facts
    """
    if returns.ndim == 1:
        returns = returns.reshape(1, -1)
    
    # Return autocorrelation (should be near zero)
    autocorr_returns_list = []
    for path in returns:
        path_clean = path[~np.isnan(path)]
        if len(path_clean) > 2:
            ac = pd.Series(path_clean).autocorr(lag=1)
            if not np.isnan(ac):
                autocorr_returns_list.append(ac)
    autocorr_returns = float(np.mean(autocorr_returns_list)) if autocorr_returns_list else 0.0
    
    # Squared returns autocorrelation (volatility clustering, should be positive)
    squared_returns = returns ** 2
    autocorr_sq_list = []
    for path in squared_returns:
        path_clean = path[~np.isnan(path)]
        if len(path_clean) > 2:
            ac = pd.Series(path_clean).autocorr(lag=1)
            if not np.isnan(ac):
                autocorr_sq_list.append(ac)
    autocorr_sq = float(np.mean(autocorr_sq_list)) if autocorr_sq_list else 0.0
    
    return {
        'autocorr_returns': autocorr_returns,
        'autocorr_squared_returns': autocorr_sq,
    }


def compare_models(
    instrument: Instrument,
    data: pd.DataFrame,
    test_window_days: int = 30,
    train_window_days: int = 90,
    n_paths: int = 1000,
    horizon_bars: int = 60,
    random_state: Optional[int] = None
) -> ModelComparison:
    """
    Run walk-forward comparison of GARCH vs A-GARCH.
    
    Args:
        instrument: Instrument to test
        data: Historical data with 'close' column
        test_window_days: Number of days for out-of-sample testing
        train_window_days: Number of days for in-sample training
        n_paths: Number of Monte Carlo paths per forecast
        horizon_bars: Forecast horizon (number of 1-minute bars)
        random_state: Random seed for reproducibility
        
    Returns:
        ModelComparison with metrics
    """
    logger.info(f"Comparing GARCH vs A-GARCH for {instrument.value}")
    
    # Calculate returns
    returns = np.diff(np.log(data['close'].values))
    
    # Set up walk-forward windows
    bars_per_day = 23 * 60  # Futures trading hours (1-min bars)
    train_bars = train_window_days * bars_per_day
    test_bars = test_window_days * bars_per_day
    
    if len(data) < train_bars + test_bars:
        logger.warning(f"Insufficient data: need {train_bars + test_bars} bars, got {len(data)}")
        test_bars = len(data) - train_bars
    
    # Storage for forecast metrics
    crps_garch_list = []
    crps_agarch_list = []
    pit_garch_list = []
    pit_agarch_list = []
    
    garch_returns_all = []
    agarch_returns_all = []
    
    # Walk-forward validation
    n_windows = min(5, test_bars // horizon_bars)  # Test up to 5 windows
    test_step = max(1, test_bars // n_windows // horizon_bars)  # Steps between windows
    
    logger.info(f"Running walk-forward test: {n_windows} windows, {horizon_bars} bars forecast horizon")
    
    for window_idx in range(n_windows):
        test_start = train_bars + window_idx * test_step * horizon_bars
        test_end = min(test_start + horizon_bars, len(data) - 1)
        
        if test_end - test_start < 10:
            break
        
        # Training data (up to test_start)
        train_returns = returns[:test_start]
        test_returns = returns[test_start:test_end]
        test_prices = data['close'].values[test_start:test_end+1]
        
        # Fit models
        try:
            garch_model = GARCHModel()
            garch_params = garch_model.fit(train_returns, method='mle')
            
            agarch_model = AGARCHModel()
            agarch_params = agarch_model.fit(train_returns, method='mle')
        except Exception as e:
            logger.warning(f"Model fitting failed for window {window_idx}: {e}")
            continue
        
        # Generate forecasts
        initial_price = test_prices[0]
        initial_variance = np.var(train_returns[-100:])  # Use recent variance
        initial_variance = max(initial_variance, 1e-8)
        last_return = train_returns[-1] if len(train_returns) > 0 else 0.0
        
        # Forecast with GARCH
        try:
            garch_prices, _ = garch_model.simulate_paths(
                n_paths=n_paths,
                horizon=len(test_prices) - 1,
                initial_price=initial_price,
                initial_variance=initial_variance,
                last_return=last_return,
                random_state=random_state + window_idx if random_state else None
            )
            garch_forecasts = garch_prices[:, -1]  # Final prices
            
            # Forecast with A-GARCH
            agarch_prices, _ = agarch_model.simulate_paths(
                n_paths=n_paths,
                horizon=len(test_prices) - 1,
                initial_price=initial_price,
                initial_variance=initial_variance,
                last_return=last_return,
                random_state=random_state + window_idx + 1000 if random_state else None
            )
            agarch_forecasts = agarch_prices[:, -1]  # Final prices
            
            # Compute CRPS for final price
            observed_final = test_prices[-1]
            crps_g = crps(garch_forecasts, observed_final)
            crps_a = crps(agarch_forecasts, observed_final)
            
            crps_garch_list.append(crps_g)
            crps_agarch_list.append(crps_a)
            
            # Compute PIT
            pit_g = pit_uniformity(garch_forecasts, observed_final)
            pit_a = pit_uniformity(agarch_forecasts, observed_final)
            
            pit_garch_list.append(pit_g)
            pit_agarch_list.append(pit_a)
            
            # Collect returns for stylized facts
            garch_returns = np.diff(np.log(garch_prices), axis=1)
            agarch_returns = np.diff(np.log(agarch_prices), axis=1)
            garch_returns_all.append(garch_returns)
            agarch_returns_all.append(agarch_returns)
            
        except Exception as e:
            logger.warning(f"Forecast generation failed for window {window_idx}: {e}")
            continue
    
    # Aggregate results
    if not crps_garch_list:
        logger.error("No successful forecasts generated")
        return ModelComparison(
            instrument=instrument.value,
            garch_crps=np.inf,
            agarch_crps=np.inf,
            crps_improvement=0.0,
            pit_uniformity_garch=1.0,
            pit_uniformity_agarch=1.0,
            garch_autocorr_returns=0.0,
            agarch_autocorr_returns=0.0,
            garch_autocorr_squared=0.0,
            agarch_autocorr_squared=0.0,
        )
    
    # Compute PIT uniformity using KS test on collected PIT values
    pit_uniformity_garch = kstest(pit_garch_list, 'uniform')[0] if pit_garch_list else 1.0
    pit_uniformity_agarch = kstest(pit_agarch_list, 'uniform')[0] if pit_agarch_list else 1.0
    
    # Aggregate returns for stylized facts
    garch_all_returns = np.concatenate(garch_returns_all, axis=0) if garch_returns_all else np.array([])
    agarch_all_returns = np.concatenate(agarch_returns_all, axis=0) if agarch_returns_all else np.array([])
    
    garch_facts = compute_stylized_facts(garch_all_returns) if len(garch_all_returns) > 0 else {}
    agarch_facts = compute_stylized_facts(agarch_all_returns) if len(agarch_all_returns) > 0 else {}
    
    avg_crps_garch = np.mean(crps_garch_list)
    avg_crps_agarch = np.mean(crps_agarch_list)
    improvement = ((avg_crps_garch - avg_crps_agarch) / avg_crps_garch) * 100 if avg_crps_garch > 0 else 0.0
    
    return ModelComparison(
        instrument=instrument.value,
        garch_crps=avg_crps_garch,
        agarch_crps=avg_crps_agarch,
        crps_improvement=improvement,
        pit_uniformity_garch=pit_uniformity_garch,
        pit_uniformity_agarch=pit_uniformity_agarch,
        garch_autocorr_returns=garch_facts.get('autocorr_returns', 0.0),
        agarch_autocorr_returns=agarch_facts.get('autocorr_returns', 0.0),
        garch_autocorr_squared=garch_facts.get('autocorr_squared_returns', 0.0),
        agarch_autocorr_squared=agarch_facts.get('autocorr_squared_returns', 0.0),
    )


def main():
    """Run A/B testing for all instruments."""
    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "synthetic_bars.db"
    
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return
    
    storage = DataStorage(str(db_path))
    instruments = [Instrument.ES, Instrument.NQ, Instrument.GC, Instrument.SI]
    
    results = {}
    
    for instrument in instruments:
        logger.info(f"\n{'='*70}")
        logger.info(f"A/B Testing: {instrument.value}")
        logger.info(f"{'='*70}")
        
        # Load data
        df = storage.load_bars(instrument.value)
        if df.empty:
            logger.warning(f"No data found for {instrument.value}")
            continue
        
        # Compare models
        comparison = compare_models(
            instrument=instrument,
            data=df,
            test_window_days=30,
            train_window_days=90,
            n_paths=1000,
            horizon_bars=60,
            random_state=42
        )
        
        results[instrument] = comparison
        
        # Print results
        logger.info(f"\n{instrument.value} Comparison Results:")
        logger.info(f"  GARCH CRPS: {comparison.garch_crps:.6f}")
        logger.info(f"  A-GARCH CRPS: {comparison.agarch_crps:.6f}")
        logger.info(f"  CRPS Improvement: {comparison.crps_improvement:+.2f}%")
        logger.info(f"  PIT Uniformity (GARCH): {comparison.pit_uniformity_garch:.4f} (lower is better)")
        logger.info(f"  PIT Uniformity (A-GARCH): {comparison.pit_uniformity_agarch:.4f} (lower is better)")
        logger.info(f"  Return AC (GARCH): {comparison.garch_autocorr_returns:.4f}")
        logger.info(f"  Return AC (A-GARCH): {comparison.agarch_autocorr_returns:.4f}")
        logger.info(f"  Squared Return AC (GARCH): {comparison.garch_autocorr_squared:.4f}")
        logger.info(f"  Squared Return AC (A-GARCH): {comparison.agarch_autocorr_squared:.4f}")
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("Summary")
    logger.info(f"{'='*70}")
    
    for instrument, comparison in results.items():
        winner = "A-GARCH" if comparison.crps_improvement > 0 else "GARCH"
        logger.info(f"{instrument.value}: {winner} wins (CRPS improvement: {comparison.crps_improvement:+.2f}%)")
    
    return results


if __name__ == "__main__":
    main()
