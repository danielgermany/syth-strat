"""
Test Jump-Diffusion Model.

This script validates the jump-diffusion model implementation by:
1. Testing parameter fitting from historical data
2. Generating paths and validating stylized facts
3. Verifying jump characteristics (frequency, size distribution)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.jump_diffusion import JumpDiffusionModel, JumpParams
from src.models.regime_config import Instrument
from src.data.storage import DataStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_stylized_facts(returns: np.ndarray) -> dict:
    """Compute stylized facts for returns."""
    if returns.ndim == 1:
        returns = returns.reshape(1, -1)
    
    returns_flat = returns.flatten()
    returns_flat = returns_flat[~np.isnan(returns_flat)]
    
    if len(returns_flat) < 10:
        return {}
    
    # Filter extreme outliers
    mean_ret = np.mean(returns_flat)
    std_ret = np.std(returns_flat)
    returns_filtered = returns_flat[
        np.abs(returns_flat - mean_ret) < 10 * std_ret
    ]
    
    # Kurtosis (fat tails - should be high due to jumps)
    kurtosis = float(pd.Series(returns_filtered).kurtosis()) if len(returns_filtered) > 10 else 0.0
    
    # Autocorrelation of returns (should be near zero)
    autocorr_returns_list = []
    for path in returns:
        path_clean = path[~np.isnan(path)]
        if len(path_clean) > 2:
            ac = pd.Series(path_clean).autocorr(lag=1)
            if not np.isnan(ac):
                autocorr_returns_list.append(ac)
    autocorr_returns = float(np.mean(autocorr_returns_list)) if autocorr_returns_list else 0.0
    
    # Autocorrelation of squared returns (volatility clustering)
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
        'kurtosis': kurtosis,
        'autocorr_returns': autocorr_returns,
        'autocorr_squared_returns': autocorr_sq,
        'mean_return': float(np.mean(returns_flat)),
        'std_return': float(np.std(returns_flat)),
    }


def test_jump_diffusion_model(instrument: Instrument, n_paths: int = 1000, horizon_bars: int = 60):
    """Test jump-diffusion model for a given instrument."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Testing Jump-Diffusion Model: {instrument.value}")
    logger.info(f"{'='*70}")
    
    # Load historical data
    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "synthetic_bars.db"
    storage = DataStorage(str(db_path))
    df = storage.load_bars(instrument.value)
    
    if df.empty:
        logger.warning(f"No data found for {instrument.value}")
        return None
    
    # Calculate returns
    returns = np.diff(np.log(df['close'].values))
    recent_returns = returns[-5000:]  # Use recent data for fitting
    
    # Fit jump-diffusion model
    logger.info("Fitting jump-diffusion model...")
    dt = 1.0 / (252 * 390)  # 1 minute in years
    
    try:
        jump_model = JumpDiffusionModel()
        jump_params = jump_model.fit(recent_returns, dt=dt, threshold_percentile=95)
        
        logger.info(f"Jump-diffusion parameters:")
        logger.info(f"  σ (sigma): {jump_params.sigma:.6f} (annualized)")
        logger.info(f"  λ (lambda): {jump_params.lambda_:.2f} jumps/year")
        logger.info(f"  μ_J (mu_jump): {jump_params.mu_jump:.4f}")
        logger.info(f"  σ_J (sigma_jump): {jump_params.sigma_jump:.4f}")
        logger.info(f"  μ (mu): {jump_params.mu:.8f} (annualized)")
        logger.info(f"  Expected jump size: {jump_params.expected_jump_size:.4f}")
        logger.info(f"  Total variance: {jump_params.total_variance:.6f}")
        
        # Generate paths
        logger.info("Generating paths...")
        initial_price = float(df['close'].iloc[-1])
        
        prices = jump_model.simulate_paths(
            n_paths=n_paths,
            horizon=horizon_bars,
            dt=dt,
            initial_price=initial_price,
            random_state=42
        )
        
        # Compute returns from paths
        path_returns = np.diff(np.log(prices), axis=1)
        
        # Compute stylized facts
        facts = compute_stylized_facts(path_returns)
        
        logger.info(f"\nStylized Facts (Jump-Diffusion Model):")
        logger.info(f"  Kurtosis: {facts.get('kurtosis', 0):.3f} (target: >3 for fat tails, higher with jumps)")
        logger.info(f"  Return AC(1): {facts.get('autocorr_returns', 0):.4f} (target: <0.1)")
        logger.info(f"  Squared Return AC(1): {facts.get('autocorr_squared_returns', 0):.4f} (target: >0.1)")
        logger.info(f"  Mean return: {facts.get('mean_return', 0):.8f}")
        logger.info(f"  Std return: {facts.get('std_return', 0):.6f}")
        
        # Validate
        validations = {
            'fat_tails': facts.get('kurtosis', 0) > 3.0,
            'no_return_autocorr': abs(facts.get('autocorr_returns', 1)) < 0.1,
            'volatility_clustering': facts.get('autocorr_squared_returns', 0) > 0.1,
        }
        
        logger.info(f"\nValidation:")
        logger.info(f"  Fat tails (kurtosis > 3): {'✓' if validations['fat_tails'] else '✗'}")
        logger.info(f"  No return autocorr (|AC| < 0.1): {'✓' if validations['no_return_autocorr'] else '✗'}")
        logger.info(f"  Volatility clustering (AC(sq) > 0.1): {'✓' if validations['volatility_clustering'] else '✗'}")
        
        return {
            'params': jump_params,
            'stylized_facts': facts,
            'validations': validations,
        }
        
    except Exception as e:
        logger.error(f"Error testing jump-diffusion model: {e}", exc_info=True)
        return None


def main():
    """Run jump-diffusion model tests for all instruments."""
    instruments = [Instrument.ES, Instrument.NQ, Instrument.GC, Instrument.SI]
    
    results = {}
    for instrument in instruments:
        result = test_jump_diffusion_model(instrument, n_paths=1000, horizon_bars=60)
        if result:
            results[instrument] = result
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("Summary")
    logger.info(f"{'='*70}")
    
    for instrument, result in results.items():
        validations = result['validations']
        passed = sum(validations.values())
        total = len(validations)
        logger.info(f"{instrument.value}: {passed}/{total} validations passed")
    
    return results


if __name__ == "__main__":
    main()
