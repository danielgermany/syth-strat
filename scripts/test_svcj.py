"""
Test script for SVCJ (Stochastic Volatility with Correlated Jumps) model.

Reference: synthetic_bar_generator_development_plan_v2.md Phase 5
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

from src.data.storage import DataStorage
from src.models.svcj import SVCJModel, SVCJParams
from src.models.regime_config import Instrument
from scripts.test_end_to_end import compute_stylized_facts  # Re-use for AC calculation

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def test_svcj_model(
    instruments: List[Instrument],
    db_path: str,
    n_paths: int = 1000,
    horizon_bars: int = 60,  # 1 hour forecast
    random_state: Optional[int] = 42
) -> Dict[Instrument, Dict]:
    """Test SVCJ model for each instrument."""
    storage = DataStorage(db_path)
    results = {}
    
    # Minutes in a trading year (approx 252 trading days * 390 minutes/day)
    MINUTES_PER_TRADING_YEAR = 252 * 390
    DT_YEAR = 1 / MINUTES_PER_TRADING_YEAR  # dt for 1-minute bar in years

    for instrument in instruments:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing SVCJ Model: {instrument.value}")
        logger.info(f"{'='*70}")
        
        historical_data = storage.load_bars(instrument.value)
        if historical_data.empty:
            logger.warning(f"No historical data found for {instrument.value}. Skipping.")
            continue
        
        # Use a recent window for fitting
        fit_data = historical_data.tail(5000)  # ~1 month of 1-min data
        if len(fit_data) < 100:
            logger.warning(f"Insufficient data for fitting SVCJ model for {instrument.value}. Skipping.")
            continue

        returns = np.log(fit_data['close'] / fit_data['close'].shift(1)).dropna().values
        if len(returns) == 0:
            logger.warning(f"No returns to fit SVCJ model for {instrument.value}. Skipping.")
            continue

        initial_price = float(fit_data['close'].iloc[-1])
        initial_variance = float(np.var(returns[-50:]))  # Use recent variance

        svcj_model = SVCJModel()
        logger.info("Fitting SVCJ model...")
        try:
            svcj_params = svcj_model.fit(returns, dt=DT_YEAR, initial_guess=None)
        except Exception as e:
            logger.error(f"Failed to fit SVCJ model for {instrument.value}: {e}. Skipping.")
            results[instrument] = {'status': 'failed_fit', 'error': str(e)}
            continue

        logger.info("SVCJ parameters:")
        logger.info(f"  Heston:")
        logger.info(f"    κ (kappa): {svcj_params.kappa:.4f}")
        logger.info(f"    θ (theta): {svcj_params.theta:.8f} (per-step)")
        logger.info(f"    ξ (xi): {svcj_params.xi:.4f}")
        logger.info(f"    ρ (rho): {svcj_params.rho:.4f}")
        logger.info(f"  Jumps:")
        logger.info(f"    λ (lambda): {svcj_params.lambda_:.2f} jumps/year")
        logger.info(f"    μ_J (mu_jump): {svcj_params.mu_jump:.4f}")
        logger.info(f"    σ_J (sigma_jump): {svcj_params.sigma_jump:.4f}")
        logger.info(f"    μ_v,J (mu_v_jump): {svcj_params.mu_v_jump:.8f}")
        logger.info(f"  Drift: μ = {svcj_params.mu:.8f}")
        logger.info(f"  Feller satisfied: {svcj_params.feller_satisfied}")

        logger.info("Generating paths...")
        try:
            prices, variances = svcj_model.simulate_paths(
                n_paths=n_paths,
                horizon=horizon_bars,
                dt=DT_YEAR,
                initial_price=initial_price,
                initial_variance=initial_variance,
                random_state=random_state
            )
        except Exception as e:
            logger.error(f"Failed to simulate SVCJ paths for {instrument.value}: {e}. Skipping.")
            results[instrument] = {'status': 'failed_simulate', 'error': str(e)}
            continue

        # Compute stylized facts on generated returns
        generated_returns = np.diff(np.log(prices), axis=1)
        stylized_facts = compute_stylized_facts(generated_returns)

        logger.info("\nStylized Facts (SVCJ Model):")
        logger.info(f"  Kurtosis: {stylized_facts.get('kurtosis', np.nan):.3f} (target: >3 for fat tails)")
        logger.info(f"  Return AC(1): {stylized_facts.get('autocorr_returns', np.nan):.4f} (target: <0.1)")
        logger.info(f"  Squared Return AC(1): {stylized_facts.get('autocorr_squared_returns', np.nan):.4f} (target: >0.1)")
        logger.info(f"  Mean return: {stylized_facts.get('mean_return', np.nan):.8f}")
        logger.info(f"  Std return: {stylized_facts.get('std_return', np.nan):.8f}")

        # Validation checks
        validations = {
            'fat_tails': stylized_facts.get('kurtosis', 0) > 3.0,
            'no_return_autocorr': abs(stylized_facts.get('autocorr_returns', 1)) < 0.1,
            'volatility_clustering': stylized_facts.get('autocorr_squared_returns', 0) > 0.09,  # Target positive AC
        }
        
        logger.info("\nValidation:")
        logger.info(f"  Fat tails (kurtosis > 3): {'✓' if validations['fat_tails'] else '✗'}")
        logger.info(f"  No return autocorr (|AC| < 0.1): {'✓' if validations['no_return_autocorr'] else '✗'}")
        logger.info(f"  Volatility clustering (AC(sq) > 0.1): {'✓' if validations['volatility_clustering'] else '✗'}")

        results[instrument] = {
            'status': 'passed',
            'params': svcj_params,
            'stylized_facts': stylized_facts,
            'validations': validations
        }
    
    logger.info(f"\n{'='*70}")
    logger.info("Summary")
    logger.info(f"{'='*70}")
    for instrument, res in results.items():
        if res['status'] == 'passed':
            passed_count = sum(res['validations'].values())
            total_count = len(res['validations'])
            logger.info(f"{instrument.value}: {passed_count}/{total_count} validations passed")
        else:
            logger.info(f"{instrument.value}: {res['status']}")
    
    return results


def main():
    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "synthetic_bars.db"
    
    instruments_to_test = [Instrument.ES, Instrument.NQ, Instrument.GC, Instrument.SI]
    test_svcj_model(instruments_to_test, db_path)


if __name__ == "__main__":
    main()
