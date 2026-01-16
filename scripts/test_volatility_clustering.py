"""
Multi-horizon volatility clustering test for all models.

Tests Heston, Jump-Diffusion, and SVCJ models with different horizons
to observe how volatility clustering (squared return autocorrelation)
changes with horizon length.

Horizons tested:
- 1 hour (60 bars)
- 1 day (1440 bars)
- 1 week (10080 bars)
- 1 month (~43200 bars, or ~30000 bars for practical testing)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.storage import DataStorage
from src.models.regime_config import Instrument
from src.models.heston import HestonModel, HestonParams
from src.models.jump_diffusion import JumpDiffusionModel, JumpParams
from src.models.svcj import SVCJModel, SVCJParams
from scripts.test_end_to_end import compute_stylized_facts

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class ModelResult:
    """Results for a single model at a single horizon."""
    model_name: str
    instrument: str
    horizon_bars: int
    horizon_name: str
    autocorr_squared_returns: float
    autocorr_returns: float
    kurtosis: float
    mean_return: float
    std_return: float
    status: str  # 'passed', 'failed_fit', 'failed_simulate', 'error'


def test_model_at_horizon(
    model_name: str,
    model,
    params,
    instrument: Instrument,
    horizon_bars: int,
    horizon_name: str,
    initial_price: float,
    initial_variance: float,
    dt_year: float,
    n_paths: int = 1000,
    random_state: Optional[int] = 42
) -> ModelResult:
    """Test a single model at a single horizon."""
    try:
        # Generate paths
        if model_name == 'Heston':
            prices, variances = model.simulate_paths(
                n_paths=n_paths,
                horizon=horizon_bars,
                dt=dt_year,
                initial_price=initial_price,
                initial_variance=initial_variance,
                random_state=random_state
            )
        elif model_name == 'JumpDiffusion':
            prices = model.simulate_paths(
                n_paths=n_paths,
                horizon=horizon_bars,
                dt=dt_year,
                initial_price=initial_price,
                random_state=random_state
            )
            # Jump-diffusion doesn't return variances
            variances = None
        elif model_name == 'SVCJ':
            prices, variances = model.simulate_paths(
                n_paths=n_paths,
                horizon=horizon_bars,
                dt=dt_year,
                initial_price=initial_price,
                initial_variance=initial_variance,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Compute returns
        generated_returns = np.diff(np.log(prices), axis=1)
        
        # Compute stylized facts
        stylized_facts = compute_stylized_facts(generated_returns)
        
        return ModelResult(
            model_name=model_name,
            instrument=instrument.value,
            horizon_bars=horizon_bars,
            horizon_name=horizon_name,
            autocorr_squared_returns=stylized_facts.get('autocorr_squared_returns', np.nan),
            autocorr_returns=stylized_facts.get('autocorr_returns', np.nan),
            kurtosis=stylized_facts.get('kurtosis', np.nan),
            mean_return=stylized_facts.get('mean_return', np.nan),
            std_return=stylized_facts.get('std_return', np.nan),
            status='passed'
        )
    
    except Exception as e:
        logger.error(f"Error testing {model_name} for {instrument.value} at {horizon_name}: {e}")
        return ModelResult(
            model_name=model_name,
            instrument=instrument.value,
            horizon_bars=horizon_bars,
            horizon_name=horizon_name,
            autocorr_squared_returns=np.nan,
            autocorr_returns=np.nan,
            kurtosis=np.nan,
            mean_return=np.nan,
            std_return=np.nan,
            status=f'error: {str(e)}'
        )


def test_all_models_multi_horizon(
    instruments: List[Instrument],
    db_path: str,
    horizons: List[Tuple[int, str]],  # List of (bars, name) tuples
    n_paths: int = 1000,
    random_state: Optional[int] = 42
) -> Dict[str, List[ModelResult]]:
    """Test all models at multiple horizons."""
    storage = DataStorage(db_path)
    results = {
        'Heston': [],
        'JumpDiffusion': [],
        'SVCJ': []
    }
    
    # Minutes in a trading year
    MINUTES_PER_TRADING_YEAR = 252 * 390
    DT_YEAR = 1 / MINUTES_PER_TRADING_YEAR
    
    for instrument in instruments:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing {instrument.value} at multiple horizons")
        logger.info(f"{'='*70}")
        
        # Load historical data
        historical_data = storage.load_bars(instrument.value)
        if historical_data.empty:
            logger.warning(f"No data for {instrument.value}. Skipping.")
            continue
        
        # Use recent window for fitting (last 5000 bars)
        fit_data = historical_data.tail(5000)
        if len(fit_data) < 100:
            logger.warning(f"Insufficient data for {instrument.value}. Skipping.")
            continue
        
        returns = np.log(fit_data['close'] / fit_data['close'].shift(1)).dropna().values
        if len(returns) == 0:
            logger.warning(f"No returns for {instrument.value}. Skipping.")
            continue
        
        initial_price = float(fit_data['close'].iloc[-1])
        initial_variance = float(np.var(returns[-50:]))
        
        # Fit models once (reuse for all horizons)
        logger.info(f"Fitting models for {instrument.value}...")
        
        # Heston
        heston_model = HestonModel()
        try:
            heston_params = heston_model.fit(returns, dt=DT_YEAR)
            logger.info(f"  Heston fitted: κ={heston_params.kappa:.4f}, θ={heston_params.theta:.8f}")
        except Exception as e:
            logger.error(f"  Failed to fit Heston: {e}")
            heston_params = None
        
        # Jump-Diffusion
        jump_model = JumpDiffusionModel()
        try:
            jump_params = jump_model.fit(returns, dt=DT_YEAR, threshold_method='std_multiple', threshold_value=5.0)
            logger.info(f"  Jump-Diffusion fitted: λ={jump_params.lambda_:.2f} jumps/year")
        except Exception as e:
            logger.error(f"  Failed to fit Jump-Diffusion: {e}")
            jump_params = None
        
        # SVCJ
        svcj_model = SVCJModel()
        try:
            svcj_params = svcj_model.fit(returns, dt=DT_YEAR)
            logger.info(f"  SVCJ fitted: λ={svcj_params.lambda_:.2f} jumps/year")
        except Exception as e:
            logger.error(f"  Failed to fit SVCJ: {e}")
            svcj_params = None
        
        # Test each model at each horizon
        for horizon_bars, horizon_name in horizons:
            logger.info(f"\n  Testing {horizon_name} ({horizon_bars} bars)...")
            
            # Heston
            if heston_params is not None:
                result = test_model_at_horizon(
                    'Heston', heston_model, heston_params, instrument,
                    horizon_bars, horizon_name, initial_price, initial_variance,
                    DT_YEAR, n_paths, random_state
                )
                results['Heston'].append(result)
                logger.info(f"    Heston: AC(sq)={result.autocorr_squared_returns:.4f}")
            
            # Jump-Diffusion
            if jump_params is not None:
                result = test_model_at_horizon(
                    'JumpDiffusion', jump_model, jump_params, instrument,
                    horizon_bars, horizon_name, initial_price, initial_variance,
                    DT_YEAR, n_paths, random_state
                )
                results['JumpDiffusion'].append(result)
                logger.info(f"    Jump-Diffusion: AC(sq)={result.autocorr_squared_returns:.4f}")
            
            # SVCJ
            if svcj_params is not None:
                result = test_model_at_horizon(
                    'SVCJ', svcj_model, svcj_params, instrument,
                    horizon_bars, horizon_name, initial_price, initial_variance,
                    DT_YEAR, n_paths, random_state
                )
                results['SVCJ'].append(result)
                logger.info(f"    SVCJ: AC(sq)={result.autocorr_squared_returns:.4f}")
    
    return results


def print_summary(results: Dict[str, List[ModelResult]], horizons: List[Tuple[int, str]]):
    """Print summary of results."""
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY: Volatility Clustering vs Horizon Length")
    logger.info(f"{'='*70}\n")
    
    # Group by instrument and horizon
    for model_name in ['Heston', 'JumpDiffusion', 'SVCJ']:
        logger.info(f"\n{model_name} Model:")
        logger.info("-" * 70)
        
        # Create table
        instruments = sorted(set(r.instrument for r in results[model_name]))
        horizon_names = [h[1] for h in horizons]
        
        # Header
        header = f"{'Instrument':<10}"
        for h_name in horizon_names:
            header += f" {h_name:>12}"
        logger.info(header)
        logger.info("-" * 70)
        
        # Rows
        for instrument in instruments:
            row = f"{instrument:<10}"
            for horizon_bars, h_name in horizons:
                # Find result
                result = next(
                    (r for r in results[model_name] 
                     if r.instrument == instrument and r.horizon_bars == horizon_bars),
                    None
                )
                if result and result.status == 'passed':
                    ac_sq = result.autocorr_squared_returns
                    row += f" {ac_sq:>12.4f}"
                else:
                    row += f" {'N/A':>12}"
            logger.info(row)
        
        # Average across instruments
        logger.info("-" * 70)
        row = f"{'Average':<10}"
        for horizon_bars, h_name in horizons:
            ac_sq_values = [
                r.autocorr_squared_returns 
                for r in results[model_name]
                if r.horizon_bars == horizon_bars and r.status == 'passed'
            ]
            if ac_sq_values:
                avg_ac_sq = np.mean(ac_sq_values)
                row += f" {avg_ac_sq:>12.4f}"
            else:
                row += f" {'N/A':>12}"
        logger.info(row)
    
    # Historical comparison
    logger.info(f"\n{'='*70}")
    logger.info("Historical Data Comparison (5000 bars ≈ 2 weeks):")
    logger.info("-" * 70)
    storage = DataStorage(project_root / "data" / "synthetic_bars.db")
    for instrument in [Instrument.ES, Instrument.NQ, Instrument.GC, Instrument.SI]:
        df = storage.load_bars(instrument.value)
        returns = np.diff(np.log(df['close'].values))
        recent_returns = returns[-5000:]
        squared_returns = recent_returns ** 2
        ac_sq = pd.Series(squared_returns).autocorr(lag=1)
        logger.info(f"  {instrument.value}: AC(sq) = {ac_sq:.4f}")


def main():
    """Main function."""
    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "synthetic_bars.db"
    
    # Define horizons
    horizons = [
        (60, "1 hour"),      # 60 bars = 1 hour
        (1440, "1 day"),     # 1440 bars = 24 hours
        (10080, "1 week"),   # 10080 bars = 7 days
        (30000, "1 month"),  # ~30000 bars = ~21 days (practical limit)
    ]
    
    instruments = [Instrument.ES, Instrument.NQ, Instrument.GC, Instrument.SI]
    
    logger.info("Starting multi-horizon volatility clustering tests...")
    logger.info(f"Horizons: {[h[1] for h in horizons]}")
    logger.info(f"Instruments: {[i.value for i in instruments]}")
    
    results = test_all_models_multi_horizon(
        instruments=instruments,
        db_path=str(db_path),
        horizons=horizons,
        n_paths=1000,
        random_state=42
    )
    
    print_summary(results, horizons)


if __name__ == "__main__":
    main()
