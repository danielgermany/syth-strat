"""
Regime-switching path generator.

This module generates Monte Carlo price paths using regime-switching GARCH
with time-varying transition probabilities (TVTP).
"""

import numpy as np
from typing import Tuple, Optional
import logging

from ..models.regime_config import Instrument, RegimeConfig
from ..models.garch_params import RegimeGARCHParams
from ..models.transition_matrix import TransitionMatrix

logger = logging.getLogger(__name__)


class RegimeSwitchingPathGenerator:
    """
    Generate Monte Carlo price paths using regime-switching GARCH with TVTP.
    
    Algorithm:
    1. For each particle and each timestep:
       a. Get TVTP-adjusted transition matrix
       b. Sample regime transition
       c. Enforce min_duration constraint
       d. Generate return using regime-specific GARCH
       e. Update variance and price
    
    Cross-instrument contagion is incorporated through TVTP adjustments
    when the partner instrument's regime state is provided.
    """
    
    def __init__(
        self,
        instrument: Instrument,
        regime_params: RegimeGARCHParams,
        transition_matrix: TransitionMatrix,
        regime_config: RegimeConfig,
    ):
        self.instrument = instrument
        self.regime_params = regime_params
        self.transition_matrix = transition_matrix
        self.regime_config = regime_config
        
    def generate_paths(
        self,
        n_paths: int,
        horizon_bars: int,
        initial_price: float,
        initial_variance: float,
        initial_regime_probs: np.ndarray,
        time_of_day_start: float,
        partner_regimes: Optional[np.ndarray] = None,
        rv_ratio: float = 1.0,
        random_state: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate regime-switching GARCH paths.
        
        Args:
            n_paths: Number of simulation paths
            horizon_bars: Number of bars to simulate
            initial_price: Starting price
            initial_variance: Current conditional variance
            initial_regime_probs: K-dimensional initial regime probabilities
            time_of_day_start: Starting hour (ET, 0-23)
            partner_regimes: Optional array of partner instrument regimes per bar
                            (for cross-instrument contagion)
            rv_ratio: Initial RV_10bar / RV_100bar ratio
            random_state: Random seed
            
        Returns:
            Tuple of:
            - prices: (n_paths, horizon_bars + 1) price paths
            - variances: (n_paths, horizon_bars + 1) variance paths
            - regimes: (n_paths, horizon_bars + 1) regime labels
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_regimes = self.regime_config.n_regimes
        min_duration = self.regime_config.min_duration_bars
        
        # Initialize arrays
        prices = np.zeros((n_paths, horizon_bars + 1))
        variances = np.zeros((n_paths, horizon_bars + 1))
        regimes = np.zeros((n_paths, horizon_bars + 1), dtype=int)
        regime_durations = np.ones(n_paths, dtype=int)
        
        # Initial conditions
        prices[:, 0] = initial_price
        variances[:, 0] = initial_variance
        
        # Sample initial regimes from probabilities
        initial_regimes = np.random.choice(
            n_regimes, size=n_paths, p=initial_regime_probs
        )
        regimes[:, 0] = initial_regimes
        
        # Time tracking (minutes per bar)
        minutes_per_day = 23 * 60  # Futures trading hours
        
        for t in range(horizon_bars):
            # Current time of day
            current_minute = (time_of_day_start * 60 + t) % minutes_per_day
            current_hour = current_minute / 60.0
            
            # Partner regime for contagion (use mode if provided)
            if partner_regimes is not None and t < len(partner_regimes):
                partner_regime = int(partner_regimes[t])
            else:
                partner_regime = 1  # Default to "normal" regime
            
            # Get TVTP-adjusted transition matrix
            P = self.transition_matrix.get_adjusted(
                time_of_day_hour=current_hour,
                partner_regime=partner_regime,
                rv_ratio=rv_ratio,
            )
            
            # For each path, handle regime transition
            for i in range(n_paths):
                current_regime = regimes[i, t]
                
                # Sample next regime from transition probabilities
                proposed_regime = np.random.choice(n_regimes, p=P[current_regime])
                
                # Apply minimum duration constraint
                if proposed_regime != current_regime:
                    if regime_durations[i] >= min_duration:
                        regimes[i, t + 1] = proposed_regime
                        regime_durations[i] = 1
                    else:
                        regimes[i, t + 1] = current_regime
                        regime_durations[i] += 1
                else:
                    regimes[i, t + 1] = current_regime
                    regime_durations[i] += 1
                
                # Get regime-specific GARCH parameters
                regime_k = regimes[i, t + 1]
                params = self.regime_params.get_regime_params(regime_k)
                
                # GARCH variance update
                # σ²_t = ω + α*ε²_{t-1} + γ*ε²_{t-1}*I(ε<0) + β*σ²_{t-1}
                # For first step, use expected shock (variance)
                if t == 0:
                    last_shock_sq = variances[i, t]
                    leverage = 0.5 * params.gamma * variances[i, t]  # Expected leverage
                else:
                    last_return = np.log(prices[i, t] / prices[i, t - 1])
                    last_shock = last_return - params.mu
                    last_shock_sq = last_shock ** 2
                    leverage = params.gamma * last_shock_sq * (last_shock < 0)
                
                new_variance = (
                    params.omega +
                    params.alpha * last_shock_sq +
                    leverage +
                    params.beta * variances[i, t]
                )
                variances[i, t + 1] = max(new_variance, 1e-10)
                
                # Generate innovation
                z = self._generate_innovation(params.nu, params.skew, params.gamma > 0)
                
                # Generate return and update price
                sigma = np.sqrt(variances[i, t + 1])
                ret = params.mu + sigma * z
                prices[i, t + 1] = prices[i, t] * np.exp(ret)
        
        return prices, variances, regimes
    
    def _generate_innovation(
        self,
        nu: float,
        skew: float,
        use_skewed_t: bool
    ) -> float:
        """
        Generate a single innovation from the appropriate distribution.
        
        ES/NQ: Skewed-t distribution (leverage effect)
        GC/SI: Generalized Error Distribution (symmetric fat tails)
        """
        if use_skewed_t:
            # Skewed-t: Use t-distribution with skewness transformation
            z = np.random.standard_t(max(nu, 2.5)) if nu > 2 else np.random.standard_t(3)
            if abs(skew) > 1e-6:
                # Apply skewness via sinh transform
                z = np.sinh(skew * np.arcsinh(z) + np.arcsinh(skew))
            return z
        else:
            # GED: Approximate with scaled t-distribution
            # nu < 2 gives fatter tails
            if nu < 2:
                df = max(2.5, nu * 2)  # Map GED shape to t df
                z = np.random.standard_t(df)
            else:
                z = np.random.standard_normal()
            return z
