"""
GARCH parameter structures for regime-switching models.

This module defines parameter structures for GARCH(1,1) and GJR-GARCH models:
- GARCHParams: Single-regime GARCH parameters with validation
- RegimeGARCHParams: Complete parameters for all regimes of an instrument
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .regime_config import Instrument


@dataclass
class GARCHParams:
    """
    GARCH(1,1) parameters for a single regime.
    
    Reference: monte_carlo_architecture.md Section 4.2 for parameter specifications.
    
    Variance equation:
        σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
    
    For GJR-GARCH (ES/NQ only):
        σ²_t = ω + α * ε²_{t-1} + γ * ε²_{t-1} * I(ε_{t-1} < 0) + β * σ²_{t-1}
    """
    omega: float      # Baseline variance constant
    alpha: float      # Reaction to shocks (ARCH term)
    beta: float       # Persistence (GARCH term)
    gamma: float = 0.0  # Leverage effect (GJR-GARCH, ES/NQ only)
    mu: float = 0.0   # Drift term
    
    # Innovation distribution parameters
    # ES/NQ: skewed-t (leverage effect)
    # GC/SI: GED (symmetric fat tails)
    nu: float = 6.0      # Degrees of freedom (t-dist) or shape (GED)
    skew: float = 0.0    # Skewness (-1 to 1, 0 = symmetric)
    
    def validate(self) -> bool:
        """Check parameter constraints."""
        if self.omega <= 0:
            return False
        if self.alpha < 0 or self.beta < 0:
            return False
        # Stationarity: α + β + γ/2 < 1
        if self.alpha + self.beta + self.gamma / 2 >= 1:
            return False
        if self.nu <= 2:  # Need nu > 2 for finite variance
            return False
        return True
    
    @property
    def persistence(self) -> float:
        """Volatility persistence: α + β + γ/2."""
        return self.alpha + self.beta + self.gamma / 2
    
    @property
    def unconditional_variance(self) -> float:
        """Long-run variance: ω / (1 - α - β - γ/2)."""
        return self.omega / (1 - self.persistence)


@dataclass 
class RegimeGARCHParams:
    """
    Complete GARCH parameters for all regimes of an instrument.
    
    Example for ES (3 regimes):
        regime_params[0] = Low volatility regime parameters
        regime_params[1] = Normal volatility regime parameters  
        regime_params[2] = High volatility regime parameters
    """
    instrument: 'Instrument'
    regime_params: List[GARCHParams]  # One per regime
    
    def get_regime_params(self, regime: int) -> GARCHParams:
        """Get parameters for a specific regime."""
        return self.regime_params[regime]
    
    @property
    def n_regimes(self) -> int:
        return len(self.regime_params)
    
    @classmethod
    def default_for_instrument(cls, instrument: 'Instrument') -> 'RegimeGARCHParams':
        """
        Create default regime-specific parameters.
        
        These are starting points - actual values should be calibrated from data.
        
        ES/NQ: Use GJR-GARCH (gamma > 0) with skewed-t innovations
        GC/SI: Use standard GARCH (gamma = 0) with GED innovations
        """
        from .regime_config import Instrument
        
        if instrument == Instrument.ES:
            return cls(
                instrument=instrument,
                regime_params=[
                    # Regime 0: Low volatility
                    GARCHParams(omega=0.00001, alpha=0.05, beta=0.92, gamma=0.08,
                               nu=8.0, skew=-0.1),
                    # Regime 1: Normal volatility
                    GARCHParams(omega=0.00002, alpha=0.10, beta=0.85, gamma=0.12,
                               nu=6.0, skew=-0.15),
                    # Regime 2: High volatility
                    GARCHParams(omega=0.00005, alpha=0.15, beta=0.78, gamma=0.15,
                               nu=4.0, skew=-0.2),
                ]
            )
        elif instrument == Instrument.NQ:
            return cls(
                instrument=instrument,
                regime_params=[
                    GARCHParams(omega=0.00001, alpha=0.06, beta=0.90, gamma=0.10,
                               nu=7.0, skew=-0.12),
                    GARCHParams(omega=0.00002, alpha=0.12, beta=0.82, gamma=0.14,
                               nu=5.0, skew=-0.18),
                    GARCHParams(omega=0.00006, alpha=0.18, beta=0.74, gamma=0.18,
                               nu=3.5, skew=-0.25),
                ]
            )
        elif instrument == Instrument.GC:
            # GC: Symmetric (no leverage effect), GED innovations
            return cls(
                instrument=instrument,
                regime_params=[
                    GARCHParams(omega=0.00001, alpha=0.06, beta=0.91, gamma=0.0,
                               nu=1.5, skew=0.0),  # GED shape
                    GARCHParams(omega=0.00002, alpha=0.09, beta=0.88, gamma=0.0,
                               nu=1.3, skew=0.0),
                    GARCHParams(omega=0.00005, alpha=0.12, beta=0.83, gamma=0.0,
                               nu=1.1, skew=0.0),
                ]
            )
        elif instrument == Instrument.SI:
            # SI: 4 regimes due to extreme kurtosis, symmetric GED
            return cls(
                instrument=instrument,
                regime_params=[
                    GARCHParams(omega=0.00002, alpha=0.06, beta=0.91, gamma=0.0,
                               nu=1.2, skew=0.0),
                    GARCHParams(omega=0.00003, alpha=0.10, beta=0.87, gamma=0.0,
                               nu=1.1, skew=0.0),
                    GARCHParams(omega=0.00008, alpha=0.15, beta=0.80, gamma=0.0,
                               nu=1.0, skew=0.0),
                    # Regime 3: Crisis (flash crash regime)
                    GARCHParams(omega=0.00020, alpha=0.25, beta=0.70, gamma=0.0,
                               nu=0.8, skew=0.0),
                ]
            )
        else:
            raise ValueError(f"Unknown instrument: {instrument}")
