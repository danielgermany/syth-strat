"""
Standalone A-GARCH (GJR-GARCH) model for comparison with symmetric GARCH.

This module provides an asymmetric GARCH implementation with leverage effect
for A/B testing against the symmetric GARCH model.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Optional, Tuple
import logging

from .garch import GARCHModel
from .garch_params import GARCHParams

logger = logging.getLogger(__name__)


class AGARCHModel(GARCHModel):
    """
    GJR-GARCH model with asymmetric volatility response (leverage effect).
    
    Variance equation:
        σ²_t = ω + α * ε²_{t-1} + γ * ε²_{t-1} * I_{ε<0} + β * σ²_{t-1}
    
    The γ term captures leverage effect: negative returns increase
    volatility more than positive returns of same magnitude.
    
    Reference: monte_carlo_architecture.md Section 4.3
    """
    
    def __init__(self, params: Optional[GARCHParams] = None):
        """
        Args:
            params: GARCH parameters with gamma > 0 for asymmetric GARCH
        """
        super().__init__(params)
    
    def fit(self, returns: np.ndarray, method: str = 'mle') -> GARCHParams:
        """
        Estimate A-GARCH (GJR-GARCH) parameters from historical returns using MLE.
        
        Args:
            returns: Array of log returns
            method: Estimation method ('mle' only currently)
            
        Returns:
            Fitted GARCHParams with gamma > 0 (asymmetric GARCH)
        """
        if method != 'mle':
            raise ValueError(f"Only 'mle' method supported, got {method}")
        
        n = len(returns)
        sample_var = np.var(returns)
        
        # Initial guess (include gamma for asymmetry)
        x0 = [
            sample_var * 0.05,  # omega
            0.05,               # alpha
            0.10,               # gamma (asymmetry)
            0.85,               # beta
            np.mean(returns)    # mu
        ]
        
        def neg_log_likelihood(params):
            omega, alpha, gamma, beta, mu = params
            
            # Constraints
            if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0:
                return 1e10
            # Stationarity with asymmetry: α + β + γ/2 < 1
            if alpha + beta + 0.5 * gamma >= 1:
                return 1e10
            
            # Initialize
            sigma2 = np.zeros(n)
            sigma2[0] = sample_var
            
            eps = returns - mu
            
            for t in range(1, n):
                # Indicator for negative shock
                indicator = 1.0 if eps[t-1] < 0 else 0.0
                # Asymmetric variance update
                sigma2[t] = (omega + 
                           alpha * eps[t-1]**2 + 
                           gamma * eps[t-1]**2 * indicator + 
                           beta * sigma2[t-1])
            
            sigma2 = np.maximum(sigma2, 1e-10)
            
            # Log-likelihood (assuming normal innovations)
            ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps**2 / sigma2)
            return -ll
        
        result = minimize(
            neg_log_likelihood,
            x0,
            method='L-BFGS-B',
            bounds=[
                (1e-10, None),  # omega > 0
                (0, 0.3),       # alpha
                (0, 0.3),       # gamma (asymmetry)
                (0, 0.999),     # beta < 1
                (None, None)    # mu (unconstrained)
            ],
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"A-GARCH optimization did not converge: {result.message}")
        
        omega, alpha, gamma, beta, mu = result.x
        
        # Ensure parameters satisfy constraints
        omega = max(omega, 1e-10)
        alpha = max(alpha, 0.0)
        gamma = max(gamma, 0.0)  # Asymmetry must be non-negative
        beta = max(beta, 0.0)
        
        # Check and enforce stationarity: α + β + γ/2 < 1
        persistence = alpha + beta + gamma / 2
        if persistence >= 0.999:
            # Scale down proportionally to enforce stationarity (target 0.99)
            scale_factor = 0.99 / persistence
            alpha = alpha * scale_factor
            beta = beta * scale_factor
            gamma = gamma * scale_factor
            logger.debug(f"Scaled down persistence from {persistence:.4f} to {alpha + beta + gamma/2:.4f}")
        
        # Clamp mu to reasonable range for 1-minute returns
        mu = np.clip(mu, -0.0005, 0.0005)
        
        params = GARCHParams(
            omega=omega,
            alpha=alpha,
            beta=beta,
            gamma=gamma,  # A-GARCH with leverage effect
            mu=mu,
            nu=6.0,       # Default (not estimated here)
            skew=0.0      # Not using skewed-t for standalone models
        )
        
        self.params = params
        self._fitted = True
        return params
    
    def simulate_paths(
        self,
        n_paths: int,
        horizon: int,
        initial_price: float,
        initial_variance: float,
        last_return: float,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Monte Carlo price paths with asymmetric variance dynamics.
        
        Args:
            n_paths: Number of simulation paths
            horizon: Number of timesteps (bars) to simulate
            initial_price: Starting price S_0
            initial_variance: Current conditional variance σ²_0
            last_return: Previous return r_{-1} for first variance calculation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (prices, variances) arrays with shape (n_paths, horizon+1)
            First column is initial values.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        omega = self.params.omega
        alpha = self.params.alpha
        beta = self.params.beta
        gamma = self.params.gamma  # Asymmetry parameter
        mu = self.params.mu
        
        prices = np.zeros((n_paths, horizon + 1))
        variances = np.zeros((n_paths, horizon + 1))
        
        prices[:, 0] = initial_price
        variances[:, 0] = max(initial_variance, 1e-10)
        
        # Track previous return for asymmetry calculation
        prev_return = np.full(n_paths, last_return)
        z = np.random.standard_normal((n_paths, horizon))
        
        for t in range(horizon):
            # Asymmetric variance update with leverage effect
            indicator = (prev_return < 0).astype(float)  # 1 if negative, 0 otherwise
            prev_shock_sq = (prev_return - mu) ** 2
            
            variances[:, t+1] = (omega + 
                                alpha * prev_shock_sq + 
                                gamma * prev_shock_sq * indicator + 
                                beta * variances[:, t])
            variances[:, t+1] = np.maximum(variances[:, t+1], 1e-10)
            
            # Generate returns
            sigma = np.sqrt(variances[:, t+1])
            returns = mu + sigma * z[:, t]
            
            # Update prices
            prices[:, t+1] = prices[:, t] * np.exp(returns)
            
            # Store for next iteration (for asymmetry)
            prev_return = returns
        
        return prices, variances
