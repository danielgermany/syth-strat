"""
Standalone GARCH(1,1) model for comparison with A-GARCH.

This module provides a simple GARCH model implementation for A/B testing.
The regime-switching implementation uses more sophisticated models, but this
standalone version allows direct comparison of symmetric vs asymmetric GARCH.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Optional, Tuple
import logging

from .garch_params import GARCHParams

logger = logging.getLogger(__name__)


class GARCHModel:
    """
    Standard GARCH(1,1) model with symmetric volatility response.
    
    Variance equation:
        σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
    
    Return equation:
        r_t = μ + σ_t * z_t,  where z_t ~ N(0,1)
    """
    
    def __init__(self, params: Optional[GARCHParams] = None):
        """
        Args:
            params: GARCH parameters (gamma should be 0 for symmetric GARCH)
        """
        self.params = params
        self._fitted = params is not None
    
    def fit(self, returns: np.ndarray, method: str = 'mle') -> GARCHParams:
        """
        Estimate GARCH parameters from historical returns using MLE.
        
        Args:
            returns: Array of log returns
            method: Estimation method ('mle' only currently)
            
        Returns:
            Fitted GARCHParams (with gamma=0 for symmetric GARCH)
        """
        if method != 'mle':
            raise ValueError(f"Only 'mle' method supported, got {method}")
        
        n = len(returns)
        sample_var = np.var(returns)
        
        # Initial guess
        x0 = [
            sample_var * 0.05,  # omega
            0.05,               # alpha
            0.85,               # beta
            np.mean(returns)    # mu
        ]
        
        def neg_log_likelihood(params):
            omega, alpha, beta, mu = params
            
            # Constraints
            if omega <= 0 or alpha < 0 or beta < 0:
                return 1e10
            if alpha + beta >= 1:  # Stationarity for symmetric GARCH
                return 1e10
            
            # Initialize
            sigma2 = np.zeros(n)
            sigma2[0] = sample_var
            
            eps = returns - mu
            
            for t in range(1, n):
                sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
            
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
                (0, 0.999),     # beta < 1
                (None, None)    # mu (unconstrained)
            ],
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"GARCH optimization did not converge: {result.message}")
        
        omega, alpha, beta, mu = result.x
        
        # Ensure parameters satisfy constraints
        omega = max(omega, 1e-10)
        alpha = max(alpha, 0.0)
        beta = max(beta, 0.0)
        
        # Enforce stationarity
        if alpha + beta >= 0.999:
            scale_factor = 0.99 / (alpha + beta)
            alpha = alpha * scale_factor
            beta = beta * scale_factor
        
        # Clamp mu to reasonable range for 1-minute returns
        mu = np.clip(mu, -0.0005, 0.0005)
        
        params = GARCHParams(
            omega=omega,
            alpha=alpha,
            beta=beta,
            gamma=0.0,  # Symmetric GARCH
            mu=mu,
            nu=6.0,     # Default (not estimated here)
            skew=0.0    # Symmetric
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
        Generate Monte Carlo price paths using GARCH dynamics.
        
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
        mu = self.params.mu
        
        prices = np.zeros((n_paths, horizon + 1))
        variances = np.zeros((n_paths, horizon + 1))
        
        prices[:, 0] = initial_price
        variances[:, 0] = max(initial_variance, 1e-10)
        
        # Track previous return for variance calculation
        prev_return = np.full(n_paths, last_return)
        z = np.random.standard_normal((n_paths, horizon))
        
        for t in range(horizon):
            # Symmetric variance update (no gamma term)
            prev_shock_sq = (prev_return - mu) ** 2
            variances[:, t+1] = omega + alpha * prev_shock_sq + beta * variances[:, t]
            variances[:, t+1] = np.maximum(variances[:, t+1], 1e-10)
            
            # Generate returns
            sigma = np.sqrt(variances[:, t+1])
            returns = mu + sigma * z[:, t]
            
            # Update prices
            prices[:, t+1] = prices[:, t] * np.exp(returns)
            
            prev_return = returns
        
        return prices, variances
