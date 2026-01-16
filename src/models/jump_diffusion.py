"""
Merton Jump-Diffusion Model.

This module implements the Merton jump-diffusion model where price dynamics
include both continuous diffusion and discrete jumps following a Poisson process.

Reference: monte_carlo_architecture.md Section 4.5
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class JumpParams:
    """
    Merton jump-diffusion parameters.
    
    Price process: dS/S = μ dt + σ dW + (J-1) dN
    
    Where:
    - dW is Brownian motion (diffusion)
    - N is Poisson process with intensity λ (jump frequency)
    - J is jump multiplier: ln(J) ~ N(μ_J, σ_J²)
    
    Reference: monte_carlo_architecture.md Section 4.5
    """
    sigma: float        # Diffusion volatility (annualized)
    lambda_: float      # Jump intensity (jumps per year)
    mu_jump: float      # Mean log jump size
    sigma_jump: float   # Std of log jump size
    mu: float = 0.0     # Drift (annualized)
    
    def validate(self) -> bool:
        """Check parameter constraints."""
        if self.sigma <= 0:
            return False
        if self.lambda_ < 0:
            return False
        if self.sigma_jump <= 0:
            return False
        return True
    
    @property
    def expected_jump_size(self) -> float:
        """
        Expected jump multiplier: E[J] = exp(μ_J + σ_J²/2).
        
        This is the expected multiplicative effect of a jump.
        """
        return np.exp(self.mu_jump + 0.5 * self.sigma_jump**2)
    
    @property
    def total_variance(self) -> float:
        """
        Total variance including jump contribution.
        
        Variance from diffusion: σ²
        Variance from jumps: λ * (σ_J² + μ_J²)
        """
        jump_var = self.lambda_ * (self.sigma_jump**2 + self.mu_jump**2)
        return self.sigma**2 + jump_var


class JumpDiffusionModel:
    """
    Merton jump-diffusion model.
    
    Handles compound Poisson process for jumps overlaid
    on standard geometric Brownian motion.
    
    Reference: monte_carlo_architecture.md Section 4.5
    """
    
    def __init__(self, params: Optional[JumpParams] = None):
        """
        Args:
            params: Jump-diffusion parameters (if None, model not fitted)
        """
        self.params = params
        self._fitted = params is not None
    
    def simulate_paths(
        self,
        n_paths: int,
        horizon: int,
        dt: float,  # Time step (in years, e.g., 1/525600 for 1 minute)
        initial_price: float,
        random_state: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate price paths with jumps.
        
        For each time step:
        1. Generate diffusion component: σ√dt Z
        2. Generate number of jumps: N ~ Poisson(λ dt)
        3. Generate jump sizes: Σ(ln J_i) where ln J ~ N(μ_J, σ_J²)
        4. Combine: S_{t+1} = S_t exp((μ - σ²/2)dt + diffusion + jumps - compensator)
        
        Args:
            n_paths: Number of simulation paths
            horizon: Number of time steps
            dt: Time step size in years
            initial_price: Starting price
            random_state: Random seed for reproducibility
            
        Returns:
            Prices array with shape (n_paths, horizon+1)
            First column is initial values.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Set params or call fit() first.")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        sigma = self.params.sigma
        lambda_ = self.params.lambda_
        mu_j = self.params.mu_jump
        sigma_j = self.params.sigma_jump
        mu = self.params.mu
        
        prices = np.zeros((n_paths, horizon + 1))
        prices[:, 0] = initial_price
        
        # Pre-compute compensator (makes price process a martingale under risk-neutral measure)
        # Compensator: λ * (E[J] - 1) = λ * (exp(μ_J + σ_J²/2) - 1)
        compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
        
        for t in range(horizon):
            # Diffusion component (continuous)
            z = np.random.standard_normal(n_paths)
            diffusion = sigma * np.sqrt(dt) * z
            
            # Jump component (discrete)
            # Number of jumps in this period: N ~ Poisson(λ dt)
            n_jumps = np.random.poisson(lambda_ * dt, n_paths)
            
            # Jump sizes: sum of log jumps
            # For each path, if n_jumps > 0, sample log jumps and sum them
            jump_component = np.zeros(n_paths)
            for i in range(n_paths):
                if n_jumps[i] > 0:
                    log_jumps = np.random.normal(mu_j, sigma_j, n_jumps[i])
                    jump_component[i] = np.sum(log_jumps)
            
            # Drift term (compensated to account for jump expectation)
            # Standard drift: (μ - σ²/2)dt
            # Minus compensator to make it a martingale
            drift = (mu - 0.5 * sigma**2 - compensator) * dt
            
            # Update prices
            log_return = drift + diffusion + jump_component
            
            # Clip extreme returns to prevent numerical issues
            log_return = np.clip(log_return, -0.1, 0.1)
            
            prices[:, t+1] = prices[:, t] * np.exp(log_return)
        
        return prices
    
    def fit(
        self,
        returns: np.ndarray,
        dt: float = 1.0 / (252 * 390),  # Default: 1 minute in years
        threshold_percentile: float = 95
    ) -> JumpParams:
        """
        Estimate jump parameters from historical returns.
        
        Simple approach:
        1. Separate returns into "normal" (diffusion) and "jumps" based on threshold
        2. Estimate diffusion σ from normal returns
        3. Estimate jump parameters (λ, μ_J, σ_J) from jump returns
        
        Args:
            returns: Array of log returns
            dt: Time step size in years
            threshold_percentile: Percentile threshold for identifying jumps (default: 95th)
            
        Returns:
            Fitted JumpParams
        """
        # Threshold for identifying jumps (absolute returns)
        abs_returns = np.abs(returns)
        threshold = np.percentile(abs_returns, threshold_percentile)
        
        # Separate returns into normal (diffusion) and jumps
        normal_mask = abs_returns <= threshold
        jump_mask = ~normal_mask
        
        normal_returns = returns[normal_mask]
        jump_returns = returns[jump_mask]
        
        # Diffusion parameters from normal returns
        # Annualize from per-step to annual
        sigma = np.std(normal_returns) / np.sqrt(dt)
        mu = np.mean(returns) / dt  # Overall drift (annualized)
        
        # Clamp mu to reasonable range
        mu = np.clip(mu, -0.5, 0.5)
        
        # Jump parameters
        n_jumps = np.sum(jump_mask)
        n_total = len(returns)
        
        # Jump intensity: λ = (number of jumps) / (total time in years)
        lambda_ = n_jumps / (n_total * dt)
        
        if n_jumps > 1:
            # Estimate jump size distribution from jump returns
            mu_jump = float(np.mean(jump_returns))
            sigma_jump = float(np.std(jump_returns))
            # Ensure minimum jump size std
            sigma_jump = max(sigma_jump, 0.01)
        else:
            # Default values if insufficient jumps detected
            logger.warning(f"Only {n_jumps} jumps detected, using default jump parameters")
            mu_jump = -0.02  # Typical: jumps are negative (crashes)
            sigma_jump = 0.05
        
        params = JumpParams(
            sigma=sigma,
            lambda_=lambda_,
            mu_jump=mu_jump,
            sigma_jump=sigma_jump,
            mu=mu
        )
        
        self.params = params
        self._fitted = True
        
        logger.info(
            f"Jump-diffusion fitted: σ={sigma:.6f}, λ={lambda_:.2f} jumps/year, "
            f"μ_J={mu_jump:.4f}, σ_J={sigma_jump:.4f}, {n_jumps} jumps detected"
        )
        
        return params
