"""
Heston Stochastic Volatility Model.

This module implements the Heston model with Quadratic-Exponential (QE)
discretization scheme for efficient and accurate simulation.

Reference: Andersen (2007) - Efficient simulation of the Heston stochastic volatility model
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.optimize import minimize
import logging
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class HestonParams:
    """
    Heston stochastic volatility parameters.
    
    Price SDE: dS = μS dt + √v S dW₁
    Variance SDE: dv = κ(θ - v) dt + ξ√v dW₂
    Correlation: Corr(dW₁, dW₂) = ρ
    
    Reference: monte_carlo_architecture.md Section 4.4
    """
    kappa: float    # Mean reversion speed (0.5 - 5.0)
    theta: float    # Long-run variance (0.01 - 0.16)
    xi: float       # Volatility of volatility (0.3 - 1.5)
    rho: float      # Price-variance correlation (-0.7 to -0.9 for ES/NQ, -0.2 to 0 for GC/SI)
    mu: float = 0.0 # Drift
    
    def validate(self) -> bool:
        """Check parameter constraints."""
        # kappa, theta, xi > 0
        if self.kappa <= 0 or self.theta <= 0 or self.xi <= 0:
            return False
        # -1 < rho < 1
        if self.rho <= -1 or self.rho >= 1:
            return False
        return True
    
    @property
    def feller_satisfied(self) -> bool:
        """
        Check Feller condition: 2κθ ≥ ξ².
        
        When satisfied, variance process stays positive.
        When violated (common in practice), use full truncation.
        """
        return 2 * self.kappa * self.theta >= self.xi ** 2


class HestonModel:
    """
    Heston stochastic volatility model.
    
    Uses Quadratic-Exponential (QE) discretization scheme from
    Andersen (2007) for efficient and accurate simulation.
    
    Reference: monte_carlo_architecture.md Section 4.4
    """
    
    # QE scheme threshold (psi_crit)
    PSI_CRIT = 1.5
    
    def __init__(self, params: Optional[HestonParams] = None):
        """
        Args:
            params: Heston parameters (if None, model not fitted)
        """
        self.params = params
        self._fitted = params is not None
    
    def simulate_paths(
        self,
        n_paths: int,
        horizon: int,
        dt: float,  # Time step (in years, e.g., 1/525600 for 1 minute ≈ 1.9e-6)
        initial_price: float,
        initial_variance: float,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate price and variance paths using QE scheme.
        
        Args:
            n_paths: Number of simulation paths
            horizon: Number of time steps
            dt: Time step size in years
            initial_price: Starting price
            initial_variance: Starting variance
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (prices, variances) with shape (n_paths, horizon+1)
            First column is initial values.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Set params or call fit() first.")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        kappa = self.params.kappa
        theta = self.params.theta
        xi = self.params.xi
        rho = self.params.rho
        mu = self.params.mu
        
        # Pre-compute constants
        exp_kappa_dt = np.exp(-kappa * dt)
        
        # Initialize arrays
        prices = np.zeros((n_paths, horizon + 1))
        variances = np.zeros((n_paths, horizon + 1))
        
        prices[:, 0] = initial_price
        variances[:, 0] = max(initial_variance, 1e-10)  # Ensure positive
        
        # Pre-generate random numbers
        # u_v: uniform for exponential scheme
        # z_s: standard normal for price increments
        u_v = np.random.uniform(size=(n_paths, horizon))
        z_s = np.random.standard_normal((n_paths, horizon))
        # Generate correlated random for variance if rho != 0
        z_v_corr = rho * z_s + np.sqrt(1 - rho**2) * np.random.standard_normal((n_paths, horizon))
        
        for t in range(horizon):
            v_t = variances[:, t]
            
            # QE scheme for variance
            # Mean and variance of v_{t+1} | v_t
            m = theta + (v_t - theta) * exp_kappa_dt
            s2 = (v_t * xi**2 * exp_kappa_dt / kappa * (1 - exp_kappa_dt) +
                  theta * xi**2 / (2 * kappa) * (1 - exp_kappa_dt)**2)
            
            # Ensure s2 is positive
            s2 = np.maximum(s2, 1e-10)
            
            # Non-centrality parameter: psi = s² / m²
            psi = s2 / (m**2 + 1e-10)
            
            # Allocate arrays for this step
            v_next = np.zeros(n_paths)
            
            # Low psi region: quadratic scheme (more efficient)
            low_psi = psi <= self.PSI_CRIT
            if np.any(low_psi):
                inv_psi = 1 / (psi[low_psi] + 1e-10)
                b2 = 2 * inv_psi - 1 + np.sqrt(2 * inv_psi * (2 * inv_psi - 1))
                a = m[low_psi] / (1 + b2)
                b = np.sqrt(b2)
                
                # Use correlated random for variance
                z_v = z_v_corr[low_psi, t]
                v_next[low_psi] = a * (b + z_v)**2
            
            # High psi region: exponential scheme (handles high volatility)
            high_psi = ~low_psi
            if np.any(high_psi):
                p = (psi[high_psi] - 1) / (psi[high_psi] + 1)
                beta = (1 - p) / (m[high_psi] + 1e-10)
                
                # Inverse CDF sampling for exponential distribution
                u = u_v[high_psi, t]
                v_next[high_psi] = np.where(
                    u <= p,
                    0,  # Point mass at zero
                    np.log((1 - p) / (1 - u + 1e-10)) / beta
                )
            
            # Full truncation: ensure variance stays positive
            # If Feller condition violated, this prevents negative variance
            variances[:, t+1] = np.maximum(v_next, 1e-10)
            
            # Price update with correlation
            # Log price increment: d(log S) = (μ - v/2)dt + √v dW₁
            # We use the integrated variance for this step (trapezoidal approximation)
            v_avg = 0.5 * (v_t + variances[:, t+1])
            
            # Correlated Brownian increments
            # dW₁ = ρ dW_v + √(1-ρ²) dZ
            # For price: dW₁ = z_s (already incorporating correlation via variance)
            # Variance uses z_v_corr, price uses z_s
            
            # Generate price increment
            dW = z_s[:, t] * np.sqrt(dt)
            log_return = (mu - 0.5 * v_avg) * dt + np.sqrt(v_avg * dt) * dW
            
            # Ensure log returns are reasonable (prevent explosions)
            log_return = np.clip(log_return, -0.1, 0.1)
            
            prices[:, t+1] = prices[:, t] * np.exp(log_return)
        
        return prices, variances
    
    def fit(
        self,
        returns: np.ndarray,
        initial_guess: Optional[HestonParams] = None,
        dt: float = 1.0 / (252 * 390)  # Default: 1 minute in years
    ) -> HestonParams:
        """
        Estimate Heston parameters from historical returns.
        
        Uses simulated method of moments targeting:
        - Mean return
        - Variance of returns
        - Autocorrelation of squared returns (volatility clustering)
        - Kurtosis (fat tails)
        
        Note: This is a simplified calibration. Production systems typically
        use particle MCMC or options-implied calibration for better accuracy.
        
        Args:
            returns: Array of log returns
            initial_guess: Initial parameter guess (uses defaults if None)
            dt: Time step size in years (default: 1 minute)
            
        Returns:
            Fitted HestonParams
        """
        sample_var = np.var(returns)
        sample_mean = np.mean(returns)
        sample_kurt = self._kurtosis(returns)
        ac1_sq = self._autocorr(returns**2, 1)
        
        # Initial guess if not provided
        if initial_guess is None:
            # Default: moderate mean reversion, reasonable volatility
            x0 = [
                2.0,              # kappa (mean reversion speed)
                sample_var * 252, # theta (long-run variance, annualized)
                0.5,              # xi (vol of vol)
                -0.7,             # rho (leverage effect for equities)
            ]
        else:
            x0 = [initial_guess.kappa, initial_guess.theta, initial_guess.xi, initial_guess.rho]
        
        def objective(params):
            kappa, theta, xi, rho = params
            
            # Constraints
            if kappa <= 0 or theta <= 0 or xi <= 0:
                return 1e10
            if rho <= -1 or rho >= 1:
                return 1e10
            
            # Simulate and compute moments
            try:
                test_params = HestonParams(
                    kappa=kappa,
                    theta=theta / 252,  # Convert annualized theta to per-step
                    xi=xi,
                    rho=rho,
                    mu=sample_mean / dt  # Drift per year
                )
                model = HestonModel(test_params)
                
                # Simulate paths
                sim_prices, _ = model.simulate_paths(
                    n_paths=1000,
                    horizon=len(returns),
                    dt=dt,
                    initial_price=100.0,
                    initial_variance=theta / 252,  # Convert to per-step variance
                    random_state=42
                )
                sim_returns = np.diff(np.log(sim_prices), axis=1)
                sim_returns_flat = sim_returns.flatten()
                
                if len(sim_returns_flat) == 0:
                    return 1e10
                
                # Moment matching
                sim_var = np.var(sim_returns_flat)
                sim_kurt = self._kurtosis(sim_returns_flat)
                sim_ac1 = self._autocorr(sim_returns_flat**2, 1)
                
                # Weighted error (normalize by target values)
                error = ((sim_var - sample_var) / (sample_var + 1e-8))**2
                error += ((sim_kurt - sample_kurt) / (abs(sample_kurt) + 1))**2
                error += (sim_ac1 - ac1_sq)**2
                
                return error
            except Exception as e:
                logger.debug(f"Objective function error: {e}")
                return 1e10
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=[
                (0.1, 10.0),      # kappa
                (1e-6, 1.0),      # theta (per-step variance)
                (0.1, 2.0),       # xi
                (-0.99, 0.99),    # rho
            ],
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        if not result.success:
            logger.warning(f"Heston optimization did not converge: {result.message}")
        
        kappa, theta_annual, xi, rho = result.x
        
        # Convert theta back to per-step variance
        theta = theta_annual / 252
        
        params = HestonParams(
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            mu=sample_mean / dt  # Drift per year
        )
        
        self.params = params
        self._fitted = True
        
        if not params.feller_satisfied:
            logger.warning(
                f"Feller condition violated: 2κθ={2*kappa*theta:.6f} < ξ²={xi**2:.6f}. "
                f"Using full truncation to handle this."
            )
        
        return params
    
    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        """Excess kurtosis (kurtosis - 3 for normal distribution)."""
        if len(x) < 4:
            return 0.0
        m = np.mean(x)
        s = np.std(x)
        if s == 0:
            return 0.0
        return np.mean(((x - m) / s)**4) - 3.0
    
    @staticmethod
    def _autocorr(x: np.ndarray, lag: int) -> float:
        """Autocorrelation at given lag."""
        n = len(x)
        if n < lag + 1:
            return 0.0
        m = np.mean(x)
        c0 = np.sum((x - m)**2)
        if c0 == 0:
            return 0.0
        cl = np.sum((x[lag:] - m) * (x[:-lag] - m))
        return cl / c0
