"""
SVCJ Model: Stochastic Volatility with Correlated Jumps.

Combines Heston stochastic volatility with Merton jump-diffusion,
including correlated jumps in both price and variance.

Reference: synthetic_bar_generator_development_plan_v2.md Phase 5
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging
import pandas as pd

from .heston import HestonModel, HestonParams
from .jump_diffusion import JumpDiffusionModel, JumpParams

logger = logging.getLogger(__name__)


@dataclass
class SVCJParams:
    """
    Stochastic Volatility with Correlated Jumps parameters.
    
    Price: dS/S = μ dt + √v dW₁ + J_S dN
    Variance: dv = κ(θ - v) dt + ξ√v dW₂ + J_v dN
    
    Correlation: Corr(dW₁, dW₂) = ρ
    Correlated jumps: J_v > 0 when J_S occurs (variance jumps up on price jumps)
    """
    # Heston parameters
    kappa: float      # Mean reversion speed
    theta: float      # Long-run variance
    xi: float         # Volatility of volatility
    rho: float        # Price-variance correlation
    
    # Jump parameters
    lambda_: float    # Jump intensity (jumps per year)
    mu_jump: float    # Mean price jump (log jump size)
    sigma_jump: float # Std of price jump
    mu_v_jump: float  # Mean variance jump (usually positive)
    
    # Drift
    mu: float = 0.0
    
    def validate(self) -> bool:
        """Check parameter constraints."""
        # Heston constraints
        if self.kappa <= 0 or self.theta <= 0 or self.xi <= 0:
            return False
        if self.rho <= -1 or self.rho >= 1:
            return False
        
        # Jump constraints
        if self.lambda_ < 0 or self.sigma_jump <= 0:
            return False
        
        return True
    
    @property
    def feller_satisfied(self) -> bool:
        """Check Feller condition: 2κθ ≥ ξ²."""
        return 2 * self.kappa * self.theta >= self.xi ** 2


class SVCJModel:
    """
    Combined Stochastic Volatility + Correlated Jumps model.
    
    This is the most complete statistical model, capturing:
    - Stochastic volatility (Heston)
    - Leverage effect (ρ < 0)
    - Volatility clustering (mean-reverting v)
    - Discontinuous price moves (jumps)
    - Volatility spikes on jumps (correlated J_v)
    
    Uses QE scheme from Heston for variance discretization.
    """
    
    # QE scheme threshold (from Heston)
    PSI_CRIT = 1.5
    
    def __init__(self, params: Optional[SVCJParams] = None):
        self.params = params
        self._fitted = params is not None
    
    def simulate_paths(
        self,
        n_paths: int,
        horizon: int,
        dt: float,  # Time step in years
        initial_price: float,
        initial_variance: float,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate price and variance paths with full SVCJ dynamics.
        
        Uses QE scheme for variance (from Heston) and adds correlated jumps.
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        p = self.params
        
        # Pre-compute constants
        exp_kappa_dt = np.exp(-p.kappa * dt)
        
        # Initialize arrays
        prices = np.zeros((n_paths, horizon + 1))
        variances = np.zeros((n_paths, horizon + 1))
        
        prices[:, 0] = initial_price
        variances[:, 0] = initial_variance
        
        # Pre-generate random numbers for QE scheme and jumps
        u_v = np.random.uniform(size=(n_paths, horizon))
        z_s = np.random.standard_normal((n_paths, horizon))
        z_v = np.random.standard_normal((n_paths, horizon))
        
        for t in range(horizon):
            v_t = variances[:, t]
            
            # 1. Variance update using QE scheme (from Heston)
            m = p.theta + (v_t - p.theta) * exp_kappa_dt
            s2 = (v_t * p.xi**2 * exp_kappa_dt / p.kappa * (1 - exp_kappa_dt) +
                  p.theta * p.xi**2 / (2 * p.kappa) * (1 - exp_kappa_dt)**2)
            psi = s2 / (m**2 + 1e-10)
            
            # QE scheme for variance diffusion
            v_next_diffusion = np.zeros(n_paths)
            
            # Low psi region: quadratic scheme
            low_psi = psi <= self.PSI_CRIT
            if np.any(low_psi):
                inv_psi = 1 / (psi[low_psi] + 1e-10)
                b2 = 2 * inv_psi - 1 + np.sqrt(2 * inv_psi * (2 * inv_psi - 1))
                a = m[low_psi] / (1 + b2)
                b = np.sqrt(b2)
                z_v_low = z_v[low_psi, t]
                v_next_diffusion[low_psi] = a * (b + z_v_low)**2
            
            # High psi region: exponential scheme
            high_psi = ~low_psi
            if np.any(high_psi):
                p_psi = (psi[high_psi] - 1) / (psi[high_psi] + 1)
                beta_exp = (1 - p_psi) / (m[high_psi] + 1e-10)
                
                # Inverse CDF sampling
                u = u_v[high_psi, t]
                v_next_diffusion[high_psi] = np.where(
                    u <= p_psi,
                    0,
                    np.log((1 - p_psi) / (1 - u + 1e-10)) / beta_exp
                )
            
            # Ensure variance stays positive
            v_next_diffusion = np.maximum(v_next_diffusion, 1e-10)
            
            # 2. Jump component
            # Number of jumps in this period
            n_jumps = np.random.poisson(p.lambda_ * dt, n_paths)
            
            # Price jumps
            jump_price = np.zeros(n_paths)
            # Variance jumps (correlated - when price jumps, variance also jumps)
            jump_var = np.zeros(n_paths)
            
            for i in range(n_paths):
                if n_jumps[i] > 0:
                    # Price jump: sum of log jumps
                    log_jumps = np.random.normal(p.mu_jump, p.sigma_jump, n_jumps[i])
                    jump_price[i] = np.sum(log_jumps)
                    
                    # Correlated variance jump: variance jumps up when price jumps
                    # Each jump contributes mu_v_jump to variance
                    jump_var[i] = n_jumps[i] * p.mu_v_jump
            
            # 3. Update variance with diffusion + jumps
            variances[:, t+1] = np.maximum(v_next_diffusion + jump_var, 1e-10)
            
            # 4. Price update with correlation and jumps
            # Correlated Brownian increments
            # dW₁ = ρ dW_v + √(1-ρ²) dZ
            # For simplicity, use z_s as independent component
            dW1 = (p.rho * z_v[:, t] + np.sqrt(1 - p.rho**2) * z_s[:, t]) * np.sqrt(dt)
            
            # Integrated variance for this step (trapezoidal)
            v_avg = 0.5 * (v_t + variances[:, t+1])
            
            # Log price increment: d(log S) = (μ - v/2)dt + √v dW₁ + J_S
            log_return = ((p.mu - 0.5 * v_avg) * dt + 
                         np.sqrt(v_avg) * dW1 + 
                         jump_price)
            
            prices[:, t+1] = prices[:, t] * np.exp(log_return)
        
        return prices, variances
    
    def fit(
        self,
        returns: np.ndarray,
        dt: float,  # Time step in years
        initial_guess: Optional[SVCJParams] = None
    ) -> SVCJParams:
        """
        Estimate SVCJ parameters from historical returns.
        
        Strategy: Fit Heston first, then fit jumps separately.
        This is simpler than joint optimization but gives reasonable results.
        
        For production, consider full joint calibration using likelihood or
        method of moments with more sophisticated optimization.
        """
        # Step 1: Fit Heston model
        heston_model = HestonModel()
        heston_params = heston_model.fit(returns, dt=dt)
        
        # Step 2: Fit jump-diffusion model
        jump_model = JumpDiffusionModel()
        # Use std-based jump detection (5 std threshold)
        # Note: jump_diffusion.fit() uses 'dt' parameter (not 'dt_year')
        jump_params = jump_model.fit(returns, dt=dt, threshold_method='std_multiple', threshold_value=5.0)
        
        # Step 3: Estimate variance jump size (mu_v_jump)
        # Simple heuristic: when large returns occur, check if variance also increases
        # For now, use a fraction of theta (long-run variance) as variance jump size
        # Literature suggests variance jumps are typically 10-50% of long-run variance
        abs_returns = np.abs(returns)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        threshold = 5.0 * std_ret
        large_returns_mask = np.abs(returns - mean_ret) > threshold
        
        if np.sum(large_returns_mask) > 0:
            # Estimate variance jump from realized volatility around large returns
            # For simplicity, use a fraction of theta
            # Typical: mu_v_jump ~ 0.2 * theta to 0.5 * theta
            mu_v_jump = 0.3 * heston_params.theta  # Default: 30% of long-run variance
        else:
            # Default if no large returns detected
            mu_v_jump = 0.3 * heston_params.theta
        
        # Combine parameters
        params = SVCJParams(
            kappa=heston_params.kappa,
            theta=heston_params.theta,
            xi=heston_params.xi,
            rho=heston_params.rho,
            lambda_=jump_params.lambda_,
            mu_jump=jump_params.mu_jump,
            sigma_jump=jump_params.sigma_jump,
            mu_v_jump=mu_v_jump,
            mu=heston_params.mu  # Use Heston's mu (or average of both)
        )
        
        self.params = params
        self._fitted = True
        
        if not params.feller_satisfied:
            logger.warning(
                f"Feller condition violated: 2κθ={2*params.kappa*params.theta:.6f} < "
                f"ξ²={params.xi**2:.6f}. Using full truncation to handle this."
            )
        
        logger.info(
            f"SVCJ fitted: κ={params.kappa:.4f}, θ={params.theta:.8f}, ξ={params.xi:.4f}, "
            f"ρ={params.rho:.4f}, λ={params.lambda_:.2f} jumps/year, "
            f"μ_J={params.mu_jump:.4f}, σ_J={params.sigma_jump:.4f}, "
            f"μ_v,J={params.mu_v_jump:.8f}"
        )
        
        return params
