"""
Extended particle filter for regime-switching GARCH models.

This module implements the particle filter with regime state tracking for
Bayesian updating of regime-switching GARCH models.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

from ..models.regime_config import Instrument
from .regime_path_generator import RegimeSwitchingPathGenerator

logger = logging.getLogger(__name__)


@dataclass
class Particle:
    """
    Single particle representing one simulation path.
    
    Extended for regime-switching GARCH:
    - price: Current simulated price
    - variance: Current conditional variance (within-regime)
    - regime_probs: K-dimensional probability vector over regimes
    - current_regime: Most likely regime (argmax of regime_probs)
    - regime_duration: Bars spent in current regime (for min_duration constraint)
    """
    price: float
    variance: float
    weight: float = 1.0
    
    # Regime state
    regime_probs: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    current_regime: int = 0
    regime_duration: int = 0
    
    # Path history
    price_history: List[float] = field(default_factory=list)
    
    def update_history(self, max_history: int = 100) -> None:
        """Append current price to history, maintaining max length."""
        self.price_history.append(self.price)
        if len(self.price_history) > max_history:
            self.price_history.pop(0)
    
    def transition_regime(
        self,
        new_regime: int,
        min_duration: int = 5
    ) -> None:
        """
        Handle regime transition with minimum duration constraint.
        
        Args:
            new_regime: Proposed new regime
            min_duration: Minimum bars before regime can change
        """
        if new_regime == self.current_regime:
            self.regime_duration += 1
        elif self.regime_duration >= min_duration:
            # Allowed to transition
            self.current_regime = new_regime
            self.regime_duration = 1
        else:
            # Must stay in current regime
            self.regime_duration += 1


class ParticleFilter:
    """
    Particle filter for regime-switching GARCH models.
    
    Performs Bayesian updating of regime probabilities and price/variance
    state using observed prices. Uses systematic resampling to maintain
    particle diversity.
    """
    
    def __init__(
        self,
        instrument: Instrument,
        path_generator: RegimeSwitchingPathGenerator,
        n_particles: int = 10000,
        resample_threshold: float = 0.5
    ):
        """
        Args:
            instrument: Instrument being filtered
            path_generator: RegimeSwitchingPathGenerator instance
            n_particles: Number of particles
            resample_threshold: ESS threshold for resampling (0.5 = 50% of n_particles)
        """
        self.instrument = instrument
        self.path_generator = path_generator
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        
        self.particles: List[Particle] = []
        
    def initialize(
        self,
        initial_price: float,
        initial_variance: float,
        initial_regime_probs: np.ndarray
    ) -> None:
        """
        Initialize particles with initial state.
        
        Args:
            initial_price: Starting price
            initial_variance: Starting variance
            initial_regime_probs: K-dimensional initial regime probabilities
        """
        self.particles = []
        n_regimes = len(initial_regime_probs)
        
        # Sample initial regimes from probabilities
        initial_regimes = np.random.choice(
            n_regimes, size=self.n_particles, p=initial_regime_probs
        )
        
        for i in range(self.n_particles):
            regime_probs = np.zeros(n_regimes)
            regime_probs[initial_regimes[i]] = 1.0
            
            particle = Particle(
                price=initial_price,
                variance=initial_variance,
                weight=1.0 / self.n_particles,
                regime_probs=regime_probs,
                current_regime=initial_regimes[i],
                regime_duration=0
            )
            self.particles.append(particle)
        
        logger.info(f"Initialized {self.n_particles} particles for {self.instrument.value}")
    
    def update(
        self,
        observed_price: float,
        observed_return: float,
        time_of_day_hour: float,
        partner_regime: int = 1,
        rv_ratio: float = 1.0
    ) -> float:
        """
        Update particle weights based on observed price/return.
        
        Likelihood: p(r_t | v_t, regime) ∝ exp(-(r_t - μ)² / (2 v_t)) / √(v_t)
        
        Args:
            observed_price: Actual market price
            observed_return: Log return since last observation
            time_of_day_hour: Current hour (ET, 0-23)
            partner_regime: Current regime of paired instrument (for TVTP)
            rv_ratio: RV_10bar / RV_100bar ratio
            
        Returns:
            Effective Sample Size (ESS)
        """
        for particle in self.particles:
            # Get regime-specific parameters
            regime_k = particle.current_regime
            params = self.path_generator.regime_params.get_regime_params(regime_k)
            
            # Likelihood of observed return given particle's variance and regime
            v = particle.variance
            expected_return = params.mu
            expected_std = np.sqrt(v)
            
            if expected_std > 1e-10:
                # Gaussian likelihood (approximate)
                likelihood = np.exp(-0.5 * ((observed_return - expected_return) / expected_std) ** 2) / expected_std
                particle.weight *= max(likelihood, 1e-10)  # Avoid zero weights
            else:
                # Very small variance - use small likelihood
                particle.weight *= 1e-10
            
            # Also adjust price toward observation (partial adjustment for diversity)
            price_diff = observed_price - particle.price
            particle.price += 0.1 * price_diff
        
        # Normalize weights
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
        else:
            # All weights zero - reset to uniform
            for particle in self.particles:
                particle.weight = 1.0 / self.n_particles
        
        # Calculate ESS
        ess = self._calculate_ess()
        
        # Resample if ESS is too low
        if ess < self.resample_threshold * self.n_particles:
            self.resample()
        
        return ess
    
    def _calculate_ess(self) -> float:
        """Calculate Effective Sample Size."""
        weights = np.array([p.weight for p in self.particles])
        ess = 1.0 / np.sum(weights ** 2)
        return ess
    
    def resample(self) -> None:
        """
        Systematic resampling to maintain particle diversity.
        
        Uses systematic resampling which is more efficient and maintains
        diversity better than multinomial resampling.
        """
        weights = np.array([p.weight for p in self.particles])
        
        # Systematic resampling
        cumulative_weights = np.cumsum(weights)
        u = (np.random.random() + np.arange(self.n_particles)) / self.n_particles
        
        new_particles = []
        j = 0
        for i in range(self.n_particles):
            while u[i] > cumulative_weights[j]:
                j += 1
            # Copy particle (with slight jitter for diversity)
            old_particle = self.particles[j]
            
            # Add small jitter to price and variance
            price_jitter = old_particle.price * 0.001 * np.random.normal()
            variance_jitter = old_particle.variance * 0.01 * np.random.normal()
            
            new_particle = Particle(
                price=max(old_particle.price + price_jitter, 1e-10),
                variance=max(old_particle.variance + variance_jitter, 1e-10),
                weight=1.0 / self.n_particles,
                regime_probs=old_particle.regime_probs.copy(),
                current_regime=old_particle.current_regime,
                regime_duration=old_particle.regime_duration,
                price_history=old_particle.price_history.copy()
            )
            new_particles.append(new_particle)
        
        self.particles = new_particles
        logger.debug(f"Resampled particles (ESS too low)")
    
    def generate_forecast(
        self,
        horizon_bars: int,
        time_of_day_start: float,
        partner_regimes: Optional[np.ndarray] = None,
        rv_ratio: float = 1.0,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecast paths using current particle states.
        
        Args:
            horizon_bars: Number of bars to forecast
            time_of_day_start: Starting hour (ET, 0-23)
            partner_regimes: Optional array of partner instrument regimes
            rv_ratio: RV_10bar / RV_100bar ratio
            random_state: Random seed
            
        Returns:
            Tuple of (prices, variances, regimes) arrays
        """
        # Get current state from particles (weighted average)
        current_price = np.average([p.price for p in self.particles], 
                                   weights=[p.weight for p in self.particles])
        current_variance = np.average([p.variance for p in self.particles],
                                     weights=[p.weight for p in self.particles])
        
        # Get regime probabilities from particles
        n_regimes = len(self.particles[0].regime_probs)
        regime_probs = np.zeros(n_regimes)
        for particle in self.particles:
            regime_probs += particle.weight * particle.regime_probs
        
        # Normalize
        regime_probs = regime_probs / regime_probs.sum()
        
        # Generate paths using path generator
        prices, variances, regimes = self.path_generator.generate_paths(
            n_paths=self.n_particles,
            horizon_bars=horizon_bars,
            initial_price=current_price,
            initial_variance=current_variance,
            initial_regime_probs=regime_probs,
            time_of_day_start=time_of_day_start,
            partner_regimes=partner_regimes,
            rv_ratio=rv_ratio,
            random_state=random_state
        )
        
        return prices, variances, regimes
    
    def get_regime_probabilities(self) -> np.ndarray:
        """
        Get current regime probabilities (weighted average across particles).
        
        Returns:
            K-dimensional array of regime probabilities
        """
        if not self.particles:
            return np.array([1.0])
        
        n_regimes = len(self.particles[0].regime_probs)
        regime_probs = np.zeros(n_regimes)
        
        for particle in self.particles:
            regime_probs += particle.weight * particle.regime_probs
        
        # Normalize
        if regime_probs.sum() > 0:
            regime_probs = regime_probs / regime_probs.sum()
        else:
            regime_probs = np.ones(n_regimes) / n_regimes
        
        return regime_probs
