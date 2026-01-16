"""
Transition matrix with time-varying transition probabilities (TVTP).

This module implements the Markov transition matrix for regime-switching models
with TVTP adjustments based on time-of-day, cross-instrument contagion, and
realized volatility.
"""

import numpy as np
from dataclasses import dataclass

from .regime_config import Instrument


@dataclass
class TransitionMatrix:
    """
    Markov transition matrix with time-varying transition probabilities (TVTP).
    
    The base matrix P[i,j] gives probability of transitioning from regime i to regime j.
    TVTP adjusts these probabilities based on:
        1. Time-of-day effects (session open/close)
        2. Cross-instrument contagion (partner's regime)
        3. Recent realized volatility ratio
    
    Research Note: Cross-instrument contagion uses SYMMETRIC bidirectional
    adjustments (equal multipliers both directions) because academic evidence
    shows conflicting results on which instrument leads.
    """
    
    base_matrix: np.ndarray  # K × K base transition probabilities
    instrument: Instrument
    
    # TVTP adjustment parameters
    tod_high_vol_boost: float = 1.3    # Multiplier for high-vol transitions at key times
    contagion_multiplier: float = 1.4  # SYMMETRIC - same both directions
    rv_adjustment_strength: float = 0.2  # How much RV ratio affects transitions
    
    def __post_init__(self):
        """Validate transition matrix."""
        assert self.base_matrix.shape[0] == self.base_matrix.shape[1], \
            "Transition matrix must be square"
        row_sums = self.base_matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0), \
            f"Rows must sum to 1, got {row_sums}"
    
    @property
    def n_regimes(self) -> int:
        return self.base_matrix.shape[0]
    
    def get_adjusted(
        self,
        time_of_day_hour: float,
        partner_regime: int,
        rv_ratio: float,
    ) -> np.ndarray:
        """
        Apply TVTP adjustments and return renormalized matrix.
        
        Args:
            time_of_day_hour: Current hour (0-23, fractional OK) in ET
            partner_regime: Current regime of paired instrument (for contagion)
            rv_ratio: RV_10bar / RV_100bar ratio (>1 means recent vol elevated)
            
        Returns:
            Adjusted and renormalized K × K transition matrix
        """
        P = self.base_matrix.copy()
        
        # 1. Time-of-day adjustments
        P = self._apply_tod_adjustment(P, time_of_day_hour)
        
        # 2. Cross-instrument contagion (SYMMETRIC)
        P = self._apply_contagion(P, partner_regime)
        
        # 3. Realized volatility adjustment
        P = self._apply_rv_adjustment(P, rv_ratio)
        
        # Renormalize rows to sum to 1
        row_sums = P.sum(axis=1, keepdims=True)
        P = P / row_sums
        
        return P
    
    def _apply_tod_adjustment(
        self, 
        P: np.ndarray, 
        hour: float
    ) -> np.ndarray:
        """
        Adjust transition probabilities based on time of day.
        
        Key volatility windows (hours in ET):
        - ES/NQ: RTH open (9:30), European close (11:30), US close (16:00)
        - GC/SI: London open (3:00), PM Fix (10:00)
        
        During these windows, increase probability of transitioning to higher regimes.
        """
        P = P.copy()
        boost = 1.0
        
        if self.instrument.is_equity_index:
            # ES/NQ volatility windows
            if 9.0 <= hour <= 10.0:     # RTH open
                boost = self.tod_high_vol_boost
            elif 11.0 <= hour <= 12.0:  # European close
                boost = self.tod_high_vol_boost * 0.8
            elif 15.5 <= hour <= 16.5:  # US close
                boost = self.tod_high_vol_boost * 0.9
        else:
            # GC/SI volatility windows
            if 2.5 <= hour <= 4.0:      # London open
                boost = self.tod_high_vol_boost
            elif 9.5 <= hour <= 10.5:   # PM Fix
                boost = self.tod_high_vol_boost
        
        if boost > 1.0:
            # Increase transitions TO higher regimes (last column = highest)
            for i in range(self.n_regimes):
                # Boost transitions to regimes higher than current
                for j in range(i + 1, self.n_regimes):
                    P[i, j] *= boost
        
        return P
    
    def _apply_contagion(
        self, 
        P: np.ndarray, 
        partner_regime: int
    ) -> np.ndarray:
        """
        Apply cross-instrument contagion effect.
        
        SYMMETRIC BIDIRECTIONAL CONTAGION:
        If partner is in a higher regime than us, increase our probability
        of transitioning UP to match. This applies equally in both directions
        (GC→SI and SI→GC get the same multiplier) because research shows
        no consistent leader.
        
        Research basis:
        - Some studies find gold leads silver (spillover direction)
        - Other studies find silver leads gold (Lau et al. 2017)
        - Studies find bidirectional causality at different time horizons
        - Solution: Use symmetric multipliers and let data drive dynamics
        """
        P = P.copy()
        
        # For each current regime, if partner is in higher regime,
        # boost our transitions toward higher regimes
        for i in range(self.n_regimes):
            if partner_regime > i:
                # Partner in higher vol regime → boost our upward transitions
                for j in range(i + 1, self.n_regimes):
                    P[i, j] *= self.contagion_multiplier
        
        return P
    
    def _apply_rv_adjustment(
        self, 
        P: np.ndarray, 
        rv_ratio: float
    ) -> np.ndarray:
        """
        Adjust based on recent realized volatility ratio.
        
        rv_ratio = RV_10bar / RV_100bar
        - If rv_ratio > 1.5: Recent vol elevated → boost upward transitions
        - If rv_ratio < 0.7: Recent vol depressed → boost downward transitions
        """
        P = P.copy()
        
        if rv_ratio > 1.5:
            # Elevated recent vol → increase upward transition probability
            adjustment = 1.0 + self.rv_adjustment_strength * (rv_ratio - 1.0)
            for i in range(self.n_regimes):
                for j in range(i + 1, self.n_regimes):
                    P[i, j] *= adjustment
                    
        elif rv_ratio < 0.7:
            # Depressed recent vol → increase downward transition probability
            adjustment = 1.0 + self.rv_adjustment_strength * (1.0 - rv_ratio)
            for i in range(1, self.n_regimes):
                for j in range(i):
                    P[i, j] *= adjustment
        
        return P
    
    @classmethod
    def default_for_instrument(cls, instrument: Instrument, n_regimes: int) -> 'TransitionMatrix':
        """
        Create default transition matrix with high self-transition probability.
        
        Diagonal elements ~0.95-0.98 ensures regimes are persistent.
        Off-diagonal transitions favor adjacent regimes.
        """
        P = np.zeros((n_regimes, n_regimes))
        
        # High self-transition probability (regime persistence)
        self_prob = 0.97
        
        for i in range(n_regimes):
            P[i, i] = self_prob
            remaining = 1.0 - self_prob
            
            # Distribute remaining probability to adjacent regimes
            neighbors = []
            if i > 0:
                neighbors.append(i - 1)
            if i < n_regimes - 1:
                neighbors.append(i + 1)
            
            if neighbors:
                for j in neighbors:
                    P[i, j] = remaining / len(neighbors)
        
        return cls(base_matrix=P, instrument=instrument)
