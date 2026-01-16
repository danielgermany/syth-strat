"""
Regime-switching configuration and enumerations.

This module defines the core type system for regime-switching GARCH models:
- Timeframe: Supported generation timeframes (1m, 5m, 15m, 1h)
- Instrument: Supported instruments with pair relationships (ES↔NQ, GC↔SI)
- RegimeConfig: Per-instrument regime configuration
- TimeframeConfig: Timeframe-specific calibration settings
"""

from enum import Enum
from dataclasses import dataclass
from typing import List


class Timeframe(Enum):
    """Supported generation timeframes."""
    M1 = "1m"    # 1,380 bars/day (23h futures)
    M5 = "5m"    # 276 bars/day
    M15 = "15m"  # 92 bars/day
    H1 = "1h"    # 23 bars/day
    
    @property
    def minutes(self) -> int:
        """Return timeframe in minutes."""
        mapping = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
        return mapping[self.value]
    
    @property
    def bars_per_day(self) -> int:
        """Approximate bars per trading day (23 hours)."""
        return (23 * 60) // self.minutes


class Instrument(Enum):
    """Supported instruments with pair relationships."""
    ES = "ES"
    NQ = "NQ"
    GC = "GC"
    SI = "SI"
    
    @property
    def pair(self) -> 'Instrument':
        """Return the paired instrument for cross-correlation."""
        pairs = {
            Instrument.ES: Instrument.NQ,
            Instrument.NQ: Instrument.ES,
            Instrument.GC: Instrument.SI,
            Instrument.SI: Instrument.GC,
        }
        return pairs[self]
    
    @property
    def is_equity_index(self) -> bool:
        """True for ES/NQ (equity indices)."""
        return self in [Instrument.ES, Instrument.NQ]
    
    @property
    def is_precious_metal(self) -> bool:
        """True for GC/SI (precious metals)."""
        return self in [Instrument.GC, Instrument.SI]
    
    @property
    def use_asymmetric_garch(self) -> bool:
        """
        ES/NQ: Use asymmetric (skewed-t) for leverage effect.
        GC/SI: Use symmetric (GED) - research shows no leverage effect.
        """
        return self.is_equity_index


@dataclass
class RegimeConfig:
    """Configuration for regime-switching model per instrument."""
    
    instrument: Instrument
    n_regimes: int                    # K regimes (3 for ES/NQ/GC, 4 for SI)
    regime_names: List[str]           # Human-readable names
    min_duration_bars: int            # Minimum bars before regime can switch
    
    # TVTP adjustment flags
    use_time_of_day: bool = True      # Adjust transitions by session
    use_cross_instrument: bool = True # Adjust based on partner regime
    use_lagged_rv: bool = True        # Adjust based on recent realized vol
    
    @classmethod
    def default_for_instrument(cls, instrument: Instrument) -> 'RegimeConfig':
        """Create default configuration for an instrument."""
        if instrument == Instrument.SI:
            # SI needs 4 regimes due to extreme tail events (flash crashes)
            return cls(
                instrument=instrument,
                n_regimes=4,
                regime_names=["Low", "Normal", "High", "Crisis"],
                min_duration_bars=5,
            )
        else:
            # ES, NQ, GC use 3 regimes
            return cls(
                instrument=instrument,
                n_regimes=3,
                regime_names=["Low", "Normal", "High"],
                min_duration_bars=5,
            )


@dataclass
class TimeframeConfig:
    """Configuration for a specific generation timeframe."""
    
    bar_size: Timeframe
    calibration_frequency: str    # '4h' | 'daily' | 'weekly'
    lookback_bars: int            # Historical bars for calibration
    min_regime_duration: int      # Minimum bars in regime
    
    @classmethod
    def default_for_timeframe(cls, tf: Timeframe) -> 'TimeframeConfig':
        """Create default configuration for a timeframe."""
        configs = {
            Timeframe.M1: cls(
                bar_size=tf,
                calibration_frequency='daily',
                lookback_bars=100_000,  # ~70 days of 1-min data
                min_regime_duration=5,
            ),
            Timeframe.M5: cls(
                bar_size=tf,
                calibration_frequency='daily',
                lookback_bars=20_000,
                min_regime_duration=5,
            ),
            Timeframe.M15: cls(
                bar_size=tf,
                calibration_frequency='daily',
                lookback_bars=6_000,
                min_regime_duration=3,
            ),
            Timeframe.H1: cls(
                bar_size=tf,
                calibration_frequency='weekly',
                lookback_bars=2_000,
                min_regime_duration=1,
            ),
        }
        return configs[tf]
