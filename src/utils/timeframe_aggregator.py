"""
Multi-timeframe aggregation utilities.

This module aggregates 1-minute generated paths to higher timeframes (5m, 15m, 1h).
Strategy: Generate at 1-minute resolution (most granular), then aggregate OHLCV
to higher timeframes as needed. This ensures consistency across timeframes while
only calibrating once at 1-minute.
"""

import numpy as np
from typing import Dict

from ..models.regime_config import Timeframe


class TimeframeAggregator:
    """
    Aggregate 1-minute generated paths to higher timeframes.
    
    Strategy: Generate at 1-minute resolution (most granular), then
    aggregate OHLCV to 5m, 15m, 1h as needed. This ensures consistency
    across timeframes while only calibrating once at 1-minute.
    """
    
    @staticmethod
    def aggregate_paths(
        prices_1m: np.ndarray,
        target_timeframe: Timeframe,
    ) -> np.ndarray:
        """
        Aggregate 1-minute price paths to target timeframe.
        
        Args:
            prices_1m: (n_paths, n_bars_1m) array of 1-minute prices
            target_timeframe: Target timeframe to aggregate to
            
        Returns:
            (n_paths, n_bars_target) array of close prices at target timeframe
        """
        if target_timeframe == Timeframe.M1:
            return prices_1m
        
        n_paths, n_bars_1m = prices_1m.shape
        agg_minutes = target_timeframe.minutes
        n_bars_target = n_bars_1m // agg_minutes
        
        if n_bars_target == 0:
            # Not enough data for even one bar
            return np.array([]).reshape(n_paths, 0)
        
        # Take close price (last price in each aggregation window)
        prices_target = np.zeros((n_paths, n_bars_target))
        
        for i in range(n_bars_target):
            start_idx = i * agg_minutes
            end_idx = (i + 1) * agg_minutes
            # Close = last price in window
            prices_target[:, i] = prices_1m[:, end_idx - 1]
        
        return prices_target
    
    @staticmethod
    def aggregate_ohlcv(
        prices_1m: np.ndarray,
        target_timeframe: Timeframe,
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate 1-minute price paths to OHLCV at target timeframe.
        
        Args:
            prices_1m: (n_paths, n_bars_1m) array of 1-minute prices
            target_timeframe: Target timeframe
            
        Returns:
            Dict with 'open', 'high', 'low', 'close' arrays
        """
        if target_timeframe == Timeframe.M1:
            return {
                'open': prices_1m[:, :-1],
                'high': prices_1m[:, :-1],
                'low': prices_1m[:, :-1],
                'close': prices_1m[:, 1:],
            }
        
        n_paths, n_bars_1m = prices_1m.shape
        agg_minutes = target_timeframe.minutes
        n_bars_target = (n_bars_1m - 1) // agg_minutes
        
        if n_bars_target == 0:
            # Not enough data for even one bar
            return {
                'open': np.array([]).reshape(n_paths, 0),
                'high': np.array([]).reshape(n_paths, 0),
                'low': np.array([]).reshape(n_paths, 0),
                'close': np.array([]).reshape(n_paths, 0),
            }
        
        opens = np.zeros((n_paths, n_bars_target))
        highs = np.zeros((n_paths, n_bars_target))
        lows = np.zeros((n_paths, n_bars_target))
        closes = np.zeros((n_paths, n_bars_target))
        
        for i in range(n_bars_target):
            start_idx = i * agg_minutes
            end_idx = (i + 1) * agg_minutes + 1  # +1 because we need close of last bar
            
            # Ensure we don't go out of bounds
            end_idx = min(end_idx, n_bars_1m)
            if start_idx >= end_idx:
                break
                
            window = prices_1m[:, start_idx:end_idx]
            
            opens[:, i] = window[:, 0]
            highs[:, i] = window.max(axis=1)
            lows[:, i] = window.min(axis=1)
            closes[:, i] = window[:, -1]
        
        return {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
        }
