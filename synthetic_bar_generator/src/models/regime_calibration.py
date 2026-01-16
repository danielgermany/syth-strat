"""
Regime-switching GARCH calibration.

This module implements the calibration pipeline for regime-switching GARCH models:
1. Regime detection using Markov-switching models
2. Per-regime GARCH parameter estimation
3. Transition matrix extraction
4. Cross-instrument lead-lag estimation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logging.warning("statsmodels not available - Markov-switching will use fallback")

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    logging.warning("arch not available - GARCH estimation will use defaults")

from .regime_config import Instrument, RegimeConfig, TimeframeConfig, Timeframe
from .garch_params import GARCHParams, RegimeGARCHParams
from .transition_matrix import TransitionMatrix
from .lead_lag_estimator import estimate_lead_lag, LeadLagResult

logger = logging.getLogger(__name__)


class RegimeSwitchingCalibrator:
    """
    Calibrates regime-switching GARCH models with TVTP.
    
    Calibration proceeds in phases:
    1. Regime Detection: Use Markov-switching model to identify regimes
    2. Per-Regime GARCH: Estimate GARCH parameters within each regime
    3. Transition Matrix: Extract base transition probabilities
    4. Cross-Instrument: Estimate lead-lag for contagion multipliers
    
    The calibration is performed at 1-minute frequency, then parameters
    are used for all timeframes (aggregation handles the rest).
    """
    
    def __init__(
        self,
        instruments: List[Instrument],
        regime_configs: Optional[Dict[Instrument, RegimeConfig]] = None,
        timeframe_config: Optional[TimeframeConfig] = None,
    ):
        """
        Args:
            instruments: List of instruments to calibrate
            regime_configs: Per-instrument regime configuration (uses defaults if None)
            timeframe_config: Timeframe configuration (defaults to 1-minute)
        """
        self.instruments = instruments
        self.regime_configs = regime_configs or {
            inst: RegimeConfig.default_for_instrument(inst) 
            for inst in instruments
        }
        self.timeframe_config = timeframe_config or TimeframeConfig.default_for_timeframe(
            Timeframe.M1
        )
        
        # Calibration results
        self.regime_params: Dict[Instrument, RegimeGARCHParams] = {}
        self.transition_matrices: Dict[Instrument, TransitionMatrix] = {}
        self.lead_lag_results: Dict[Tuple[Instrument, Instrument], LeadLagResult] = {}
        self.calibration_date: Optional[datetime] = None
        self._last_data: Dict[Instrument, pd.DataFrame] = {}  # Store last calibration data
    
    def calibrate_all(
        self,
        data: Dict[Instrument, pd.DataFrame],
        as_of_date: Optional[datetime] = None
    ) -> None:
        """
        Run full calibration for all instruments.
        
        Args:
            data: Dict mapping Instrument to DataFrame with 'close' and 'volume' columns
            as_of_date: Calibration date (uses end of data if None)
        """
        self.calibration_date = as_of_date or datetime.now()
        self._last_data = data  # Store for use in calibration script
        
        # Phase 1 & 2: Regime detection and per-regime GARCH for each instrument
        for instrument in self.instruments:
            logger.info(f"Calibrating {instrument.value}...")
            
            df = data[instrument]
            config = self.regime_configs[instrument]
            
            # Calculate returns
            returns = np.log(df['close'] / df['close'].shift(1)).dropna().values
            
            # Detect regimes and estimate parameters
            regime_labels, transition_matrix = self._detect_regimes(
                returns, instrument, config.n_regimes
            )
            
            # Estimate per-regime GARCH parameters
            regime_garch = self._estimate_regime_garch(
                returns, regime_labels, instrument, config.n_regimes
            )
            
            self.regime_params[instrument] = regime_garch
            self.transition_matrices[instrument] = transition_matrix
        
        # Phase 3: Cross-instrument lead-lag estimation
        self._calibrate_cross_instrument(data)
        
        logger.info("Calibration complete")
    
    def _detect_regimes(
        self,
        returns: np.ndarray,
        instrument: Instrument,
        n_regimes: int
    ) -> Tuple[np.ndarray, TransitionMatrix]:
        """
        Detect volatility regimes using Markov-switching model.
        
        Uses statsmodels.tsa.regime_switching.MarkovRegression with
        switching_variance=True to identify different volatility states.
        """
        # Fit Markov-switching model
        if HAS_STATSMODELS:
            try:
                mod = sm.tsa.MarkovRegression(
                    returns * 100,  # Scale for numerical stability
                    k_regimes=n_regimes,
                    trend='n',
                    switching_variance=True
                )
                result = mod.fit(disp=False)
                
                # Extract regime probabilities and most likely sequence
                regime_probs = result.smoothed_marginal_probabilities
                regime_labels = np.argmax(regime_probs, axis=1)
                
                # Extract transition matrix
                base_matrix = result.regime_transition
                
                # Sort regimes by variance (regime 0 = lowest vol)
                regime_variances = []
                for k in range(n_regimes):
                    mask = regime_labels == k
                    if mask.sum() > 0:
                        regime_variances.append(returns[mask].var())
                    else:
                        regime_variances.append(0)
                
                sort_order = np.argsort(regime_variances)
                
                # Remap regime labels
                remap = {old: new for new, old in enumerate(sort_order)}
                regime_labels = np.array([remap[r] for r in regime_labels])
                
                # Reorder transition matrix
                base_matrix = base_matrix[sort_order][:, sort_order]
                
            except Exception as e:
                logger.warning(f"Markov switching failed for {instrument}: {e}. Using fallback.")
                # Fallback: simple variance-based regime assignment
                regime_labels = self._simple_regime_detection(returns, n_regimes)
                base_matrix = TransitionMatrix.default_for_instrument(
                    instrument, n_regimes
                ).base_matrix
        else:
            logger.warning(f"statsmodels not available for {instrument}. Using fallback.")
            # Fallback: simple variance-based regime assignment
            regime_labels = self._simple_regime_detection(returns, n_regimes)
            base_matrix = TransitionMatrix.default_for_instrument(
                instrument, n_regimes
            ).base_matrix
        
        transition_matrix = TransitionMatrix(
            base_matrix=base_matrix,
            instrument=instrument
        )
        
        return regime_labels, transition_matrix
    
    def _simple_regime_detection(
        self,
        returns: np.ndarray,
        n_regimes: int
    ) -> np.ndarray:
        """
        Fallback regime detection using rolling volatility percentiles.
        """
        # Calculate rolling volatility
        window = 60  # 1 hour of 1-min bars
        rolling_vol = pd.Series(returns).rolling(window).std().values
        
        # Assign regimes by percentile
        regime_labels = np.zeros(len(returns), dtype=int)
        percentiles = np.linspace(0, 100, n_regimes + 1)[1:-1]
        thresholds = np.nanpercentile(rolling_vol, percentiles)
        
        for i, threshold in enumerate(thresholds):
            regime_labels[rolling_vol > threshold] = i + 1
        
        return regime_labels
    
    def _estimate_regime_garch(
        self,
        returns: np.ndarray,
        regime_labels: np.ndarray,
        instrument: Instrument,
        n_regimes: int
    ) -> RegimeGARCHParams:
        """
        Estimate GARCH parameters for each regime separately.
        
        For ES/NQ: Use GJR-GARCH (asymmetric) with skewed-t distribution
        For GC/SI: Use standard GARCH (symmetric) with GED distribution
        """
        regime_params_list = []
        
        use_gjr = instrument.use_asymmetric_garch
        
        for k in range(n_regimes):
            mask = regime_labels == k
            returns_k = returns[mask]
            
            if len(returns_k) < 100:
                # Insufficient data - use defaults scaled by regime
                logger.warning(f"Insufficient data for {instrument} regime {k}, using defaults")
                default = RegimeGARCHParams.default_for_instrument(instrument)
                regime_params_list.append(default.regime_params[k])
                continue
            
            if HAS_ARCH:
                try:
                    # Scale returns for numerical stability
                    scale = 100
                    
                    if use_gjr:
                        # GJR-GARCH for ES/NQ
                        am = arch_model(
                            returns_k * scale,
                            vol='GARCH',
                            p=1, o=1, q=1,  # o=1 gives GJR term
                            dist='skewt'
                        )
                    else:
                        # Standard GARCH for GC/SI
                        am = arch_model(
                            returns_k * scale,
                            vol='GARCH',
                            p=1, q=1,
                            dist='ged'
                        )
                    
                    result = am.fit(disp='off')
                    
                    # Extract parameters (scale back omega)
                    omega = result.params.get('omega', 0.01) / (scale ** 2)
                    alpha = result.params.get('alpha[1]', 0.1)
                    beta = result.params.get('beta[1]', 0.85)
                    gamma = result.params.get('gamma[1]', 0.0) if use_gjr else 0.0
                    mu = result.params.get('mu', 0.0) / scale
                    nu = result.params.get('nu', 6.0)
                    skew = result.params.get('lambda', 0.0) if use_gjr else 0.0
                    
                    # Validate stationarity
                    if alpha + beta + gamma / 2 >= 0.999:
                        total = alpha + beta + gamma / 2
                        factor = 0.99 / total
                        alpha *= factor
                        beta *= factor
                        gamma *= factor
                    
                    params = GARCHParams(
                        omega=max(omega, 1e-8),
                        alpha=max(alpha, 0.01),
                        beta=max(beta, 0.5),
                        gamma=max(gamma, 0.0),
                        mu=mu,
                        nu=max(nu, 2.5),
                        skew=np.clip(skew, -0.5, 0.5)
                    )
                    
                    logger.info(
                        f"{instrument.value} Regime {k}: "
                        f"ω={params.omega:.6f}, α={params.alpha:.3f}, "
                        f"β={params.beta:.3f}, γ={params.gamma:.3f}"
                    )
                    
                    regime_params_list.append(params)
                    
                except Exception as e:
                    logger.warning(f"GARCH estimation failed for {instrument} regime {k}: {e}")
                    default = RegimeGARCHParams.default_for_instrument(instrument)
                    regime_params_list.append(default.regime_params[k])
            else:
                # arch package not available - use defaults
                logger.warning(f"arch package not available for {instrument} regime {k}. Using defaults.")
                default = RegimeGARCHParams.default_for_instrument(instrument)
                regime_params_list.append(default.regime_params[k])
        
        return RegimeGARCHParams(
            instrument=instrument,
            regime_params=regime_params_list
        )
    
    def _calibrate_cross_instrument(
        self,
        data: Dict[Instrument, pd.DataFrame]
    ) -> None:
        """
        Estimate cross-instrument lead-lag relationships.
        
        Pairs: GC↔SI, ES↔NQ
        
        Research Note: We estimate from data rather than assuming a direction
        because academic evidence is conflicting.
        """
        pairs = [
            (Instrument.GC, Instrument.SI),
            (Instrument.ES, Instrument.NQ),
        ]
        
        for inst1, inst2 in pairs:
            if inst1 not in data or inst2 not in data:
                continue
            
            returns_1 = np.log(data[inst1]['close'] / data[inst1]['close'].shift(1)).dropna().values
            returns_2 = np.log(data[inst2]['close'] / data[inst2]['close'].shift(1)).dropna().values
            
            result = estimate_lead_lag(
                returns_1, returns_2,
                inst1.value, inst2.value,
                max_lag=10
            )
            
            self.lead_lag_results[(inst1, inst2)] = result
            
            # Update contagion multipliers in transition matrices
            if inst1 in self.transition_matrices:
                self.transition_matrices[inst1].contagion_multiplier = result.contagion_2_to_1
            if inst2 in self.transition_matrices:
                self.transition_matrices[inst2].contagion_multiplier = result.contagion_1_to_2
    
    def get_contagion_multiplier(
        self,
        from_instrument: Instrument,
        to_instrument: Instrument
    ) -> float:
        """
        Get the calibrated contagion multiplier for regime spillover.
        
        Args:
            from_instrument: Instrument whose regime affects the other
            to_instrument: Instrument being affected
            
        Returns:
            Contagion multiplier (default 1.4 if not calibrated)
        """
        # Check both orderings
        key1 = (from_instrument, to_instrument)
        key2 = (to_instrument, from_instrument)
        
        if key1 in self.lead_lag_results:
            return self.lead_lag_results[key1].contagion_1_to_2
        elif key2 in self.lead_lag_results:
            return self.lead_lag_results[key2].contagion_2_to_1
        else:
            return 1.4  # Default symmetric
