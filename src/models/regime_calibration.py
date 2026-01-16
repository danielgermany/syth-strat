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
                    # Scale returns for numerical stability (arch recommends 1-1000 range)
                    # Using 1000x to avoid scaling warnings
                    scale = 1000
                    
                    # Suppress DataScaleWarning since we're handling scaling manually
                    import warnings
                    try:
                        from arch.univariate.base import DataScaleWarning
                    except ImportError:
                        # If DataScaleWarning doesn't exist, create a dummy class
                        class DataScaleWarning(UserWarning):
                            pass
                    
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
                    
                    # Validate data before optimization
                    if np.any(np.isnan(returns_k)) or np.any(np.isinf(returns_k)):
                        raise ValueError(f"Returns contain NaN or Inf values for {instrument} regime {k}")
                    
                    if np.var(returns_k * scale) < 1e-10:
                        raise ValueError(f"Returns have near-zero variance for {instrument} regime {k}")
                    
                    # Fit the model with convergence options
                    # arch package uses scipy.optimize internally and doesn't accept 'method' parameter
                    # We can only control convergence via 'options' dict
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=DataScaleWarning)
                        try:
                            result = am.fit(
                                disp='off',
                                show_warning=False,
                                options={'maxiter': 1000, 'ftol': 1e-6}
                            )
                        except Exception as e:
                            raise RuntimeError(f"GARCH optimization failed: {e}")
                    
                    # Extract parameters using pandas Series access
                    # For GJR-GARCH: ['mu', 'omega', 'alpha[1]', 'gamma[1]', 'beta[1]', 'eta', 'lambda']
                    # For standard GARCH: ['mu', 'omega', 'alpha[1]', 'beta[1]', 'nu']
                    
                    omega = float(result.params.get('omega', 0.01)) / (scale ** 2)
                    alpha_raw = float(result.params.get('alpha[1]', 0.1))
                    beta_raw = float(result.params.get('beta[1]', 0.85))
                    mu_scaled = float(result.params.get('mu', 0.0))
                    mu = mu_scaled / scale
                    
                    # Clamp mu to reasonable range for 1-minute returns
                    # Typical 1-min returns are on order of 1e-5 (1 bp) to 1e-4 (10 bp)
                    # Cap mu at ±0.0005 (50 bp) which is extreme but possible
                    mu = np.clip(mu, -0.0005, 0.0005)
                    
                    # GJR-specific parameters
                    if use_gjr:
                        gamma_raw = float(result.params.get('gamma[1]', 0.0))
                        # Skewed-t uses 'eta' for degrees of freedom, 'lambda' for skewness
                        nu = float(result.params.get('eta', 6.0))  # eta, not nu
                        skew = float(result.params.get('lambda', 0.0))
                    else:
                        gamma_raw = 0.0
                        # GED uses 'nu' for shape parameter
                        nu = float(result.params.get('nu', 1.5))
                        skew = 0.0
                    
                    # Ensure non-negative (but don't force minimums that break stationarity)
                    omega = max(omega, 1e-8)  # Only omega needs minimum > 0
                    alpha_raw = max(alpha_raw, 0.0)  # Just ensure non-negative
                    beta_raw = max(beta_raw, 0.0)
                    gamma_raw = max(gamma_raw, 0.0)
                    
                    # Check and enforce stationarity: α + β + γ/2 < 1
                    persistence = alpha_raw + beta_raw + gamma_raw / 2
                    if persistence >= 0.999:
                        # Scale down proportionally to enforce stationarity (target 0.99)
                        scale_factor = 0.99 / persistence
                        alpha = alpha_raw * scale_factor
                        beta = beta_raw * scale_factor
                        gamma = gamma_raw * scale_factor
                    else:
                        alpha = alpha_raw
                        beta = beta_raw
                        gamma = gamma_raw
                    
                    # Ensure nu is reasonable (for skewed-t, nu > 2 for finite variance; for GED, nu >= 1.5 to avoid overflow)
                    if use_gjr:
                        nu = max(nu, 2.5)  # Skewed-t: need nu > 2
                    else:
                        nu = max(nu, 1.5)  # GED: need nu >= 1.5 to avoid numerical overflow in likelihood calculation
                    
                    params = GARCHParams(
                        omega=omega,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                        mu=mu,
                        nu=nu,
                        skew=np.clip(skew, -0.5, 0.5)
                    )
                    
                    # Verify stationarity
                    final_persistence = params.alpha + params.beta + params.gamma / 2
                    
                    # Format omega with scientific notation if very small
                    if params.omega < 1e-5:
                        omega_str = f"ω={params.omega:.2e}"
                    else:
                        omega_str = f"ω={params.omega:.6f}"
                    
                    logger.info(
                        f"{instrument.value} Regime {k}: "
                        f"{omega_str}, α={params.alpha:.3f}, "
                        f"β={params.beta:.3f}, γ={params.gamma:.3f}, "
                        f"μ={params.mu:.6f}, ν={params.nu:.2f}, "
                        f"skew={params.skew:.3f}, persistence={final_persistence:.4f}"
                    )
                    
                    if final_persistence >= 1.0:
                        logger.warning(
                            f"{instrument.value} Regime {k}: Stationarity violation! "
                            f"Persistence={final_persistence:.4f} >= 1.0"
                        )
                    
                    regime_params_list.append(params)
                    
                except Exception as e:
                    logger.warning(f"GARCH estimation failed for {instrument} regime {k}: {e}")
                    logger.debug(f"  Returns length: {len(returns_k)}, variance: {np.var(returns_k):.2e}, "
                                f"mean: {np.mean(returns_k):.2e}")
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
