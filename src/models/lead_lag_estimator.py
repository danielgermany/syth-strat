"""
Cross-instrument lead-lag estimation.

This module estimates lead-lag relationships between instrument pairs using
cross-correlation and Granger causality tests. Results determine contagion
multipliers for regime-switching models.
"""

import numpy as np
from scipy.signal import correlate
from dataclasses import dataclass
from typing import Optional
import logging

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    HAS_GRANGER = True
except ImportError:
    HAS_GRANGER = False
    logging.warning("statsmodels not available - Granger causality tests will be skipped")

logger = logging.getLogger(__name__)


@dataclass
class LeadLagResult:
    """Results from lead-lag estimation between two instruments."""
    
    instrument_1: str
    instrument_2: str
    
    # Cross-correlation results
    cross_corr_peak_lag: int      # Positive = instrument_1 leads instrument_2
    peak_correlation: float
    
    # Granger causality results (p-values)
    p_1_granger_causes_2: float   # p-value: does 1 Granger-cause 2?
    p_2_granger_causes_1: float   # p-value: does 2 Granger-cause 1?
    
    # Interpretation
    is_bidirectional: bool        # Both directions significant
    leader: Optional[str]         # None if bidirectional or unclear
    confidence: str               # "high", "moderate", "low"
    
    # Recommended contagion multipliers
    contagion_1_to_2: float       # Multiplier when 1's regime affects 2
    contagion_2_to_1: float       # Multiplier when 2's regime affects 1


def estimate_lead_lag(
    returns_1: np.ndarray,
    returns_2: np.ndarray,
    instrument_1: str,
    instrument_2: str,
    max_lag: int = 10,
    significance_level: float = 0.05
) -> LeadLagResult:
    """
    Estimate lead-lag relationship between two instruments.
    
    This function uses both cross-correlation and Granger causality to
    determine if one instrument systematically leads the other.
    
    Args:
        returns_1: Log returns of instrument 1
        returns_2: Log returns of instrument 2  
        instrument_1: Name of instrument 1 (e.g., "GC")
        instrument_2: Name of instrument 2 (e.g., "SI")
        max_lag: Maximum lag for Granger causality test
        significance_level: p-value threshold for significance
        
    Returns:
        LeadLagResult with cross-correlation, Granger causality, and
        recommended contagion multipliers.
        
    Research Note:
        Academic evidence on gold-silver and ES-NQ lead-lag is CONFLICTING.
        This function estimates from YOUR data rather than assuming a
        relationship. If results are unclear or bidirectional, we default
        to symmetric contagion multipliers.
    """
    # Ensure equal length
    min_len = min(len(returns_1), len(returns_2))
    r1 = returns_1[-min_len:]
    r2 = returns_2[-min_len:]
    
    # 1. Cross-correlation analysis
    r1_centered = r1 - r1.mean()
    r2_centered = r2 - r2.mean()
    cc = correlate(r1_centered, r2_centered, mode='full')
    cc = cc / (len(r1) * r1.std() * r2.std())  # Normalize
    
    lags = np.arange(-len(r1) + 1, len(r1))
    
    # Find peak within reasonable lag range
    center = len(r1) - 1
    search_range = slice(max(0, center - max_lag), min(len(cc), center + max_lag + 1))
    local_cc = cc[search_range]
    local_lags = lags[search_range]
    
    peak_idx = np.argmax(np.abs(local_cc))
    peak_lag = int(local_lags[peak_idx])
    peak_corr = float(local_cc[peak_idx])
    
    # 2. Granger causality tests
    p_1_causes_2 = 1.0
    p_2_causes_1 = 1.0
    
    if HAS_GRANGER:
        try:
            # Test: does instrument_1 Granger-cause instrument_2?
            # Data format: [effect, cause] for grangercausalitytests
            data_12 = np.column_stack([r2, r1])  # [effect, cause]
            gc_1_causes_2 = grangercausalitytests(data_12, maxlag=max_lag, verbose=False)
            
            # Extract p-value at lag 1 (first lag tested)
            # grangercausalitytests returns dict with keys 1, 2, ..., maxlag
            # Each value is a dict with test results
            if 1 in gc_1_causes_2:
                test_results = gc_1_causes_2[1][0]  # First test result (ssr_ftest)
                p_1_causes_2 = test_results[1]  # p-value
            
            # Test: does instrument_2 Granger-cause instrument_1?
            data_21 = np.column_stack([r1, r2])  # [effect, cause]
            gc_2_causes_1 = grangercausalitytests(data_21, maxlag=max_lag, verbose=False)
            
            if 1 in gc_2_causes_1:
                test_results = gc_2_causes_1[1][0]  # First test result (ssr_ftest)
                p_2_causes_1 = test_results[1]  # p-value
                
        except Exception as e:
            logger.warning(f"Granger causality test failed: {e}. Using defaults.")
            p_1_causes_2 = 1.0
            p_2_causes_1 = 1.0
    else:
        logger.warning("statsmodels not available - skipping Granger causality tests")
    
    # 3. Interpret results
    sig_1_causes_2 = p_1_causes_2 < significance_level
    sig_2_causes_1 = p_2_causes_1 < significance_level
    
    is_bidirectional = sig_1_causes_2 and sig_2_causes_1
    
    # Determine leader (if any)
    leader = None
    confidence = "low"
    
    if is_bidirectional:
        # Both directions significant - no clear leader
        leader = None
        confidence = "moderate"  # Bidirectional is a valid finding
    elif sig_1_causes_2 and not sig_2_causes_1:
        leader = instrument_1
        confidence = "high" if p_1_causes_2 < 0.01 else "moderate"
    elif sig_2_causes_1 and not sig_1_causes_2:
        leader = instrument_2
        confidence = "high" if p_2_causes_1 < 0.01 else "moderate"
    else:
        # Neither significant - independent or weak relationship
        leader = None
        confidence = "low"
    
    # 4. Determine contagion multipliers
    # Default: symmetric (1.4x both directions)
    base_multiplier = 1.4
    
    if leader == instrument_1:
        # 1 leads 2: 1's regime has stronger effect on 2
        contagion_1_to_2 = base_multiplier * 1.2  # 1.68x
        contagion_2_to_1 = base_multiplier * 0.8  # 1.12x
    elif leader == instrument_2:
        # 2 leads 1: 2's regime has stronger effect on 1
        contagion_1_to_2 = base_multiplier * 0.8
        contagion_2_to_1 = base_multiplier * 1.2
    else:
        # Symmetric (bidirectional or unclear)
        contagion_1_to_2 = base_multiplier
        contagion_2_to_1 = base_multiplier
    
    result = LeadLagResult(
        instrument_1=instrument_1,
        instrument_2=instrument_2,
        cross_corr_peak_lag=peak_lag,
        peak_correlation=peak_corr,
        p_1_granger_causes_2=p_1_causes_2,
        p_2_granger_causes_1=p_2_causes_1,
        is_bidirectional=is_bidirectional,
        leader=leader,
        confidence=confidence,
        contagion_1_to_2=contagion_1_to_2,
        contagion_2_to_1=contagion_2_to_1,
    )
    
    logger.info(
        f"Lead-lag estimation {instrument_1}-{instrument_2}: "
        f"leader={leader}, bidirectional={is_bidirectional}, "
        f"confidence={confidence}, "
        f"contagion multipliers: {contagion_1_to_2:.2f}/{contagion_2_to_1:.2f}"
    )
    
    return result
