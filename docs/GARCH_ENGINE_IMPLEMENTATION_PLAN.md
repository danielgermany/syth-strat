# GARCH Engine Implementation Plan
## Based on synthetic_bar_generator_development_plan_v2.md

**Reference:** `synthetic_bar_generator_development_plan_v2.md` Section 3 (Phase 1: GARCH MVP)

---

## Overview

This plan implements a **regime-switching GARCH** engine with:
- Time-varying transition probabilities (TVTP)
- Symmetric bidirectional cross-instrument contagion
- Multi-timeframe support (1m base, aggregate to 5m, 15m, 1h)
- Extended particle filter with regime state

**Key Difference from Simple GARCH:**
- Not a single-regime GARCH model
- Multiple volatility regimes per instrument (3 for ES/NQ/GC, 4 for SI)
- Regimes switch based on TVTP adjustments (time-of-day, cross-instrument, realized vol)
- Cross-instrument contagion affects regime transitions

---

## Implementation Phases

### Phase 1: Core Data Structures and Configuration (Foundation)
**Goal:** Establish the type system and configuration framework

#### 1.1 Enumerations and Base Classes
**Files:**
- `src/models/regime_config.py`

**Components:**
- `Timeframe` enum (M1, M5, M15, H1)
- `Instrument` enum (ES, NQ, GC, SI) with pair relationships
- `RegimeConfig` dataclass (per-instrument regime configuration)
- `TimeframeConfig` dataclass (timeframe-specific settings)

**Dependencies:** None

**Validation:**
- Unit tests for enum properties
- Test default configurations for all instruments
- Verify pair relationships (ES↔NQ, GC↔SI)

---

#### 1.2 GARCH Parameter Structures
**Files:**
- `src/models/garch_params.py`

**Components:**
- `GARCHParams` dataclass (single-regime parameters)
  - omega, alpha, beta, gamma (leverage), mu
  - nu (degrees of freedom / GED shape)
  - skew (for skewed-t distribution)
  - Validation methods
  - Property methods (persistence, unconditional_variance)
- `RegimeGARCHParams` dataclass (all regimes for an instrument)
  - List of GARCHParams (one per regime)
  - Default factory methods per instrument

**Dependencies:** `regime_config.py` (Instrument enum)

**Validation:**
- Parameter constraint validation (stationarity, positivity)
- Default parameter generation for all instruments
- Verify ES/NQ use gamma > 0 (GJR-GARCH)
- Verify GC/SI use gamma = 0 (symmetric)

---

#### 1.3 Transition Matrix with TVTP
**Files:**
- `src/models/transition_matrix.py`

**Components:**
- `TransitionMatrix` class
  - Base K×K transition matrix
  - `get_adjusted()` method (applies TVTP)
  - `_apply_tod_adjustment()` (time-of-day effects)
  - `_apply_contagion()` (cross-instrument effects)
  - `_apply_rv_adjustment()` (realized volatility effects)
  - Default factory method

**Dependencies:** `regime_config.py` (Instrument)

**Validation:**
- Matrix normalization (rows sum to 1)
- TVTP adjustments preserve row sums
- Time-of-day windows for ES/NQ and GC/SI
- Contagion multiplier effects

---

### Phase 2: Cross-Instrument Lead-Lag Estimation
**Goal:** Empirical estimation of cross-instrument relationships

#### 2.1 Lead-Lag Estimator
**Files:**
- `src/models/lead_lag_estimator.py`

**Components:**
- `LeadLagResult` dataclass
  - Cross-correlation results
  - Granger causality p-values
  - Interpretation (bidirectional, leader, confidence)
  - Recommended contagion multipliers
- `estimate_lead_lag()` function
  - Cross-correlation analysis
  - Granger causality tests
  - Symmetric default if unclear

**Dependencies:** 
- `scipy.signal.correlate`
- `statsmodels.tsa.stattools.grangercausalitytests`

**Validation:**
- Test with synthetic data (known lead-lag)
- Test bidirectional case
- Test unclear case (defaults to symmetric)
- Verify contagion multipliers are reasonable

---

### Phase 3: Regime Detection and Calibration
**Goal:** Calibrate regime-switching GARCH parameters from historical data

#### 3.1 Regime Detection
**Files:**
- `src/models/regime_calibration.py` (partial)

**Components:**
- `_detect_regimes()` method
  - Uses `statsmodels.tsa.regime_switching.MarkovRegression`
  - Fallback to simple variance-based detection
  - Regime sorting by volatility (low to high)
- `_simple_regime_detection()` fallback
  - Rolling volatility percentiles

**Dependencies:**
- `statsmodels.tsa.regime_switching.MarkovRegression`
- `regime_config.py`, `transition_matrix.py`

**Validation:**
- Test regime detection on synthetic data
- Verify regimes are sorted by volatility
- Test fallback when Markov switching fails
- Check minimum regime duration constraints

---

#### 3.2 Per-Regime GARCH Estimation
**Files:**
- `src/models/regime_calibration.py` (partial)

**Components:**
- `_estimate_regime_garch()` method
  - For each regime: fit GARCH/GJR-GARCH using `arch` package
  - ES/NQ: GJR-GARCH with skewed-t (`o=1`, `dist='skewt'`)
  - GC/SI: Standard GARCH with GED (`dist='ged'`)
  - Parameter extraction and scaling
  - Stationarity enforcement

**Dependencies:**
- `arch.arch_model`
- `garch_params.py`

**Validation:**
- Test GARCH fitting on synthetic data
- Verify parameter constraints are satisfied
- Test with insufficient data (uses defaults)
- Verify ES/NQ get gamma > 0, GC/SI get gamma = 0

---

#### 3.3 Complete Calibration Pipeline
**Files:**
- `src/models/regime_calibration.py` (complete)

**Components:**
- `RegimeSwitchingCalibrator` class
  - `calibrate_all()` method (orchestrates full calibration)
  - `_calibrate_cross_instrument()` (lead-lag estimation)
  - Stores results: `regime_params`, `transition_matrices`, `lead_lag_results`

**Dependencies:** All previous components

**Validation:**
- End-to-end calibration on historical data
- Verify all instruments calibrated
- Check cross-instrument results are stored
- Test calibration date tracking

---

### Phase 4: Path Generation Engine
**Goal:** Generate Monte Carlo paths using regime-switching GARCH

#### 4.1 Regime-Switching Path Generator
**Files:**
- `src/simulation/regime_path_generator.py`

**Components:**
- `RegimeSwitchingPathGenerator` class
  - `generate_paths()` method
    - Initialize particles with regime probabilities
    - For each timestep:
      - Get TVTP-adjusted transition matrix
      - Sample regime transition (with min_duration constraint)
      - Generate return using regime-specific GARCH
      - Update variance and price
  - `_generate_innovation()` method
    - Skewed-t for ES/NQ
    - GED for GC/SI

**Dependencies:**
- `regime_config.py`, `garch_params.py`, `transition_matrix.py`
- `lead_lag_estimator.py` (for partner regime info)

**Validation:**
- Test path generation with known parameters
- Verify regime transitions respect min_duration
- Check TVTP adjustments affect transitions
- Verify innovation distributions (skewed-t vs GED)
- Test cross-instrument contagion effects

---

#### 4.2 Multi-Timeframe Aggregation
**Files:**
- `src/utils/timeframe_aggregator.py`

**Components:**
- `TimeframeAggregator` class
  - `aggregate_paths()` static method (close prices only)
  - `aggregate_ohlcv()` static method (full OHLCV)

**Dependencies:** `regime_config.py` (Timeframe enum)

**Validation:**
- Test aggregation from 1m to 5m, 15m, 1h
- Verify OHLCV relationships (high >= all, low <= all)
- Test edge cases (incomplete windows)

---

### Phase 5: Extended Particle Filter
**Goal:** Bayesian updating with regime state

#### 5.1 Extended Particle State
**Files:**
- `src/simulation/particle_filter.py` (partial)

**Components:**
- `Particle` dataclass (extended)
  - price, variance, weight
  - regime_probs (K-dimensional)
  - current_regime, regime_duration
  - price_history
  - `update_history()` method
  - `transition_regime()` method (with min_duration)

**Dependencies:** None (pure data structure)

**Validation:**
- Test regime transition logic
- Verify min_duration constraint enforcement
- Test history management

---

#### 5.2 Particle Filter with Regime State
**Files:**
- `src/simulation/particle_filter.py` (complete)

**Components:**
- `ParticleFilter` class (extended)
  - `update()` method (likelihood calculation with regime)
  - `resample()` method (systematic resampling)
  - `generate_forecast()` method (uses RegimeSwitchingPathGenerator)
  - Regime probability tracking

**Dependencies:**
- `regime_path_generator.py`
- `resampler.py` (if separate)

**Validation:**
- Test likelihood calculation
- Verify resampling preserves regime diversity
- Test forecast generation
- Check ESS (Effective Sample Size) tracking

---

### Phase 6: Integration and Testing
**Goal:** End-to-end testing and validation

#### 6.1 Calibration Script
**Files:**
- `scripts/calibrate_models.py`

**Components:**
- Load historical data from database
- Initialize `RegimeSwitchingCalibrator`
- Run calibration for all instruments
- Save parameters to database/disk

**Dependencies:** All model components, data storage

**Validation:**
- Run calibration on real data
- Verify parameters are saved
- Check calibration reports/logs

---

#### 6.2 Unit Tests
**Files:**
- `tests/unit/test_regime_config.py`
- `tests/unit/test_garch_params.py`
- `tests/unit/test_transition_matrix.py`
- `tests/unit/test_lead_lag_estimator.py`
- `tests/unit/test_regime_calibration.py`
- `tests/unit/test_regime_path_generator.py`
- `tests/unit/test_particle_filter.py`

**Coverage Target:** 90% for models/, 85% for simulation/

---

#### 6.3 Integration Tests
**Files:**
- `tests/integration/test_regime_garch_pipeline.py`

**Components:**
- End-to-end: data → calibration → path generation → particle filter
- Cross-instrument contagion test
- Multi-timeframe aggregation test

---

#### 6.4 Validation Tests
**Files:**
- `tests/validation/test_stylized_facts.py`

**Components:**
- Fat tails (kurtosis > 3)
- Volatility clustering (ARCH-LM test)
- Leverage effect (ES/NQ only)
- Regime persistence
- Cross-instrument correlation

---

## Implementation Order

### Sprint 1: Foundation (Phases 1.1-1.3)
1. `regime_config.py` - Enumerations and configuration
2. `garch_params.py` - Parameter structures
3. `transition_matrix.py` - TVTP logic
4. Unit tests for all three

**Deliverable:** Type system and configuration framework complete

---

### Sprint 2: Cross-Instrument Analysis (Phase 2)
1. `lead_lag_estimator.py` - Lead-lag estimation
2. Unit tests

**Deliverable:** Can estimate cross-instrument relationships from data

---

### Sprint 3: Calibration (Phase 3)
1. `regime_calibration.py` - Complete calibration pipeline
2. `scripts/calibrate_models.py` - Calibration script
3. Integration tests

**Deliverable:** Can calibrate regime-switching GARCH from historical data

---

### Sprint 4: Path Generation (Phase 4)
1. `regime_path_generator.py` - Path generation engine
2. `timeframe_aggregator.py` - Multi-timeframe support
3. Unit and integration tests

**Deliverable:** Can generate regime-switching paths at multiple timeframes

---

### Sprint 5: Particle Filter (Phase 5)
1. `particle_filter.py` - Extended particle filter
2. `resampler.py` - Resampling logic (if separate)
3. Integration tests

**Deliverable:** Complete particle filter with regime state

---

### Sprint 6: Validation (Phase 6)
1. Validation test suite
2. Stylized facts verification
3. End-to-end pipeline test
4. Documentation

**Deliverable:** Validated, production-ready GARCH engine

---

## Key Design Decisions

### 1. Symmetric Bidirectional Contagion
- **Decision:** Equal multipliers both directions (GC↔SI, ES↔NQ)
- **Rationale:** Academic evidence is conflicting; let data decide via lead-lag estimation
- **Implementation:** Default 1.4x both directions, adjust if lead-lag detected

### 2. Regime Count
- **ES/NQ/GC:** 3 regimes (Low, Normal, High)
- **SI:** 4 regimes (Low, Normal, High, Crisis)
- **Rationale:** SI has extreme tail events requiring crisis regime

### 3. Innovation Distributions
- **ES/NQ:** Skewed-t (captures leverage effect)
- **GC/SI:** GED (symmetric fat tails, no leverage effect)
- **Rationale:** Research shows no leverage effect in precious metals

### 4. Calibration Base Timeframe
- **Base:** 1-minute (most granular)
- **Aggregation:** Generate at 1m, aggregate to 5m, 15m, 1h
- **Rationale:** Single calibration, multiple outputs, ensures consistency

### 5. TVTP Adjustments
- **Time-of-day:** Session-specific volatility windows
- **Cross-instrument:** Partner's regime affects transitions
- **Realized volatility:** Recent RV ratio affects transitions
- **Rationale:** Captures known market dynamics (session effects, contagion, vol clustering)

---

## Dependencies

### Python Packages
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `scipy>=1.11.0`
- `statsmodels>=0.14.0` (for Markov-switching)
- `arch>=6.0.0` (for GARCH estimation)

### Internal Dependencies
- `src/data/storage.py` (for loading historical data)
- `src/data/historical_loader.py` (already implemented)

---

## Testing Strategy

### Unit Tests
- Each component tested in isolation
- Mock dependencies where appropriate
- Synthetic data for deterministic tests

### Integration Tests
- Test component interactions
- Use real historical data (small samples)
- Verify end-to-end pipeline

### Validation Tests
- Stylized facts verification
- Parameter constraint checks
- Cross-instrument relationship validation

---

## Success Criteria

1. ✅ All data structures implemented and tested
2. ✅ Calibration pipeline runs successfully on historical data
3. ✅ Path generation produces valid regime-switching paths
4. ✅ Particle filter tracks regime probabilities correctly
5. ✅ Multi-timeframe aggregation works correctly
6. ✅ Stylized facts validated (fat tails, vol clustering, leverage effect)
7. ✅ Cross-instrument contagion effects visible in simulations
8. ✅ Unit test coverage > 85%
9. ✅ Integration tests pass
10. ✅ Documentation complete

---

## Notes

- **Markov-Switching Fallback:** If `statsmodels` Markov-switching fails, use simple variance-based regime detection
- **Parameter Constraints:** Always enforce stationarity (α + β + γ/2 < 1)
- **Numerical Stability:** Scale returns by 100 for GARCH estimation, scale back omega
- **Min Duration:** Enforce minimum bars in regime before allowing transition (prevents noise-driven switches)
- **Symmetric Contagion:** Default assumption unless lead-lag estimation shows clear direction

---

**Plan Version:** 1.0  
**Date:** January 2026  
**Based On:** `synthetic_bar_generator_development_plan_v2.md` Section 3
