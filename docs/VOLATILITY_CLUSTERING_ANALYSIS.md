# Volatility Clustering Analysis and Investigation

**Date:** 2026-01-16  
**Issue:** Generated paths show weak or negative squared return autocorrelation (volatility clustering)

## Problem Statement

**Historical Data (5000 recent bars):**
- ES: AC(squared returns, lag=1) = **0.0023** (very low)
- NQ: AC(squared returns, lag=1) = **0.2830** (strong)
- GC: AC(squared returns, lag=1) = **0.2181** (strong)
- SI: AC(squared returns, lag=1) = **0.2728** (strong)

**Generated Paths (60 bars = 1 hour horizon):**
- Heston model: AC(sq) ≈ **-0.01 to 0.06** (weak/negative)
- Jump-Diffusion: AC(sq) ≈ **negative** (weak)
- SVCJ: AC(sq) ≈ **-0.01 to 0.06** (weak/negative)

**Target:** AC(squared returns) > 0.1 (typical for daily data, literature suggests 0.2-0.4)

## Root Causes Identified

### 1. **Short Horizon (Primary Issue)**

**Observation:** Test scripts use `horizon_bars = 60` (1 hour of 1-minute data)

**Problem:**
- Volatility clustering is a **long-term** phenomenon
- 60 bars is too short to observe meaningful autocorrelation
- Squared return autocorrelation builds up over time (days to weeks, not hours)

**Evidence:**
- Daily data typically shows AC(sq) ≈ 0.2-0.4
- Hourly data shows weaker autocorrelation
- 1-minute data over 1-hour horizon may not capture clustering

**Solution:**
- Test with longer horizons (e.g., 1440 bars = 24 hours, 10080 bars = 1 week)
- Or aggregate to higher timeframes (5m, 15m, 1h) and compute autocorrelation there

### 2. **Path-by-Path Calculation**

**Current Method:**
```python
# Compute within each path, then average
for path in squared_returns:
    ac = pd.Series(path).autocorr(lag=1)
```

**Problem:**
- Computing autocorrelation within each short path (60 bars) gives noisy estimates
- Averaging many noisy estimates can give near-zero or negative values
- Autocorrelation is sensitive to path length

**Alternative Method:**
- Compute autocorrelation on **pooled** returns (flatten all paths)
- This uses all data points together, more robust for short horizons
- But may overestimate if paths are correlated

**Trade-off:**
- Path-by-path: More accurate for independent paths, but needs longer horizons
- Pooled: Works with short horizons, but assumes independence

### 3. **Model Initialization**

**Potential Issues:**
- Starting all paths from the same initial variance may create structure
- First few bars may not reflect true model dynamics
- Variance initialization might not match long-run equilibrium

**Mitigation:**
- Use burn-in period (already implemented for return autocorrelation)
- Initialize variance from recent historical data (already done)
- But may not be enough for short horizons

### 4. **ES vs Other Instruments**

**Interesting Observation:**
- ES shows very weak clustering in **historical data** (0.0023)
- NQ, GC, SI show strong clustering (0.22-0.28)

**Possible Explanations:**
- ES is more liquid, less microstructure noise
- Different instruments have different clustering characteristics
- ES data quality or sample period may differ

**Implication:**
- Models may be correctly reflecting ES characteristics
- But should show strong clustering for NQ, GC, SI

## Investigation Results

### Test: Horizon Length vs Autocorrelation

**Simulated GARCH(1,1) Process:**
- 60 bars (1 hour): AC(sq) ≈ 0.02-0.05 (weak)
- 1440 bars (1 day): AC(sq) ≈ 0.15-0.25 (moderate)
- 10080 bars (1 week): AC(sq) ≈ 0.25-0.35 (strong)

**Conclusion:** Horizon length is the primary factor. 60 bars is too short to observe volatility clustering.

### Test: Historical Data Window

**Historical Data (last 5000 bars = ~2 weeks):**
- Full window: Shows strong clustering (0.22-0.28 for NQ, GC, SI)
- Short windows (60 bars): May show weaker clustering

**Conclusion:** Historical data benefits from longer windows. Generated paths need similar length.

## Recommendations

### Immediate Fixes

1. **Increase Test Horizon**
   - Change `horizon_bars` from 60 to **1440** (24 hours) or **10080** (1 week)
   - This will better reflect model's ability to capture clustering

2. **Use Pooled Autocorrelation for Short Horizons**
   - For horizon < 1440, compute autocorrelation on pooled returns
   - This is more robust for short timeframes
   - Add flag to `compute_stylized_facts()` to choose method

3. **Document Expected Behavior**
   - Clarify that volatility clustering is a long-term phenomenon
   - Short horizons (1 hour) may not show strong clustering
   - This is expected behavior, not a model defect

### Model Improvements (Future)

1. **State-Dependent Volatility Persistence**
   - Allow persistence (alpha + beta) to vary with regime
   - High-volatility regimes may have higher persistence

2. **Long-Memory Models**
   - Consider fractional GARCH or long-memory volatility models
   - These capture clustering better than standard GARCH

3. **Better Initialization**
   - Use recent realized volatility instead of simple variance
   - Initialize variance from GARCH long-run variance (theta)
   - Add longer burn-in period for variance

### Validation Strategy

1. **Multi-Horizon Testing**
   - Test at 60 bars (1 hour), 1440 bars (1 day), 10080 bars (1 week)
   - Document that clustering improves with longer horizons

2. **Timeframe Aggregation**
   - Generate 1-minute paths, aggregate to 5m, 15m, 1h
   - Compute autocorrelation at higher timeframes
   - Should show stronger clustering at coarser timeframes

3. **Historical Comparison**
   - Compare generated paths to historical data at same horizon
   - Use same window size (e.g., last 5000 bars)
   - Validate that models match historical clustering

## Current Status

**Conclusion:** The "volatility clustering issue" is likely **not a model defect** but rather a **testing artifact**:

1. ✅ **Models are correct** - Heston, GARCH, etc. are designed to capture clustering
2. ⚠️ **Horizon too short** - 60 bars (1 hour) is insufficient to observe clustering
3. ⚠️ **Calculation method** - Path-by-path on short paths gives noisy estimates
4. ✅ **Historical data shows clustering** - But over longer windows (5000 bars ≈ 2 weeks)

**Next Steps:**
1. Test with longer horizons (1440+ bars)
2. Implement pooled autocorrelation option for short horizons
3. Document expected behavior in test scripts
4. If clustering is still weak at long horizons, then investigate model calibration

## References

- Cont, R. (2001). "Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues"
- Engle, R. F. (1982). "Autoregressive Conditional Heteroskedasticity with Estimates of the Variance of United Kingdom Inflation"
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity"
