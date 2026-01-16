# Testing Methodology Review: Volatility Clustering Best Practices

**Date:** 2026-01-16  
**Purpose:** Compare our testing methodology to academic best practices and identify improvements

## Current Methodology

### What We're Doing

1. **Multi-horizon testing** ✅
   - Horizons: 1 hour (60), 1 day (1440), 1 week (10080), 1 month (30000)
   - Tests: Heston, Jump-Diffusion, SVCJ models

2. **Autocorrelation of squared returns**
   - Metric: AC(squared returns, lag=1)
   - Method: Path-by-path calculation, then average
   - Formula: `pd.Series(squared_returns).autocorr(lag=1)`

3. **Historical comparison**
   - Compare generated paths vs historical data
   - Same calculation method for both

## Best Practices from Literature

### 1. Multiple Lags (Not Just Lag-1) ⚠️ **MISSING**

**Literature:** Autocorrelation should be computed at **multiple lags** (e.g., lags 1, 5, 10, 20, 30+) to observe decay patterns.

**Current Status:** We only compute lag-1 autocorrelation.

**Why It Matters:**
- Volatility clustering shows **slow decay** across many lags
- Power-law vs exponential decay reveals long-memory vs short-memory
- Lag-1 alone doesn't capture persistence structure

**Recommendation:** Compute ACF at multiple lags (e.g., 1, 5, 10, 20, 30) and compare decay patterns.

### 2. Absolute Returns vs Squared Returns ⚠️ **PARTIAL**

**Literature:** Use **both** absolute returns `|r_t|` and squared returns `r_t²`. Absolute returns are less noisy for high-frequency data.

**Current Status:** We only use squared returns.

**Why It Matters:**
- Absolute returns are more robust to outliers
- Different instruments may show stronger clustering in `|r_t|` vs `r_t²`
- Literature suggests using both for comprehensive validation

**Recommendation:** Add autocorrelation of absolute returns alongside squared returns.

### 3. Pooled vs Path-by-Path ⚠️ **NEEDS VALIDATION**

**Literature:** 
- **Path-by-path**: Better for independent paths, but needs sufficient path length
- **Pooled**: Better for short horizons, uses all data points together

**Current Status:** We use path-by-path, then average.

**Why It Matters:**
- Short horizons (60 bars) give noisy path-by-path estimates
- Pooling may overestimate if paths are correlated
- Best practice: Use both methods and compare

**Recommendation:** 
- For short horizons (<1440 bars): Use pooled autocorrelation
- For long horizons (≥1440 bars): Use path-by-path (more accurate for independent paths)
- Report both for comparison

### 4. Decay Pattern Analysis ⚠️ **MISSING**

**Literature:** Examine **decay pattern** of autocorrelation (power-law vs exponential).

**Current Status:** We only report lag-1 value.

**Why It Matters:**
- Power-law decay indicates long-memory (stylized fact)
- Exponential decay indicates short-memory (GARCH-style)
- Visual inspection of ACF plots reveals clustering quality

**Recommendation:** 
- Plot ACF decay across multiple lags
- Fit decay pattern (power-law or exponential)
- Compare decay rates: historical vs generated

### 5. Stationarity Checks ⚠️ **MISSING**

**Literature:** Ensure returns are **stationary** before computing autocorrelation.

**Current Status:** We assume stationarity (returns are already differenced).

**Why It Matters:**
- Nonstationarity (trends, structural breaks) can bias autocorrelation
- Should test for unit roots or trends
- Detrending may be needed for long horizons

**Recommendation:** 
- Add stationarity tests (ADF, KPSS) for long horizons
- Detrend if necessary
- Document stationarity assumptions

### 6. Hurst Exponent / Long-Memory ⚠️ **MISSING**

**Literature:** Estimate **Hurst exponent** or long-memory parameter of volatility series.

**Current Status:** We don't compute this.

**Why It Matters:**
- Hurst > 0.5 indicates long-memory (clustering)
- Quantitative measure of clustering strength
- More robust than single-lag autocorrelation

**Recommendation:** Add Hurst exponent calculation for volatility series.

### 7. Bootstrap / Confidence Intervals ⚠️ **MISSING**

**Literature:** Use **bootstrap** or simulation to compute confidence intervals for autocorrelation estimates.

**Current Status:** We report point estimates without confidence intervals.

**Why It Matters:**
- Small sample sizes give noisy estimates
- Confidence intervals show statistical significance
- Can test if generated clustering differs significantly from historical

**Recommendation:** 
- Bootstrap autocorrelation estimates
- Compute confidence intervals
- Test significance of differences

### 8. Multiple Variance Proxies ⚠️ **PARTIAL**

**Literature:** Use multiple variance proxies: squared returns, absolute returns, realized volatility.

**Current Status:** We only use squared returns.

**Why It Matters:**
- Different proxies capture different aspects of volatility
- Realized volatility (sum of squared intraday returns) is less noisy
- Robustness check across different measures

**Recommendation:** Add realized volatility calculation for intraday data.

## Comparison Table

| Best Practice | Our Status | Priority | Impact |
|--------------|------------|----------|--------|
| **Multiple horizons** | ✅ Done | - | High |
| **Multiple lags** | ❌ Missing | High | High |
| **Absolute returns** | ❌ Missing | Medium | Medium |
| **Decay pattern** | ❌ Missing | High | High |
| **Pooled autocorrelation** | ⚠️ Partial | Medium | Medium |
| **Stationarity checks** | ❌ Missing | Low | Low |
| **Hurst exponent** | ❌ Missing | Medium | Medium |
| **Bootstrap CIs** | ❌ Missing | Medium | Medium |
| **Multiple proxies** | ⚠️ Partial | Low | Low |
| **Visual diagnostics** | ❌ Missing | Medium | Medium |

## Recommendations: Priority Order

### High Priority (Do First)

1. **Add Multiple Lags** ⚠️ **CRITICAL**
   - Compute ACF at lags 1, 5, 10, 20, 30
   - Compare decay patterns: historical vs generated
   - This will reveal if clustering persists or decays too fast

2. **Add Decay Pattern Analysis**
   - Plot ACF curves
   - Fit decay patterns (power-law vs exponential)
   - Compare decay rates

3. **Add Pooled Autocorrelation for Short Horizons**
   - Use pooled for <1440 bars
   - Use path-by-path for ≥1440 bars
   - Report both for comparison

### Medium Priority (Do Next)

4. **Add Absolute Returns Autocorrelation**
   - Compute AC(|returns|) alongside AC(returns²)
   - More robust to outliers
   - Literature standard

5. **Add Hurst Exponent**
   - Estimate Hurst for volatility series
   - Quantitative measure of clustering
   - Target: Hurst > 0.5

6. **Add Bootstrap Confidence Intervals**
   - Bootstrap autocorrelation estimates
   - Compute 95% confidence intervals
   - Test statistical significance

### Low Priority (Nice to Have)

7. **Add Stationarity Tests**
   - ADF, KPSS tests for long horizons
   - Detrend if necessary
   - Document assumptions

8. **Add Visual Diagnostics**
   - Plot ACF curves
   - Volatility time-series plots
   - QQ plots for comparison

9. **Add Multiple Variance Proxies**
   - Realized volatility
   - Range-based estimators
   - Jump-adjusted measures

## Expected Impact of Changes

### Multiple Lags
- **Current:** Only see lag-1 (may miss persistence)
- **After:** See full decay pattern (reveals long-memory vs short-memory)
- **Impact:** Better understanding of clustering quality

### Pooled Autocorrelation
- **Current:** Noisy estimates at short horizons
- **After:** More stable estimates
- **Impact:** Better validation at 1-hour horizon

### Absolute Returns
- **Current:** Only squared returns (noisy for high-frequency)
- **After:** More robust measure
- **Impact:** More accurate clustering detection

### Hurst Exponent
- **Current:** No long-memory measure
- **After:** Quantitative clustering measure
- **Impact:** Better model comparison

## Next Steps

1. **Update `compute_stylized_facts()`** to:
   - Compute ACF at multiple lags (1, 5, 10, 20, 30)
   - Add autocorrelation of absolute returns
   - Add pooled autocorrelation option for short horizons
   - Compute Hurst exponent

2. **Update test scripts** to:
   - Plot ACF decay curves
   - Report multiple lags
   - Compare decay patterns

3. **Add bootstrap** for confidence intervals

4. **Document** all assumptions and methodology

## References

- Cont (2001): "Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues"
- "Quantifying volatility clustering in financial time series" (2011)
- Engle (1982): "Autoregressive Conditional Heteroskedasticity"
- Bollerslev (1986): "Generalized Autoregressive Conditional Heteroskedasticity"
- Chen et al. (2005): Conditional probability measures of volatility clustering
