# Phase 4: Jump-Diffusion Model - Known Issues for Future Resolution

**Date:** 2026-01-16  
**Status:** Functional but calibration method improved

## Issue Identified and Fixed

### Jump Detection Method Issue (FIXED)
**Problem:** All instruments had identical jump intensity (4914 jumps/year)

**Root Cause:**
- Using 95th percentile threshold always gives exactly 5% of returns as "jumps"
- For 5000 recent returns, this always gives 250 jumps
- Calculation: λ = 250 / (5000 * dt) = 4914 jumps/year for all instruments
- This is a **calibration bug**, not a data quality issue

**Fix Applied:**
- Changed from percentile-based to standard deviation-based jump detection
- Use 5 standard deviations threshold (instrument-specific)
- Now produces different jump intensities per instrument:
  - ES: 216 jumps/year (0.22%)
  - NQ: 373 jumps/year (0.38%)
  - GC: 157 jumps/year (0.16%)
  - SI: 334 jumps/year (0.34%)

**Why This is Better:**
- Instrument-specific thresholds (based on each instrument's volatility)
- More robust to outliers
- Better reflects actual jump frequency differences between instruments

## Remaining Considerations

### Jump Intensity Still High
**Observation:** Even with 5 std threshold, jump intensities are 150-370 jumps/year

**Possible Reasons:**
1. **1-minute data characteristics**: Large moves are more common at high frequency
2. **Market microstructure**: Bid-ask bounces, quote updates can appear as jumps
3. **Data quality**: May include some noise/gaps that look like jumps
4. **Threshold too low**: 5 std might not be strict enough for true jumps

**Potential Solutions:**
- Use higher threshold (e.g., 6-7 std for very rare jumps)
- Use 99.9th percentile instead (would give ~100 jumps/year)
- Consider using Bipower Variation or other jump detection methods from high-frequency finance literature
- Cap maximum jump intensity to reasonable values (e.g., <100 jumps/year)

### Volatility Clustering
**Same Issue as Heston Model:**
- Squared return autocorrelation is negative instead of positive
- Likely due to short horizon or calibration method limitations

## Data Quality Check

**Data Loader Investigation:**
- Data loader only removes invalid OHLC rows (high < low, etc.)
- Does NOT filter or replace jumps
- Large gaps (>5%) are flagged but not removed
- Data appears clean - jumps are present in the data

**Conclusion:** The issue was calibration method, not data quality.

## What's Working Well

✅ **Model Implementation:** Jump-diffusion working correctly  
✅ **Path Generation:** Successfully generates paths with jumps  
✅ **Fat Tails:** Captured excellently (kurtosis > 46 for all instruments)  
✅ **Return Autocorrelation:** Near zero as expected (<0.02)  
✅ **Jump Simulation:** Compound Poisson process working correctly  
✅ **Fixed Calibration:** Now produces instrument-specific jump intensities  

## Priority for Future Work

1. **Medium Priority:** Consider using higher threshold (6-7 std) or 99.9th percentile for rarer jumps
2. **Low Priority:** Implement Bipower Variation or other advanced jump detection methods
3. **Low Priority:** Cap maximum jump intensity to reasonable bounds

## Notes

- Model is functional and generates paths correctly
- Calibration method has been improved (fixed the identical jump intensity bug)
- Jump intensities are now instrument-specific, which is correct
- May still want to refine threshold based on production requirements
