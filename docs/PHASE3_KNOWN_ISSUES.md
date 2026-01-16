# Phase 3: Heston Model - Known Issues for Future Resolution

**Date:** 2026-01-16  
**Status:** Functional but with limitations

## Issues Identified

### 1. Volatility Clustering Failure
**Problem:** Squared return autocorrelation is negative (~-0.03) instead of positive (>0.1)

**Root Causes:**
- Short horizon (60 bars ≈ 1 hour) may not capture long-term volatility persistence
- Calibration method (simulated method of moments) may be insufficient
- Need to target longer-horizon moments or use different calibration approach

**Potential Solutions:**
- Increase test horizon to 200+ bars (longer paths)
- Calibrate to longer-horizon moments (e.g., 1-day, 1-week autocorrelation)
- Consider particle MCMC or options-implied calibration for better accuracy
- Add volatility clustering as explicit moment target in objective function

### 2. Calibration Accuracy Issues
**Problem:** Mu (drift) values appear unrealistic:
- NQ: μ = -0.526 (should be near zero for 1-minute returns)
- GC: μ = 1.96 (extremely high)
- ES: μ = 0.022 (reasonable)
- SI: μ = 0.213 (high)

**Root Causes:**
- Simulated method of moments may be overfitting to noise
- No constraints on mu parameter during optimization
- Calibration may be capturing spurious drift

**Potential Solutions:**
- Add mu constraints (clip to ±0.0005 per step, equivalent to ±50 bp/min)
- Use empirical mean instead of calibrated mu (similar to GARCH fix)
- Add regularization term to objective function
- Consider using arch package or other established calibration methods

### 3. Feller Condition Violations
**Status:** Handled correctly but worth noting

**Observation:** All instruments violate Feller condition (2κθ < ξ²)
- This is common in practice and expected
- Full truncation scheme handles this correctly
- No action needed, but could document typical violation magnitudes

## What's Working Well

✅ **Model Implementation:** QE discretization working correctly  
✅ **Path Generation:** Successfully generates 1000 paths  
✅ **Fat Tails:** Captured excellently (kurtosis > 40 for all instruments)  
✅ **Return Autocorrelation:** Near zero as expected (<0.05)  
✅ **Feller Handling:** Full truncation prevents negative variance  
✅ **Correlation Structure:** Negative ρ for equity instruments (leverage effect)  

## Priority for Future Work

1. **High Priority:** Fix volatility clustering (core stylized fact)
2. **Medium Priority:** Add mu constraints to calibration
3. **Low Priority:** Document typical Feller violation magnitudes

## Notes

- Model is functional and generates paths correctly
- Issues are primarily calibration-related, not implementation bugs
- Can proceed to Phase 4 (Jump-Diffusion) while these are documented
- May need to revisit calibration when combining with other models (SVCJ)
