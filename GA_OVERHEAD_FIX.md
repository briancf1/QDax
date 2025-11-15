# Critical Fix: Competition-GA Overhead Calculation

## Problem Identified

The original GA overhead calculation was dramatically wrong - **off by 300-2000x**!

### Original (Incorrect) Formula
```python
evals_per_ga_call = k * num_ga_children * num_ga_generations
# Example: 3 * 2 * 2 = 12 evaluations per call
```

This assumed only k=3 selected parents were evolved.

### Corrected Formula
```python
# Competition-GA evolves the ENTIRE population!
offspring_per_call = population_size * num_ga_children * (num_ga_children^num_ga_generations - 1) / (num_ga_children - 1)
# Example: 1024 * 2 * (2^2 - 1) / (2 - 1) = 1024 * 6 = 6,144 evaluations per call
```

## Why The Original Was Wrong

By examining `/qdax/core/containers/dns_repertoire_ga.py`, the `_competition_ga` function clearly shows:

1. **ALL individuals in the population are evolved** (line ~87)
2. Each individual generates offspring through a tree structure
3. Total offspring per generation i: `population_size * num_children^i`
4. Across all generations: `population_size * sum(num_children^i for i=1 to num_ga_generations)`

## Corrected Overhead Values

For population_size=1024, num_children=2:

| Generations | Old (Wrong) | **New (Correct)** | Ratio |
|-------------|-------------|-------------------|-------|
| 1           | 6           | **2,048**         | 341x  |
| 2           | 12          | **6,144**         | 512x  |
| 3           | 18          | **14,336**        | 797x  |
| 4           | 24          | **30,720**        | 1,280x|
| 5           | 30          | **62,464**        | 2,082x|

## Impact on Experiments

### Before Fix (Using Wrong Overhead)

Example: g_n=300, num_ga_generations=2, converges at iteration 1500
- Main evals: 1500 * 100 = 150,000
- GA overhead (WRONG): 5 calls * 12 = **60 evals**
- Total (WRONG): 150,060 evals
- Net savings (WRONG): 50% (looked great!)

### After Fix (Using Correct Overhead)

Same scenario with corrected calculation:
- Main evals: 1500 * 100 = 150,000
- GA overhead (CORRECT): 5 calls * 6,144 = **30,720 evals**
- Total (CORRECT): 180,720 evals
- Net savings (CORRECT): 40% (still positive, but much less impressive)

### Worst Case Scenarios

Deep foresight configs with frequent GA calls could have NEGATIVE net savings!

Example: g_n=150, num_ga_generations=4, converges at 1500
- Main evals: 150,000
- GA calls: 10
- GA overhead: 10 * 30,720 = **307,200 evals**
- **Total: 457,200 evals** (52% MORE than baseline!)

## Files Updated

All tier experiment scripts now use corrected formula:
- `examples/run_tier1_proven_winners.py` ✓
- `examples/run_tier2_promising.py` ✓
- `examples/run_tier3_deep_foresight.py` ✓
- `examples/run_tier4_aggressive.py` ✓
- `examples/run_tier5_rare_deep.py` ✓

## Key Changes

1. **Function signature** changed from:
   ```python
   calculate_ga_overhead_evals(g_n, num_iterations, k, ...)
   ```
   to:
   ```python
   calculate_ga_overhead_evals(g_n, num_iterations, population_size, ...)
   ```

2. **Calculation** uses geometric series formula for offspring tree

3. **Function calls** pass `FIXED_PARAMS['population_size']` instead of `FIXED_PARAMS['k']`

## Next Steps

1. **MUST re-run ALL experiments** with corrected overhead calculations
2. Many "winning" configurations may actually be losers when accounting for true overhead
3. Tier 4 (aggressive) and Tier 5 (deep foresight) likely have negative net savings
4. Need to identify sweet spot: configurations that converge fast enough to offset massive overhead
5. Update all documentation with realistic overhead expectations

## Research Implications

This fix reveals Competition-GA is **computationally expensive**:
- Not a "free lunch" - significant evaluation cost
- Only beneficial when convergence speedup > overhead cost
- Optimal configs likely have:
  - Large g_n (infrequent GA calls)
  - Small num_ga_generations (shallow foresight)
  - Fast convergence (strong exploitation)

The corrected metrics will show which configurations actually provide net benefit.
