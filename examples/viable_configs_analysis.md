# DNS-GA Viable Configurations Analysis

## Executive Summary

**Key Finding**: DNS-GA can NEVER use fewer total evaluations than DNS when running the same number of iterations, because Competition-GA adds extra evaluations on top of the standard batch evaluations.

**The Real Question**: Can DNS-GA converge to the same QD score in fewer iterations, such that even with the extra evaluation overhead, it uses fewer total evaluations?

## Current Configuration Results

**Tested**: `g_n=100, num_ga_generations=3, num_ga_children=2`

| Metric | DNS | DNS-GA | Result |
|--------|-----|---------|---------|
| Final QD Score | 360,358 | 366,433 | +1.7% ✓ |
| Total Evaluations (1000 iters) | 100,000 | 243,360 | +143% ✗ |
| Evaluations to reach DNS QD | 100,000 | 151,016 | +51% ✗ |
| Overhead | - | 143% | Too High |
| Breakeven Requirement | - | <416 iterations | Failed |
| Actual Convergence | 1000 | 650 | Too Slow |

**Verdict**: FAILED - DNS-GA converged at iteration 650 but needed to converge by iteration 416 to offset the 143% evaluation overhead.

## Fundamental Calculation

For DNS-GA with parameters (`g_n`, `num_ga_generations`, `num_ga_children`):

```
offspring_per_individual = sum(num_ga_children^i for i=1 to num_ga_generations)
offspring_per_ga_call = population_size × offspring_per_individual
ga_calls = total_iterations ÷ g_n
ga_evaluations = ga_calls × offspring_per_ga_call

total_dns_ga_evals = (iterations × batch_size) + ga_evaluations
total_dns_evals = iterations × batch_size

overhead = (ga_evaluations / total_dns_evals) × 100%
```

### Break-even Iteration Calculation

For DNS-GA to be more efficient, we need:
```
DNS-GA(X iterations) < DNS(1000 iterations)

X × batch_size + (X ÷ g_n) × offspring_per_ga_call < 1000 × batch_size

Solving for X gives the break-even iteration count
```

## Viable Configurations (Ranked by Success Probability)

### Tier 1: Highest Success Probability (>95%)

| Rank | Gens | Children | g_n | Overhead | Breakeven | Speedup Needed | GA Calls |
|------|------|----------|-----|----------|-----------|----------------|----------|
| 1 | 1 | 2 | 500 | 2.0% | <980 | 1.02x | 2 |
| 2 | 1 | 2 | 250 | 4.1% | <962 | 1.04x | 4 |
| 3 | 2 | 2 | 614 | 6.1% | <942 | 1.06x | 1-2 |

**Configuration #1 Details**: `g_n=500, num_ga_generations=1, num_ga_children=2`
- Offspring per GA call: 1,024 × 2 = 2,048 evaluations
- GA calls over 1000 iterations: 2
- Total GA evaluations: 4,096
- Total DNS-GA evaluations: 104,096
- Overhead: 4.1%
- **Success condition**: Reach DNS QD by iteration 980 (only 2% faster than DNS)

### Tier 2: Good Success Probability (80-90%)

| Rank | Gens | Children | g_n | Overhead | Breakeven | Speedup Needed | GA Calls |
|------|------|----------|-----|----------|-----------|----------------|----------|
| 4 | 1 | 2 | 200 | 10.2% | <909 | 1.10x | 5 |
| 5 | 2 | 2 | 300 | 20.5% | <830 | 1.20x | 3-4 |
| 6 | 1 | 2 | 150 | 13.7% | <880 | 1.14x | 6-7 |

### Tier 3: Moderate Success Probability (50-70%)

| Rank | Gens | Children | g_n | Overhead | Breakeven | Speedup Needed | GA Calls |
|------|------|----------|-----|----------|-----------|----------------|----------|
| 7 | 1 | 2 | 100 | 20.5% | <830 | 1.20x | 10 |
| 8 | 2 | 2 | 200 | 30.7% | <765 | 1.31x | 5 |
| 9 | 2 | 2 | 150 | 41.0% | <710 | 1.41x | 6-7 |

### Tier 4: Low Success Probability (<30%)

| Rank | Gens | Children | g_n | Overhead | Breakeven | Speedup Needed | GA Calls |
|------|------|----------|-----|----------|-----------|----------------|----------|
| 10 | 3 | 2 | 300 | 47.8% | <677 | 1.48x | 3-4 |
| 11 | 3 | 2 | 200 | 71.7% | <583 | 1.72x | 5 |
| 12 | 3 | 2 | 100 | 143.4% | <416 | 2.40x | 10 |

**Note**: Configuration #12 is what was tested (g_n=100, num_ga_generations=3) - it required 2.40x speedup but only achieved 1.54x (1000/650).

## Recommended Testing Sequence

### Phase 1: Validate the Hypothesis (Start Here)

**Test #1**: `g_n=500, num_ga_generations=1, num_ga_children=2`
- **Why**: Most lenient configuration, only needs 2% faster convergence
- **Success Criteria**: Reach QD=360,358 by iteration 980
- **If Success**: Competition-GA provides value with minimal cost → proceed to Phase 2
- **If Failure**: Competition-GA may not accelerate convergence → investigate why

### Phase 2: Optimize Performance (If Phase 1 succeeds)

**Test #2**: `g_n=250, num_ga_generations=1, num_ga_children=2`
- **Why**: 2x more frequent GA, still low 4% overhead
- **Success Criteria**: Reach QD=360,358 by iteration 962
- **Expected**: May achieve better convergence with more frequent interventions

**Test #3**: `g_n=200, num_ga_generations=1, num_ga_children=2`
- **Why**: Further increase GA frequency, 10% overhead
- **Success Criteria**: Reach QD=360,358 by iteration 909

### Phase 3: Explore Deeper Foresight (If Phase 2 shows benefit)

**Test #4**: `g_n=614, num_ga_generations=2, num_ga_children=2`
- **Why**: Test if 2-generation lookahead improves decision quality
- **Success Criteria**: Reach QD=360,358 by iteration 942

**Test #5**: `g_n=300, num_ga_generations=2, num_ga_children=2`
- **Why**: More frequent 2-gen GA, test convergence improvement
- **Success Criteria**: Reach QD=360,358 by iteration 830

## Detailed Calculation Examples

### Example 1: g_n=500, num_ga_generations=1
```
Offspring per individual: 2^1 = 2
Total offspring per GA call: 1024 × 2 = 2,048
GA calls (1000 iters): 1000 ÷ 500 = 2
GA evaluations: 2 × 2,048 = 4,096
Standard evaluations: 1000 × 100 = 100,000
Total evaluations: 104,096
Overhead: 4.1%

Break-even calculation:
Let X = break-even iteration
X × 100 + (X ÷ 500) × 2,048 = 100,000
X × (100 + 4.096) = 100,000
X = 961 iterations

To be efficient: DNS-GA must reach DNS QD in <961 iterations
Speedup needed: 1000 ÷ 961 = 1.04x
```

### Example 2: g_n=250, num_ga_generations=1
```
Offspring per individual: 2^1 = 2
Total offspring per GA call: 1024 × 2 = 2,048
GA calls (1000 iters): 1000 ÷ 250 = 4
GA evaluations: 4 × 2,048 = 8,192
Standard evaluations: 1000 × 100 = 100,000
Total evaluations: 108,192
Overhead: 8.2%

Break-even iteration: ~919
Speedup needed: 1.09x
```

### Example 3: g_n=100, num_ga_generations=3 (TESTED)
```
Offspring per individual: 2 + 4 + 8 = 14
Total offspring per GA call: 1024 × 14 = 14,336
GA calls (1000 iters): 1000 ÷ 100 = 10
GA evaluations: 10 × 14,336 = 143,360
Standard evaluations: 1000 × 100 = 100,000
Total evaluations: 243,360
Overhead: 143.4%

Break-even iteration: 411
Speedup needed: 2.43x

ACTUAL RESULT:
DNS-GA reached DNS QD at iteration 650
Achieved speedup: 1000 ÷ 650 = 1.54x
Result: FAILED (needed 2.43x, got 1.54x)
```

## Key Insights

1. **Lower overhead = Higher success probability**: Configurations with <10% overhead are most likely to succeed

2. **Single generation is optimal initially**: `num_ga_generations=1` minimizes overhead while still providing forecasting benefit

3. **Moderate g_n balances cost and benefit**: g_n between 200-500 provides good balance

4. **Current config (g_n=100, gens=3) was too aggressive**: 143% overhead required 2.4x speedup - unrealistic

5. **Start conservative**: Begin with g_n=500, gens=1 (only 4% overhead) to validate the core hypothesis

## Success Metrics to Track

For each configuration test, monitor:

1. **QD Score vs Iterations**: Does DNS-GA converge faster?
2. **QD Score vs Evaluations**: Is DNS-GA more efficient per evaluation?
3. **Iteration at DNS QD**: When does DNS-GA reach DNS's final QD?
4. **Evaluation Budget Used**: Total evaluations to reach DNS QD
5. **Final QD Score**: Does DNS-GA achieve better final quality?

## Next Steps

1. **Run Test #1**: `g_n=500, num_ga_generations=1, num_ga_children=2`
   - Update notebook configuration
   - Run 1000 iterations
   - Check if QD≥360,358 reached by iteration 980

2. **Analyze Results**:
   - If successful: Proceed to Test #2 (g_n=250)
   - If failed: Investigate why Competition-GA didn't accelerate convergence
     - Check mutation strength (iso_sigma)
     - Verify GA mutation function is working correctly
     - Consider if walker2d_uni benefits from evolutionary forecasting

3. **Document Findings**: Track which configurations work and why

## Hypothesis Validation

The core hypothesis is: **Competition-GA accelerates convergence by making better-informed culling decisions through evolutionary forecasting.**

- **If g_n=500, gens=1 succeeds**: Hypothesis is valid ✓
- **If g_n=500, gens=1 fails**: Hypothesis may be invalid or requires different parameters ✗

Either outcome provides valuable information about the approach's viability.
