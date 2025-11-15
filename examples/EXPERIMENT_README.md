# DNS-GA Parameter Exploration Suite

This directory contains scripts for comprehensive testing of DNS-GA parameter configurations to identify optimal settings for evolutionary forecasting.

## Overview

The DNS-GA approach uses Competition-GA (evolutionary forecasting) to make more informed culling decisions. This experiment suite tests multiple parameter combinations to find configurations that:

1. **Achieve comparable or better QD scores** than baseline DNS
2. **Use fewer total evaluations** (converge faster)
3. **Balance computational cost** vs performance gains

## Key Insight

**DNS-GA can never use fewer evaluations than DNS at the same iteration count** because Competition-GA adds extra evaluations. The only way DNS-GA can be more efficient is through **faster convergence** - reaching target QD scores in fewer iterations such that even with the overhead, total evaluations are lower.

## Files

### `run_dns_ga_experiments.py`
Main experiment runner that:
- Tests 15 different parameter combinations
- Runs 2000 iterations per configuration (vs 1000 in original tests)
- Saves detailed logs for each experiment
- Generates summary results

**Usage:**
```bash
python run_dns_ga_experiments.py
```

**Output:**
Creates a directory `dns_ga_experiments_YYYYMMDD_HHMMSS/` containing:
- Individual CSV logs for each experiment
- `results_summary.json` - Complete results
- `summary_table.csv` - Quick comparison table
- `experiment_config.json` - Configuration used

**Estimated runtime:** ~3.2 hours for all 26 main experiments (based on 3.5 min/1000 iters)
**Adaptive tests:** +0.5 hours (run separately with `run_adaptive_experiments.py`)

### `analyze_dns_ga_experiments.py`
Analysis script that:
- Loads all experiment results
- Analyzes convergence speed and efficiency
- Generates visualizations
- Identifies best configurations

**Usage:**
```bash
python analyze_dns_ga_experiments.py dns_ga_experiments_YYYYMMDD_HHMMSS/
```

**Output:**
- `convergence_curves.png` - QD score over iterations
- `final_comparison.png` - Bar charts of final metrics
- `efficiency_analysis.png` - Speedup vs GA frequency
- `convergence_analysis.csv` - Detailed convergence data

## Parameter Configurations Tested

### Fixed Parameters (All Experiments)
- `batch_size`: 100
- `population_size`: 1024
- `num_iterations`: 2000 (increased from 1000)
- `k`: 3 (for novelty calculation)
- `line_sigma`: 0.05

### Variable Parameters

#### 1. **g_n** (Competition-GA Frequency)
- `500`: Very infrequent (2% overhead, 2 GA calls)
- `250`: Infrequent (4% overhead, 8 GA calls)
- `200`: Moderate (10% overhead, 10 GA calls)
- `100`: Frequent (20% overhead, 20 GA calls)

#### 2. **num_ga_generations** (Forecast Horizon)
- `1`: Short-term forecast (low cost)
- `2`: Medium-term forecast (moderate cost)

#### 3. **iso_sigma** (Mutation Strength)
- `0.003`: Low mutation
- `0.005`: Standard mutation (baseline)
- `0.01`: High mutation

#### 4. **mutation_eta** (Currently unused, but tested)
- `0.05`, `0.1`, `0.2`

#### 5. **mutation_proportion** (Currently unused, but tested)
- `0.005`, `0.01`, `0.02`

## Experiment Configurations

**Total: 26 experiments in main suite + 3 adaptive experiments**
**Estimated runtime: ~3.2 hours (main) + ~0.5 hours (adaptive) = 3.7 hours total**

### Tier 1: High Success Probability (Low Overhead)
1. **g500_gen1_iso0.005** - Most conservative (2% overhead)
2. **g500_gen1_iso0.01** - Higher mutation
3. **g500_gen1_iso0.003** - Lower mutation
4. **g250_gen1_iso0.005** - More frequent (4% overhead)
5. **g250_gen1_iso0.01** - More frequent + higher mutation

### Tier 2: Better Foresight (Moderate Overhead)
6. **g614_gen2_iso0.005** - 2 generations, infrequent (6% overhead)
7. **g614_gen2_iso0.01** - 2 generations + higher mutation
8. **g300_gen2_iso0.005** - 2 generations, moderate frequency

### Tier 3: Parameter Sensitivity Tests
9. **g250_gen1_iso0.005_eta0.05** - Lower eta
10. **g250_gen1_iso0.005_eta0.2** - Higher eta
11. **g250_gen1_iso0.005_prop0.005** - Lower mutation proportion
12. **g250_gen1_iso0.005_prop0.02** - Higher mutation proportion

### Tier 4: Aggressive (For Comparison)
13. **g100_gen1_iso0.005** - Frequent GA (20% overhead)
14. **g200_gen1_iso0.005** - Moderate frequency (10% overhead)

### Tier 5: Deep Foresight (Edge Cases)
15. **g1000_gen5_iso0.005** - Very deep lookahead (5 gens), rare execution
16. **g2000_gen10_iso0.005** - Extreme depth (10 gens), single call at end
17. **g1500_gen3_iso0.005** - 3 generations, minimal overhead (~1 call)
18. **g1000_gen4_iso0.005** - 4 generations, rare execution (2 calls)

### Tier 6: Mutation Extremes
19. **g250_gen1_iso0.02** - Very high mutation (4x baseline)
20. **g250_gen1_iso0.001** - Very low mutation (0.2x baseline)

### Tier 7: More Offspring
21. **g250_gen1_children3** - 3 children per parent
22. **g250_gen1_children4** - 4 children per parent

### Tier 8: Combined Optimizations
23. **g400_gen2_iso0.007** - Balanced middle-ground configuration

### Baseline
24. **DNS_baseline** - Standard DNS (no Competition-GA)

### Adaptive Frequency Tests (Separate Script)
25. **adaptive_early_frequent** - g_n=100 first 1000 iters, then g_n=500
26. **adaptive_late_frequent** - g_n=500 first 1000 iters, then g_n=100
27. **adaptive_three_phase** - g_n=100→250→500 progression

## Success Criteria

For each configuration, success is determined by:

1. **Convergence Speed**: Does it reach baseline DNS QD score faster?
2. **Evaluation Efficiency**: Does it use fewer total evaluations?
3. **Final Quality**: Does it achieve equal or better final QD score?

### Break-even Calculations

| g_n | Generations | Overhead | Break-even Iteration | Required Speedup |
|-----|-------------|----------|---------------------|------------------|
| 500 | 1 | 2.0% | <1960 (of 2000) | 1.02x |
| 250 | 1 | 4.1% | <1923 | 1.04x |
| 614 | 2 | 6.1% | <1886 | 1.06x |
| 200 | 1 | 10.2% | <1818 | 1.10x |
| 100 | 1 | 20.5% | <1661 | 1.20x |

## Running Overnight Tests

### Step 1: Start Main Experiments
```bash
cd /Users/briancf/Desktop/source/EvoAlgsAndSwarm/lib-qdax/QDax/examples
source ../.venv/bin/activate
nohup python run_dns_ga_experiments.py > experiment_output.log 2>&1 &
```

This will:
- Run all 26 main experiments sequentially
- Take approximately 3.2 hours
- Save output to `experiment_output.log`
- Continue running even if terminal is closed

### Step 1b: Start Adaptive Experiments (Optional)
```bash
# Run after main experiments or in parallel in another terminal
nohup python run_adaptive_experiments.py > adaptive_output.log 2>&1 &
```

This will:
- Run 3 adaptive frequency experiments
- Take approximately 30 minutes
- Test dynamic g_n schedules
- Save to separate directory

### Step 2: Monitor Progress
```bash
# Check if still running
ps aux | grep run_dns_ga_experiments

# View progress
tail -f experiment_output.log

# Check completion
ls -lh dns_ga_experiments_*/
```

### Step 3: Analyze Results
```bash
# Find the experiment directory
ls -d dns_ga_experiments_*

# Run analysis
python analyze_dns_ga_experiments.py dns_ga_experiments_YYYYMMDD_HHMMSS/
```

## Expected Outcomes

### If Tier 1 Configs Succeed (g_n=500, g_n=250)
- **Hypothesis validated**: Competition-GA accelerates convergence with minimal cost
- **Next steps**: Test even more frequent GA (g_n=100, g_n=50)
- **Publication value**: Strong evidence for evolutionary forecasting approach

### If Tier 1 Configs Fail
- **Hypothesis questioned**: Competition-GA may not accelerate convergence enough
- **Investigate**: 
  - Is walker2d_uni task suitable for evolutionary forecasting?
  - Does mutation strength need tuning?
  - Are 2000 iterations sufficient to see benefits?

### If Only Aggressive Configs Work (g_n=100)
- **Mixed results**: Benefits exist but at high computational cost
- **Trade-off**: Better quality vs more evaluations
- **Consider**: Different evaluation metrics or longer-term benefits

## Interpreting Results

### Key Metrics to Examine

1. **QD Score Progression**
   - Does DNS-GA show steeper initial growth?
   - Does it plateau at a higher level?
   - Are gains visible early or only after many iterations?

2. **Convergence Iteration**
   - At what iteration does DNS-GA reach DNS baseline QD?
   - Is this before the break-even iteration?

3. **Final Performance**
   - Does DNS-GA achieve significantly better final QD?
   - Is max fitness higher?
   - Is coverage maintained?

4. **Parameter Sensitivity**
   - Which iso_sigma works best?
   - Does g_n matter more than num_ga_generations?
   - Are mutation_eta and mutation_proportion relevant?

### Success Patterns

**Strong Success:**
- g_n=500 converges 5%+ faster
- Final QD is 2%+ better than baseline
- All low-overhead configs show benefits

**Moderate Success:**
- g_n=250 converges 3%+ faster
- Final QD is comparable to baseline
- Trade-off between overhead and benefits

**Weak/No Success:**
- No configuration converges faster than break-even
- Final QD similar or worse than baseline
- High-overhead configs don't justify cost

## Troubleshooting

### Out of Memory Errors
```bash
# Reduce batch_size or population_size in run_dns_ga_experiments.py
batch_size = 50  # Instead of 100
population_size = 512  # Instead of 1024
```

### Experiments Taking Too Long
```bash
# Reduce num_iterations
num_iterations = 1000  # Instead of 2000

# Or test fewer configurations
# Comment out experiments in EXPERIMENT_CONFIGS list
```

### Analysis Script Errors
```bash
# Install required packages
pip install pandas matplotlib seaborn

# Check experiment directory exists
ls dns_ga_experiments_*/
```

## Next Steps After Analysis

1. **Identify best configuration** from results
2. **Run extended test** (5000+ iterations) with best config
3. **Test on different tasks** (other Brax environments)
4. **Tune mutation operators** based on findings
5. **Write up results** for paper/documentation

## Questions to Answer

1. Does Competition-GA accelerate convergence?
2. What's the optimal g_n for this task?
3. Does deeper forecasting (num_ga_generations=2) help?
4. What mutation strength (iso_sigma) works best?
5. Are there late-stage benefits that only appear after 1000+ iterations?
6. Is the computational overhead justified by performance gains?

## Contact

For questions about this experiment suite, refer to:
- `viable_configs_analysis.md` - Theoretical analysis
- Paper: research_papers/2502.00593v1.md - DNS-GA original paper
