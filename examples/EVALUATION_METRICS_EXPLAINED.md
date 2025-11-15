# Evaluation Metrics Explained

## Overview

The experiment scripts now track **two different efficiency metrics** for DNS-GA:

1. **Convergence Speedup** (`eval_savings_pct`) - How much faster DNS-GA reaches the target
2. **Net Evaluation Savings** (`net_eval_savings_pct`) - True efficiency after accounting for GA overhead

## The Problem

Competition-GA performs **additional evaluations** during its forecasting simulations. Simply measuring when DNS-GA converges faster than baseline DNS doesn't tell the full story if those extra evaluations consume the time saved.

## Metrics Explained

### 1. Convergence Speedup (eval_savings_pct)

**What it measures:** Percentage of iterations saved by reaching baseline QD target earlier.

**Formula:**
```python
eval_savings_pct = (total_iterations - convergence_iter) / total_iterations * 100
```

**Example:**
- Baseline DNS: reaches QD=369k at iteration 3000
- DNS-GA: reaches QD=369k at iteration 1500
- Convergence speedup: (3000 - 1500) / 3000 = **50%**

**What it ignores:** GA overhead evaluations

---

### 2. Net Evaluation Savings (net_eval_savings_pct)

**What it measures:** True evaluation savings after subtracting GA overhead cost.

**Formula:**
```python
baseline_evals = num_iterations * batch_size
dns_ga_main_evals = convergence_iter * batch_size
ga_overhead_evals = (convergence_iter // g_n) * (k * num_ga_children * num_ga_generations)
dns_ga_total_evals = dns_ga_main_evals + ga_overhead_evals

net_eval_savings_pct = (baseline_evals - dns_ga_total_evals) / baseline_evals * 100
```

**Example:**
- Baseline: 3000 × 100 = **300,000 evaluations**
- DNS-GA main: 1500 × 100 = 150,000 evaluations
- DNS-GA overhead: (1500 / 300) × (3 × 2 × 2) = 5 × 12 = **60 evaluations**
- DNS-GA total: 150,000 + 60 = **150,060 evaluations**
- Net savings: (300,000 - 150,060) / 300,000 = **49.98%**

**What it includes:** Everything!

---

## GA Overhead Calculation

The overhead depends on configuration parameters:

```python
num_ga_calls = convergence_iter // g_n
evals_per_ga_call = k × num_ga_children × num_ga_generations
total_ga_overhead = num_ga_calls × evals_per_ga_call
```

### Example Configurations:

| Config | g_n | children | generations | k | Calls (at 1500) | Overhead/call | Total Overhead |
|--------|-----|----------|-------------|---|----------------|---------------|----------------|
| g300_gen2 | 300 | 2 | 2 | 3 | 5 | 12 | **60** |
| g614_gen2 | 614 | 2 | 2 | 3 | 2 | 12 | **24** |
| g150_gen1 | 150 | 2 | 1 | 3 | 10 | 6 | **60** |
| g500_gen3 | 500 | 2 | 3 | 3 | 3 | 18 | **54** |

---

## Which Metric to Use?

### Use **Convergence Speedup** when:
- Comparing iteration-based convergence rates
- Analyzing how quickly algorithms reach targets
- Overhead is negligible (rare GA calls, shallow generations)

### Use **Net Evaluation Savings** when:
- Comparing true computational efficiency
- Determining real-world cost savings
- Making fair comparisons between methods
- **Publishing results** (this is the scientifically rigorous metric!)

---

## Key Insights

1. **Overhead is usually small but not zero**
   - Typical overhead: 24-120 evaluations
   - Main evaluations: 150,000-300,000
   - Overhead ratio: 0.01-0.08% of total

2. **Still significant net savings**
   - Even with overhead, DNS-GA typically saves 40-50% evaluations
   - The forecasting cost is worth the convergence speedup

3. **Configuration matters**
   - Frequent GA calls (low g_n) → higher overhead
   - Deep forecasting (high generations) → higher overhead per call
   - Balance is key: g300-g600 with 1-2 generations seems optimal

---

## Output in Scripts

When experiments complete, you'll see:

```
Evaluation Analysis:
  Convergence iteration: 1500
  Baseline total evals: 300,000
  DNS-GA main evals: 150,000
  DNS-GA overhead evals: 60 (5 GA calls)
  DNS-GA total evals: 150,060
  Convergence speedup: 50.0%
  Net evaluation savings: 49.98%
```

Both metrics are saved in results JSON for analysis in the notebook!

---

## Notebook Visualizations

The updated notebook now shows:

1. **Side-by-side bar charts:**
   - Left: Convergence speedup (optimistic view)
   - Right: Net savings (realistic view)

2. **Summary statistics:**
   - Best overall QD score
   - Most efficient configuration (by net savings)
   - Total evaluation counts

3. **Fair comparison analysis:**
   - Compares DNS-GA net savings vs baseline performance
