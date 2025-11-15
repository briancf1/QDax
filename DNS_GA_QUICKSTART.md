# DNS-GA Quick Start Guide

## TL;DR

Competition-GA is now implemented as an extension to DNS. It uses evolutionary forecasting to make better culling decisions.

## Quick Setup

```python
from qdax.core.dns_ga import DominatedNoveltySearchGA
from qdax.core.emitters.mutation_operators import polynomial_mutation
from jax.flatten_util import ravel_pytree

# Define mutation for micro-GA
def mutation_fn(genotype, key):
    flat, unravel = ravel_pytree(genotype)
    mutated = polynomial_mutation(
        flat[None, :], key, proportion_to_mutate=0.5, 
        eta=1.0, minval=-jnp.inf, maxval=jnp.inf
    )[0]
    return unravel(mutated)

# Create DNS-GA
dns_ga = DominatedNoveltySearchGA(
    scoring_function=your_scoring_fn,
    emitter=your_emitter,
    metrics_function=your_metrics_fn,
    population_size=1024,
    k=3,
    g_n=5,                    # Competition-GA every 5 generations
    num_ga_children=2,        # 2 offspring per parent
    num_ga_generations=1,     # 1-step forecast
    mutation_fn=mutation_fn,
)

# Initialize and run (same as standard DNS)
repertoire, emitter_state, metrics = dns_ga.init(init_genotypes, key)
repertoire, emitter_state, metrics = dns_ga.update(repertoire, emitter_state, key)
```

## Key Parameters

| Parameter | Description | Recommended | Effect |
|-----------|-------------|-------------|---------|
| `g_n` | Competition-GA frequency | 5 | Lower = more exploration, higher cost |
| `num_ga_children` | Offspring per solution | 2 | Higher = better forecast, higher cost |
| `num_ga_generations` | Forecast depth | 1 | Higher = deeper forecast, exponential cost |
| `mutation_fn` | Mutation operator | polynomial_mutation | Controls variation in micro-GA |

## Quick Experiments

### Experiment 1: Standard DNS Baseline
```python
dns = DominatedNoveltySearch(...)  # No Competition-GA
# Run and record: QD score, max fitness, generations to converge
```

### Experiment 2: DNS-GA with Balanced Settings
```python
dns_ga = DominatedNoveltySearchGA(..., g_n=5, num_ga_children=2, num_ga_generations=1)
# Run and compare with baseline
```

### Experiment 3: High Exploration
```python
dns_ga = DominatedNoveltySearchGA(..., g_n=1, num_ga_children=2, num_ga_generations=1)
# More frequent Competition-GA, higher cost
```

### Experiment 4: Deep Forecasting
```python
dns_ga = DominatedNoveltySearchGA(..., g_n=5, num_ga_children=2, num_ga_generations=2)
# Deeper forecast, higher accuracy, much higher cost
```

## Files to Check

- **Implementation**: `qdax/core/dns_ga.py`, `qdax/core/containers/dns_repertoire_ga.py`
- **Example**: `examples/dns_ga.ipynb`
- **Documentation**: `DNS_GA_README.md`
- **Tests**: `test_dns_ga.py`

## Expected Results

| Metric | Standard DNS | DNS-GA |
|--------|--------------|---------|
| Convergence Speed | Baseline | Faster (fewer generations) |
| QD Score | Baseline | Similar or better |
| Per-Generation Time | Baseline | Higher (by factor of ~1/g_n) |
| Solution Quality | Baseline | Better (preserves high-potential solutions) |

## Common Issues

**Issue**: Out of memory
- **Fix**: Increase `g_n` or decrease `num_ga_children`

**Issue**: Too slow
- **Fix**: Increase `g_n` (less frequent Competition-GA)

**Issue**: Not seeing improvement
- **Fix**: Decrease `g_n` (more frequent Competition-GA)

**Issue**: Want standard DNS behavior
- **Fix**: Set `g_n=float('inf')` or use `DominatedNoveltySearch` directly

## Running the Example

```bash
# Open the example notebook
jupyter notebook examples/dns_ga.ipynb

# Or run the test
python test_dns_ga.py
```

## Comparison Workflow

1. Run standard DNS, save logs
2. Run DNS-GA with same settings (except g_n, mutation_fn), save logs
3. Compare:
   - QD score progression
   - Max fitness progression
   - Coverage progression
   - Time to convergence
   - Total computation time

## Citation

Based on the DNS framework from:
- Bahlous-Boldi et al. "Dominated Novelty Search: Rethinking Quality-Diversity Algorithms" (2025)
- https://arxiv.org/abs/2502.00593
