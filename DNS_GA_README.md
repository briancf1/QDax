# DNS-GA: Dominated Novelty Search with Competition-GA

## Overview

This implementation extends the standard Dominated Novelty Search (DNS) algorithm with **Competition-GA**, a novel competition function that uses evolutionary forecasting to make more informed culling decisions during selection.

## Key Concepts

### Standard DNS (Bahlous-Boldi et al.)

DNS reframes Quality-Diversity algorithms as genetic algorithms with location-based competition. Instead of using raw fitness for selection, DNS computes **dominated novelty**:

- For each solution, identify all fitter solutions in the population
- Calculate distances to these fitter solutions in behavioral descriptor space
- The dominated novelty score is the average distance to the k-nearest-fitter neighbors
- Higher dominated novelty = farther from better solutions = higher selection priority

This encourages exploration of underperforming regions of the behavior space.

### Competition-GA Extension

Competition-GA extends DNS by performing **short-term evolutionary forecasting** to estimate the future potential of solutions:

1. **Micro-GA Forecasting**: For each solution, run a mutation-only micro-GA for a limited number of generations
2. **Future Fitness**: Calculate `max(current_fitness, max_offspring_fitness)` for each solution
3. **Future-Based Competition**: Use future fitness (instead of current fitness) in dominated novelty calculations
4. **Selective Application**: Run Competition-GA every `g_n` generations to balance exploration and computational cost

### Benefits

- **Better Convergence**: Fewer generations to reach performance plateau
- **Preserved Quality**: Solutions with high-performing offspring potential are retained
- **Comparable/Superior QD-Scores**: Maintains or improves overall quality-diversity
- **Tunable Trade-offs**: Balance exploration vs. exploitation via `g_n` parameter

## File Structure

```
qdax/
├── core/
│   ├── dns.py                      # Standard DNS algorithm
│   ├── dns_ga.py                   # DNS-GA algorithm (new)
│   └── containers/
│       ├── dns_repertoire.py       # Standard DNS repertoire
│       └── dns_repertoire_ga.py    # DNS-GA repertoire (new)
examples/
├── dns.ipynb                       # Standard DNS example
└── dns_ga.ipynb                    # DNS-GA example (new)
```

## Implementation Details

### DominatedNoveltyGARepertoire

Located in `qdax/core/containers/dns_repertoire_ga.py`

**Key Features:**
- Extends `DominatedNoveltyRepertoire`
- Maintains generation counter for alternating competition functions
- Supports both standard Competition and Competition-GA
- JIT-compatible for performance

**Additional Attributes:**
- `generation_counter`: Tracks current generation
- `g_n`: Generation frequency (Competition-GA runs every g_n generations)
- `num_ga_children`: Offspring per solution in micro-GA
- `num_ga_generations`: Forecast horizon (GA tree depth)
- `mutation_fn`: Mutation function for Competition-GA
- `scoring_fn`: Scoring function for evaluating offspring

### _competition_ga Function

The core Competition-GA logic:

```python
def _competition_ga(
    genotypes,           # Population genotypes
    fitness,             # Current fitness values
    descriptor,          # Behavioral descriptors
    dominated_novelty_k, # Number of neighbors
    mutation_fn,         # Mutation operator
    scoring_fn,          # Evaluation function
    num_children,        # Offspring per parent
    num_generations,     # Forecast depth
    key,                 # Random key
) -> jax.Array:         # Returns competition fitness
```

**Algorithm:**
1. For each solution in the population:
   - Generate `num_children` offspring via mutation
   - Evaluate offspring fitness
   - For `num_generations`, repeat mutation on offspring
   - Track maximum fitness across all descendants
2. Compute dominated novelty using future fitness instead of current fitness
3. Return dominated novelty as competition fitness

**Computational Complexity:**
- Without parent reuse: For 2 children and 2 generations:
  - Gen 0: 1 parent
  - Gen 1: 2 children  
  - Gen 2: 4 grandchildren
  - Total: 7 evaluations per solution
- Standard GA would be: 1 + 3 + 9 = 13 evaluations (parents participate)

### DominatedNoveltySearchGA

Located in `qdax/core/dns_ga.py`

**Key Features:**
- Compatible with standard QDax emitters
- Alternates between Competition and Competition-GA based on `generation_counter`
- Maintains same interface as standard DNS for easy comparison

**Parameters:**
- `g_n`: Generation frequency (1 = every generation, inf = standard DNS)
- `num_ga_children`: Offspring per solution (default: 2)
- `num_ga_generations`: Forecast depth (default: 1)
- `mutation_fn`: Mutation function for micro-GA

## Usage

### Basic Usage

```python
from qdax.core.dns_ga import DominatedNoveltySearchGA
from qdax.core.emitters.mutation_operators import polynomial_mutation
import functools

# Define mutation function for Competition-GA
def mutation_fn(genotype, key):
    flat_genotype, unravel = ravel_pytree(genotype)
    mutated = polynomial_mutation(
        flat_genotype[None, :],
        key,
        proportion_to_mutate=0.5,
        eta=1.0,
        minval=-jnp.inf,
        maxval=jnp.inf,
    )[0]
    return unravel(mutated)

# Create DNS-GA instance
dns_ga = DominatedNoveltySearchGA(
    scoring_function=scoring_fn,
    emitter=emitter,
    metrics_function=metrics_fn,
    population_size=1024,
    k=3,
    g_n=5,                    # Competition-GA every 5 generations
    num_ga_children=2,        # 2 offspring per solution
    num_ga_generations=1,     # 1 generation forecast
    mutation_fn=mutation_fn,  # Mutation for micro-GA
)

# Initialize and run
repertoire, emitter_state, metrics = dns_ga.init(init_genotypes, key)
for _ in range(num_iterations):
    repertoire, emitter_state, metrics = dns_ga.update(
        repertoire, emitter_state, key
    )
```

### Parameter Tuning Guidelines

**g_n (Generation Frequency):**
- `g_n = 1`: Maximum exploration, highest computation cost
- `g_n = 5`: Balanced exploration-exploitation (recommended starting point)
- `g_n = 10`: More exploitation, lower computation cost
- `g_n = float('inf')`: Reduces to standard DNS (no Competition-GA)

**num_ga_children:**
- `num_ga_children = 2`: Standard binary offspring (recommended)
- Higher values: More thorough exploration but higher computational cost

**num_ga_generations:**
- `num_ga_generations = 1`: Short-term forecast (less computation)
- `num_ga_generations = 2`: Medium-term forecast (balanced)
- Higher values: Better future prediction but exponential cost increase

**mutation_fn parameters:**
- `proportion_to_mutate`: Fraction of genome to mutate (0.5 is typical)
- `eta`: Mutation strength (1.0 is typical for polynomial mutation)

## Comparison with Standard DNS

Run both algorithms with identical settings to compare:

```python
# Standard DNS
dns = DominatedNoveltySearch(
    scoring_function=scoring_fn,
    emitter=emitter,
    metrics_function=metrics_fn,
    population_size=1024,
    k=3,
)

# DNS-GA (with Competition-GA)
dns_ga = DominatedNoveltySearchGA(
    scoring_function=scoring_fn,
    emitter=emitter,
    metrics_function=metrics_fn,
    population_size=1024,
    k=3,
    g_n=5,
    num_ga_children=2,
    num_ga_generations=1,
    mutation_fn=mutation_fn,
)
```

**Expected Differences:**
- DNS-GA converges faster (fewer generations to plateau)
- DNS-GA may achieve higher or comparable QD-scores
- DNS-GA has higher per-generation computation cost (proportional to 1/g_n)
- DNS-GA better preserves solutions with high-performing offspring

## Example Notebook

See `examples/dns_ga.ipynb` for a complete working example including:
- Environment setup (Walker2D, Ant, etc.)
- Parameter configuration
- Side-by-side DNS vs DNS-GA comparison
- Visualization of results
- Hyperparameter tuning guidelines

## Implementation Notes

### JAX Compatibility
All functions are designed to be JIT-compatible for performance:
- Use `jax.vmap` for vectorization
- Avoid Python loops where possible
- Use JAX random key splitting properly

### Memory Considerations
Competition-GA temporarily creates additional genotypes for forecasting:
- Memory usage: `O(population_size × num_children × num_ga_generations)`
- For typical settings (pop=1024, children=2, gens=1): ~2048 extra evaluations
- Consider reducing `g_n` if memory is constrained

### Backward Compatibility
- Standard DNS remains unchanged
- DNS-GA with `g_n=inf` and no `mutation_fn` behaves like standard DNS
- All existing DNS code continues to work

## Future Extensions

Potential improvements:
1. **Adaptive g_n**: Automatically adjust frequency based on convergence
2. **Multi-objective forecasting**: Consider both fitness and novelty in micro-GA
3. **Crossover support**: Add crossover operations to micro-GA
4. **Parallel evaluation**: Exploit JAX parallelism for offspring evaluation
5. **Archive-based forecasting**: Use archive history to predict future value

## References

- Bahlous-Boldi et al. "Dominated Novelty Search: Rethinking Quality-Diversity Algorithms" (2025)
- Original DNS paper: https://arxiv.org/abs/2502.00593

## Citation

If you use this implementation, please cite both the original DNS paper and acknowledge this extension:

```bibtex
@article{bahlousboldi2025dns,
  title={Dominated Novelty Search: Rethinking Quality-Diversity Algorithms},
  author={Bahlous-Boldi, [...]},
  journal={arXiv preprint arXiv:2502.00593},
  year={2025}
}
```

## License

This implementation follows the QDax license (Apache 2.0).
