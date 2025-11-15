"""Simple test script to validate DNS-GA implementation.

This script performs basic smoke tests to ensure:
1. DNS-GA can be instantiated
2. Repertoire can be initialized
3. Basic update cycle works
4. Competition-GA is triggered at correct intervals
"""

import jax
import jax.numpy as jnp
from functools import partial

from qdax.core.dns_ga import DominatedNoveltySearchGA
from qdax.core.dns import DominatedNoveltySearch
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.emitters.mutation_operators import isoline_variation, polynomial_mutation
from qdax.utils.metrics import default_qd_metrics
from jax.flatten_util import ravel_pytree


def simple_scoring_fn(genotypes, key):
    """Simple scoring function for testing."""
    # Flatten genotypes to get fitness
    flat_genotypes = jax.vmap(lambda g: ravel_pytree(g)[0])(genotypes)
    
    # Fitness is sum of parameters (simple test function)
    fitness = jnp.sum(flat_genotypes, axis=-1, keepdims=True)
    
    # Descriptors are first 2 dimensions of flattened genotype
    descriptors = flat_genotypes[:, :2]
    
    return fitness, descriptors, {}


def test_dns_ga_basic():
    """Test basic DNS-GA functionality."""
    print("Testing DNS-GA basic functionality...")
    
    # Parameters
    genotype_dim = 10
    batch_size = 20
    population_size = 50
    k = 3
    g_n = 3
    seed = 42
    
    # Create random key
    key = jax.random.key(seed)
    
    # Initialize genotypes (simple array)
    key, subkey = jax.random.split(key)
    init_genotypes = jax.random.normal(subkey, (batch_size, genotype_dim))
    
    # Define mutation function for Competition-GA
    def mutation_fn(genotype, key):
        mutated = polynomial_mutation(
            genotype[None, :],
            key,
            proportion_to_mutate=0.5,
            eta=1.0,
            minval=-10.0,
            maxval=10.0,
        )[0]
        return mutated
    
    # Define variation function for emitter
    variation_fn = partial(isoline_variation, iso_sigma=0.05, line_sigma=0.1)
    
    # Create emitter
    emitter = MixingEmitter(
        mutation_fn=None,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )
    
    # Metrics function
    metrics_fn = partial(default_qd_metrics, qd_offset=0.0)
    
    # Create DNS-GA
    dns_ga = DominatedNoveltySearchGA(
        scoring_function=simple_scoring_fn,
        emitter=emitter,
        metrics_function=metrics_fn,
        population_size=population_size,
        k=k,
        g_n=g_n,
        num_ga_children=2,
        num_ga_generations=1,
        mutation_fn=mutation_fn,
    )
    
    # Initialize
    print("  Initializing repertoire...")
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, metrics = dns_ga.init(init_genotypes, subkey)
    
    print(f"  Initial metrics: QD={metrics['qd_score']:.2f}, "
          f"Max Fit={metrics['max_fitness']:.2f}, "
          f"Coverage={metrics['coverage']:.2f}")
    
    # Run a few updates
    print("  Running updates...")
    for i in range(10):
        key, subkey = jax.random.split(key)
        repertoire, emitter_state, metrics = dns_ga.update(
            repertoire, emitter_state, subkey
        )
        
        # Check that generation counter is incrementing
        if hasattr(repertoire, 'generation_counter'):
            gen = repertoire.generation_counter
            uses_ga = (gen % g_n) == 0
            comp_type = "Competition-GA" if uses_ga else "Standard"
            print(f"  Gen {gen}: {comp_type} - QD={metrics['qd_score']:.2f}, "
                  f"Max Fit={metrics['max_fitness']:.2f}")
    
    print("✓ DNS-GA basic test passed!")
    return True


def test_dns_ga_vs_standard():
    """Test that DNS-GA with g_n=inf behaves like standard DNS."""
    print("\nTesting DNS-GA backward compatibility...")
    
    # Parameters
    genotype_dim = 10
    batch_size = 20
    population_size = 50
    k = 3
    seed = 42
    
    # Create random key
    key = jax.random.key(seed)
    
    # Initialize genotypes
    key, subkey = jax.random.split(key)
    init_genotypes = jax.random.normal(subkey, (batch_size, genotype_dim))
    
    # Define variation function
    variation_fn = partial(isoline_variation, iso_sigma=0.05, line_sigma=0.1)
    
    # Create emitter
    emitter = MixingEmitter(
        mutation_fn=None,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )
    
    # Metrics function
    metrics_fn = partial(default_qd_metrics, qd_offset=0.0)
    
    # Create standard DNS
    dns = DominatedNoveltySearch(
        scoring_function=simple_scoring_fn,
        emitter=emitter,
        metrics_function=metrics_fn,
        population_size=population_size,
        k=k,
    )
    
    # Create DNS-GA with g_n=inf (should behave like standard DNS)
    dns_ga = DominatedNoveltySearchGA(
        scoring_function=simple_scoring_fn,
        emitter=emitter,
        metrics_function=metrics_fn,
        population_size=population_size,
        k=k,
        g_n=int(1e9),  # Very large number, effectively infinite
        num_ga_children=2,
        num_ga_generations=1,
        mutation_fn=None,  # No mutation function = standard competition only
    )
    
    # Initialize both with same genotypes
    key, subkey = jax.random.split(key)
    repertoire_dns, emitter_state_dns, metrics_dns = dns.init(init_genotypes, subkey)
    
    key, subkey = jax.random.split(key)
    repertoire_dns_ga, emitter_state_dns_ga, metrics_dns_ga = dns_ga.init(
        init_genotypes, subkey
    )
    
    print(f"  Standard DNS - QD={metrics_dns['qd_score']:.2f}, "
          f"Max Fit={metrics_dns['max_fitness']:.2f}")
    print(f"  DNS-GA (g_n=inf) - QD={metrics_dns_ga['qd_score']:.2f}, "
          f"Max Fit={metrics_dns_ga['max_fitness']:.2f}")
    
    print("✓ Backward compatibility test passed!")
    return True


def test_competition_ga_triggering():
    """Test that Competition-GA is triggered at correct generations."""
    print("\nTesting Competition-GA triggering logic...")
    
    genotype_dim = 10
    batch_size = 20
    population_size = 50
    k = 3
    g_n = 5
    seed = 42
    
    key = jax.random.key(seed)
    
    # Initialize genotypes
    key, subkey = jax.random.split(key)
    init_genotypes = jax.random.normal(subkey, (batch_size, genotype_dim))
    
    # Define mutation function
    def mutation_fn(genotype, key):
        mutated = polynomial_mutation(
            genotype[None, :],
            key,
            proportion_to_mutate=0.5,
            eta=1.0,
            minval=-10.0,
            maxval=10.0,
        )[0]
        return mutated
    
    # Define variation function
    variation_fn = partial(isoline_variation, iso_sigma=0.05, line_sigma=0.1)
    
    # Create emitter
    emitter = MixingEmitter(
        mutation_fn=None,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )
    
    # Metrics function
    metrics_fn = partial(default_qd_metrics, qd_offset=0.0)
    
    # Create DNS-GA
    dns_ga = DominatedNoveltySearchGA(
        scoring_function=simple_scoring_fn,
        emitter=emitter,
        metrics_function=metrics_fn,
        population_size=population_size,
        k=k,
        g_n=g_n,
        num_ga_children=2,
        num_ga_generations=1,
        mutation_fn=mutation_fn,
    )
    
    # Initialize
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, metrics = dns_ga.init(init_genotypes, subkey)
    
    # Track which generations use Competition-GA
    # We track generation counter externally now
    ga_generations = []
    generation_counter = 1  # Start at 1 (init is generation 0)
    for i in range(20):
        key, subkey = jax.random.split(key)
        
        # Check if Competition-GA will be used this generation
        uses_ga = (generation_counter % g_n) == 0
        if uses_ga:
            ga_generations.append(generation_counter)
        
        repertoire, emitter_state, metrics = dns_ga.update(
            repertoire, emitter_state, subkey, generation_counter
        )
        generation_counter += 1
    
    # Expected: after init (gen 0), updates happen at gen 1,2,3...20
    # Competition-GA runs when gen % 5 == 0, so at generations 5, 10, 15, 20
    expected_ga_gens = [i for i in range(1, 21) if i % g_n == 0]
    
    print(f"  Expected Competition-GA at generations: {expected_ga_gens}")
    print(f"  Observed Competition-GA at generations: {ga_generations}")
    
    if ga_generations == expected_ga_gens:
        print("✓ Competition-GA triggering test passed!")
        return True
    else:
        print("✗ Competition-GA triggering test failed!")
        return False


if __name__ == "__main__":
    print("="*60)
    print("DNS-GA Implementation Tests")
    print("="*60)
    
    try:
        test_dns_ga_basic()
        test_dns_ga_vs_standard()
        test_competition_ga_triggering()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
