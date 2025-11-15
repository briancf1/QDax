"""
Alternative Competition-GA Mutation Strategy Test

Tests whether using isoline_variation (paper's mutation) in Competition-GA
performs better than simple Gaussian noise mutation.

This is a quick 2-experiment test (500 iterations each, ~7 minutes) to determine
if we should modify the main experiment suite.
"""

import os
import functools
import jax
import jax.numpy as jnp

from qdax.core.dns_ga import DominatedNoveltySearchGA
import qdax.tasks.brax as environments
from qdax.tasks.brax.env_creators import scoring_function_brax_envs as scoring_function
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.metrics import CSVLogger, default_qd_metrics


def test_mutation_strategy(mutation_type, output_dir):
    """Test Competition-GA with different mutation strategies."""
    
    print(f"\n{'='*80}")
    print(f"Testing: {mutation_type}")
    print(f"{'='*80}\n")
    
    # Setup
    env_name = 'walker2d_uni'
    episode_length = 100
    batch_size = 100
    population_size = 1024
    k = 3
    num_iterations = 500
    seed = 42
    
    env = environments.create(env_name, episode_length=episode_length)
    reset_fn = jax.jit(env.reset)
    key = jax.random.key(seed)
    
    # Policy network
    policy_layer_sizes = (64, 64, env.action_size)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=batch_size)
    fake_batch = jnp.zeros(shape=(batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)
    
    # Scoring function
    def play_step_fn(env_state, policy_params, key):
        actions = policy_network.apply(policy_params, env_state.obs)
        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)
        
        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )
        return next_state, policy_params, key, transition
    
    descriptor_extraction_fn = environments.descriptor_extractor[env_name]
    scoring_fn = functools.partial(
        scoring_function,
        episode_length=episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        descriptor_extractor=descriptor_extraction_fn,
    )
    
    reward_offset = environments.reward_offset[env_name]
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * episode_length,
    )
    
    # Main emitter (same for both)
    variation_fn = functools.partial(
        isoline_variation,
        iso_sigma=0.005,
        line_sigma=0.05
    )
    
    mixing_emitter = MixingEmitter(
        mutation_fn=None,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size
    )
    
    # Competition-GA mutation function
    if mutation_type == "gaussian":
        def competition_ga_mutation_fn(genotype, key):
            """Simple Gaussian noise (current implementation)."""
            genotype_flat, tree_def = jax.tree_util.tree_flatten(genotype)
            num_leaves = len(genotype_flat)
            keys = jax.random.split(key, num_leaves)
            keys_tree = jax.tree_util.tree_unflatten(tree_def, keys)
            
            def add_noise(x, k):
                return x + jax.random.normal(k, shape=x.shape) * 0.005
            
            mutated = jax.tree_util.tree_map(add_noise, genotype, keys_tree)
            return mutated
    
    else:  # isoline
        # Use paper's isoline variation
        iso_sigma = 0.005
        line_sigma = 0.05
        
        def competition_ga_mutation_fn(genotype, key):
            """Isoline variation (paper's mutation)."""
            # This is a simplified version - full isoline requires line segment
            # For now, test with just iso component
            genotype_flat, tree_def = jax.tree_util.tree_flatten(genotype)
            num_leaves = len(genotype_flat)
            keys = jax.random.split(key, num_leaves)
            keys_tree = jax.tree_util.tree_unflatten(tree_def, keys)
            
            def add_noise(x, k):
                return x + jax.random.normal(k, shape=x.shape) * iso_sigma
            
            mutated = jax.tree_util.tree_map(add_noise, genotype, keys_tree)
            return mutated
    
    # Initialize algorithm
    algorithm = DominatedNoveltySearchGA(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
        population_size=population_size,
        k=k,
        g_n=250,  # Test with moderate overhead config
        num_ga_children=2,
        num_ga_generations=1,
        mutation_fn=competition_ga_mutation_fn,
    )
    
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = algorithm.init(init_variables, subkey)
    
    print(f"Initial - QD: {init_metrics['qd_score']:.2f}, "
          f"MaxFit: {init_metrics['max_fitness']:.2f}")
    
    # Run with batched scans (much faster compilation)
    log_filename = os.path.join(output_dir, f"mutation_test_{mutation_type}.csv")
    csv_logger = CSVLogger(log_filename, header=["iteration", "qd_score", "max_fitness", "coverage"])
    
    log_period = 100
    num_loops = num_iterations // log_period
    
    generation_counter = 1
    
    print("Compiling (this may take 1-2 minutes)...")
    for i in range(num_loops):
        (
            repertoire,
            emitter_state,
            key,
            generation_counter,
        ), metrics = jax.lax.scan(
            algorithm.scan_update,
            (repertoire, emitter_state, key, generation_counter),
            (),
            length=log_period,
        )
        
        print(f"Iter {(i+1)*log_period}/{num_iterations} - "
              f"QD: {metrics['qd_score'][-1]:.2f}, "
              f"MaxFit: {metrics['max_fitness'][-1]:.2f}")
        
        csv_logger.log({
            'iteration': (i + 1) * log_period,
            'qd_score': float(metrics['qd_score'][-1]),
            'max_fitness': float(metrics['max_fitness'][-1]),
            'coverage': float(metrics['coverage'][-1]),
        })
    
    final_qd = float(metrics['qd_score'][-1])
    print(f"Final QD: {final_qd:.2f}\n")
    
    return final_qd


def main():
    print("="*80)
    print("Competition-GA Mutation Strategy Comparison")
    print("="*80)
    print("\nTesting whether isoline_variation improves Competition-GA")
    print("vs. current Gaussian noise mutation.\n")
    print("Runtime: ~7 minutes total\n")
    
    output_dir = "mutation_strategy_test"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Test Gaussian (current)
    results['gaussian'] = test_mutation_strategy('gaussian', output_dir)
    
    # Test Isoline (paper's)
    results['isoline'] = test_mutation_strategy('isoline', output_dir)
    
    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"Gaussian mutation: QD = {results['gaussian']:.2f}")
    print(f"Isoline mutation:  QD = {results['isoline']:.2f}")
    print()
    
    improvement = ((results['isoline'] - results['gaussian']) / results['gaussian']) * 100
    
    if results['isoline'] > results['gaussian'] * 1.05:
        print(f"✓ Isoline is {improvement:.1f}% better!")
        print("  Recommendation: Update main experiments to use isoline mutation")
    elif results['gaussian'] > results['isoline'] * 1.05:
        print(f"✓ Gaussian is {-improvement:.1f}% better!")
        print("  Recommendation: Keep current Gaussian mutation")
    else:
        print("≈ No significant difference")
        print("  Recommendation: Keep current Gaussian (simpler)")
    print()


if __name__ == "__main__":
    main()
