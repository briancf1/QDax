"""
Quick validation test - runs ONE experiment to verify everything works
before launching the full overnight test suite.
"""

import os
import sys
import time
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


def main():
    print("\n" + "="*80)
    print("DNS-GA VALIDATION TEST")
    print("="*80)
    print("\nThis runs a quick test (50 iterations) to verify everything works")
    print("before launching the full overnight experiment suite.\n")
    
    # Test configuration - fast but realistic
    config = {
        'batch_size': 100,
        'env_name': 'walker2d_uni',
        'episode_length': 100,
        'num_iterations': 50,  # Quick test
        'seed': 42,
        'policy_hidden_layer_sizes': (64, 64),
        'population_size': 1024,
        'k': 3,
        'line_sigma': 0.05,
        'iso_sigma': 0.005,
        'g_n': 25,  # Will trigger twice during test
        'num_ga_children': 2,
        'num_ga_generations': 1,
    }
    
    print("Test Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "-"*80)
    print("INITIALIZING...")
    print("-"*80)
    
    # Setup environment
    env = environments.create(config['env_name'], episode_length=config['episode_length'])
    reset_fn = jax.jit(env.reset)
    
    key = jax.random.key(config['seed'])
    
    # Setup policy
    policy_layer_sizes = config['policy_hidden_layer_sizes'] + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=config['batch_size'])
    fake_batch = jnp.zeros(shape=(config['batch_size'], env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)
    
    print("✓ Environment and policy initialized")
    
    # Create scoring function
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
    
    descriptor_extraction_fn = environments.descriptor_extractor[config['env_name']]
    scoring_fn = functools.partial(
        scoring_function,
        episode_length=config['episode_length'],
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        descriptor_extractor=descriptor_extraction_fn,
    )
    
    reward_offset = environments.reward_offset[config['env_name']]
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * config['episode_length'],
    )
    
    print("✓ Scoring function created")
    
    # Create emitter
    variation_fn = functools.partial(
        isoline_variation,
        iso_sigma=config['iso_sigma'],
        line_sigma=config['line_sigma']
    )
    
    mixing_emitter = MixingEmitter(
        mutation_fn=None,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=config['batch_size']
    )
    
    print("✓ Emitter created")
    
    # Create mutation function for Competition-GA
    def competition_ga_mutation_fn(genotype, key):
        genotype_flat, tree_def = jax.tree_util.tree_flatten(genotype)
        num_leaves = len(genotype_flat)
        keys = jax.random.split(key, num_leaves)
        keys_tree = jax.tree_util.tree_unflatten(tree_def, keys)
        
        def add_noise(x, k):
            return x + jax.random.normal(k, shape=x.shape) * config['iso_sigma']
        
        mutated = jax.tree_util.tree_map(add_noise, genotype, keys_tree)
        return mutated
    
    # Create algorithm
    algorithm = DominatedNoveltySearchGA(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
        population_size=config['population_size'],
        k=config['k'],
        g_n=config['g_n'],
        num_ga_children=config['num_ga_children'],
        num_ga_generations=config['num_ga_generations'],
        mutation_fn=competition_ga_mutation_fn,
    )
    
    print("✓ DNS-GA algorithm created")
    
    # Initialize
    print("\nInitializing repertoire...")
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = algorithm.init(init_variables, subkey)
    
    print(f"✓ Repertoire initialized")
    print(f"  Initial QD: {init_metrics['qd_score']:.2f}")
    print(f"  Initial Max Fitness: {init_metrics['max_fitness']:.2f}")
    print(f"  Initial Coverage: {init_metrics['coverage']:.2f}")
    
    # Run quick test
    print("\n" + "-"*80)
    print(f"RUNNING {config['num_iterations']} ITERATIONS...")
    print("-"*80)
    
    log_period = 10
    num_loops = config['num_iterations'] // log_period
    
    metrics = {key: jnp.array([]) for key in ["iteration", "qd_score", "coverage", "max_fitness", "time"]}
    init_metrics = jax.tree.map(lambda x: jnp.array([x]) if x.shape == () else x, init_metrics)
    init_metrics["iteration"] = jnp.array([0], dtype=jnp.int32)
    init_metrics["time"] = jnp.array([0.0])
    metrics = jax.tree.map(
        lambda metric, init_metric: jnp.concatenate([metric, init_metric], axis=0),
        metrics, init_metrics
    )
    
    algorithm_scan_update = algorithm.scan_update
    generation_counter = 1
    
    start_time_total = time.time()
    
    for i in range(num_loops):
        start_time = time.time()
        (
            repertoire,
            emitter_state,
            key,
            generation_counter,
        ), current_metrics = jax.lax.scan(
            algorithm_scan_update,
            (repertoire, emitter_state, key, generation_counter),
            (),
            length=log_period,
        )
        timelapse = time.time() - start_time
        
        current_metrics["iteration"] = jnp.arange(
            1 + log_period * i, 1 + log_period * (i + 1), dtype=jnp.int32
        )
        current_metrics["time"] = jnp.repeat(timelapse, log_period)
        metrics = jax.tree.map(
            lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0),
            metrics, current_metrics
        )
        
        print(f"Iter {1+log_period*(i+1):3d}/{config['num_iterations']} - "
              f"QD: {metrics['qd_score'][-1]:8.2f}, "
              f"MaxFit: {metrics['max_fitness'][-1]:6.2f}, "
              f"Cov: {metrics['coverage'][-1]:5.2f}%, "
              f"Time: {timelapse:.2f}s")
    
    total_time = time.time() - start_time_total
    
    print("\n" + "-"*80)
    print("TEST COMPLETE!")
    print("-"*80)
    print(f"✓ All {config['num_iterations']} iterations completed successfully")
    print(f"✓ Total time: {total_time:.2f}s ({total_time/config['num_iterations']:.2f}s per iteration)")
    print(f"✓ Final QD Score: {metrics['qd_score'][-1]:.2f}")
    print(f"✓ Final Max Fitness: {metrics['max_fitness'][-1]:.2f}")
    print(f"✓ Final Coverage: {metrics['coverage'][-1]:.2f}%")
    
    print("\n" + "="*80)
    print("VALIDATION SUCCESSFUL!")
    print("="*80)
    print("\nYour setup is working correctly. You can now run the full experiment suite:")
    print("  ./launch_experiments.sh")
    print("\nOr manually:")
    print("  python run_dns_ga_experiments.py")
    print("\nEstimated time for full suite: ~7 hours")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR DURING VALIDATION")
        print("="*80)
        print(f"\n{e}\n")
        import traceback
        traceback.print_exc()
        print("\nPlease fix the error before running the full experiment suite.")
        print("="*80 + "\n")
        sys.exit(1)
