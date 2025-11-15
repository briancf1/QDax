"""
DNS-GA Adaptive Frequency Experiments

Tests adaptive g_n strategies where Competition-GA frequency changes during evolution.
These tests require modifications to the DNS-GA algorithm to support dynamic g_n.

Key hypotheses:
1. Early frequent GA (exploration) + late rare GA (exploitation)
2. Early rare GA (build repertoire) + late frequent GA (refinement)
3. QD-score-triggered GA (only when improvement slows)
"""

import os
import json
import time
from datetime import datetime
import functools

import jax
import jax.numpy as jnp
import pandas as pd

from qdax.core.dns_ga import DominatedNoveltySearchGA
from qdax.core.dns import DominatedNoveltySearch
import qdax.tasks.brax as environments
from qdax.tasks.brax.env_creators import scoring_function_brax_envs as scoring_function
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.metrics import CSVLogger, default_qd_metrics


FIXED_PARAMS = {
    'batch_size': 100,
    'env_name': 'walker2d_uni',
    'episode_length': 100,
    'num_iterations': 3000,
    'seed': 42,
    'policy_hidden_layer_sizes': (64, 64),
    'population_size': 1024,
    'k': 3,
    'line_sigma': 0.05,
    'iso_sigma': 0.005,
}


def run_adaptive_experiment(name, g_n_schedule, output_dir):
    """
    Run experiment with adaptive g_n schedule.
    
    Args:
        name: Experiment name
        g_n_schedule: Function (iteration -> g_n) or list of (threshold, g_n) tuples
        output_dir: Output directory
    """
    print(f"\n{'='*80}")
    print(f"Running: {name}")
    print(f"{'='*80}")
    
    # Setup environment (same as main script)
    env = environments.create(FIXED_PARAMS['env_name'], 
                             episode_length=FIXED_PARAMS['episode_length'])
    reset_fn = jax.jit(env.reset)
    
    key = jax.random.key(FIXED_PARAMS['seed'])
    
    policy_layer_sizes = FIXED_PARAMS['policy_hidden_layer_sizes'] + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=FIXED_PARAMS['batch_size'])
    fake_batch = jnp.zeros(shape=(FIXED_PARAMS['batch_size'], env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)
    
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
    
    descriptor_extraction_fn = environments.descriptor_extractor[FIXED_PARAMS['env_name']]
    scoring_fn = functools.partial(
        scoring_function,
        episode_length=FIXED_PARAMS['episode_length'],
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        descriptor_extractor=descriptor_extraction_fn,
    )
    
    reward_offset = environments.reward_offset[FIXED_PARAMS['env_name']]
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * FIXED_PARAMS['episode_length'],
    )
    
    variation_fn = functools.partial(
        isoline_variation,
        iso_sigma=FIXED_PARAMS['iso_sigma'],
        line_sigma=FIXED_PARAMS['line_sigma']
    )
    
    mixing_emitter = MixingEmitter(
        mutation_fn=None,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=FIXED_PARAMS['batch_size']
    )
    
    def competition_ga_mutation_fn(genotype, key):
        genotype_flat, tree_def = jax.tree_util.tree_flatten(genotype)
        num_leaves = len(genotype_flat)
        keys = jax.random.split(key, num_leaves)
        keys_tree = jax.tree_util.tree_unflatten(tree_def, keys)
        
        def add_noise(x, k):
            return x + jax.random.normal(k, shape=x.shape) * FIXED_PARAMS['iso_sigma']
        
        mutated = jax.tree_util.tree_map(add_noise, genotype, keys_tree)
        return mutated
    
    # Initial algorithm with first g_n value
    current_g_n = g_n_schedule(0) if callable(g_n_schedule) else g_n_schedule[0][1]
    
    algorithm = DominatedNoveltySearchGA(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
        population_size=FIXED_PARAMS['population_size'],
        k=FIXED_PARAMS['k'],
        g_n=current_g_n,
        num_ga_children=2,
        num_ga_generations=1,
        mutation_fn=competition_ga_mutation_fn,
    )
    
    print(f"Starting with g_n={current_g_n}")
    
    # Initialize
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = algorithm.init(init_variables, subkey)
    
    print(f"Initial - QD: {init_metrics['qd_score']:.2f}, "
          f"MaxFit: {init_metrics['max_fitness']:.2f}, "
          f"Coverage: {init_metrics['coverage']:.2f}")
    
    # Setup logging
    log_period = 10
    num_loops = FIXED_PARAMS['num_iterations'] // log_period
    
    metrics = {key: jnp.array([]) for key in ["iteration", "qd_score", "coverage", "max_fitness", "time", "g_n"]}
    init_metrics = jax.tree.map(lambda x: jnp.array([x]) if x.shape == () else x, init_metrics)
    init_metrics["iteration"] = jnp.array([0], dtype=jnp.int32)
    init_metrics["time"] = jnp.array([0.0])
    init_metrics["g_n"] = jnp.array([current_g_n], dtype=jnp.int32)
    metrics = jax.tree.map(
        lambda metric, init_metric: jnp.concatenate([metric, init_metric], axis=0),
        metrics, init_metrics
    )
    
    log_filename = os.path.join(output_dir, f"{name}_logs.csv")
    csv_logger = CSVLogger(log_filename, header=list(metrics.keys()))
    csv_logger.log(jax.tree.map(lambda x: x[-1], metrics))
    
    # Main loop with adaptive g_n
    algorithm_scan_update = algorithm.scan_update
    generation_counter = 1
    start_time_total = time.time()
    
    for i in range(num_loops):
        # Check if g_n should change
        current_iter = (i + 1) * log_period
        new_g_n = g_n_schedule(current_iter) if callable(g_n_schedule) else current_g_n
        
        # Update schedule if threshold crossed
        if not callable(g_n_schedule):
            for threshold, g_n_val in g_n_schedule:
                if current_iter >= threshold:
                    new_g_n = g_n_val
        
        if new_g_n != current_g_n:
            print(f"\nIteration {current_iter}: Changing g_n from {current_g_n} to {new_g_n}")
            current_g_n = new_g_n
            # Need to reinitialize algorithm with new g_n
            # This is a limitation - in production would modify algorithm state
            algorithm = DominatedNoveltySearchGA(
                scoring_function=scoring_fn,
                emitter=mixing_emitter,
                metrics_function=metrics_function,
                population_size=FIXED_PARAMS['population_size'],
                k=FIXED_PARAMS['k'],
                g_n=current_g_n,
                num_ga_children=2,
                num_ga_generations=1,
                mutation_fn=competition_ga_mutation_fn,
            )
            algorithm_scan_update = algorithm.scan_update
        
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
        current_metrics["g_n"] = jnp.repeat(current_g_n, log_period)
        metrics = jax.tree.map(
            lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0),
            metrics, current_metrics
        )
        
        csv_logger.log(jax.tree.map(lambda x: x[-1], metrics))
        
        if (i + 1) % 10 == 0:
            print(f"Iter {1+log_period*(i+1)}/{FIXED_PARAMS['num_iterations']} (g_n={current_g_n}) - "
                  f"QD: {metrics['qd_score'][-1]:.2f}, "
                  f"MaxFit: {metrics['max_fitness'][-1]:.2f}, "
                  f"Cov: {metrics['coverage'][-1]:.2f}%")
    
    total_time = time.time() - start_time_total
    
    print(f"\nCompleted in {total_time:.2f}s")
    print(f"Final - QD: {metrics['qd_score'][-1]:.2f}, "
          f"MaxFit: {metrics['max_fitness'][-1]:.2f}, "
          f"Cov: {metrics['coverage'][-1]:.2f}%")
    
    return {
        'name': name,
        'final_qd_score': float(metrics['qd_score'][-1]),
        'final_max_fitness': float(metrics['max_fitness'][-1]),
        'final_coverage': float(metrics['coverage'][-1]),
        'total_time': total_time,
        'log_file': log_filename,
    }


def main():
    """Run adaptive frequency experiments."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"dns_ga_adaptive_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"DNS-GA Adaptive Frequency Experiments - {timestamp}")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    
    experiments = [
        # Early exploration, late exploitation
        {
            'name': 'adaptive_early_frequent_g100to500',
            'schedule': [(0, 100), (1500, 500)],  # g_n=100 until iter 1500, then g_n=500
            'description': 'Frequent GA early for exploration, rare GA late for stability'
        },
        # Early exploitation, late exploration
        {
            'name': 'adaptive_late_frequent_g500to100',
            'schedule': [(0, 500), (1500, 100)],  # g_n=500 until iter 1500, then g_n=100
            'description': 'Rare GA early to build repertoire, frequent GA late for refinement'
        },
        # Three-phase approach
        {
            'name': 'adaptive_three_phase',
            'schedule': [(0, 100), (1000, 250), (2000, 500)],
            'description': 'Frequent -> Moderate -> Rare as evolution progresses'
        },
    ]
    
    results = []
    
    for exp in experiments:
        print(f"\n{'='*80}")
        print(f"Experiment: {exp['name']}")
        print(f"Description: {exp['description']}")
        print(f"{'='*80}")
        
        try:
            result = run_adaptive_experiment(
                exp['name'],
                exp['schedule'],
                output_dir
            )
            results.append(result)
        except Exception as e:
            print(f"ERROR in {exp['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    results_file = os.path.join(output_dir, "adaptive_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'experiments': experiments,
            'results': results,
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("ADAPTIVE EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    
    if results:
        print("\nFinal QD Scores:")
        for r in results:
            print(f"  {r['name']:<40}: {r['final_qd_score']:.2f}")


if __name__ == "__main__":
    main()
