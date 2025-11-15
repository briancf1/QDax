"""
DNS Baseline Experiments - Test mutation strength effect on DNS alone

This script tests if iso_sigma changes improve DNS performance independently
of Competition-GA. This establishes fair comparison baselines.
"""

import os
import json
import time
from datetime import datetime
import functools

import jax
import jax.numpy as jnp
import pandas as pd

from qdax.core.dns import DominatedNoveltySearch
import qdax.tasks.brax as environments
from qdax.tasks.brax.env_creators import scoring_function_brax_envs as scoring_function
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.metrics import CSVLogger, default_qd_metrics


# Fixed parameters
FIXED_PARAMS = {
    'batch_size': 100,
    'env_name': 'walker2d_uni',
    'episode_length': 100,
    'num_iterations': 500,
    'seed': 42,
    'policy_hidden_layer_sizes': (64, 64),
    'population_size': 1024,
    'k': 3,
    'line_sigma': 0.05,
}

# Target QD score for convergence tracking (will be set from baseline results)
BASELINE_TARGET_QD = None

# Baseline experiments
EXPERIMENT_CONFIGS = [
    {
        'name': 'DNS_baseline_iso0.005',
        'iso_sigma': 0.005,
    },
    {
        'name': 'DNS_baseline_iso0.01',
        'iso_sigma': 0.01,
    },
    {
        'name': 'DNS_baseline_iso0.003',
        'iso_sigma': 0.003,
    },
]


def setup_environment(env_name, episode_length, policy_hidden_layer_sizes, batch_size, seed):
    """Initialize environment and policy network."""
    env = environments.create(env_name, episode_length=episode_length)
    reset_fn = jax.jit(env.reset)
    
    key = jax.random.key(seed)
    
    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=batch_size)
    fake_batch = jnp.zeros(shape=(batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)
    
    return env, policy_network, reset_fn, init_variables, key


def create_scoring_function(env, policy_network, reset_fn, episode_length):
    """Create scoring function for fitness evaluation."""
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
        episode_length=episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        descriptor_extractor=descriptor_extraction_fn,
    )
    
    return scoring_fn


def run_experiment(config, output_dir):
    """Run a single baseline experiment."""
    print(f"\n{'='*80}")
    print(f"Running: {config['name']}")
    print(f"{'='*80}")
    
    # Setup
    env, policy_network, reset_fn, init_variables, key = setup_environment(
        FIXED_PARAMS['env_name'],
        FIXED_PARAMS['episode_length'],
        FIXED_PARAMS['policy_hidden_layer_sizes'],
        FIXED_PARAMS['batch_size'],
        FIXED_PARAMS['seed']
    )
    
    scoring_fn = create_scoring_function(env, policy_network, reset_fn, FIXED_PARAMS['episode_length'])
    
    reward_offset = environments.reward_offset[FIXED_PARAMS['env_name']]
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * FIXED_PARAMS['episode_length'],
    )
    
    # Create emitter
    variation_fn = functools.partial(
        isoline_variation,
        iso_sigma=config['iso_sigma'],
        line_sigma=FIXED_PARAMS['line_sigma']
    )
    
    mixing_emitter = MixingEmitter(
        mutation_fn=None,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=FIXED_PARAMS['batch_size']
    )
    
    # Create algorithm
    algorithm = DominatedNoveltySearch(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
        population_size=FIXED_PARAMS['population_size'],
        k=FIXED_PARAMS['k'],
    )
    print(f"Config: Standard DNS, iso_sigma={config['iso_sigma']}")
    
    # Initialize
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = algorithm.init(init_variables, subkey)
    
    print(f"Initial - QD: {init_metrics['qd_score']:.2f}, "
          f"MaxFit: {init_metrics['max_fitness']:.2f}, "
          f"Coverage: {init_metrics['coverage']:.2f}")
    
    # Setup logging
    log_period = 100
    num_loops = FIXED_PARAMS['num_iterations'] // log_period
    
    metrics = {key: jnp.array([]) for key in ["iteration", "qd_score", "coverage", "max_fitness", "time"]}
    init_metrics = jax.tree.map(lambda x: jnp.array([x]) if x.shape == () else x, init_metrics)
    init_metrics["iteration"] = jnp.array([0], dtype=jnp.int32)
    init_metrics["time"] = jnp.array([0.0])
    metrics = jax.tree.map(
        lambda metric, init_metric: jnp.concatenate([metric, init_metric], axis=0),
        metrics, init_metrics
    )
    
    log_filename = os.path.join(output_dir, f"{config['name']}_logs.csv")
    csv_logger = CSVLogger(log_filename, header=list(metrics.keys()))
    csv_logger.log(jax.tree.map(lambda x: x[-1], metrics))
    
    # Main training loop
    algorithm_scan_update = algorithm.scan_update
    start_time_total = time.time()
    convergence_iter = None
    
    for i in range(num_loops):
        start_time = time.time()
        (
            repertoire,
            emitter_state,
            key,
        ), current_metrics = jax.lax.scan(
            algorithm_scan_update,
            (repertoire, emitter_state, key),
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
        
        csv_logger.log(jax.tree.map(lambda x: x[-1], metrics))
        
        # Track convergence (baseline tracks its own final QD score)
        if convergence_iter is None and BASELINE_TARGET_QD is not None:
            if float(metrics['qd_score'][-1]) >= BASELINE_TARGET_QD:
                convergence_iter = int(metrics['iteration'][-1])
                print(f"\n>>> Converged at iteration {convergence_iter} <<<\n")
        
        if (i + 1) % 10 == 0:
            print(f"Iter {1+log_period*(i+1)}/{FIXED_PARAMS['num_iterations']} - "
                  f"QD: {metrics['qd_score'][-1]:.2f}, "
                  f"MaxFit: {metrics['max_fitness'][-1]:.2f}, "
                  f"Cov: {metrics['coverage'][-1]:.2f}%")
    
    total_time = time.time() - start_time_total
    
    print(f"\nCompleted in {total_time:.2f}s")
    print(f"Final - QD: {metrics['qd_score'][-1]:.2f}, "
          f"MaxFit: {metrics['max_fitness'][-1]:.2f}, "
          f"Cov: {metrics['coverage'][-1]:.2f}%")
    
    # Calculate evaluation savings if converged
    eval_savings_pct = None
    if convergence_iter is not None:
        eval_savings_pct = (FIXED_PARAMS['num_iterations'] - convergence_iter) / FIXED_PARAMS['num_iterations'] * 100
        print(f"Convergence: iter {convergence_iter}, savings: {eval_savings_pct:.1f}%")
    
    return {
        'name': config['name'],
        'final_qd_score': float(metrics['qd_score'][-1]),
        'final_max_fitness': float(metrics['max_fitness'][-1]),
        'final_coverage': float(metrics['coverage'][-1]),
        'total_time': total_time,
        'convergence_iter': convergence_iter,
        'eval_savings_pct': eval_savings_pct,
        'log_file': log_filename,
    }


def main():
    """Run all baseline experiments."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"dns_baselines_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"DNS Baseline Experiments - {timestamp}")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Total experiments: {len(EXPERIMENT_CONFIGS)}")
    print(f"Purpose: Test if iso_sigma affects DNS performance")
    
    results = []
    for i, config in enumerate(EXPERIMENT_CONFIGS, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Experiment {i}/{len(EXPERIMENT_CONFIGS)}")
        print(f"{'#'*80}")
        
        try:
            result = run_experiment(config, output_dir)
            results.append(result)
            
            # Set baseline target from iso0.005 (first experiment)
            global BASELINE_TARGET_QD
            if i == 1 and 'iso0.005' in config['name']:
                BASELINE_TARGET_QD = result['final_qd_score']
                print(f"\n>>> Baseline target QD set to: {BASELINE_TARGET_QD:.2f} <<<\n")
        except Exception as e:
            print(f"ERROR in {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    results_file = os.path.join(output_dir, "baseline_results.json")
    with open(results_file, 'w') as f:
        json.dump({'results': results}, f, indent=2)
    
    print(f"\n\n{'='*80}")
    print(f"BASELINE RESULTS")
    print(f"{'='*80}")
    for r in sorted(results, key=lambda x: x['final_qd_score'], reverse=True):
        print(f"{r['name']:<30} QD={r['final_qd_score']:>10.2f}")
    
    print(f"\n\nResults saved to: {output_dir}")


if __name__ == "__main__":
    print('reached')
    main()
