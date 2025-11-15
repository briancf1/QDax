"""
DNS-GA Tier 1: Proven Winners

Top 3 configurations from initial experiments that showed best results
with fair comparison (iso_sigma=0.005).
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
}

def load_baseline_target():
    """
    Load baseline target QD score from most recent baseline results.
    Falls back to hardcoded value if baseline results not found.
    """
    import glob
    baseline_dirs = sorted(glob.glob('dns_baselines_*'))
    if baseline_dirs:
        latest_baseline = baseline_dirs[-1]
        results_file = os.path.join(latest_baseline, 'baseline_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                data = json.load(f)
                # Find iso0.005 result (standard baseline)
                for result in data.get('results', []):
                    if 'iso0.005' in result['name']:
                        print(f"Loaded baseline target from {results_file}: {result['final_qd_score']:.2f}")
                        return result['final_qd_score']
    # Fallback to hardcoded value
    print("Warning: Could not find baseline results, using hardcoded target: 369000")
    return 369000

BASELINE_TARGET_QD = load_baseline_target()

def calculate_ga_overhead_evals(g_n, num_iterations, population_size, num_ga_children, num_ga_generations):
    """
    Calculate total evaluations performed by Competition-GA.
    
    The Competition-GA evolves the ENTIRE population through offspring trees.
    Each individual produces num_ga_children offspring per generation.
    
    Total offspring per call = population_size * sum(num_children^i for i=1 to num_ga_generations)
    
    Example with population_size=1024, num_children=2, num_generations=2:
    - Gen 1: 1024 * 2^1 = 2,048 offspring
    - Gen 2: 1024 * 2^2 = 4,096 grandchildren  
    - Total: 6,144 evaluations per GA call
    
    Args:
        g_n: Frequency of GA calls (every g_n iterations)
        num_iterations: Total iterations run
        population_size: DNS population size (all individuals are evolved)
        num_ga_children: Children per parent per generation
        num_ga_generations: Generations of GA evolution
    
    Returns:
        (total_ga_evals, num_ga_calls, evals_per_ga_call)
    """
    num_ga_calls = num_iterations // g_n
    
    # Geometric series: sum(num_children^i for i=1 to num_ga_generations)
    # = num_children * (num_children^num_ga_generations - 1) / (num_children - 1)
    if num_ga_children == 1:
        offspring_per_call = population_size * num_ga_generations
    else:
        offspring_per_call = population_size * num_ga_children * (num_ga_children**num_ga_generations - 1) // (num_ga_children - 1)
    
    evals_per_ga_call = offspring_per_call
    total_ga_evals = num_ga_calls * evals_per_ga_call
    return total_ga_evals, num_ga_calls, evals_per_ga_call

# Top 3 proven winners
EXPERIMENT_CONFIGS = [
    {
        'name': 'DNS-GA_g300_gen2_iso0.005',
        'g_n': 300,
        'num_ga_children': 2,
        'num_ga_generations': 2,
        'iso_sigma': 0.005,
    },
    {
        'name': 'DNS-GA_g614_gen2_iso0.005',
        'g_n': 614,
        'num_ga_children': 2,
        'num_ga_generations': 2,
        'iso_sigma': 0.005,
    },
    {
        'name': 'DNS-GA_g250_gen1_iso0.005',
        'g_n': 250,
        'num_ga_children': 2,
        'num_ga_generations': 1,
        'iso_sigma': 0.005,
    },
]


def setup_environment(env_name, episode_length, policy_hidden_layer_sizes, batch_size, seed):
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


def create_mutation_function(iso_sigma):
    def competition_ga_mutation_fn(genotype, key):
        genotype_flat, tree_def = jax.tree_util.tree_flatten(genotype)
        num_leaves = len(genotype_flat)
        keys = jax.random.split(key, num_leaves)
        keys_tree = jax.tree_util.tree_unflatten(tree_def, keys)
        
        def add_noise(x, k):
            return x + jax.random.normal(k, shape=x.shape) * iso_sigma
        
        mutated = jax.tree_util.tree_map(add_noise, genotype, keys_tree)
        return mutated
    
    return competition_ga_mutation_fn


def run_experiment(config, output_dir):
    print(f"\n{'='*80}")
    print(f"Running: {config['name']}")
    print(f"{'='*80}")
    
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
    
    mutation_fn = create_mutation_function(config['iso_sigma'])
    
    algorithm = DominatedNoveltySearchGA(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
        population_size=FIXED_PARAMS['population_size'],
        k=FIXED_PARAMS['k'],
        g_n=config['g_n'],
        num_ga_children=config['num_ga_children'],
        num_ga_generations=config['num_ga_generations'],
        mutation_fn=mutation_fn,
    )
    print(f"Config: g_n={config['g_n']}, gens={config['num_ga_generations']}, iso_sigma={config['iso_sigma']}")
    
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = algorithm.init(init_variables, subkey)
    
    print(f"Initial - QD: {init_metrics['qd_score']:.2f}, "
          f"MaxFit: {init_metrics['max_fitness']:.2f}, "
          f"Coverage: {init_metrics['coverage']:.2f}")
    
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
    
    algorithm_scan_update = algorithm.scan_update
    start_time_total = time.time()
    generation_counter = 1
    convergence_iter = None
    
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
        
        csv_logger.log(jax.tree.map(lambda x: x[-1], metrics))
        
        # Track convergence to baseline target
        if convergence_iter is None and float(metrics['qd_score'][-1]) >= BASELINE_TARGET_QD:
            convergence_iter = int(metrics['iteration'][-1])
            print(f"\n>>> Converged at iteration {convergence_iter} (reached baseline QD target) <<<\n")
        
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
    
    # Calculate GA overhead
    ga_total_evals, ga_num_calls, ga_evals_per_call = calculate_ga_overhead_evals(
        config['g_n'],
        FIXED_PARAMS['num_iterations'],
        FIXED_PARAMS['population_size'],
        config['num_ga_children'],
        config['num_ga_generations']
    )
    
    # Calculate evaluation counts and savings
    baseline_total_evals = FIXED_PARAMS['num_iterations'] * FIXED_PARAMS['batch_size']
    
    # Convergence-based savings (ignoring GA overhead)
    eval_savings_pct = None
    net_eval_savings_pct = None
    dns_ga_total_evals = None
    
    if convergence_iter is not None:
        # Main DNS-GA evaluations up to convergence
        dns_ga_main_evals = convergence_iter * FIXED_PARAMS['batch_size']
        # Add GA overhead for calls made up to convergence
        ga_calls_until_convergence = convergence_iter // config['g_n']
        ga_evals_until_convergence = ga_calls_until_convergence * ga_evals_per_call
        dns_ga_total_evals = dns_ga_main_evals + ga_evals_until_convergence
        
        # Convergence savings (doesn't account for GA overhead)
        eval_savings_pct = (FIXED_PARAMS['num_iterations'] - convergence_iter) / FIXED_PARAMS['num_iterations'] * 100
        
        # Net savings (accounts for GA overhead)
        net_eval_savings_pct = (baseline_total_evals - dns_ga_total_evals) / baseline_total_evals * 100
        
        print(f"\nEvaluation Analysis:")
        print(f"  Convergence iteration: {convergence_iter}")
        print(f"  Baseline total evals: {baseline_total_evals:,}")
        print(f"  DNS-GA main evals: {dns_ga_main_evals:,}")
        print(f"  DNS-GA overhead evals: {ga_evals_until_convergence:,} ({ga_calls_until_convergence} GA calls)")
        print(f"  DNS-GA total evals: {dns_ga_total_evals:,}")
        print(f"  Convergence speedup: {eval_savings_pct:.1f}%")
        print(f"  Net evaluation savings: {net_eval_savings_pct:.1f}%")
    else:
        print(f"\nDid not converge to baseline target ({BASELINE_TARGET_QD:.0f}) within {FIXED_PARAMS['num_iterations']} iterations")
        print(f"Total GA overhead if run to completion: {ga_total_evals:,} evals ({ga_num_calls} calls)")
    
    return {
        'name': config['name'],
        'final_qd_score': float(metrics['qd_score'][-1]),
        'final_max_fitness': float(metrics['max_fitness'][-1]),
        'final_coverage': float(metrics['coverage'][-1]),
        'total_time': total_time,
        'convergence_iter': convergence_iter,
        'eval_savings_pct': eval_savings_pct,
        'net_eval_savings_pct': net_eval_savings_pct,
        'dns_ga_total_evals': dns_ga_total_evals,
        'baseline_total_evals': baseline_total_evals,
        'ga_overhead_evals': ga_total_evals,
        'baseline_target_qd': BASELINE_TARGET_QD,
        'log_file': log_filename,
    }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"dns_ga_tier1_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"DNS-GA Tier 1: Proven Winners - {timestamp}")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Total experiments: {len(EXPERIMENT_CONFIGS)}")
    
    results = []
    for i, config in enumerate(EXPERIMENT_CONFIGS, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Experiment {i}/{len(EXPERIMENT_CONFIGS)}")
        print(f"{'#'*80}")
        
        try:
            result = run_experiment(config, output_dir)
            results.append(result)
        except Exception as e:
            print(f"ERROR in {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    results_file = os.path.join(output_dir, "tier1_results.json")
    with open(results_file, 'w') as f:
        json.dump({'results': results}, f, indent=2)
    
    print(f"\n\n{'='*80}")
    print(f"TIER 1 RESULTS")
    print(f"{'='*80}")
    for r in sorted(results, key=lambda x: x['final_qd_score'], reverse=True):
        print(f"{r['name']:<35} QD={r['final_qd_score']:>10.2f}")
    
    print(f"\n\nResults saved to: {output_dir}")


if __name__ == "__main__":
    print('reached')
    main()
