"""
DNS-GA Comprehensive Parameter Exploration

Tests multiple parameter combinations to find configurations that:
1. Achieve comparable or better QD scores than baseline DNS
2. Use fewer total evaluations (faster convergence)

The script runs all configurations sequentially and saves results for analysis.
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


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Fixed parameters (same across all experiments)
FIXED_PARAMS = {
    'batch_size': 100,
    'env_name': 'walker2d_uni',
    'episode_length': 100,
    'num_iterations': 2000,  # Increased to capture late-stage improvements
    'seed': 42,
    'policy_hidden_layer_sizes': (64, 64),
    'population_size': 1024,
    'k': 3,
    'line_sigma': 0.05,
}

# Parameter combinations to test
# FAIR COMPARISON: All configs use iso_sigma=0.005 to isolate Competition-GA effect
# Total: 15 experiments (~2 hours on your hardware)
EXPERIMENT_CONFIGS = [
    # =========================================================================
    # BASELINES - Compare mutation strength effect on DNS alone
    # =========================================================================
    {
        'name': 'DNS_baseline_iso0.005',
        'use_competition_ga': False,
        'g_n': None,
        'num_ga_children': None,
        'num_ga_generations': None,
        'iso_sigma': 0.005,  # Standard
        'mutation_eta': None,
        'mutation_proportion': None,
    },
    {
        'name': 'DNS_baseline_iso0.01',
        'use_competition_ga': False,
        'g_n': None,
        'num_ga_children': None,
        'num_ga_generations': None,
        'iso_sigma': 0.01,  # Test if higher mutation helps DNS too
        'mutation_eta': None,
        'mutation_proportion': None,
    },
    {
        'name': 'DNS_baseline_iso0.003',
        'use_competition_ga': False,
        'g_n': None,
        'num_ga_children': None,
        'num_ga_generations': None,
        'iso_sigma': 0.003,  # Lower mutation baseline
        'mutation_eta': None,
        'mutation_proportion': None,
    },
    
    # =========================================================================
    # TIER 1: PROVEN WINNERS (from initial results)
    # =========================================================================
    # Top 3 performers with fair comparison (iso_sigma=0.005)
    {
        'name': 'DNS-GA_g300_gen2_iso0.005',
        'use_competition_ga': True,
        'g_n': 300,
        'num_ga_children': 2,
        'num_ga_generations': 2,
        'iso_sigma': 0.005,
        'mutation_eta': 0.1,
        'mutation_proportion': 0.01,
    },
    {
        'name': 'DNS-GA_g614_gen2_iso0.005',
        'use_competition_ga': True,
        'g_n': 614,
        'num_ga_children': 2,
        'num_ga_generations': 2,
        'iso_sigma': 0.005,
        'mutation_eta': 0.1,
        'mutation_proportion': 0.01,
    },
    {
        'name': 'DNS-GA_g250_gen1_iso0.005',
        'use_competition_ga': True,
        'g_n': 250,
        'num_ga_children': 2,
        'num_ga_generations': 1,
        'iso_sigma': 0.005,
        'mutation_eta': 0.1,
        'mutation_proportion': 0.01,
    },
    
    # =========================================================================
    # TIER 2: PROMISING VARIANTS
    # =========================================================================
    # Low overhead configurations
    {
        'name': 'DNS-GA_g500_gen1_iso0.005',
        'use_competition_ga': True,
        'g_n': 500,
        'num_ga_children': 2,
        'num_ga_generations': 1,
        'iso_sigma': 0.005,
        'mutation_eta': 0.1,
        'mutation_proportion': 0.01,
    },
    {
        'name': 'DNS-GA_g400_gen2_iso0.005',
        'use_competition_ga': True,
        'g_n': 400,
        'num_ga_children': 2,
        'num_ga_generations': 2,
        'iso_sigma': 0.005,
        'mutation_eta': 0.1,
        'mutation_proportion': 0.01,
    },
    
    # =========================================================================
    # TIER 3: DEEPER FORESIGHT
    # =========================================================================
    # Test if 3+ generations improves predictions
    {
        'name': 'DNS-GA_g500_gen3_iso0.005',
        'use_competition_ga': True,
        'g_n': 500,
        'num_ga_children': 2,
        'num_ga_generations': 3,
        'iso_sigma': 0.005,
        'mutation_eta': 0.1,
        'mutation_proportion': 0.01,
    },
    {
        'name': 'DNS-GA_g700_gen3_iso0.005',
        'use_competition_ga': True,
        'g_n': 700,
        'num_ga_children': 2,
        'num_ga_generations': 3,
        'iso_sigma': 0.005,
        'mutation_eta': 0.1,
        'mutation_proportion': 0.01,
    },
    {
        'name': 'DNS-GA_g1000_gen4_iso0.005',
        'use_competition_ga': True,
        'g_n': 1000,
        'num_ga_children': 2,
        'num_ga_generations': 4,
        'iso_sigma': 0.005,
        'mutation_eta': 0.1,
        'mutation_proportion': 0.01,
    },
    
    # =========================================================================
    # TIER 4: AGGRESSIVE FREQUENCY (test overhead limits)
    # =========================================================================
    {
        'name': 'DNS-GA_g150_gen1_iso0.005',
        'use_competition_ga': True,
        'g_n': 150,
        'num_ga_children': 2,
        'num_ga_generations': 1,
        'iso_sigma': 0.005,
        'mutation_eta': 0.1,
        'mutation_proportion': 0.01,
    },
    {
        'name': 'DNS-GA_g200_gen2_iso0.005',
        'use_competition_ga': True,
        'g_n': 200,
        'num_ga_children': 2,
        'num_ga_generations': 2,
        'iso_sigma': 0.005,
        'mutation_eta': 0.1,
        'mutation_proportion': 0.01,
    },
    
    # =========================================================================
    # TIER 5: RARE BUT DEEP (minimal overhead, max foresight)
    # =========================================================================
    {
        'name': 'DNS-GA_g1000_gen5_iso0.005',
        'use_competition_ga': True,
        'g_n': 1000,
        'num_ga_children': 2,
        'num_ga_generations': 5,
        'iso_sigma': 0.005,
        'mutation_eta': 0.1,
        'mutation_proportion': 0.01,
    },
    {
        'name': 'DNS-GA_g1500_gen5_iso0.005',
        'use_competition_ga': True,
        'g_n': 1500,
        'num_ga_children': 2,
        'num_ga_generations': 5,
        'iso_sigma': 0.005,
        'mutation_eta': 0.1,
        'mutation_proportion': 0.01,
    },
]

# Note: Adaptive frequency configurations require code modifications
# and will be added in a separate experiment script if needed


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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


def create_mutation_function(iso_sigma):
    """Create mutation function for Competition-GA."""
    def competition_ga_mutation_fn(genotype, key):
        """Mutation function for micro-GA in Competition-GA."""
        genotype_flat, tree_def = jax.tree_util.tree_flatten(genotype)
        num_leaves = len(genotype_flat)
        keys = jax.random.split(key, num_leaves)
        keys_tree = jax.tree_util.tree_unflatten(tree_def, keys)
        
        def add_noise(x, k):
            return x + jax.random.normal(k, shape=x.shape) * iso_sigma
        
        mutated = jax.tree_util.tree_map(add_noise, genotype, keys_tree)
        return mutated
    
    return competition_ga_mutation_fn


def calculate_evaluation_budget(config, num_iterations, batch_size, population_size):
    """Calculate total evaluations for a configuration."""
    standard_evals = num_iterations * batch_size
    
    if not config['use_competition_ga']:
        return standard_evals, 0, standard_evals
    
    # Calculate offspring per GA call
    offspring_per_individual = sum(
        config['num_ga_children']**i 
        for i in range(1, config['num_ga_generations'] + 1)
    )
    offspring_per_ga_call = population_size * offspring_per_individual
    
    # Number of GA calls
    num_ga_calls = num_iterations // config['g_n']
    ga_evals = num_ga_calls * offspring_per_ga_call
    
    total_evals = standard_evals + ga_evals
    return standard_evals, ga_evals, total_evals


def run_experiment(config, output_dir):
    """Run a single experiment configuration."""
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
    if config['use_competition_ga']:
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
        print(f"Config: g_n={config['g_n']}, gens={config['num_ga_generations']}, "
              f"iso_sigma={config['iso_sigma']}")
    else:
        algorithm = DominatedNoveltySearch(
            scoring_function=scoring_fn,
            emitter=mixing_emitter,
            metrics_function=metrics_function,
            population_size=FIXED_PARAMS['population_size'],
            k=FIXED_PARAMS['k'],
        )
        print("Config: Standard DNS (baseline)")
    
    # Initialize
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = algorithm.init(init_variables, subkey)
    
    print(f"Initial - QD: {init_metrics['qd_score']:.2f}, "
          f"MaxFit: {init_metrics['max_fitness']:.2f}, "
          f"Coverage: {init_metrics['coverage']:.2f}")
    
    # Setup logging
    log_period = 100  # Increased from 10 to reduce memory usage
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
    
    if config['use_competition_ga']:
        generation_counter = 1
        
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
            
            if (i + 1) % 10 == 0:
                print(f"Iter {1+log_period*(i+1)}/{FIXED_PARAMS['num_iterations']} - "
                      f"QD: {metrics['qd_score'][-1]:.2f}, "
                      f"MaxFit: {metrics['max_fitness'][-1]:.2f}, "
                      f"Cov: {metrics['coverage'][-1]:.2f}%")
    else:
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
            
            if (i + 1) % 10 == 0:
                print(f"Iter {1+log_period*(i+1)}/{FIXED_PARAMS['num_iterations']} - "
                      f"QD: {metrics['qd_score'][-1]:.2f}, "
                      f"MaxFit: {metrics['max_fitness'][-1]:.2f}, "
                      f"Cov: {metrics['coverage'][-1]:.2f}%")
    
    total_time = time.time() - start_time_total
    
    # Calculate evaluation budget
    standard_evals, ga_evals, total_evals = calculate_evaluation_budget(
        config, FIXED_PARAMS['num_iterations'], 
        FIXED_PARAMS['batch_size'], FIXED_PARAMS['population_size']
    )
    
    print(f"\nCompleted in {total_time:.2f}s")
    print(f"Final - QD: {metrics['qd_score'][-1]:.2f}, "
          f"MaxFit: {metrics['max_fitness'][-1]:.2f}, "
          f"Cov: {metrics['coverage'][-1]:.2f}%")
    print(f"Total evaluations: {total_evals:,}")
    
    # Return summary
    return {
        'name': config['name'],
        'config': config,
        'final_qd_score': float(metrics['qd_score'][-1]),
        'final_max_fitness': float(metrics['max_fitness'][-1]),
        'final_coverage': float(metrics['coverage'][-1]),
        'total_time': total_time,
        'standard_evals': standard_evals,
        'ga_evals': ga_evals,
        'total_evals': total_evals,
        'log_file': log_filename,
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all experiments and generate summary."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"dns_ga_experiments_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"DNS-GA Parameter Exploration - {timestamp}")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Total experiments: {len(EXPERIMENT_CONFIGS)}")
    print(f"Iterations per experiment: {FIXED_PARAMS['num_iterations']}")
    print(f"Estimated time: ~{len(EXPERIMENT_CONFIGS) * 30} minutes")
    
    # Save experiment configuration
    config_file = os.path.join(output_dir, "experiment_config.json")
    with open(config_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'fixed_params': FIXED_PARAMS,
            'experiments': EXPERIMENT_CONFIGS,
        }, f, indent=2)
    
    # Run all experiments
    results = []
    start_time_all = time.time()
    
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
            results.append({
                'name': config['name'],
                'error': str(e),
            })
    
    total_time_all = time.time() - start_time_all
    
    # Generate summary
    print(f"\n\n{'='*80}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {total_time_all/3600:.2f} hours")
    
    # Save results
    results_file = os.path.join(output_dir, "results_summary.json")
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_time_hours': total_time_all / 3600,
            'results': results,
        }, f, indent=2)
    
    # Create summary table
    successful_results = [r for r in results if 'error' not in r]
    
    if successful_results:
        summary_df = pd.DataFrame([{
            'Name': r['name'],
            'Final QD': r['final_qd_score'],
            'Max Fitness': r['final_max_fitness'],
            'Coverage': r['final_coverage'],
            'Total Evals': r['total_evals'],
            'Time (min)': r['total_time'] / 60,
        } for r in successful_results])
        
        summary_csv = os.path.join(output_dir, "summary_table.csv")
        summary_df.to_csv(summary_csv, index=False)
        
        print("\nTop 5 by Final QD Score:")
        print(summary_df.sort_values('Final QD', ascending=False).head().to_string())
        
        # Find baseline DNS result
        baseline = next((r for r in successful_results if r['name'] == 'DNS_baseline'), None)
        
        if baseline:
            print(f"\n\nComparison to Baseline DNS (QD={baseline['final_qd_score']:.2f}):")
            print(f"{'Name':<40} {'QD Diff':<12} {'Eval Efficiency':<20}")
            print("-" * 72)
            
            for r in successful_results:
                if r['name'] != 'DNS_baseline':
                    qd_diff = r['final_qd_score'] - baseline['final_qd_score']
                    
                    # Find iteration where DNS-GA reaches baseline QD
                    df = pd.read_csv(r['log_file'])
                    converged = df[df['qd_score'] >= baseline['final_qd_score']]
                    
                    if len(converged) > 0:
                        conv_iter = int(converged.iloc[0]['iteration'])
                        # Calculate evals to convergence
                        config = next(c for c in EXPERIMENT_CONFIGS if c['name'] == r['name'])
                        offspring_per_indiv = sum(
                            config['num_ga_children']**i 
                            for i in range(1, config['num_ga_generations'] + 1)
                        )
                        offspring_per_ga = FIXED_PARAMS['population_size'] * offspring_per_indiv
                        ga_calls_to_conv = conv_iter // config['g_n']
                        evals_to_conv = conv_iter * FIXED_PARAMS['batch_size'] + ga_calls_to_conv * offspring_per_ga
                        
                        efficiency = f"Iter {conv_iter} ({evals_to_conv:,} evals)"
                        if evals_to_conv < baseline['total_evals']:
                            efficiency += " ✓"
                        else:
                            efficiency += " ✗"
                    else:
                        efficiency = "Did not converge"
                    
                    print(f"{r['name']:<40} {qd_diff:>+10.2f}  {efficiency}")
    
    print(f"\n\nAll results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    print('reached')
    main()
