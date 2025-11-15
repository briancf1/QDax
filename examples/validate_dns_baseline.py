"""
Validate DNS Baseline Performance

Quick 100-iteration test to verify baseline DNS achieves reasonable QD scores
before running full experiment suite. Compares against expected performance
from the paper's experiments.
"""

import functools
import jax
import jax.numpy as jnp

from qdax.core.dns import DominatedNoveltySearch
import qdax.tasks.brax as environments
from qdax.tasks.brax.env_creators import scoring_function_brax_envs as scoring_function
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.metrics import default_qd_metrics


def main():
    print("="*80)
    print("DNS Baseline Validation")
    print("="*80)
    print("\nVerifying baseline DNS implementation before full experiment suite...")
    print("This should complete in ~2 minutes.\n")
    
    # Setup (same as full experiments)
    env_name = 'walker2d_uni'
    episode_length = 100
    batch_size = 100
    population_size = 1024
    k = 3
    num_iterations = 100  # Quick test
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
    
    # Emitter
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
    
    # Initialize DNS
    print("Initializing DNS algorithm...")
    algorithm = DominatedNoveltySearch(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
        population_size=population_size,
        k=k,
    )
    
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = algorithm.init(init_variables, subkey)
    
    print(f"Initial metrics:")
    print(f"  QD Score: {init_metrics['qd_score']:.2f}")
    print(f"  Max Fitness: {init_metrics['max_fitness']:.2f}")
    print(f"  Coverage: {init_metrics['coverage']:.2f}%")
    
    # Run for 100 iterations
    print(f"\nRunning {num_iterations} iterations...")
    
    for i in range(num_iterations):
        (
            repertoire,
            emitter_state,
            key,
        ), metrics = jax.lax.scan(
            algorithm.scan_update,
            (repertoire, emitter_state, key),
            (),
            length=1,
        )
        
        if (i + 1) % 20 == 0:
            print(f"  Iteration {i+1}/{num_iterations}: "
                  f"QD={metrics['qd_score'][0]:.2f}, "
                  f"MaxFit={metrics['max_fitness'][0]:.2f}, "
                  f"Cov={metrics['coverage'][0]:.2f}%")
    
    final_qd = float(metrics['qd_score'][0])
    final_fitness = float(metrics['max_fitness'][0])
    final_coverage = float(metrics['coverage'][0])
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"Final metrics after {num_iterations} iterations:")
    print(f"  QD Score: {final_qd:.2f}")
    print(f"  Max Fitness: {final_fitness:.2f}")
    print(f"  Coverage: {final_coverage:.2f}%")
    print()
    
    # Sanity checks based on expected performance
    checks_passed = True
    
    if final_qd < 1000:
        print("⚠️  WARNING: QD score seems low (< 1000)")
        print("   Expected: ~2000-5000 after 100 iterations")
        checks_passed = False
    else:
        print(f"✓ QD score looks reasonable ({final_qd:.2f})")
    
    if final_fitness < 50:
        print("⚠️  WARNING: Max fitness seems low (< 50)")
        print("   Expected: ~100-300 after 100 iterations")
        checks_passed = False
    else:
        print(f"✓ Max fitness looks reasonable ({final_fitness:.2f})")
    
    if final_coverage < 5:
        print("⚠️  WARNING: Coverage seems low (< 5%)")
        print("   Expected: ~10-30% after 100 iterations")
        checks_passed = False
    else:
        print(f"✓ Coverage looks reasonable ({final_coverage:.2f}%)")
    
    print()
    if checks_passed:
        print("✓✓✓ All sanity checks passed!")
        print("    DNS baseline appears to be working correctly.")
        print("    Proceed with full experiment suite.")
    else:
        print("⚠️⚠️⚠️ Some checks failed!")
        print("    Review DNS implementation before running full experiments.")
        print("    These are rough heuristics - review metrics above.")
    
    print()
    return 0 if checks_passed else 1


if __name__ == "__main__":
    exit(main())
