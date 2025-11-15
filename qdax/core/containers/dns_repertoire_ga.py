"""Dominated Novelty Search with Competition-GA (DNS-GA) repertoire.

This container extends the standard DNS repertoire by implementing Competition-GA,
a novel competition function that performs short-term evolutionary forecasting to
better evaluate solution value. Competition-GA selectively runs a micro-GA to
predict the future fitness potential of solutions and their k-nearest-fitter
neighbors, enabling more informed culling decisions.

The Competition-GA function:
1. Performs asexual (mutation-only) evolution for a limited number of generations
2. Calculates dominated novelty using "future fitness" (max of current and offspring)
3. Operates selectively every g_n generations to balance exploration and computation

This enables improved convergence while preserving solutions with high-performing
offspring potential.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import flax.struct
import jax
import jax.numpy as jnp

from qdax.core.containers.dns_repertoire import (
    DominatedNoveltyRepertoire,
    _novelty_and_dominated_novelty,
)
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


def _competition_ga(
    genotypes: Genotype,
    fitness: jax.Array,
    descriptor: jax.Array,
    dominated_novelty_k: int,
    mutation_fn: Callable[[Genotype, RNGKey], Genotype],
    scoring_fn: Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores]],
    num_children: int,
    num_generations: int,
    key: RNGKey,
) -> jax.Array:
    """Compute competition fitness using genetic algorithm forecasting.

    This function performs short-term evolutionary forecasting by:
    1. For each solution, generating offspring through mutation-only GA
    2. Computing "future fitness" as mean(current_fitness, all_offspring_fitness)
    3. Using future fitness in dominated novelty calculation

    The GA is structured as a tree where each parent has `num_children` offspring,
    but offspring do not become parents (unlike standard GA). This limits
    computational growth: with 2 children and 2 generations, we have:
    - Generation 0: 1 parent
    - Generation 1: 2 children
    - Generation 2: 4 grandchildren
    Total: 7 evaluations vs. standard GA's 9 (with parents participating)

    Args:
        genotypes: population genotypes, pytree (N, ...)
        fitness: shape (N,) fitness values, higher is better
        descriptor: shape (N, D) descriptors in behavior space
        dominated_novelty_k: number of fitter-neighbors for dominated novelty
        mutation_fn: function to mutate genotypes, takes (genotype, key) -> genotype
        scoring_fn: function to evaluate genotypes, takes (genotypes, key) -> (fitness, descriptors, extra_scores)
        num_children: number of offspring per parent in GA tree
        num_generations: depth of GA tree (forecast horizon)
        key: random key for stochastic operations

    Returns:
        competition_fitness: shape (N,) - dominated novelty scores using future fitness
    """

    valid = fitness != -jnp.inf
    population_size = fitness.shape[0]

    # Generate all offspring in batches per generation, then evaluate all at once
    keys = jax.random.split(key, num_generations + 2)
    eval_key = keys[-1]
    gen_keys = keys[:-1]
    
    def evolve_and_evaluate_batch(genotypes_pop, fitness_pop):
        """Evolve entire population through all generations and compute mean fitness.
        
        Uses a simpler approach: generate all offspring sequentially by generation,
        collect them all, evaluate in one batch, then compute means.
        """
        valid_mask = fitness_pop != -jnp.inf
        
        # Lists to collect offspring genotypes and track their organization
        all_offspring_genotypes = []
        
        # Generation 0: all parents produce children
        current_parents = genotypes_pop
        n_current = population_size
        
        for gen in range(num_generations):
            # Generate keys for this generation
            n_offspring = n_current * num_children
            mutation_keys = jax.random.split(gen_keys[gen], n_offspring)
            
            # Create offspring: each parent produces num_children offspring
            # Reshape keys to (n_current, num_children)
            mutation_keys_reshaped = mutation_keys.reshape(n_current, num_children)
            
            # Generate all offspring for this generation using vmap
            # Outer vmap: over parents
            # Inner vmap: over children per parent
            def mutate_one_parent(parent_genotype, parent_keys):
                """Generate num_children offspring from one parent."""
                return jax.vmap(mutation_fn, in_axes=(None, 0))(parent_genotype, parent_keys)
            
            # Apply to all current parents
            offspring_nested = jax.vmap(mutate_one_parent, in_axes=(0, 0))(
                current_parents, mutation_keys_reshaped
            )
            
            # Flatten from (n_current, num_children, ...) to (n_current * num_children, ...)
            offspring_flat = jax.tree.map(
                lambda x: x.reshape((n_current * num_children,) + x.shape[2:]),
                offspring_nested
            )
            
            # Store these offspring
            all_offspring_genotypes.append(offspring_flat)
            
            # These offspring become parents for next generation
            current_parents = offspring_flat
            n_current = n_current * num_children
        
        # Concatenate all offspring across all generations
        all_offspring_concat = jax.tree.map(
            lambda *arrs: jnp.concatenate(arrs, axis=0),
            *all_offspring_genotypes
        )
        
        # SINGLE BATCH EVALUATION: Evaluate all offspring at once
        offspring_fitness, _, _ = scoring_fn(all_offspring_concat, eval_key)
        offspring_fitness = jnp.reshape(offspring_fitness, (-1,))
        
        # Compute generation sizes and offsets statically
        gen_sizes = [population_size * (num_children ** (i+1)) for i in range(num_generations)]
        gen_offsets = [0] + [sum(gen_sizes[:i+1]) for i in range(num_generations - 1)]
        
        # Compute mean fitness for each individual
        def compute_individual_mean(idx):
            """Compute mean of parent and all offspring fitness for one individual."""
            is_valid = valid_mask[idx]
            parent_fitness = fitness_pop[idx]
            
            def do_compute(_):
                # Collect this individual's offspring across all generations
                individual_offspring_fitnesses = []
                
                for gen_idx in range(num_generations):
                    # Static offset to start of this generation's offspring
                    gen_offset = gen_offsets[gen_idx]
                    offspring_per_indiv = num_children ** (gen_idx + 1)
                    
                    # Calculate start position for this individual's offspring in this generation
                    start_in_gen = idx * offspring_per_indiv
                    global_start = gen_offset + start_in_gen
                    
                    # Extract this individual's offspring from this generation
                    indiv_gen_offspring = jax.lax.dynamic_slice(
                        offspring_fitness,
                        (global_start,),
                        (offspring_per_indiv,)
                    )
                    individual_offspring_fitnesses.append(indiv_gen_offspring)
                
                # Concatenate all offspring fitnesses
                all_offspring_fit = jnp.concatenate(individual_offspring_fitnesses)
                
                # Compute mean of parent + all offspring
                all_fit = jnp.concatenate([jnp.array([parent_fitness]), all_offspring_fit])
                return jnp.mean(all_fit)
            
            return jax.lax.cond(is_valid, do_compute, lambda _: -jnp.inf, None)
        
        # Vectorize over all individuals
        future_fitness = jax.vmap(compute_individual_mean)(jnp.arange(population_size))
        return future_fitness
    
    future_fitness = evolve_and_evaluate_batch(genotypes, fitness)

    # Compute dominated novelty using future fitness
    # Identify fitter neighbors based on future fitness
    fitter = future_fitness[:, None] <= future_fitness[None, :]
    neighbor = valid[:, None] & valid[None, :]
    neighbor = jnp.fill_diagonal(neighbor, False, inplace=False)
    fitter = jnp.where(neighbor, fitter, False)

    # Pairwise distances in descriptor space
    distance = jnp.linalg.norm(descriptor[:, None, :] - descriptor[None, :, :], axis=-1)
    distance = jnp.where(neighbor, distance, jnp.inf)

    # Distances to fitter neighbors only
    distance_fitter = jnp.where(fitter, distance, jnp.inf)

    # Dominated novelty: mean distance to k nearest fitter neighbors
    values_fit, indices_fit = jax.vmap(
        lambda x: jax.lax.top_k(-x, dominated_novelty_k)
    )(distance_fitter)
    dominated_novelty = jnp.mean(
        -values_fit,
        axis=-1,
        where=jnp.take_along_axis(fitter, indices_fit, axis=-1),
    )

    return dominated_novelty


class DominatedNoveltyGARepertoire(DominatedNoveltyRepertoire):
    """DNS repertoire with Competition-GA support.

    Extends the standard DNS repertoire to support alternating between standard
    Competition (dominated novelty) and Competition-GA (forecasting-based
    dominated novelty) based on generation frequency.

    Additional attributes:
        g_n: generation frequency - Competition-GA runs every g_n generations
        num_ga_children: number of offspring per solution in micro-GA
        num_ga_generations: number of generations in evolutionary forecast
        mutation_fn: function to mutate genotypes in Competition-GA
        scoring_fn: function to evaluate genotypes in Competition-GA
    """

    g_n: int = flax.struct.field(pytree_node=False)
    num_ga_children: int = flax.struct.field(pytree_node=False)
    num_ga_generations: int = flax.struct.field(pytree_node=False)
    mutation_fn: Optional[Callable] = flax.struct.field(pytree_node=False, default=None)
    scoring_fn: Optional[Callable] = flax.struct.field(pytree_node=False, default=None)

    def add(  # type: ignore
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
        key: Optional[RNGKey] = None,
        generation_counter: int = 0,
    ) -> DominatedNoveltyGARepertoire:
        """Add a batch and keep the top individuals by competition fitness.

        Competition fitness alternates between:
        - Standard dominated novelty (Competition)
        - GA-forecasting dominated novelty (Competition-GA)

        Args:
            batch_of_genotypes: new genotypes to add
            batch_of_descriptors: descriptors of new genotypes
            batch_of_fitnesses: fitness values of new genotypes
            batch_of_extra_scores: optional extra scores
            key: random key for Competition-GA (required if using Competition-GA)
            generation_counter: current generation number for alternation logic

        Returns:
            Updated repertoire with incremented generation counter
        """

        if batch_of_extra_scores is None:
            batch_of_extra_scores = {}

        filtered_batch_of_extra_scores = self.filter_extra_scores(batch_of_extra_scores)

        batch_of_fitnesses = jnp.reshape(
            batch_of_fitnesses, (batch_of_fitnesses.shape[0], 1)
        )

        # Gather candidates
        candidates_genotypes = jax.tree.map(
            lambda x, y: jnp.concatenate((x, y), axis=0),
            self.genotypes,
            batch_of_genotypes,
        )
        candidates_fitnesses = jnp.concatenate(
            (self.fitnesses, batch_of_fitnesses), axis=0
        )
        candidates_descriptors = jnp.concatenate(
            (self.descriptors, batch_of_descriptors), axis=0
        )
        candidates_extra_scores = jax.tree.map(
            lambda x, y: jnp.concatenate((x, y), axis=0),
            self.extra_scores,
            filtered_batch_of_extra_scores,
        )

        # Determine which competition function to use
        has_ga_functions = (self.mutation_fn is not None) and (self.scoring_fn is not None)

        # Compute competition fitness
        # Use Python if for checking function availability (not traced)
        # but use jax.lax.cond for generation counter to avoid computing both branches
        if has_ga_functions:
            if key is None:
                raise ValueError("Random key required for Competition-GA")
            
            # Define functions for jax.lax.cond
            def compute_standard(_):
                return _novelty_and_dominated_novelty(
                    fitness=candidates_fitnesses[:, 0],
                    descriptor=candidates_descriptors,
                    novelty_k=self.k,
                    dominated_novelty_k=self.k,
                )[1]
            
            def compute_ga(_):
                return _competition_ga(
                    genotypes=candidates_genotypes,
                    fitness=candidates_fitnesses[:, 0],
                    descriptor=candidates_descriptors,
                    dominated_novelty_k=self.k,
                    mutation_fn=self.mutation_fn,
                    scoring_fn=self.scoring_fn,
                    num_children=self.num_ga_children,
                    num_generations=self.num_ga_generations,
                    key=key,
                )
            
            # Use jax.lax.cond to conditionally compute only one branch
            use_ga_competition = (generation_counter % self.g_n) == 0
            dominated_novelty = jax.lax.cond(
                use_ga_competition,
                compute_ga,
                compute_standard,
                None,
            )
        else:
            # Use standard Competition (dominated novelty)
            _, dominated_novelty = _novelty_and_dominated_novelty(
                fitness=candidates_fitnesses[:, 0],
                descriptor=candidates_descriptors,
                novelty_k=self.k,
                dominated_novelty_k=self.k,
            )

        # Use dominated novelty as meta-fitness
        valid = candidates_fitnesses[:, 0] != -jnp.inf
        meta_fitness = jnp.where(valid, dominated_novelty, -jnp.inf)

        # Select survivors
        indices = jnp.argsort(meta_fitness)[::-1]
        survivor_indices = indices[: self.size]

        new_genotypes = jax.tree.map(
            lambda x: x[survivor_indices], candidates_genotypes
        )
        new_fitnesses = candidates_fitnesses[survivor_indices]
        new_descriptors = candidates_descriptors[survivor_indices]
        new_extra_scores = jax.tree.map(
            lambda x: x[survivor_indices], candidates_extra_scores,
        )

        return self.replace(  # type: ignore
            genotypes=new_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            extra_scores=new_extra_scores,
        )

    @classmethod
    def init(  # type: ignore
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        population_size: int,
        k: int,
        g_n: int = 1,
        num_ga_children: int = 2,
        num_ga_generations: int = 1,
        mutation_fn: Optional[Callable] = None,
        scoring_fn: Optional[Callable] = None,
        extra_scores: Optional[ExtraScores] = None,
        keys_extra_scores: Tuple[str, ...] = (),
        key: Optional[RNGKey] = None,
    ) -> DominatedNoveltyGARepertoire:
        """Initialize the DNS-GA repertoire and add the first batch.

        Args:
            genotypes: first batch of genotypes (batch_size, ...)
            fitnesses: fitnesses of shape (batch_size, fitness_dim)
            descriptors: descriptors of shape (batch_size, num_descriptors)
            population_size: maximum number of individuals kept
            k: number of neighbors for novelty metrics
            g_n: generation frequency for Competition-GA (1 = every generation,
                 float('inf') = never, reduces to standard DNS)
            num_ga_children: offspring per solution in micro-GA
            num_ga_generations: forecast horizon (depth of GA tree)
            mutation_fn: function to mutate genotypes in Competition-GA
            scoring_fn: function to evaluate genotypes in Competition-GA
            extra_scores: extra scores of the first batch
            keys_extra_scores: keys of extra scores to store
            key: random key for Competition-GA

        Returns:
            Initialized DNS-GA repertoire
        """

        if extra_scores is None:
            extra_scores = {}

        # retrieve one genotype and one extra score prototype
        first_genotype = jax.tree.map(lambda x: x[0], genotypes)
        first_extra_scores = jax.tree.map(lambda x: x[0], extra_scores)

        # create a repertoire with default values
        repertoire = cls.init_default(
            genotype=first_genotype,
            descriptor_dim=descriptors.shape[-1],
            population_size=population_size,
            one_extra_score=first_extra_scores,
            keys_extra_scores=keys_extra_scores,
            k=k,
            g_n=g_n,
            num_ga_children=num_ga_children,
            num_ga_generations=num_ga_generations,
            mutation_fn=mutation_fn,
            scoring_fn=scoring_fn,
        )

        # add initial population to the repertoire
        return repertoire.add(  # type: ignore
            genotypes, descriptors, fitnesses, extra_scores, key
        )

    @classmethod
    def init_default(
        cls,
        genotype: Genotype,
        descriptor_dim: int,
        population_size: int,
        one_extra_score: Optional[ExtraScores] = None,
        keys_extra_scores: Tuple[str, ...] = (),
        k: int = 15,
        g_n: int = 1,
        num_ga_children: int = 2,
        num_ga_generations: int = 1,
        mutation_fn: Optional[Callable] = None,
        scoring_fn: Optional[Callable] = None,
    ) -> DominatedNoveltyGARepertoire:
        """Create a DNS-GA repertoire with default values.

        Args:
            genotype: a representative genotype PyTree
            descriptor_dim: number of descriptor dimensions
            population_size: maximum number of individuals kept
            one_extra_score: a representative extra score PyTree
            keys_extra_scores: keys of extra scores to store
            k: number of neighbors for novelty metrics
            g_n: generation frequency for Competition-GA
            num_ga_children: offspring per solution in micro-GA
            num_ga_generations: forecast horizon
            mutation_fn: mutation function for Competition-GA
            scoring_fn: scoring function for Competition-GA

        Returns:
            A repertoire filled with default values
        """
        if one_extra_score is None:
            one_extra_score = {}

        one_extra_score = {
            key: value
            for key, value in one_extra_score.items()
            if key in keys_extra_scores
        }

        # default fitness is -inf
        default_fitnesses = -jnp.inf * jnp.ones(shape=(population_size, 1))

        # default genotypes is all zeros
        default_genotypes = jax.tree.map(
            lambda x: jnp.zeros(shape=(population_size,) + x.shape, dtype=x.dtype),
            genotype,
        )

        # default descriptors is NaN (uninitialized)
        default_descriptors = jnp.full(
            shape=(population_size, descriptor_dim), fill_value=jnp.nan
        )

        # default extra scores buffers
        default_extra_scores = jax.tree.map(
            lambda x: jnp.zeros(shape=(population_size,) + x.shape, dtype=x.dtype),
            one_extra_score,
        )

        return cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            extra_scores=default_extra_scores,
            keys_extra_scores=keys_extra_scores,
            k=k,
            g_n=g_n,
            num_ga_children=num_ga_children,
            num_ga_generations=num_ga_generations,
            mutation_fn=mutation_fn,
            scoring_fn=scoring_fn,
        )
