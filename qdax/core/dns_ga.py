"""Dominated Novelty Search with Competition-GA (DNS-GA) algorithm.

This module implements DNS-GA, which extends the standard Dominated Novelty Search
algorithm with a novel Competition-GA function. DNS-GA alternates between:
- Standard Competition: uses dominated novelty based on current fitness
- Competition-GA: uses dominated novelty based on forecasted fitness from micro-GA

This enables more informed culling decisions that preserve solutions with
high-performing offspring potential, leading to improved convergence with
comparable or superior QD-scores.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import jax

from qdax.core.containers.dns_repertoire_ga import DominatedNoveltyGARepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import (
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)


class DominatedNoveltySearchGA:
    """Dominated Novelty Search with Competition-GA.

    DNS-GA maintains a flat population without tessellation and selects survivors
    using either standard dominated novelty or GA-forecasting dominated novelty,
    alternating based on generation frequency.

    Args:
        scoring_function: a function that takes a batch of genotypes and computes
            their fitnesses, descriptors and optional extra scores.
        emitter: an emitter used to propose offspring and update its internal state.
        metrics_function: a function that takes a DNS-GA repertoire and computes
            metrics to track the evolution.
        population_size: maximum number of individuals maintained.
        k: number of nearest neighbors used to compute novelty/dominated novelty.
        g_n: generation frequency for Competition-GA (1 = every generation,
             float('inf') = standard DNS without GA competition).
        num_ga_children: number of offspring per solution in micro-GA.
        num_ga_generations: forecast horizon (depth of GA tree).
        mutation_fn: function to mutate genotypes in Competition-GA.
        repertoire_init: optional custom initialization function for the repertoire.
    """

    def __init__(
        self,
        scoring_function: Optional[
            Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores]]
        ],
        emitter: Emitter,
        metrics_function: Callable[[DominatedNoveltyGARepertoire], Metrics],
        population_size: int,
        k: int,
        g_n: int = 1,
        num_ga_children: int = 2,
        num_ga_generations: int = 1,
        mutation_fn: Optional[Callable[[Genotype, RNGKey], Genotype]] = None,
        repertoire_init: Callable[
            [
                Genotype,
                Fitness,
                Descriptor,
                int,
                int,
                int,
                int,
                int,
                Optional[Callable],
                Optional[Callable],
                Optional[ExtraScores],
                Tuple[str, ...],
                Optional[RNGKey],
            ],
            DominatedNoveltyGARepertoire,
        ] = DominatedNoveltyGARepertoire.init,
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._population_size = population_size
        self._k = k
        self._g_n = g_n
        self._num_ga_children = num_ga_children
        self._num_ga_generations = num_ga_generations
        self._mutation_fn = mutation_fn

        # Wrapper to bind parameters
        def _repertoire_init(
            genotypes: Genotype,
            fitnesses: Fitness,
            descriptors: Descriptor,
            _population_size: int,
            _k: int,
            _g_n: int,
            _num_ga_children: int,
            _num_ga_generations: int,
            _mutation_fn: Optional[Callable],
            _scoring_fn: Optional[Callable],
            extra_scores: Optional[ExtraScores] = None,
            keys_extra_scores: Tuple[str, ...] = (),
            key: Optional[RNGKey] = None,
        ) -> DominatedNoveltyGARepertoire:
            del _population_size, _k, _g_n, _num_ga_children, _num_ga_generations
            del _mutation_fn, _scoring_fn
            return repertoire_init(
                genotypes=genotypes,
                fitnesses=fitnesses,
                descriptors=descriptors,
                population_size=self._population_size,
                k=self._k,
                g_n=self._g_n,
                num_ga_children=self._num_ga_children,
                num_ga_generations=self._num_ga_generations,
                mutation_fn=self._mutation_fn,
                scoring_fn=self._scoring_function,
                extra_scores=extra_scores,
                keys_extra_scores=keys_extra_scores,
                key=key,
            )

        self._repertoire_init = _repertoire_init

    def init(
        self,
        genotypes: Genotype,
        key: RNGKey,
    ) -> Tuple[DominatedNoveltyGARepertoire, Optional[EmitterState], Metrics]:
        """
        Initialize a DNS-GA repertoire with an initial population of genotypes.

        Args:
            genotypes: initial genotypes, pytree (batch_size, ...)
            key: a random key used for stochastic operations.

        Returns:
            An initialized DNS-GA repertoire with the initial state of the emitter
            and the initial metrics.
        """
        if self._scoring_function is None:
            raise ValueError("Scoring function is not set.")

        # score initial genotypes
        key, subkey = jax.random.split(key)
        (fitnesses, descriptors, extra_scores) = self._scoring_function(
            genotypes, subkey
        )

        key, subkey = jax.random.split(key)
        repertoire, emitter_state, metrics = self.init_ask_tell(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            key=subkey,
            extra_scores=extra_scores,
        )
        return repertoire, emitter_state, metrics

    def init_ask_tell(
        self,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        key: RNGKey,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Tuple[DominatedNoveltyGARepertoire, Optional[EmitterState], Metrics]:
        """Initialize a DNS-GA repertoire with evaluated initial genotypes."""
        if extra_scores is None:
            extra_scores = {}

        # init the repertoire
        key, subkey = jax.random.split(key)
        repertoire = self._repertoire_init(
            genotypes,
            fitnesses,
            descriptors,
            self._population_size,
            self._k,
            self._g_n,
            self._num_ga_children,
            self._num_ga_generations,
            self._mutation_fn,
            self._scoring_function,
            extra_scores,
            key=subkey,
        )

        # get initial state of the emitter
        key, subkey = jax.random.split(key)
        emitter_state = self._emitter.init(
            key=subkey,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # calculate the initial metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics

    def update(
        self,
        repertoire: DominatedNoveltyGARepertoire,
        emitter_state: Optional[EmitterState],
        key: RNGKey,
        generation_counter: int = 0,
    ) -> Tuple[DominatedNoveltyGARepertoire, Optional[EmitterState], Metrics]:
        """
        Performs one iteration of DNS-GA:
        1. Ask the emitter for offsprings based on the current repertoire.
        2. Score offsprings to obtain fitnesses and descriptors.
        3. Add them to the repertoire using Competition or Competition-GA.
        4. Update the emitter state and compute metrics.
        """
        if self._scoring_function is None:
            raise ValueError("Scoring function is not set.")

        # generate offsprings with the emitter
        key, subkey = jax.random.split(key)
        genotypes, extra_info = self.ask(repertoire, emitter_state, subkey)

        # score the offsprings
        key, subkey = jax.random.split(key)
        (fitnesses, descriptors, extra_scores) = self._scoring_function(
            genotypes, subkey
        )

        key, subkey = jax.random.split(key)
        repertoire, emitter_state, metrics = self.tell(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            repertoire=repertoire,
            emitter_state=emitter_state,
            extra_scores=extra_scores,
            extra_info=extra_info,
            key=subkey,
            generation_counter=generation_counter,
        )
        return repertoire, emitter_state, metrics

    def scan_update(
        self,
        carry: Tuple[DominatedNoveltyGARepertoire, Optional[EmitterState], RNGKey, int],
        _: Any,
    ) -> Tuple[
        Tuple[DominatedNoveltyGARepertoire, Optional[EmitterState], RNGKey, int], Metrics
    ]:
        """scan-compatible wrapper around update."""
        repertoire, emitter_state, key, generation_counter = carry
        key, subkey = jax.random.split(key)
        (
            repertoire,
            emitter_state,
            metrics,
        ) = self.update(
            repertoire,
            emitter_state,
            subkey,
            generation_counter,
        )

        return (repertoire, emitter_state, key, generation_counter + 1), metrics

    def ask(
        self,
        repertoire: DominatedNoveltyGARepertoire,
        emitter_state: Optional[EmitterState],
        key: RNGKey,
    ) -> Tuple[Genotype, ExtraScores]:
        """Ask the emitter to generate a new batch of genotypes."""
        key, subkey = jax.random.split(key)
        genotypes, extra_info = self._emitter.emit(repertoire, emitter_state, subkey)
        return genotypes, extra_info

    def tell(
        self,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        repertoire: DominatedNoveltyGARepertoire,
        emitter_state: Optional[EmitterState],
        extra_scores: Optional[ExtraScores] = None,
        extra_info: Optional[ExtraScores] = None,
        key: Optional[RNGKey] = None,
        generation_counter: int = 0,
    ) -> Tuple[DominatedNoveltyGARepertoire, Optional[EmitterState], Metrics]:
        """Add new genotypes to the repertoire and update the emitter state."""
        if extra_scores is None:
            extra_scores = {}
        if extra_info is None:
            extra_info = {}

        # add genotypes in the repertoire (pass key for Competition-GA)
        repertoire = repertoire.add(
            genotypes, descriptors, fitnesses, extra_scores, key, generation_counter
        )

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores={**extra_scores, **extra_info},
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics
