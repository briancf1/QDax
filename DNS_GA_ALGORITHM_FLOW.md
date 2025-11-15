# DNS-GA Algorithm Flow

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DNS-GA Main Loop                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. REPRODUCTION                                                │
│     ┌──────────────────┐                                        │
│     │ Emitter          │ → Generate B offspring from population │
│     │ (Isoline, etc.)  │                                        │
│     └──────────────────┘                                        │
│                                                                 │
│  2. CONCATENATION                                               │
│     ┌──────────────────────────────────────┐                   │
│     │ Population (N) + Offspring (B)       │ → N+B candidates  │
│     └──────────────────────────────────────┘                   │
│                                                                 │
│  3. EVALUATION                                                  │
│     ┌──────────────────────────────────────┐                   │
│     │ Scoring Function                     │ → Fitness & BDs   │
│     └──────────────────────────────────────┘                   │
│                                                                 │
│  4. COMPETITION (Alternating)                ┌─────────────┐   │
│                                              │ Gen counter │   │
│     ┌─────────────────────────────────┐     └─────────────┘   │
│     │ If gen % g_n == 0:              │                        │
│     │   → COMPETITION-GA              │  ← New!               │
│     │ Else:                           │                        │
│     │   → Standard Competition        │  ← Original DNS       │
│     └─────────────────────────────────┘                        │
│                          ↓                                      │
│                Competition Fitness (f̃)                          │
│                                                                 │
│  5. SELECTION                                                   │
│     ┌──────────────────────────────────────┐                   │
│     │ Keep top-N by competition fitness    │ → New population  │
│     └──────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Standard Competition (Original DNS)

```
For each solution i:
┌─────────────────────────────────────────────────────────────┐
│ 1. Find fitter solutions: D_i = {j | f_j > f_i}            │
│                                                             │
│ 2. Calculate distances in BD space: d_ij = ||d_i - d_j||   │
│                                                             │
│ 3. Find k-nearest-fitter: K_i ⊂ D_i                        │
│                                                             │
│ 4. Dominated novelty: f̃_i = (1/k) Σ d_ij for j ∈ K_i      │
│                                                             │
│    High f̃_i = Far from better solutions = Keep             │
│    Low f̃_i = Close to better solutions = Discard           │
└─────────────────────────────────────────────────────────────┘
```

## Competition-GA (New)

```
For each solution i:
┌──────────────────────────────────────────────────────────────────┐
│ 1. EVOLUTIONARY FORECASTING (Micro-GA)                           │
│    ┌────────────────────────────────────────────────────┐        │
│    │ Gen 0:  [Parent_i] (fitness = f_i)                 │        │
│    │           ↓ mutate (num_ga_children times)         │        │
│    │ Gen 1:  [Child_1, Child_2, ...] → evaluate         │        │
│    │           ↓ mutate each child                      │        │
│    │ Gen 2:  [GrandChild_1, GrandChild_2, ...] → eval   │        │
│    │           ... (repeat for num_ga_generations)      │        │
│    │                                                     │        │
│    │ Future fitness: f'_i = max(f_i, max_offspring_fit) │        │
│    └────────────────────────────────────────────────────┘        │
│                                                                   │
│ 2. FUTURE-BASED DOMINATED NOVELTY                                │
│    ┌────────────────────────────────────────────────────┐        │
│    │ Find fitter solutions using FUTURE fitness:        │        │
│    │ D'_i = {j | f'_j > f'_i}                           │        │
│    │                                                     │        │
│    │ Calculate distances in BD space: d_ij              │        │
│    │                                                     │        │
│    │ Find k-nearest-fitter by future fitness            │        │
│    │                                                     │        │
│    │ Competition fitness: f̃_i = (1/k) Σ d_ij           │        │
│    └────────────────────────────────────────────────────┘        │
│                                                                   │
│ Key insight: Solutions with high-performing offspring are        │
│              valued even if current fitness is lower             │
└──────────────────────────────────────────────────────────────────┘
```

## Micro-GA Tree Structure

```
                    Parent (f=5.0)
                         │
         ┌───────────────┴───────────────┐
         │                               │
    Child 1 (f=6.0)                 Child 2 (f=4.5)
         │                               │
    ┌────┴────┐                     ┌────┴────┐
    │         │                     │         │
GC_1      GC_2                   GC_3      GC_4
(f=7.0)   (f=5.5)               (f=4.0)   (f=6.5)

Future fitness = max(5.0, 6.0, 4.5, 7.0, 5.5, 4.0, 6.5) = 7.0

Note: Unlike standard GA, children don't become parents
      (reduces computational growth)
```

## Alternation Example (g_n = 3)

```
Generation:  0   1   2   3   4   5   6   7   8   9
            ─┴───┴───┴───┴───┴───┴───┴───┴───┴───┴─
Competition: GA  Std Std GA  Std Std GA  Std Std GA
             ↑               ↑               ↑
             └── Competition-GA runs every 3 generations

GA  = Competition-GA (evolutionary forecasting, slower)
Std = Standard Competition (current fitness, faster)
```

## Computational Cost Comparison

### Standard DNS (per generation)
```
Population: N solutions
Offspring:  B evaluations
Total:      B evaluations
```

### DNS-GA with Competition-GA (generation when gen % g_n == 0)
```
Population: N solutions
Offspring:  B evaluations
Extra:      N × num_ga_children × num_ga_generations evaluations
            (for micro-GA forecasting)

Example:    N=1024, children=2, gens=1
Total:      B + 2048 evaluations
```

### DNS-GA Amortized Cost (over g_n generations)
```
If g_n = 5:
- 1 generation with Competition-GA: B + N × 2 × 1 evals
- 4 generations with Standard:     B evals each

Total over 5 gens:  5B + 2N
Avg per generation: B + 0.4N

Overhead: 0.4N / B ≈ 0.4 × 1024 / 100 ≈ 4x per generation
But amortized over g_n=5: ~0.8x overhead
```

## Decision Flow: When to Use What

```
                ┌─────────────────────┐
                │ Start DNS-GA        │
                └──────────┬──────────┘
                           │
                    ┌──────┴──────┐
                    │ Generation  │
                    │   Counter   │
                    └──────┬──────┘
                           │
                  ┌────────┴────────┐
                  │ gen % g_n == 0? │
                  └────────┬────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
           YES│                         │NO
              │                         │
    ┌─────────▼──────────┐    ┌────────▼─────────┐
    │ Competition-GA     │    │ Standard         │
    │ (Forecasting)      │    │ Competition      │
    └─────────┬──────────┘    └────────┬─────────┘
              │                         │
              │                         │
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │  Selection  │
                    │  (top-N)    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────────┐
                    │ Increment gen   │
                    │    counter      │
                    └─────────────────┘
```

## Key Innovation

```
┌─────────────────────────────────────────────────────────────┐
│ Standard DNS Problem:                                       │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ Solution A: Low fitness NOW, but has genetic        │    │
│ │             material for high-performing offspring  │    │
│ │             → Gets CULLED ✗                         │    │
│ └─────────────────────────────────────────────────────┘    │
│                                                             │
│ Competition-GA Solution:                                    │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ Solution A: Low fitness NOW, but forecasting shows  │    │
│ │             high-performing offspring potential     │    │
│ │             → Gets PRESERVED ✓                      │    │
│ └─────────────────────────────────────────────────────┘    │
│                                                             │
│ Result: Better convergence + preserved quality              │
└─────────────────────────────────────────────────────────────┘
```
