# Future Research Directions for DNS-GA

## Meta-Learning and Adaptive Forecasting

### Core Idea
Instead of using fixed Competition-GA parameters (g_n, num_generations, mutation rates), make these parameters **adaptive** based on the current state of evolution.

### Inspiration
From DNS paper: *"the competition mechanism could be represented as a neural network and trained to maximize population diversity and performance"*

**Extension**: Learn not just the competition function, but the **forecasting strategy itself** - when and how to apply Competition-GA.

### Three Approaches to Explore

#### 1. Persistent GA State Across Iterations
**Current**: Competition-GA restarts fresh every g_n iterations
**Proposed**: Maintain GA state across invocations, building "institutional knowledge"

**Implementation Ideas:**
- Save GA population state between runs
- Implement "staleness detection" to reset when population changes too much
- Use exponential decay: blend old GA state with fresh population

**Potential Benefits:**
- Transfer learning between GA runs
- Reduced overhead from rediscovery
- Natural adaptation to evolution phases

**Challenges:**
- When to reset vs continue?
- How to detect when GA state is obsolete?
- Balancing exploration (fresh start) vs exploitation (continue)

#### 2. Neural Network Meta-Controller
**Goal**: Learn optimal Competition-GA parameters from population state

**NN Architecture:**
```
Input: [current_qd_score, qd_score_trend, coverage, iteration, 
        fitness_variance, descriptor_space_spread, ...]
        ↓
Hidden layers (learn evolution phase patterns)
        ↓
Output: [predicted_g_n, predicted_num_generations, predicted_iso_sigma]
```

**Training Approach:**
- Reinforcement Learning: state=population metrics, action=GA params, reward=QD improvement
- Supervised Learning: train on results from current experiments (which configs worked when)
- Online Learning: update after each iteration based on QD improvement

**Key Questions:**
- What features best capture evolution phase?
- How to handle exploration-exploitation in meta-level?
- Does learned strategy transfer across tasks?

#### 3. Evolutionary Meta-Optimization
**Concept**: Evolve the GA hyperparameters themselves using an outer evolution loop

**Implementation:**
- Meta-population: Each individual is a set of hyperparameters
- Fitness: Run full DNS-GA experiment, measure final QD score
- Evolution: Standard GA operators on hyperparameter space

**Advantages:**
- No need to design meta-learning architecture
- Natural handling of discrete/continuous parameters
- Parallelizable across multiple runs

**Disadvantages:**
- Extremely expensive (N full experiments per generation)
- May overfit to specific task
- Requires many evolution runs to converge

### Phase-Specific Strategy Discovery

**Hypothesis**: Optimal forecasting strategy varies by evolution phase

**Early Phase (0-500 iters):**
- Rapid exploration, building initial diversity
- Hypothesis: Frequent shallow GA (g_n=100, gen=1)
- Goal: Sample many possible directions quickly

**Middle Phase (500-1500 iters):**
- Balancing exploration and exploitation
- Hypothesis: Moderate deep GA (g_n=500, gen=3)
- Goal: Strategic forecasting of promising regions

**Late Phase (1500+ iters):**
- Fine-tuning, local optimization
- Hypothesis: Rare deep GA (g_n=1500, gen=5) OR no GA
- Goal: Precision refinement without waste

**Current Experiments Test This:**
- `run_adaptive_experiments.py` has rule-based transitions
- Could extend with learned transitions
- Could make truly continuous (parameters change every iteration)

### Population-State Triggers

**Stagnation Detection:**
```python
if qd_score_unchanged_for(100 iterations):
    increase_ga_depth()  # Need deeper forecasting
    decrease_g_n()       # Need more frequent checks
```

**Rapid Improvement Detection:**
```python
if qd_score_delta > threshold:
    decrease_ga_depth()  # Don't waste compute on deep forecasting
    maintain_g_n()       # Current strategy working
```

**Diversity Crisis Detection:**
```python
if coverage_dropped_by(10%):
    reset_ga_with_high_mutation()
    force_ga_immediately()
```

### Connection to Current Work

**Step 1 (Current):** Test fixed parameter combinations
- Determine if Competition-GA helps at all
- Identify which parameter ranges are promising
- Build intuition about phase-dependent performance

**Step 2 (Next):** Analyze patterns in results
- Which configs dominate in early/middle/late phases?
- Are there clear transitions points?
- What population metrics correlate with success?

**Step 3 (Rule-Based Adaptive):** Implement simple adaptive rules
- Hand-crafted thresholds based on Step 2 insights
- E.g., "if QD hasn't improved in 100 iters, switch to deeper GA"
- Low overhead, interpretable, easy to debug

**Step 4 (Learned Adaptive):** Meta-learning approach
- Train NN or evolve meta-parameters
- Discover non-obvious patterns
- Potential for superhuman performance

### Biological Inspiration

Natural evolution exhibits **evolvability** - the capacity to evolve itself:
- Mutation rates themselves evolve
- Genetic architecture (modularity, etc.) evolves
- Developmental programs evolve

Analogously:
- Competition-GA parameters could evolve
- Forecasting strategy could adapt
- Meta-optimization of optimization

### Practical Considerations

**Sample Efficiency:**
- Need many runs to learn meta-strategy
- Each run is expensive (hours)
- Solution: Transfer learning across related tasks?

**Overfitting:**
- Strategy learned on walker2d may not generalize
- Need diverse task suite for robust learning
- Or: Learn task-agnostic features

**Computational Cost:**
- Meta-learning adds overhead
- Must justify with performance gains
- Consider: Is 10% better QD worth 2x compute?

**Simplicity Bias:**
- Fixed low-overhead configs might be "good enough"
- E.g., g_n=500, gen=1 with 2% overhead
- Adaptive strategy must significantly outperform

### Research Questions to Answer

1. **Does phase matter?** Do optimal parameters change over evolution?
2. **What features predict success?** Which population metrics are informative?
3. **Is it learnable?** Can NN/GA discover better strategies than manual tuning?
4. **Does it transfer?** Do learned strategies generalize across tasks?
5. **Is it worth it?** Does performance gain justify complexity?

### Concrete Next Steps (After Current Experiments)

1. ✅ Analyze current experiment results by phase
2. ✅ Plot parameter performance vs iteration number
3. ✅ Identify correlation between population state and config success
4. 🔄 Implement simple rule-based adaptive strategy
5. 🔄 Compare rule-based to best fixed config
6. 🚀 If promising: Design meta-learning experiment
7. 🚀 If very promising: This becomes PhD thesis topic

### Related Work to Review

- **Population-Based Training (PBT)**: Adapts hyperparameters during training
- **AutoML**: Learning learning algorithms
- **Meta-evolution**: Evolution of evolutionary algorithms
- **Adaptive operator selection**: Dynamically choosing genetic operators
- **Online algorithm configuration**: Real-time parameter tuning

### Potential Publications

1. **Short paper**: "Adaptive Forecasting Frequency in Quality-Diversity"
   - Rule-based adaptive strategies
   - Comparison to fixed configs
   - Phase analysis

2. **Full paper**: "Meta-Learning Competition Strategies for Quality-Diversity"
   - NN-based meta-controller
   - Transfer across tasks
   - Theoretical analysis

3. **Major paper**: "Evolvable Quality-Diversity: Meta-Optimization of Evolutionary Forecasting"
   - Comprehensive meta-learning framework
   - Multiple approaches compared
   - New theoretical insights

---

## Notes from November 14, 2025

**Context**: Running experiments to validate if Competition-GA (evolutionary forecasting) helps DNS convergence.

**Key Insight**: If current experiments show phase-dependent performance, that's strong evidence for adaptive approaches.

**Question to revisit**: Can the forecasting mechanism maintain state across iterations rather than restarting? Would persistent GA population improve over time?

**Meta-learning vision**: Learn the competition strategy itself - when to forecast, how deep to go, what parameters to use. This aligns with paper's suggestion but goes further.

**Biological analogy**: Natural evolution's mutation rates evolve. Similarly, Competition-GA's parameters could evolve/adapt during the run.

**Priority**: HIGH - but only after current experiments show signal to learn from. Don't optimize in a vacuum.
