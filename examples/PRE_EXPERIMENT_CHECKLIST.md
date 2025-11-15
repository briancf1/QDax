# Pre-Experiment Checklist

Based on paper review (Dominated Novelty Search: Rethinking Local Competition in Quality-Diversity)

## ✅ Already Correct

1. **Baseline DNS Implementation** - Our DNS baseline uses the paper's algorithm
2. **Parameter k=3** - Matches paper for k-nearest-fitter solutions
3. **Environment walker2d_uni** - Matches paper's experiments
4. **Population size 1024** - Matches paper
5. **Isoline variation** - Main emitter uses paper's mutation operator

## 🔍 Clarifications

### What We're Actually Testing

- **Paper's Method**: Dominated Novelty Search (DNS) - fitness transformation based on distance to better solutions
- **Our Extension**: DNS + Competition-GA (evolutionary forecasting via mini-GA within iterations)
- **Goal**: Test if evolutionary forecasting accelerates DNS convergence

### Key Difference
- **Paper**: No mention of Competition-GA or multi-generation lookahead
- **Our Work**: Testing whether GA-based forecasting helps DNS find better solutions faster
- This is a **novel extension**, not directly from the paper

## 📋 Recommended Pre-Flight Tests

### Test 1: Validate DNS Baseline (2 minutes) ⭐ CRITICAL

```bash
cd /Users/briancf/Desktop/source/EvoAlgsAndSwarm/lib-qdax/QDax/examples
python validate_dns_baseline.py
```

**Purpose**: Verify baseline DNS achieves reasonable performance
**Expected Output**:
- QD Score: ~2000-5000 after 100 iterations
- Max Fitness: ~100-300 after 100 iterations  
- Coverage: ~10-30% after 100 iterations

**Action**: 
- ✅ If all checks pass → Proceed with main experiments
- ⚠️ If checks fail → Investigate DNS implementation

### Test 2: Competition-GA Mutation Strategy (7 minutes) - OPTIONAL

```bash
python test_mutation_strategy.py
```

**Purpose**: Test if isoline_variation (paper's mutation) works better in Competition-GA than simple Gaussian
**Current**: Competition-GA uses Gaussian noise
**Alternative**: Use isoline_variation (paper's approach)

**Action**:
- If isoline is >5% better → Update main experiments to use it
- If Gaussian is better or same → Keep current implementation

## 🎯 Experiment Design Validation

### Our Hypothesis (Not from Paper)
> Adding Competition-GA (evolutionary forecasting) to DNS will enable faster convergence to comparable QD scores, resulting in fewer total evaluations needed.

### Why This Might Work
1. **Lookahead**: GA can "test" evolutionary paths before committing
2. **Adaptive**: Can adjust GA frequency and depth based on results
3. **Compatible**: Works within DNS's competition framework

### Why This Might Fail
1. **Overhead**: Extra evaluations may outweigh benefits
2. **DNS Already Optimal**: Paper shows DNS is state-of-the-art
3. **Walker2d Specifics**: Environment may not benefit from forecasting

## 📊 Success Criteria

### Minimal Success
- At least 1 config (e.g., g500_gen1) achieves DNS final QD with <100% overhead
- Evidence that Competition-GA helps in some phase of evolution

### Strong Success  
- Multiple configs achieve DNS QD with <50% overhead
- Clear pattern: low overhead + shallow GA = win
- Adaptive strategies show benefit

### Publication-Worthy
- Competition-GA reaches higher final QD than DNS baseline
- OR reaches same QD in <80% of iterations
- Clear understanding of when/why it works

## 🚀 Launch Sequence

### Recommended Order

1. **Validate DNS Baseline** (2 min) - MUST DO
   ```bash
   python validate_dns_baseline.py
   ```

2. **Test Mutation Strategy** (7 min) - OPTIONAL but recommended
   ```bash
   python test_mutation_strategy.py
   ```

3. **Quick Full-Setup Test** (2 min) - Already created
   ```bash
   python validate_setup.py
   ```

4. **Launch Main Experiments** (3.2 hours)
   ```bash
   ./launch_experiments.sh
   ```

## 📖 Paper Key Findings (for Reference)

From "Dominated Novelty Search: Rethinking Local Competition in Quality-Diversity"

### DNS Competition Function
For each solution i:
1. Find all fitter solutions: D_i = {j | f_j > f_i}
2. Compute distances: d_ij = ||descriptor_i - descriptor_j||
3. Competition fitness: f̃_i = (1/k) × sum of k-nearest-fitter distances

### Performance (from paper)
- **Walker2d**: DNS significantly outperforms MAP-Elites
- **High-dimensional**: DNS scales better than grid methods
- **Unsupervised**: DNS works without predefined bounds

### Our Extension
We test whether adding evolutionary forecasting (Competition-GA) on top of DNS's competition mechanism provides additional benefits.

## ⚠️ Important Notes

1. **Competition-GA is NOT from the paper** - It's our novel extension
2. **DNS baseline should already be strong** - Paper shows it's state-of-the-art
3. **High bar for success** - Need to beat already-optimal DNS
4. **Valuable negative result** - Even if Competition-GA doesn't help, understanding why is useful

## 🔬 Alternative If DNS Baseline Looks Wrong

If `validate_dns_baseline.py` shows poor performance:

1. Check DNS implementation in `qdax/core/dns.py`
2. Verify competition fitness calculation matches paper (Section 4.2)
3. Compare against paper's Figure 4 (Walker results)
4. Consider testing with MAP-Elites baseline for comparison

## ✨ Ready to Launch?

- [ ] Ran `validate_dns_baseline.py` - passed
- [ ] (Optional) Ran `test_mutation_strategy.py` - reviewed results  
- [ ] Ran `validate_setup.py` - passed
- [ ] Reviewed this checklist
- [ ] Understanding: We're testing novel extension, not replicating paper
- [ ] Ready for 3.7 hour experiment run

**Launch command:**
```bash
./launch_experiments.sh
```

**Monitor progress:**
```bash
tail -f experiment_logs/pipeline_*.log
```
