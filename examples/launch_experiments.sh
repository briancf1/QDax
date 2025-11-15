#!/bin/bash

# DNS-GA Complete Experiment Pipeline
# Runs main experiments, analyzes results, then runs adaptive experiments

set -e

echo "=========================================="
echo "DNS-GA Complete Experiment Pipeline"
echo "=========================================="
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "../.venv" ]; then
    echo "Error: Virtual environment not found at ../.venv"
    echo "Please create and activate the virtual environment first"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ../.venv/bin/activate

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import jax, qdax, pandas, matplotlib" 2>/dev/null || {
    echo "Error: Missing required packages"
    echo "Install with: pip install jax qdax pandas matplotlib seaborn"
    exit 1
}

# Check available disk space (need ~5GB for results)
echo "Checking disk space..."
AVAILABLE=$(df -h . | tail -1 | awk '{print $4}')
echo "Available disk space: $AVAILABLE"

# Estimate runtime
NUM_MAIN_EXPERIMENTS=26
NUM_ADAPTIVE_EXPERIMENTS=3
ITERATIONS=2000
EST_TIME_MAIN=3.5
EST_TIME_ADAPTIVE=0.5
EST_TIME_TOTAL=$(echo "$EST_TIME_MAIN + $EST_TIME_ADAPTIVE" | bc)

echo ""
echo "=========================================="
echo "Experiment Configuration"
echo "=========================================="
echo "Main experiments: $NUM_MAIN_EXPERIMENTS (~$EST_TIME_MAIN hours)"
echo "Adaptive experiments: $NUM_ADAPTIVE_EXPERIMENTS (~$EST_TIME_ADAPTIVE hours)"
echo "Iterations per experiment: $ITERATIONS"
echo "Estimated total time: ~$EST_TIME_TOTAL hours"
echo ""
echo "Pipeline:"
echo "  1. Run main parameter exploration experiments"
echo "  2. Analyze main experiment results"
echo "  3. Run adaptive frequency experiments"
echo "  4. Generate final summary"
echo ""

# Ask for confirmation
read -p "Start complete pipeline now? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

# Create log directory
LOG_DIR="experiment_logs"
mkdir -p "$LOG_DIR"

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG_FILE="$LOG_DIR/main_experiments_$TIMESTAMP.log"
ANALYSIS_LOG_FILE="$LOG_DIR/analysis_$TIMESTAMP.log"
ADAPTIVE_LOG_FILE="$LOG_DIR/adaptive_experiments_$TIMESTAMP.log"
PIPELINE_LOG_FILE="$LOG_DIR/pipeline_$TIMESTAMP.log"

echo ""
echo "=========================================="
echo "Starting Pipeline"
echo "=========================================="
echo "Pipeline log: $PIPELINE_LOG_FILE"
echo ""

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$PIPELINE_LOG_FILE"
}

# Function to check if process succeeded
check_success() {
    if [ $? -eq 0 ]; then
        log_message "✓ $1 completed successfully"
        return 0
    else
        log_message "✗ $1 failed"
        return 1
    fi
}

log_message "Starting DNS-GA complete experiment pipeline"
log_message "Main log: $MAIN_LOG_FILE"
log_message "Analysis log: $ANALYSIS_LOG_FILE"
log_message "Adaptive log: $ADAPTIVE_LOG_FILE"

# Stage 1: Run main experiments
log_message ""
log_message "=========================================="
log_message "STAGE 1: Main Parameter Exploration ($NUM_MAIN_EXPERIMENTS experiments)"
log_message "=========================================="
log_message "Estimated time: $EST_TIME_MAIN hours"
log_message "Log file: $MAIN_LOG_FILE"
log_message ""

python -u run_dns_ga_experiments.py > "$MAIN_LOG_FILE" 2>&1
check_success "Main experiments" || exit 1

# Extract output directory from main experiments
MAIN_OUTPUT_DIR=$(grep -o "dns_ga_experiments_[0-9_]*" "$MAIN_LOG_FILE" | head -1)
if [ -z "$MAIN_OUTPUT_DIR" ]; then
    log_message "✗ Could not find output directory in logs"
    exit 1
fi
log_message "Main experiments output: $MAIN_OUTPUT_DIR"

# Stage 2: Analyze main experiments
log_message ""
log_message "=========================================="
log_message "STAGE 2: Analyzing Main Experiment Results"
log_message "=========================================="
log_message "Log file: $ANALYSIS_LOG_FILE"
log_message ""

python -u analyze_dns_ga_experiments.py "$MAIN_OUTPUT_DIR" > "$ANALYSIS_LOG_FILE" 2>&1
check_success "Main experiment analysis" || exit 1

log_message "Analysis plots saved to: $MAIN_OUTPUT_DIR/"

# Stage 3: Run adaptive experiments
log_message ""
log_message "=========================================="
log_message "STAGE 3: Adaptive Frequency Experiments ($NUM_ADAPTIVE_EXPERIMENTS experiments)"
log_message "=========================================="
log_message "Estimated time: $EST_TIME_ADAPTIVE hours"
log_message "Log file: $ADAPTIVE_LOG_FILE"
log_message ""

python -u run_adaptive_experiments.py > "$ADAPTIVE_LOG_FILE" 2>&1
check_success "Adaptive experiments" || exit 1

# Extract adaptive output directory
ADAPTIVE_OUTPUT_DIR=$(grep -o "dns_ga_adaptive_[0-9_]*" "$ADAPTIVE_LOG_FILE" | head -1)
if [ -z "$ADAPTIVE_OUTPUT_DIR" ]; then
    log_message "✗ Could not find adaptive output directory in logs"
    exit 1
fi
log_message "Adaptive experiments output: $ADAPTIVE_OUTPUT_DIR"

# Stage 4: Generate final summary
log_message ""
log_message "=========================================="
log_message "STAGE 4: Generating Final Summary"
log_message "=========================================="

SUMMARY_FILE="experiment_summary_$TIMESTAMP.txt"

{
    echo "=========================================="
    echo "DNS-GA Complete Experiment Summary"
    echo "=========================================="
    echo "Date: $(date)"
    echo ""
    echo "Main Experiments: $MAIN_OUTPUT_DIR"
    echo "Adaptive Experiments: $ADAPTIVE_OUTPUT_DIR"
    echo ""
    echo "=========================================="
    echo "Main Experiment Results (Top 5)"
    echo "=========================================="
    # Extract top 5 from main experiments
    if [ -f "$MAIN_OUTPUT_DIR/experiment_summary.json" ]; then
        python -c "
import json
with open('$MAIN_OUTPUT_DIR/experiment_summary.json') as f:
    data = json.load(f)
    results = data.get('results', [])
    sorted_results = sorted(results, key=lambda x: x['final_qd_score'], reverse=True)[:5]
    for i, r in enumerate(sorted_results, 1):
        print(f\"{i}. {r['name']:<40} QD: {r['final_qd_score']:.2f} | MaxFit: {r['final_max_fitness']:.2f}\")
"
    fi
    echo ""
    echo "=========================================="
    echo "Adaptive Experiment Results"
    echo "=========================================="
    # Extract adaptive results
    if [ -f "$ADAPTIVE_OUTPUT_DIR/adaptive_results.json" ]; then
        python -c "
import json
with open('$ADAPTIVE_OUTPUT_DIR/adaptive_results.json') as f:
    data = json.load(f)
    results = data.get('results', [])
    for r in results:
        print(f\"{r['name']:<40} QD: {r['final_qd_score']:.2f} | MaxFit: {r['final_max_fitness']:.2f}\")
"
    fi
    echo ""
    echo "=========================================="
    echo "Key Findings"
    echo "=========================================="
    echo "See detailed analysis in:"
    echo "  - $MAIN_OUTPUT_DIR/convergence_comparison.png"
    echo "  - $MAIN_OUTPUT_DIR/efficiency_analysis.png"
    echo "  - $MAIN_OUTPUT_DIR/configuration_ranking.csv"
    echo ""
    echo "Logs:"
    echo "  - Main experiments: $MAIN_LOG_FILE"
    echo "  - Analysis: $ANALYSIS_LOG_FILE"
    echo "  - Adaptive experiments: $ADAPTIVE_LOG_FILE"
    echo "  - Pipeline: $PIPELINE_LOG_FILE"
    echo ""
} > "$SUMMARY_FILE"

log_message "Summary saved to: $SUMMARY_FILE"

# Display summary
cat "$SUMMARY_FILE"

log_message ""
log_message "=========================================="
log_message "PIPELINE COMPLETE"
log_message "=========================================="
log_message "Total runtime: $SECONDS seconds ($(($SECONDS / 3600))h $(($SECONDS % 3600 / 60))m)"
log_message ""
log_message "Next steps:"
log_message "  1. Review summary: cat $SUMMARY_FILE"
log_message "  2. Check plots: open $MAIN_OUTPUT_DIR/*.png"
log_message "  3. Review detailed results: cat $MAIN_OUTPUT_DIR/experiment_summary.json"
log_message ""

echo ""
echo "✓ All experiments and analysis complete!"
echo ""
echo "Summary: $SUMMARY_FILE"
echo ""
