"""
Analyze DNS-GA experiment results and generate visualizations
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_experiment_results(experiment_dir):
    """Load all experiment results from directory."""
    # Load summary
    with open(os.path.join(experiment_dir, 'results_summary.json'), 'r') as f:
        summary = json.load(f)
    
    # Load all log files
    results = []
    for result in summary['results']:
        if 'error' in result:
            continue
        
        df = pd.read_csv(result['log_file'])
        df['experiment'] = result['name']
        df['config'] = json.dumps(result['config'])
        results.append(df)
    
    all_data = pd.concat(results, ignore_index=True)
    return summary, all_data


def analyze_convergence(all_data, baseline_name='DNS_baseline'):
    """Analyze convergence speed and efficiency."""
    baseline = all_data[all_data['experiment'] == baseline_name]
    if len(baseline) == 0:
        print(f"Warning: Baseline {baseline_name} not found")
        return None
    
    baseline_final_qd = baseline['qd_score'].iloc[-1]
    baseline_final_iter = baseline['iteration'].iloc[-1]
    
    print(f"\nBaseline DNS:")
    print(f"  Final QD Score: {baseline_final_qd:.2f}")
    print(f"  Final Iteration: {int(baseline_final_iter)}")
    
    convergence_results = []
    
    experiments = all_data['experiment'].unique()
    for exp_name in experiments:
        if exp_name == baseline_name:
            continue
        
        exp_data = all_data[all_data['experiment'] == exp_name]
        final_qd = exp_data['qd_score'].iloc[-1]
        
        # Find when this experiment reaches baseline QD
        converged = exp_data[exp_data['qd_score'] >= baseline_final_qd]
        
        if len(converged) > 0:
            conv_iter = int(converged.iloc[0]['iteration'])
            conv_qd = converged.iloc[0]['qd_score']
            speedup = baseline_final_iter / conv_iter
            
            convergence_results.append({
                'experiment': exp_name,
                'final_qd': final_qd,
                'qd_improvement': final_qd - baseline_final_qd,
                'convergence_iter': conv_iter,
                'convergence_qd': conv_qd,
                'iteration_speedup': speedup,
                'converged': True,
            })
        else:
            convergence_results.append({
                'experiment': exp_name,
                'final_qd': final_qd,
                'qd_improvement': final_qd - baseline_final_qd,
                'convergence_iter': None,
                'convergence_qd': None,
                'iteration_speedup': None,
                'converged': False,
            })
    
    return pd.DataFrame(convergence_results).sort_values('iteration_speedup', ascending=False)


def plot_convergence_curves(all_data, output_dir):
    """Plot QD score convergence over iterations."""
    plt.figure(figsize=(14, 8))
    
    experiments = all_data['experiment'].unique()
    baseline_name = 'DNS_baseline'
    
    # Plot baseline prominently
    if baseline_name in experiments:
        baseline_data = all_data[all_data['experiment'] == baseline_name]
        plt.plot(baseline_data['iteration'], baseline_data['qd_score'], 
                linewidth=3, label='DNS (Baseline)', color='black', linestyle='--')
    
    # Plot all other experiments
    colors = sns.color_palette('husl', len(experiments) - 1)
    color_idx = 0
    
    for exp_name in experiments:
        if exp_name == baseline_name:
            continue
        
        exp_data = all_data[all_data['experiment'] == exp_name]
        
        # Determine line style based on configuration
        if '_g500_' in exp_name:
            linestyle = '-'
            alpha = 0.8
        elif '_g250_' in exp_name:
            linestyle = '-.'
            alpha = 0.7
        elif '_gen2_' in exp_name:
            linestyle = ':'
            alpha = 0.6
        else:
            linestyle = '-'
            alpha = 0.5
        
        plt.plot(exp_data['iteration'], exp_data['qd_score'],
                label=exp_name.replace('DNS-GA_', ''), 
                color=colors[color_idx], linestyle=linestyle, alpha=alpha)
        color_idx += 1
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('QD Score', fontsize=12)
    plt.title('DNS-GA Convergence Comparison', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'convergence_curves.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved convergence curves to {output_file}")
    plt.close()


def plot_final_comparison(summary, output_dir):
    """Plot bar chart comparing final QD scores."""
    results = [r for r in summary['results'] if 'error' not in r]
    
    df = pd.DataFrame([{
        'Experiment': r['name'].replace('DNS-GA_', '').replace('DNS_baseline', 'DNS'),
        'QD Score': r['final_qd_score'],
        'Max Fitness': r['final_max_fitness'],
        'Coverage': r['final_coverage'],
    } for r in results])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # QD Score
    df_sorted = df.sort_values('QD Score', ascending=False)
    colors = ['red' if x == 'DNS' else 'steelblue' for x in df_sorted['Experiment']]
    axes[0].barh(df_sorted['Experiment'], df_sorted['QD Score'], color=colors)
    axes[0].set_xlabel('QD Score', fontsize=12)
    axes[0].set_title('Final QD Score Comparison', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Max Fitness
    df_sorted = df.sort_values('Max Fitness', ascending=False)
    colors = ['red' if x == 'DNS' else 'steelblue' for x in df_sorted['Experiment']]
    axes[1].barh(df_sorted['Experiment'], df_sorted['Max Fitness'], color=colors)
    axes[1].set_xlabel('Max Fitness', fontsize=12)
    axes[1].set_title('Final Max Fitness Comparison', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Coverage
    df_sorted = df.sort_values('Coverage', ascending=False)
    colors = ['red' if x == 'DNS' else 'steelblue' for x in df_sorted['Experiment']]
    axes[2].barh(df_sorted['Experiment'], df_sorted['Coverage'], color=colors)
    axes[2].set_xlabel('Coverage (%)', fontsize=12)
    axes[2].set_title('Final Coverage Comparison', fontsize=12, fontweight='bold')
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'final_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved final comparison to {output_file}")
    plt.close()


def plot_efficiency_analysis(convergence_df, output_dir):
    """Plot efficiency analysis (speedup vs overhead)."""
    if convergence_df is None or len(convergence_df) == 0:
        return
    
    # Extract configuration parameters from experiment names
    def extract_g_n(name):
        if '_g' in name:
            return int(name.split('_g')[1].split('_')[0])
        return None
    
    def extract_gens(name):
        if '_gen' in name:
            return int(name.split('_gen')[1].split('_')[0])
        return None
    
    convergence_df['g_n'] = convergence_df['experiment'].apply(extract_g_n)
    convergence_df['num_ga_generations'] = convergence_df['experiment'].apply(extract_gens)
    
    converged = convergence_df[convergence_df['converged']]
    
    if len(converged) == 0:
        print("No experiments converged to baseline")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Color by num_ga_generations
    colors = {1: 'green', 2: 'orange', 3: 'red'}
    
    for gens in converged['num_ga_generations'].unique():
        if pd.isna(gens):
            continue
        subset = converged[converged['num_ga_generations'] == gens]
        plt.scatter(subset['g_n'], subset['iteration_speedup'], 
                   s=200, alpha=0.7, color=colors.get(int(gens), 'gray'),
                   label=f'{int(gens)} generation(s)', edgecolors='black', linewidth=1.5)
    
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Break-even (1.0x)')
    plt.xlabel('g_n (Competition-GA Frequency)', fontsize=12)
    plt.ylabel('Iteration Speedup (x)', fontsize=12)
    plt.title('DNS-GA Efficiency: Convergence Speedup vs GA Frequency', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Annotate points with experiment names
    for _, row in converged.iterrows():
        if not pd.isna(row['g_n']):
            label = row['experiment'].replace('DNS-GA_', '').split('_iso')[0]
            plt.annotate(label, (row['g_n'], row['iteration_speedup']),
                        fontsize=7, alpha=0.7, 
                        xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'efficiency_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved efficiency analysis to {output_file}")
    plt.close()


def generate_report(experiment_dir):
    """Generate comprehensive analysis report."""
    print(f"\n{'='*80}")
    print("DNS-GA EXPERIMENT ANALYSIS")
    print(f"{'='*80}\n")
    
    # Load data
    summary, all_data = load_experiment_results(experiment_dir)
    
    print(f"Experiment directory: {experiment_dir}")
    print(f"Total experiments: {len(summary['results'])}")
    print(f"Total time: {summary['total_time_hours']:.2f} hours")
    
    # Convergence analysis
    print(f"\n{'='*80}")
    print("CONVERGENCE ANALYSIS")
    print(f"{'='*80}")
    
    convergence_df = analyze_convergence(all_data)
    
    if convergence_df is not None:
        print("\nSuccessfully converged experiments (sorted by speedup):")
        converged = convergence_df[convergence_df['converged']]
        if len(converged) > 0:
            print(converged.to_string())
            
            # Save to CSV
            conv_csv = os.path.join(experiment_dir, 'convergence_analysis.csv')
            convergence_df.to_csv(conv_csv, index=False)
            print(f"\nSaved convergence analysis to {conv_csv}")
        else:
            print("No experiments converged to baseline QD score")
    
    # Generate plots
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    plot_convergence_curves(all_data, experiment_dir)
    plot_final_comparison(summary, experiment_dir)
    
    if convergence_df is not None:
        plot_efficiency_analysis(convergence_df, experiment_dir)
    
    # Best configurations
    print(f"\n{'='*80}")
    print("TOP CONFIGURATIONS")
    print(f"{'='*80}")
    
    results = [r for r in summary['results'] if 'error' not in r]
    results_sorted = sorted(results, key=lambda x: x['final_qd_score'], reverse=True)
    
    print("\nTop 5 by Final QD Score:")
    for i, r in enumerate(results_sorted[:5], 1):
        print(f"{i}. {r['name']}")
        print(f"   QD Score: {r['final_qd_score']:.2f}")
        print(f"   Max Fitness: {r['final_max_fitness']:.2f}")
        print(f"   Coverage: {r['final_coverage']:.2f}%")
        print(f"   Total Evaluations: {r['total_evals']:,}")
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze DNS-GA experiment results')
    parser.add_argument('experiment_dir', help='Directory containing experiment results')
    args = parser.parse_args()
    
    if not os.path.exists(args.experiment_dir):
        print(f"Error: Directory {args.experiment_dir} does not exist")
        return
    
    generate_report(args.experiment_dir)


if __name__ == "__main__":
    main()
