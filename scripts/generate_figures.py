"""Generate figures from experiment results."""

import argparse
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from loguru import logger

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def load_results(results_dir: Path):
    """Load experiment results."""
    with open(results_dir / "metrics.pkl", 'rb') as f:
        metrics = pickle.load(f)
    
    return metrics


def plot_rc_curves(metrics: dict, output_path: Path):
    """Generate Risk-Coverage curves (Figure 1)."""
    logger.info("Generating RC curves...")
    
    plt.figure(figsize=(10, 6))
    
    colors = {
        'heuristic': 'gray',
        'unicr': 'blue',
        'unicr_filter': 'green',
        'unicr_necrc': 'orange',
        's_unicr': 'red',
    }
    
    labels = {
        'heuristic': 'Heuristic',
        'unicr': 'UniCR',
        'unicr_filter': 'UniCR+Filter',
        'unicr_necrc': 'UniCR+NE-CRC',
        's_unicr': 'S-UniCR (Proposed)',
    }
    
    for system, system_metrics in metrics.items():
        if '_rc_curve' not in system_metrics:
            continue
        
        curve_data = system_metrics['_rc_curve']
        coverages = curve_data['coverages']
        risks = curve_data['risks']
        
        plt.plot(
            coverages, risks,
            label=f"{labels.get(system, system)} (AURC={system_metrics['aurc']:.4f})",
            color=colors.get(system, 'black'),
            linewidth=2,
        )
    
    plt.xlabel('Coverage', fontsize=14)
    plt.ylabel('Selective Risk', fontsize=14)
    plt.title('Risk-Coverage Curves', fontsize=16)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path / 'rc_curves.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'rc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved RC curves to {output_path}")


def plot_coverage_at_risk(metrics: dict, output_path: Path):
    """Plot coverage at different risk levels (Figure 2 component)."""
    logger.info("Generating coverage at risk plot...")
    
    risk_levels = [0.01, 0.02, 0.05, 0.10]
    systems = list(metrics.keys())
    
    coverages = {risk: [] for risk in risk_levels}
    
    for system in systems:
        for risk in risk_levels:
            key = f'coverage@risk<={risk:.2f}'
            cov = metrics[system].get(key, 0.0)
            coverages[risk].append(cov)
    
    # Create grouped bar plot
    x = np.arange(len(systems))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, risk in enumerate(risk_levels):
        offset = width * (i - 1.5)
        ax.bar(x + offset, coverages[risk], width, label=f'Risk≤{risk:.0%}')
    
    ax.set_xlabel('System', fontsize=14)
    ax.set_ylabel('Coverage', fontsize=14)
    ax.set_title('Coverage at Different Risk Levels', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / 'coverage_at_risk.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'coverage_at_risk.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved coverage at risk plot to {output_path}")


def plot_risk_violation(metrics: dict, output_path: Path):
    """Plot risk violation metrics (Figure 2)."""
    logger.info("Generating risk violation plot...")
    
    systems = list(metrics.keys())
    risk_gaps = []
    violation_rates = []
    
    for system in systems:
        risk_gaps.append(metrics[system].get('risk_gap', 0.0))
        violation_rates.append(metrics[system].get('violation_rate_bootstrap', 0.0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Risk gap
    ax1.bar(range(len(systems)), risk_gaps, color='steelblue')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Target Risk')
    ax1.set_xlabel('System', fontsize=12)
    ax1.set_ylabel('Risk Gap (Actual - Target)', fontsize=12)
    ax1.set_title('Risk Gap by System', fontsize=14)
    ax1.set_xticks(range(len(systems)))
    ax1.set_xticklabels(systems, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Violation rate
    ax2.bar(range(len(systems)), violation_rates, color='coral')
    ax2.set_xlabel('System', fontsize=12)
    ax2.set_ylabel('Violation Rate (Bootstrap)', fontsize=12)
    ax2.set_title('Risk Violation Rate', fontsize=14)
    ax2.set_xticks(range(len(systems)))
    ax2.set_xticklabels(systems, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / 'risk_violation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'risk_violation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved risk violation plot to {output_path}")


def plot_cost_efficiency(metrics: dict, output_path: Path):
    """Plot cost-efficiency trade-off (Figure 4)."""
    logger.info("Generating cost-efficiency plot...")
    
    systems = list(metrics.keys())
    coverages = []
    risks = []
    efficiencies = []
    
    for system in systems:
        coverages.append(metrics[system].get('coverage', 0.0))
        risks.append(metrics[system].get('selective_risk', 0.0))
        efficiencies.append(metrics[system].get('efficiency_score', 0.0))
    
    # Scatter plot: coverage vs risk, size by efficiency
    plt.figure(figsize=(10, 7))
    
    scatter = plt.scatter(
        coverages, risks,
        s=[e * 1000 for e in efficiencies],  # Scale for visibility
        alpha=0.6,
        c=range(len(systems)),
        cmap='viridis'
    )
    
    # Annotate points
    for i, system in enumerate(systems):
        plt.annotate(
            system,
            (coverages[i], risks[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
    
    plt.xlabel('Coverage', fontsize=14)
    plt.ylabel('Selective Risk', fontsize=14)
    plt.title('Cost-Risk-Coverage Frontier\n(Marker size = Efficiency)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='System Index')
    
    plt.tight_layout()
    plt.savefig(output_path / 'cost_efficiency.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'cost_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved cost-efficiency plot to {output_path}")


def generate_all_figures(results_dir: Path, output_dir: Path):
    """Generate all figures."""
    logger.info(f"Loading results from {results_dir}")
    metrics = load_results(results_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all figures
    plot_rc_curves(metrics, output_dir)
    plot_coverage_at_risk(metrics, output_dir)
    plot_risk_violation(metrics, output_dir)
    
    if 'efficiency_score' in metrics[list(metrics.keys())[0]]:
        plot_cost_efficiency(metrics, output_dir)
    
    logger.info(f"All figures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate figures from experiment results")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to experiment results directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for figures (default: results_dir/figures)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    output_dir = Path(args.output) if args.output else results_dir / "figures"
    
    generate_all_figures(results_dir, output_dir)


if __name__ == "__main__":
    main()

