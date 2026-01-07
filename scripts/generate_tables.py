"""Generate LaTeX tables from experiment results."""

import argparse
import pickle
import json
from pathlib import Path
from typing import Dict
from loguru import logger


def load_results(results_dir: Path) -> Dict:
    """Load experiment results."""
    with open(results_dir / "metrics.pkl", 'rb') as f:
        metrics = pickle.load(f)
    
    return metrics


def format_number(value, decimal_places: int = 3) -> str:
    """Format number for LaTeX."""
    if isinstance(value, bool):
        return "\\checkmark" if value else "\\times"
    elif isinstance(value, (int, float)):
        return f"{value:.{decimal_places}f}"
    else:
        return str(value)


def generate_main_results_table(metrics: Dict, output_path: Path):
    """Generate main results table (Table 1)."""
    logger.info("Generating main results table...")
    
    systems = list(metrics.keys())
    
    # Metrics to include
    metric_keys = [
        ('coverage', 'Coverage', 3),
        ('selective_risk', 'Selective Risk', 3),
        ('aurc', 'AURC', 4),
        ('coverage@risk<=0.05', 'Cov@Risk≤5%', 3),
        ('risk_gap', 'Risk Gap', 4),
    ]
    
    # Start LaTeX table
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Main Results: Coverage, Risk, and AURC}")
    lines.append("\\label{tab:main_results}")
    
    # Table header
    num_cols = len(metric_keys) + 1
    lines.append(f"\\begin{{tabular}}{{l{'c' * len(metric_keys)}}}")
    lines.append("\\toprule")
    
    header = "System & " + " & ".join([name for _, name, _ in metric_keys]) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    
    # Table rows
    for system in systems:
        row_values = [system]
        for key, _, decimals in metric_keys:
            value = metrics[system].get(key, 0.0)
            row_values.append(format_number(value, decimals))
        
        row = " & ".join(row_values) + " \\\\"
        lines.append(row)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    # Write to file
    latex_content = "\n".join(lines)
    
    with open(output_path / "table_main_results.tex", 'w') as f:
        f.write(latex_content)
    
    logger.info(f"Saved main results table to {output_path}")
    
    return latex_content


def generate_coverage_at_risk_table(metrics: Dict, output_path: Path):
    """Generate coverage at different risk levels table (Table 2)."""
    logger.info("Generating coverage at risk table...")
    
    systems = list(metrics.keys())
    risk_levels = [0.01, 0.02, 0.05, 0.10]
    
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Coverage at Different Risk Levels}")
    lines.append("\\label{tab:coverage_at_risk}")
    
    # Table header
    lines.append(f"\\begin{{tabular}}{{l{'c' * len(risk_levels)}}}")
    lines.append("\\toprule")
    
    header = "System & " + " & ".join([f"Risk≤{r:.0%}" for r in risk_levels]) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    
    # Table rows
    for system in systems:
        row_values = [system]
        for risk in risk_levels:
            key = f'coverage@risk<={risk:.2f}'
            value = metrics[system].get(key, 0.0)
            row_values.append(format_number(value, 3))
        
        row = " & ".join(row_values) + " \\\\"
        lines.append(row)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    latex_content = "\n".join(lines)
    
    with open(output_path / "table_coverage_at_risk.tex", 'w') as f:
        f.write(latex_content)
    
    logger.info(f"Saved coverage at risk table to {output_path}")
    
    return latex_content


def generate_ablation_table(metrics: Dict, output_path: Path):
    """Generate ablation study table (Table 3)."""
    logger.info("Generating ablation table...")
    
    # Focus on systems relevant for ablation
    ablation_systems = {
        'unicr': 'UniCR (baseline)',
        'unicr_filter': '+ Filter',
        'unicr_necrc': '+ NE-CRC',
        's_unicr': '+ Both (S-UniCR)',
    }
    
    metric_keys = [
        ('coverage', 'Coverage', 3),
        ('selective_risk', 'Risk', 3),
        ('aurc', 'AURC', 4),
    ]
    
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation Study: Component Contributions}")
    lines.append("\\label{tab:ablation}")
    
    lines.append(f"\\begin{{tabular}}{{l{'c' * len(metric_keys)}}}")
    lines.append("\\toprule")
    
    header = "Configuration & " + " & ".join([name for _, name, _ in metric_keys]) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    
    for system_key, system_name in ablation_systems.items():
        if system_key not in metrics:
            continue
        
        row_values = [system_name]
        for key, _, decimals in metric_keys:
            value = metrics[system_key].get(key, 0.0)
            row_values.append(format_number(value, decimals))
        
        row = " & ".join(row_values) + " \\\\"
        lines.append(row)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    latex_content = "\n".join(lines)
    
    with open(output_path / "table_ablation.tex", 'w') as f:
        f.write(latex_content)
    
    logger.info(f"Saved ablation table to {output_path}")
    
    return latex_content


def generate_all_tables(results_dir: Path, output_dir: Path):
    """Generate all tables."""
    logger.info(f"Loading results from {results_dir}")
    metrics = load_results(results_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all tables
    generate_main_results_table(metrics, output_dir)
    generate_coverage_at_risk_table(metrics, output_dir)
    generate_ablation_table(metrics, output_dir)
    
    # Also save as Markdown for easy viewing
    generate_markdown_summary(metrics, output_dir)
    
    logger.info(f"All tables saved to {output_dir}")


def generate_markdown_summary(metrics: Dict, output_dir: Path):
    """Generate Markdown summary table."""
    logger.info("Generating Markdown summary...")
    
    systems = list(metrics.keys())
    
    lines = []
    lines.append("# Experiment Results Summary\n")
    lines.append("## Main Metrics\n")
    
    # Create Markdown table
    header = "| System | Coverage | Sel. Risk | AURC | Cov@5% | Risk Gap |"
    separator = "|--------|----------|-----------|------|--------|----------|"
    
    lines.append(header)
    lines.append(separator)
    
    for system in systems:
        m = metrics[system]
        row = (
            f"| {system} "
            f"| {m.get('coverage', 0):.3f} "
            f"| {m.get('selective_risk', 0):.3f} "
            f"| {m.get('aurc', 0):.4f} "
            f"| {m.get('coverage@risk<=0.05', 0):.3f} "
            f"| {m.get('risk_gap', 0):.4f} |"
        )
        lines.append(row)
    
    markdown_content = "\n".join(lines)
    
    with open(output_dir / "results_summary.md", 'w') as f:
        f.write(markdown_content)
    
    logger.info(f"Saved Markdown summary to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate tables from experiment results")
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
        help="Output directory for tables (default: results_dir/tables)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    output_dir = Path(args.output) if args.output else results_dir / "tables"
    
    generate_all_tables(results_dir, output_dir)


if __name__ == "__main__":
    main()

