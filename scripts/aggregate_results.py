#!/usr/bin/env python3
"""Aggregate results across multiple experiments."""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
from loguru import logger


def load_all_metrics(results_dir: Path):
    """Load metrics from all experiment subdirectories."""
    all_metrics = {}
    
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        metrics_file = exp_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        all_metrics[exp_dir.name] = metrics
    
    return all_metrics


def aggregate_by_dataset(all_metrics):
    """Aggregate metrics by dataset."""
    by_dataset = defaultdict(lambda: defaultdict(list))
    
    for exp_name, metrics in all_metrics.items():
        # Parse experiment name: benchmark_<dataset>_<shift>
        parts = exp_name.split('_')
        if len(parts) < 3:
            continue
        
        dataset = parts[1]
        
        for system, system_metrics in metrics.items():
            for metric_name, value in system_metrics.items():
                by_dataset[dataset][f"{system}_{metric_name}"].append(value)
    
    # Compute means
    aggregated = {}
    for dataset, metrics in by_dataset.items():
        aggregated[dataset] = {
            metric: sum(values) / len(values)
            for metric, values in metrics.items()
        }
    
    return aggregated


def aggregate_by_shift(all_metrics):
    """Aggregate metrics by shift type."""
    by_shift = defaultdict(lambda: defaultdict(list))
    
    for exp_name, metrics in all_metrics.items():
        parts = exp_name.split('_')
        if len(parts) < 3:
            continue
        
        shift = parts[2]
        
        for system, system_metrics in metrics.items():
            for metric_name, value in system_metrics.items():
                by_shift[shift][f"{system}_{metric_name}"].append(value)
    
    # Compute means
    aggregated = {}
    for shift, metrics in by_shift.items():
        aggregated[shift] = {
            metric: sum(values) / len(values)
            for metric, values in metrics.items()
        }
    
    return aggregated


def generate_aggregate_report(results_dir: Path):
    """Generate aggregate report."""
    logger.info(f"Loading results from {results_dir}")
    
    all_metrics = load_all_metrics(results_dir)
    logger.info(f"Found {len(all_metrics)} experiments")
    
    if not all_metrics:
        logger.warning("No experiment results found")
        return
    
    # Aggregate by dataset
    by_dataset = aggregate_by_dataset(all_metrics)
    
    # Aggregate by shift
    by_shift = aggregate_by_shift(all_metrics)
    
    # Create DataFrames
    df_dataset = pd.DataFrame(by_dataset).T
    df_shift = pd.DataFrame(by_shift).T
    
    # Save CSV files
    output_dir = results_dir / "aggregate"
    output_dir.mkdir(exist_ok=True)
    
    df_dataset.to_csv(output_dir / "by_dataset.csv")
    df_shift.to_csv(output_dir / "by_shift.csv")
    
    logger.info(f"Aggregate results saved to {output_dir}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS SUMMARY")
    print("=" * 80)
    
    print("\nBy Shift Type:")
    print(df_shift.to_string())
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing experiment results"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    generate_aggregate_report(results_dir)


if __name__ == "__main__":
    main()

