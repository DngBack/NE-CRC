#!/usr/bin/env python3
"""Run experiment from configuration file."""

import argparse
import yaml
from pathlib import Path
import sys

# Add project root to path (not src, to allow relative imports within src)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import ExperimentConfig, ShiftType
from src.pipeline import create_experiment_pipeline
from loguru import logger


def load_config(config_path: Path) -> ExperimentConfig:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Convert shift_type string to enum
    shift_type_str = config_dict.get('shift_type', 'id')
    config_dict['shift_type'] = ShiftType[shift_type_str.upper()]
    
    # Create config object
    config = ExperimentConfig(**config_dict)
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Run S-UniCR experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration YAML file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # Load configuration
    config_path = Path(args.config)
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Create and run pipeline
    pipeline = create_experiment_pipeline(config)
    results, metrics = pipeline.run()
    
    logger.info("Experiment completed successfully!")
    logger.info(f"Results saved to: {pipeline.output_dir}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    for system_name, system_metrics in metrics.items():
        print(f"\n{system_name.upper()}:")
        print(f"  Coverage:        {system_metrics['coverage']:.3f}")
        print(f"  Selective Risk:  {system_metrics['selective_risk']:.3f}")
        print(f"  AURC:            {system_metrics['aurc']:.4f}")
        print(f"  Cov@Risk≤5%:     {system_metrics.get('coverage@risk<=0.05', 0):.3f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

