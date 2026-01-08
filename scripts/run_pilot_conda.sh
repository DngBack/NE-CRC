#!/bin/bash
# Pilot experiment runner script (using conda)
# Note: Assumes conda environment is already activated

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (parent of scripts/)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root directory
cd "$PROJECT_ROOT"

echo "========================================="
echo "S-UniCR Pilot Experiment (conda version)"
echo "========================================="
echo "Working directory: $PROJECT_ROOT"
echo ""

# Verify Python is available
if ! command -v python &> /dev/null; then
    echo "Error: python not found."
    echo "Please activate your conda environment first:"
    echo "  conda activate your_env_name"
    exit 1
fi

# Show current environment info
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
fi
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Run pilot experiment
echo "[1/3] Running pilot experiment..."
python scripts/run_experiment.py \
    --config experiments/config_templates/pilot.yaml \
    --verbose

# Get the latest experiment directory
EXPERIMENT_DIR=$(ls -td outputs/pilot_* 2>/dev/null | head -1)

if [ -z "$EXPERIMENT_DIR" ]; then
    echo "Error: No experiment directory found"
    exit 1
fi

echo ""
echo "[2/3] Generating figures..."
python scripts/generate_figures.py \
    --results "$EXPERIMENT_DIR"

echo ""
echo "[3/3] Generating tables..."
python scripts/generate_tables.py \
    --results "$EXPERIMENT_DIR"

echo ""
echo "========================================="
echo "Pilot experiment completed!"
echo "Results: $EXPERIMENT_DIR"
echo "Figures: $EXPERIMENT_DIR/figures/"
echo "Tables: $EXPERIMENT_DIR/tables/"
echo "========================================="

