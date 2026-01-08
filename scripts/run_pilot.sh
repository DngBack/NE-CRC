#!/bin/bash
# Pilot experiment runner script

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the project root directory (parent of scripts/)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root directory
cd "$PROJECT_ROOT"

echo "========================================="
echo "S-UniCR Pilot Experiment"
echo "========================================="
echo "Working directory: $PROJECT_ROOT"
echo ""

# Initialize USE_UV flag
USE_UV=false

# Activate virtual environment (if available)
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "Using existing virtual environment: $VIRTUAL_ENV"
elif command -v uv &> /dev/null; then
    echo "No .venv found, using uv to run commands..."
    USE_UV=true
else
    echo "Warning: No virtual environment found and uv is not available."
    echo "Please run 'uv sync' first or activate a virtual environment."
    exit 1
fi

# Run pilot experiment
echo "[1/3] Running pilot experiment..."
if [ "$USE_UV" = true ]; then
    uv run python scripts/run_experiment.py \
        --config experiments/config_templates/pilot.yaml \
        --verbose
else
    python scripts/run_experiment.py \
        --config experiments/config_templates/pilot.yaml \
        --verbose
fi

# Get the latest experiment directory
EXPERIMENT_DIR=$(ls -td outputs/pilot_* 2>/dev/null | head -1)

if [ -z "$EXPERIMENT_DIR" ]; then
    echo "Error: No experiment directory found"
    exit 1
fi

echo ""
echo "[2/3] Generating figures..."
if [ "$USE_UV" = true ]; then
    uv run python scripts/generate_figures.py \
        --results "$EXPERIMENT_DIR"
else
    python scripts/generate_figures.py \
        --results "$EXPERIMENT_DIR"
fi

echo ""
echo "[3/3] Generating tables..."
if [ "$USE_UV" = true ]; then
    uv run python scripts/generate_tables.py \
        --results "$EXPERIMENT_DIR"
else
    python scripts/generate_tables.py \
        --results "$EXPERIMENT_DIR"
fi

echo ""
echo "========================================="
echo "Pilot experiment completed!"
echo "Results: $EXPERIMENT_DIR"
echo "Figures: $EXPERIMENT_DIR/figures/"
echo "Tables: $EXPERIMENT_DIR/tables/"
echo "========================================="

