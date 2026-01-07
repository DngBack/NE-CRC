#!/bin/bash
# Pilot experiment runner script

set -e  # Exit on error

echo "========================================="
echo "S-UniCR Pilot Experiment"
echo "========================================="
echo ""

# Activate virtual environment
source .venv/bin/activate

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

