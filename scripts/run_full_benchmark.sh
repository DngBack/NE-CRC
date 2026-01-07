#!/bin/bash
# Full benchmark runner script

set -e

echo "========================================="
echo "S-UniCR Full Benchmark"
echo "========================================="
echo ""

# Activate virtual environment
source .venv/bin/activate

# Define datasets and shift types
DATASETS=("unknown" "false_premise" "outdated" "underspecified" "multi_hop")
SHIFT_TYPES=("id" "mild" "strong")

echo "Running benchmark on ${#DATASETS[@]} datasets with ${#SHIFT_TYPES[@]} shift types..."
echo ""

# Counter for total experiments
TOTAL_EXPERIMENTS=$((${#DATASETS[@]} * ${#SHIFT_TYPES[@]}))
CURRENT=0

# Run experiments for each combination
for DATASET in "${DATASETS[@]}"; do
    for SHIFT in "${SHIFT_TYPES[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        echo "----------------------------------------"
        echo "Experiment $CURRENT/$TOTAL_EXPERIMENTS"
        echo "Dataset: $DATASET, Shift: $SHIFT"
        echo "----------------------------------------"
        
        # Create experiment config
        EXP_NAME="benchmark_${DATASET}_${SHIFT}"
        CONFIG_FILE="experiments/temp_${EXP_NAME}.yaml"
        
        # Generate config from template
        cat > "$CONFIG_FILE" << EOF
dataset_names:
  - $DATASET

shift_type: $SHIFT
split_ratios: [0.4, 0.3, 0.3]

model_name: "meta-llama/Llama-3.2-1B"
num_samples: 10

calibration_head: mlp
alpha: 0.05
delta: 0.05

uncertainty_method: semantic_entropy
weight_method: similarity

experiment_name: "$EXP_NAME"
seed: 42
cache_dir: ".cache"
output_dir: "outputs"
EOF
        
        # Run experiment
        python scripts/run_experiment.py --config "$CONFIG_FILE"
        
        # Get experiment directory
        EXPERIMENT_DIR="outputs/$EXP_NAME"
        
        # Generate visualizations
        echo "Generating visualizations..."
        python scripts/generate_figures.py --results "$EXPERIMENT_DIR"
        python scripts/generate_tables.py --results "$EXPERIMENT_DIR"
        
        # Clean up temp config
        rm "$CONFIG_FILE"
        
        echo "Completed: $EXP_NAME"
        echo ""
    done
done

echo "========================================="
echo "Full benchmark completed!"
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Results directory: outputs/"
echo "========================================="

# Generate aggregate report
echo ""
echo "Generating aggregate report..."
python scripts/aggregate_results.py --results-dir outputs/

