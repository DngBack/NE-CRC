# Running Experiments

This guide explains how to run experiments with the S-UniCR framework.

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### 2. Run Pilot Experiment

```bash
# Run pilot on single dataset (fastest, ~10-30 min depending on hardware)
./scripts/run_pilot.sh
```

This will:
- Run LLM inference on a small dataset
- Train all system variants
- Evaluate metrics
- Generate figures and tables

### 3. Run Full Benchmark

```bash
# Run full benchmark on all datasets and shift types (~several hours)
./scripts/run_full_benchmark.sh
```

## Manual Experiment Running

### Using Configuration Files

Create a YAML config file (or use templates in `experiments/config_templates/`):

```yaml
# my_experiment.yaml
dataset_names:
  - unknown

shift_type: strong  # id, mild, or strong
split_ratios: [0.4, 0.3, 0.3]

model_name: "meta-llama/Llama-3.2-1B"
num_samples: 10

calibration_head: mlp  # logistic or mlp
alpha: 0.05  # Risk level
delta: 0.05  # Outlier threshold

uncertainty_method: semantic_entropy
weight_method: similarity

experiment_name: "my_experiment"
seed: 42
cache_dir: ".cache"
output_dir: "outputs"
```

Run the experiment:

```bash
python scripts/run_experiment.py --config my_experiment.yaml --verbose
```

### Generate Visualizations

After running an experiment:

```bash
# Generate figures
python scripts/generate_figures.py --results outputs/my_experiment/

# Generate tables
python scripts/generate_tables.py --results outputs/my_experiment/
```

## Available Datasets

From AbstentionBench:
- `unknown` - Questions about unknown information
- `false_premise` - Questions with incorrect assumptions
- `outdated` - Questions about outdated information
- `underspecified` - Ambiguous queries
- `multi_hop` - Multi-hop reasoning questions

## Shift Types

- **ID** (`id`): In-distribution (random split, no shift)
- **MILD** (`mild`): Cross-topic/difficulty shift
- **STRONG** (`strong`): Temporal/domain shift (outdated + cross-domain)

## System Variants

All experiments automatically run these systems:
1. **Heuristic**: Direct uncertainty threshold
2. **UniCR**: Standard CRC (baseline)
3. **UniCR+Filter**: With SConU outlier filtering
4. **UniCR+NE-CRC**: With non-exchangeable CRC
5. **S-UniCR**: Full system (Filter + NE-CRC)

## Output Structure

```
outputs/
└── experiment_name/
    ├── results.pkl          # Raw prediction results
    ├── metrics.pkl          # All computed metrics
    ├── metrics.json         # Metrics in JSON format
    ├── figures/
    │   ├── rc_curves.pdf
    │   ├── coverage_at_risk.pdf
    │   ├── risk_violation.pdf
    │   └── cost_efficiency.pdf
    └── tables/
        ├── table_main_results.tex
        ├── table_coverage_at_risk.tex
        ├── table_ablation.tex
        └── results_summary.md
```

## Caching

LLM outputs are automatically cached in `.cache/` to avoid recomputation:
- Generations cached by model + dataset + config hash
- Features cached separately
- Reusing same config will load from cache

To clear cache:

```python
from src.pipeline import create_cache_manager

cache = create_cache_manager()
cache.clear_cache("all")  # or "generations", "features", "results"
```

## Performance Tips

1. **Use vLLM**: Automatically used if available (faster inference)
2. **Cache Aggressively**: LLM inference is the bottleneck
3. **Start Small**: Test with pilot experiment first
4. **GPU Recommended**: For reasonable runtime with open-source models
5. **Batch Size**: Increase in `llm_wrapper.py` if you have more VRAM

## Ablation Studies

To run ablations on uncertainty methods:

```bash
for method in semantic_entropy token_entropy dispersion; do
    # Create config with different uncertainty method
    python scripts/run_experiment.py --config experiments/ablation_${method}.yaml
done
```

To run ablations on weight schemes:

```bash
for weights in similarity density_ratio kernel; do
    # Create config with different weight method
    python scripts/run_experiment.py --config experiments/weights_${weights}.yaml
done
```

## Troubleshooting

### Out of Memory

- Reduce `num_samples` (K)
- Use smaller model
- Use quantization (8-bit or 4-bit)
- Reduce batch size in LLM wrapper

### Slow Inference

- Install vLLM: `uv add vllm`
- Use GPU acceleration
- Check cache is being used
- Consider smaller model for testing

### Missing Dependencies

```bash
# Reinstall all dependencies
uv sync --reinstall
```

## Next Steps

After running experiments:

1. Review results in `outputs/experiment_name/`
2. Check `results_summary.md` for quick overview
3. Use LaTeX tables in paper
4. Analyze RC curves and ablations
5. Run statistical significance tests if needed

For questions or issues, check `IMPLEMENTATION_STATUS.md` or open an issue.

