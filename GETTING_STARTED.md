# Getting Started with S-UniCR

This guide will help you get started with the Shift-Aware UniCR (S-UniCR) framework.

## Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU (recommended for open-source LLMs)
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# 1. Clone the repository (if not already done)
cd /home/admin1/Desktop/NE-CRC

# 2. Install dependencies with uv
uv sync

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Quick Start: Run a Pilot Experiment

The fastest way to see the system in action:

```bash
# Run pilot experiment on one dataset
./scripts/run_pilot.sh
```

This will:
1. Load the "unknown" dataset from AbstentionBench
2. Run LLM inference with a small model (cached after first run)
3. Train all 5 system variants
4. Evaluate comprehensive metrics
5. Generate figures and tables

**Expected runtime**: 10-30 minutes (depending on GPU)

## Understanding the Output

After running an experiment, you'll find:

```
outputs/pilot_id_unknown/
├── results.pkl              # Raw predictions
├── metrics.pkl              # All metrics
├── metrics.json             # Human-readable metrics
├── figures/
│   ├── rc_curves.pdf       # Risk-Coverage curves
│   ├── coverage_at_risk.pdf
│   ├── risk_violation.pdf
│   └── cost_efficiency.pdf
└── tables/
    ├── table_main_results.tex      # LaTeX table
    ├── table_coverage_at_risk.tex
    ├── table_ablation.tex
    └── results_summary.md          # Quick overview
```

**Start here**: Open `results_summary.md` for a quick overview of results.

## Key Concepts

### System Variants

The framework implements 5 systems:

1. **Heuristic**: Simple uncertainty threshold (no calibration)
2. **UniCR**: Standard CRC (baseline from paper)
3. **UniCR+Filter**: UniCR with SConU outlier detection
4. **UniCR+NE-CRC**: UniCR with non-exchangeable CRC
5. **S-UniCR**: Full proposed system (Filter + NE-CRC)

### Distribution Shift Types

- **ID** (In-Distribution): Random split, no shift
- **MILD**: Cross-topic or difficulty-based shift
- **STRONG**: Temporal + domain shift (most challenging)

### Key Metrics

- **Coverage**: Fraction of samples answered (higher = more answers)
- **Selective Risk**: Error rate on answered samples (lower = better)
- **AURC**: Area Under Risk-Coverage curve (lower = better)
- **Coverage@Risk≤ε**: Maximum coverage while keeping risk ≤ ε

## Running Custom Experiments

### 1. Create a Configuration File

```yaml
# experiments/my_config.yaml
dataset_names:
  - outdated  # Choose dataset

shift_type: strong  # id, mild, or strong
split_ratios: [0.4, 0.3, 0.3]

model_name: "meta-llama/Llama-3.2-1B"
num_samples: 10  # K for uncertainty estimation

calibration_head: mlp  # logistic or mlp
alpha: 0.05  # Target risk level
delta: 0.05  # Outlier threshold

uncertainty_method: semantic_entropy
weight_method: similarity

experiment_name: "my_experiment"
seed: 42
cache_dir: ".cache"
output_dir: "outputs"
```

### 2. Run the Experiment

```bash
python scripts/run_experiment.py --config experiments/my_config.yaml
```

### 3. Generate Visualizations

```bash
python scripts/generate_figures.py --results outputs/my_experiment/
python scripts/generate_tables.py --results outputs/my_experiment/
```

## Running the Full Benchmark

To reproduce full paper results:

```bash
# This will run 15 experiments (5 datasets × 3 shift types)
# Expected runtime: Several hours depending on hardware
./scripts/run_full_benchmark.sh
```

Results will be aggregated in `outputs/aggregate/`.

## Using the Framework in Your Code

### Basic Usage

```python
from src.data import create_default_loader, create_default_splitter, ShiftType
from src.variants import create_s_unicr
from src.conformal.weights import create_similarity_weights

# 1. Load data
loader = create_default_loader()
samples = loader.load_dataset("unknown", max_samples=1000)

# 2. Create shift split
splitter = create_default_splitter()
data_split = splitter.create_split(samples, ShiftType.STRONG)

# 3. [Run LLM inference - see full example in EXPERIMENTS.md]

# 4. Create S-UniCR system
weight_fn = create_similarity_weights()
s_unicr = create_s_unicr(
    alpha=0.05,  # 5% risk target
    delta=0.05,  # 5% outlier threshold
    weight_fn=weight_fn
)

# 5. Fit on train/calibration data
s_unicr.fit(
    train_features, train_labels,
    cal_features, cal_labels,
    cal_uncertainties, cal_weight_features
)

# 6. Predict on test data
predictions = s_unicr.predict(
    test_features, test_uncertainties, test_weight_features
)

# 7. Evaluate
from src.metrics import create_metrics_computer

metrics_computer = create_metrics_computer()
metrics = metrics_computer.compute_all(
    decisions=[r.is_answered() for r in predictions],
    confidences=[r.confidence for r in predictions],
    correctness=test_labels,
    alpha=0.05
)

print(metrics_computer.format_metrics_table(metrics))
```

## Common Tasks

### Change the LLM

Edit your config file:

```yaml
model_name: "meta-llama/Llama-3-8B"  # or any HF model
```

### Try Different Uncertainty Methods

```yaml
uncertainty_method: token_entropy  # or dispersion
```

### Change Weight Scheme for NE-CRC

```yaml
weight_method: density_ratio  # or kernel
```

### Adjust Risk/Outlier Thresholds

```yaml
alpha: 0.02  # 2% risk target (more strict)
delta: 0.10  # 10% outlier threshold (more permissive)
```

## Performance Optimization

### Speed Up Inference

1. **Use vLLM** (automatically detected if installed):
   ```bash
   uv add vllm
   ```

2. **Enable caching** (automatic):
   - First run generates and caches outputs
   - Subsequent runs with same config load from cache

3. **Use smaller models** for testing:
   ```yaml
   model_name: "meta-llama/Llama-3.2-1B"  # Fast
   ```

### Reduce Memory Usage

1. **Quantization** (edit `src/models/llm_wrapper.py`):
   ```python
   llm = TransformersInference(model_name, load_in_8bit=True)
   ```

2. **Reduce K** (number of samples):
   ```yaml
   num_samples: 5  # Lower = faster
   ```

## Troubleshooting

### "CUDA out of memory"
- Reduce `num_samples` in config
- Use smaller model
- Enable 8-bit quantization
- Clear GPU cache: `torch.cuda.empty_cache()`

### "Module not found"
```bash
# Reinstall dependencies
uv sync --reinstall

# Verify environment
source .venv/bin/activate
python -c "import src; print('OK')"
```

### Slow first run
- First run does LLM inference (cached afterward)
- Consider using smaller dataset for testing
- Check GPU is being used: `nvidia-smi`

## Next Steps

1. ✅ Run pilot experiment: `./scripts/run_pilot.sh`
2. 📊 Review results in `outputs/pilot_*/results_summary.md`
3. 🔬 Try different configurations
4. 📈 Run full benchmark for paper results
5. 📝 Use generated LaTeX tables in your paper

## Documentation

- [`README.md`](README.md) - Project overview
- [`EXPERIMENTS.md`](EXPERIMENTS.md) - Detailed experiment guide
- [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md) - Complete implementation details
- Individual module docstrings - See `src/*/` for API documentation

## Support

For issues or questions:
1. Check documentation above
2. Review example scripts in `scripts/`
3. Examine module `__main__` blocks for usage examples
4. Check logs in `.cache/` for debugging

Happy experimenting! 🚀

