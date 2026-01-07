# Shift-Aware UniCR (S-UniCR)

**Non-Exchangeable Conformal Risk Control for Universal Confidence under Distribution Shift**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

This repository implements **Shift-Aware UniCR (S-UniCR)**, a method for calibrated confidence and selective prediction under distribution shift. It extends Universal Conformal Risk Control (UniCR) to handle non-exchangeable data through:

1. **Non-Exchangeable CRC (NE-CRC)**: Test-specific adaptive thresholds via weighted conformal risk control
2. **SConU-style Outlier Filtering**: Conformal p-value detection of exchangeability violations  
3. **Multiple Uncertainty Measures**: Semantic entropy, token entropy, and dispersion
4. **Comprehensive Evaluation**: Full metrics suite on AbstentionBench with shift scenarios

## ✨ Key Features

- 🔧 **5 System Variants**: From heuristic baselines to full S-UniCR
- 📊 **Complete Metrics**: Coverage@Risk, AURC, RC curves, risk violation, cost tracking
- 🚀 **Production Ready**: Caching, logging, modular design, GPU acceleration
- 📈 **Full Pipeline**: Data → Inference → Calibration → Evaluation → Visualization
- 🎨 **Auto Visualization**: Publication-ready figures and LaTeX tables
- ⚡ **Fast Inference**: vLLM support with intelligent caching

## 🚀 Quick Start

```bash
# 1. Install dependencies (using uv)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate

# 2. Run pilot experiment (~10-30 min)
./scripts/run_pilot.sh

# 3. View results
cat outputs/pilot_*/results_summary.md
```

**That's it!** Results, figures, and tables are automatically generated in `outputs/`.

## 📋 Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone <your-repo-url>
cd NE-CRC

# Install all dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Verify installation
python -c "import torch; print('✓ Setup complete!')"
```

## Project Structure

```
NE-CRC/
├── src/
│   ├── data/              # AbstentionBench loaders & shift splits
│   ├── models/            # LLM inference & evidence extraction
│   ├── uncertainty/       # Semantic entropy, token entropy, dispersion
│   ├── calibration/       # UniCR calibration heads (logistic, MLP)
│   ├── conformal/         # CRC, NE-CRC, weight schemes
│   ├── filtering/         # SConU p-value filtering
│   ├── metrics/           # Coverage@Risk, AURC, RC curves
│   ├── pipeline/          # End-to-end experiment orchestration
│   ├── baselines/         # Baseline systems
│   └── variants/          # System variants (UniCR+filter, etc.)
├── experiments/           # Experiment configurations
├── scripts/               # CLI scripts for running experiments
├── outputs/               # Results, figures, tables
└── tests/                 # Unit tests
```

## 📖 Documentation

- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Installation and first steps
- **[EXPERIMENTS.md](EXPERIMENTS.md)** - Running experiments and configurations  
- **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** - Complete technical details

## 🧪 Running Experiments

### Pilot Experiment (Recommended First Step)

```bash
./scripts/run_pilot.sh
```

Runs a complete experiment on one dataset with all systems, generates all metrics, figures, and tables.

### Custom Experiment

```bash
# 1. Create config (or use template)
cp experiments/config_templates/pilot.yaml experiments/my_experiment.yaml

# 2. Edit configuration
nano experiments/my_experiment.yaml

# 3. Run
python scripts/run_experiment.py --config experiments/my_experiment.yaml

# 4. Generate visualizations
python scripts/generate_figures.py --results outputs/my_experiment/
python scripts/generate_tables.py --results outputs/my_experiment/
```

### Full Benchmark

```bash
# Runs 15 experiments (5 datasets × 3 shift types)
./scripts/run_full_benchmark.sh
```

## System Variants

The codebase implements 6 system configurations:

1. **Heuristic Threshold**: Direct threshold on uncertainty (no calibration)
2. **UniCR (baseline)**: Standard CRC with exchangeability assumption
3. **UniCR + SConU Filter**: Adds p-value outlier filtering
4. **UniCR + NE-CRC**: Non-exchangeable CRC without filtering
5. **S-UniCR (proposed)**: Full system with both NE-CRC and filtering
6. **Trivial baselines**: Always-answer / Always-abstain

## Key Features

### Uncertainty Measures
- **Semantic Entropy**: Cluster-based entropy over sampled generations
- **Token Entropy**: Average per-token entropy
- **Dispersion**: Variance in semantic embedding space

### Weight Schemes for NE-CRC
- **Similarity Weights**: Cosine similarity on embeddings
- **Density Ratio**: Domain classifier-based importance weighting
- **Kernel Weights**: RBF kernel similarity

### Evaluation Metrics
- Coverage@Risk≤ε (ε ∈ {1%, 2%, 5%, 10%})
- AURC (Area Under Risk-Coverage curve)
- Risk violation rate & gap
- Outlier abstention rates
- Cost metrics (tokens, latency)

## Benchmarks

Evaluated on **AbstentionBench** with focus on:
- Unknown information
- False premises
- Outdated information
- Underspecified queries
- Multi-hop reasoning

Shift scenarios:
- **ID**: Random split (no shift)
- **Mild shift**: Cross-topic/domain
- **Strong shift**: Temporal drift (outdated) + domain shift

## Citation

```bibtex
@article{your-paper,
  title={Shift-Aware Universal Confidence via Non-Exchangeable Conformal Risk Control},
  author={Your Name},
  year={2025}
}
```

## License

See [LICENSE](LICENSE) file.

## Acknowledgments

Built on:
- [UniCR](https://arxiv.org/abs/2410.02982) - Universal Conformal Risk Control
- [NE-CRC](https://arxiv.org/abs/2411.16653) - Non-Exchangeable Conformal Risk Control
- [SConU](https://arxiv.org/abs/2408.03259) - Conformal uncertainty quantification under distribution shift
- [AbstentionBench](https://arxiv.org/abs/2502.00058) - Benchmark for abstention decisions
