# SAE Feature Dynamics Analysis

Analyze whether Sparse Autoencoder (SAE) feature trajectories in language models exhibit persistent internal modes beyond immediate token forcing.

## Overview

This project tests whether SAE features evolve according to internal state dynamics or are primarily driven by the immediate input token. We compare two predictors:

1. **Token-only baseline**: `x_{t+1} ≈ B u_t`
2. **State+token model**: `x_{t+1} ≈ A x_t + B u_t`

where:
- `x_t` = SAE encoder activations at time t
- `u_t` = current token (one-hot encoded)
- `A` = state transition matrix
- `B` = token input matrix

## Experimental Design

1. **Model**: Gemma 3 27B-IT with Gemma Scope 2 SAE
2. **Layer**: One layer with available SAE (configurable, default: layer 20)
3. **Features**: Top-K features by variance (default: K=512)
4. **Corpus**: ~200 prompts across 4 styles:
   - Plain Q&A
   - Creative writing ("write a story...")
   - Policy/safety questions
   - Paraphrases of the above
5. **Generation**: 200-400 tokens per prompt
6. **Data logged per token**:
   - SAE encoder activations (`x_t`)
   - Token ID and embedding (`u_t`)
   - Metadata (position, newline, punctuation flags)
7. **Train/test split**: By prompt family (75%/25%), ensuring test set contains unseen paraphrases/styles

## Setup

### Prerequisites

- Python 3.10+
- UV package manager
- GPU recommended (but can run on CPU for testing)

### Installation

```bash
# Clone the repository
cd feature-dynamics

# Activate the UV environment (automatically created)
source .venv/bin/activate

# Verify installation
uv run feature-dynamics --help
```

## Usage

### Basic Usage (CPU, for testing)

```bash
# Run full pipeline on CPU with reduced parameters
uv run feature-dynamics \
    --device cpu \
    --layer 20 \
    --top-k 256 \
    --num-prompts 50 \
    --alpha 10.0
```

### Production Usage (GPU)

```bash
# Run full pipeline on GPU
uv run feature-dynamics \
    --device cuda \
    --layer 20 \
    --top-k 512 \
    --num-prompts 200 \
    --alpha 10.0
```

### Using Cached Data

After initial data collection, you can skip collection and rerun analysis:

```bash
uv run feature-dynamics \
    --skip-collection \
    --alpha 5.0
```

### Command-Line Arguments

- `--layer`: Layer index to analyze (default: 20)
- `--top-k`: Number of top features to select (default: 512)
- `--num-prompts`: Total number of prompts (default: 200)
- `--alpha`: Ridge regularization parameter (default: 10.0)
- `--device`: Device (cpu/cuda/mps, default: cpu)
- `--skip-collection`: Skip data collection (use cached data)
- `--cache-dir`: Cache directory (default: .cache)
- `--output-dir`: Output directory (default: outputs)

## Output

The analysis produces several outputs in the `outputs/` directory:

1. **results.pkl**: Complete results dictionary
2. **results_by_style.csv**: Per-style performance breakdown
3. **r2_comparison.png**: Scatter plot comparing model R² scores
4. **token_predictor.pkl**: Trained token-only model
5. **state_token_predictor.pkl**: Trained state+token model

Cached data is stored in `.cache/`:
- `train_data.pkl`: Training dataset
- `test_data.pkl`: Test dataset
- `feature_indices.pkl`: Selected feature indices and variances

## Interpreting Results

### Key Metrics

1. **Median R²**: Median variance explained across features
2. **Fraction improved**: % of features where state+token beats token-only
3. **Fraction improved by >ε**: % of features with improvement > 0.05

### Decision Criteria

**Evidence of persistent internal modes:**
- Median improvement > 0.05
- >30% of features improve by >0.05
- State+token model shows good cross-style generalization

**Limited evidence (token forcing dominates):**
- Small or negative median improvement
- <30% of features improve meaningfully
- Performance similar across models

### Example Output

```
EVALUATION RESULTS
===========================================================

Token-Only Model:
----------------------------------------
  mean_r2             : 0.1234
  median_r2           : 0.0856
  frac_positive       : 0.7800
  frac_good           : 0.3200

State+Token Model:
----------------------------------------
  mean_r2             : 0.2145
  median_r2           : 0.1523
  frac_positive       : 0.8900
  frac_good           : 0.5600

Comparison (State+Token vs Token-Only):
----------------------------------------
  median_improvement              : 0.0667
  frac_improved                   : 0.7200
  frac_improved_by_0.05           : 0.4500

KEY FINDINGS
===========================================================
✓ EVIDENCE OF PERSISTENT INTERNAL MODES
  Adding state significantly improves prediction across styles.
```

## Project Structure

```
feature-dynamics/
├── src/feature_dynamics/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration dataclass
│   ├── prompts.py            # Prompt generation across styles
│   ├── data_collection.py    # Model inference + SAE activation collection
│   ├── predictors.py         # Token-only and state+token predictors
│   ├── evaluation.py         # R² metrics and comparison
│   └── main.py               # Main entry point
├── outputs/                  # Results and trained models
├── .cache/                   # Cached datasets
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## Advanced Usage

### Custom Analysis

You can import the modules and run custom analyses:

```python
from feature_dynamics import Config, PromptGenerator, DataCollector
from feature_dynamics import TokenOnlyPredictor, StateTokenPredictor
from feature_dynamics import evaluate_predictors

# Create custom config
config = Config(
    layer_idx=15,
    top_k_features=1024,
    ridge_alpha=5.0,
)

# Generate prompts
generator = PromptGenerator()
corpus = generator.generate_corpus(num_prompts=100)
train_corpus, test_corpus = generator.split_corpus(corpus)

# Collect data
collector = DataCollector(config, device="cuda")
train_data = collector.collect_corpus_data(train_corpus)

# ... train and evaluate models
```

### Modifying Prompt Styles

Edit `src/feature_dynamics/prompts.py` to add new prompt styles or customize paraphrasing.

### Adjusting SAE Selection

Modify `src/feature_dynamics/config.py` to change:
- SAE release name
- Layer selection
- Feature selection criteria

## Notes

### Memory Requirements

- **CPU mode**: Requires ~32GB RAM for Gemma 27B (with offloading)
- **GPU mode**: Requires GPU with >40GB VRAM for fp16, or use model sharding
- Reduce `--top-k` and `--num-prompts` for lower memory usage

### Runtime

- Initial data collection: ~2-4 hours on GPU for 200 prompts
- Model fitting: ~1-5 minutes depending on feature count
- Evaluation: <1 minute

### Gemma Scope 2 SAE

The code assumes Gemma Scope 2 SAE format. If using different SAEs:
1. Update `sae_release` in `Config`
2. Modify SAE loading in `DataCollector._load_sae()`
3. Adjust hook point naming if needed

## Citation

If you use this code, please cite:

```
[Your paper citation here]
```

## License

MIT License

## Contact

[Your contact information]
# semantic-dynamics
