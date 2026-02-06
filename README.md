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

### Installation

```bash
# Clone the repository
cd semantic-dynamics

# Activate the virtual environment
source .venv/bin/activate

# Change to the feature_dynamics source directory
cd src/feature_dynamics
```

### Configuration

Copy the example environment file and set your paths:

```bash
cp .env.example .env
# Edit .env with your cache and output directories
```

Environment variables:
- `FEATURE_DYNAMICS_CACHE_DIR`: Directory for cached datasets
- `FEATURE_DYNAMICS_OUTPUT_DIR`: Directory for outputs (models, results)

If not set, defaults to `./cache` and `./outputs` in the current directory.

## Usage

All commands should be run from `src/feature_dynamics/` with the environment activated.

### Main Analysis Pipeline

```bash
# Run full pipeline (prompts -> collect -> train -> evaluate)
python -m feature_dynamics.main --help

# Example: Run on GPU with default parameters
python -m feature_dynamics.main --device cuda --layer 31 --top-k 512
```

### Text Generation with Learned Dynamics

```bash
# Generate text using a trained predictor
python -m feature_dynamics.generation.generate \
    --predictor /path/to/predictor.pkl \
    --sae-model /path/to/sae_weights.pkl \
    --prompt "Your prompt here" \
    --max-tokens 100
```

### LFADS Training

```bash
# Train LFADS model on collected trial data
python -m feature_dynamics.lfads \
    --data-dir /path/to/trials_data \
    --output-dir /path/to/output \
    --epochs 100
```

### Data Collection

```bash
# Collect multi-trial data for LFADS training
python -m feature_dynamics.generation.collect_trials --help
```

### Analysis & Visualization

```bash
# Evaluate SAE reconstruction ceiling
python -m feature_dynamics.analysis.evaluate_sae_ceiling --help

# Plot feature dynamics
python -m feature_dynamics.analysis.plot_dynamics --help
```



## Project Structure

```
semantic-dynamics/
├── src/feature_dynamics/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration dataclass
│   ├── prompts.py            # Prompt generation across styles
│   ├── data_collection.py    # Model inference + SAE activation collection
│   ├── predictors.py         # Linear predictors (TokenOnly, StateToken, LFADS)
│   ├── evaluation.py         # R² metrics and comparison
│   ├── main.py               # Main entry point
│   ├── analysis/             # Analysis and visualization scripts
│   │   ├── evaluate_sae_ceiling.py
│   │   ├── scale_features.py
│   │   ├── plot_dynamics.py
│   │   └── plot_dynamics_comparison.py
│   ├── generation/           # Text generation and data collection
│   │   ├── generate.py       # Predictor-driven text generation
│   │   └── collect_trials.py # Multi-trial data collection for LFADS
│   ├── rnn/                  # RNN-based predictors
│   │   └── train_rnn.py
│   ├── utils/                # Utility scripts
│   │   └── save_sae_weights.py
│   ├── lfads/                # LFADS model and training
│   │   ├── model.py          # LFADS architecture
│   │   ├── training.py       # Dataset and trainer classes
│   │   └── train.py          # Training entry point
│   └── notebooks/            # Jupyter notebooks for exploration
├── outputs/                  # Results and trained models
├── .cache/                   # Cached datasets
├── pyproject.toml            # Project configuration
└── README.md                 # This file
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