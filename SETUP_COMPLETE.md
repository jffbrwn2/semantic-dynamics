# Setup Complete! 🎉

Your SAE feature dynamics analysis project is ready to use.

## What's Been Set Up

### ✅ UV Environment
- **Python**: 3.12
- **Key packages**: torch, transformers, sae-lens, scikit-learn, pandas, numpy, matplotlib
- **Location**: `.venv/` (automatically managed by UV)

### ✅ Project Structure

```
feature-dynamics/
├── src/feature_dynamics/
│   ├── config.py              # Configuration (layer, features, hyperparams)
│   ├── prompts.py             # Generate 4 prompt styles + paraphrases
│   ├── data_collection.py     # Run Gemma 27B + collect SAE activations
│   ├── predictors.py          # Token-only & state+token models
│   ├── evaluation.py          # R² metrics & comparison
│   └── main.py                # Main pipeline
├── test_setup.py              # Validation script
├── README.md                  # Full documentation
├── CLUSTER_SETUP.md           # GPU cluster guide
└── pyproject.toml             # UV configuration

Outputs:
├── outputs/                   # Results, plots, trained models
└── .cache/                    # Cached datasets
```

### ✅ Core Functionality

The project implements your experimental design:

1. **Model**: Gemma 3 27B-IT with Gemma Scope 2 SAE
2. **Corpus**: ~200 prompts across 4 styles (Q&A, story, policy, paraphrases)
3. **Data**: SAE activations (x_t) + tokens (u_t) for 200-400 tokens/prompt
4. **Predictors**:
   - Token-only: `x_{t+1} ≈ B u_t`
   - State+token: `x_{t+1} ≈ A x_t + B u_t`
5. **Evaluation**: R² per feature, aggregate metrics, style-wise analysis
6. **Output**: Evidence of persistent modes vs token forcing

## Quick Start

### 1. Verify Setup
```bash
uv run python test_setup.py
```

### 2. Test Locally (CPU, small scale)
```bash
# This will take a while even with small params
uv run feature-dynamics \
    --device cpu \
    --layer 10 \
    --top-k 128 \
    --num-prompts 20
```

### 3. Production Run (GPU cluster)
See `CLUSTER_SETUP.md` for full cluster instructions.

```bash
# On cluster with GPU
uv run feature-dynamics \
    --device cuda \
    --layer 20 \
    --top-k 512 \
    --num-prompts 200 \
    --alpha 10.0
```

## What Happens When You Run

1. **Prompt Generation** (~1 second)
   - Creates corpus across 4 styles
   - Splits into train/test by prompt family

2. **Data Collection** (⚠️ 2-4 hours on GPU)
   - Loads Gemma 27B + Gemma Scope 2 SAE
   - Generates text for each prompt
   - Collects SAE activations at each token
   - Saves to `.cache/` for reuse

3. **Feature Selection** (~1 minute)
   - Selects top-K features by variance
   - Default K=512 (configurable)

4. **Model Fitting** (~1-5 minutes)
   - Fits token-only baseline
   - Fits state+token model
   - Uses ridge regression with regularization

5. **Evaluation** (<1 minute)
   - Computes R² per feature
   - Generates comparison metrics
   - Analyzes by style
   - Creates plots

6. **Results**
   ```
   outputs/
   ├── results.pkl              # Full results dict
   ├── results_by_style.csv     # Per-style breakdown
   ├── r2_comparison.png        # Scatter plot
   ├── token_predictor.pkl      # Trained baseline
   └── state_token_predictor.pkl # Trained state model
   ```

## Key Parameters

Adjust these for your needs:

- `--layer`: Which layer's SAE to analyze (0-42 for Gemma 27B)
- `--top-k`: Number of features to keep (128, 256, 512, 1024)
- `--num-prompts`: Total prompts across all styles (20, 100, 200)
- `--alpha`: Ridge regularization strength (1.0, 10.0, 100.0)
- `--device`: cpu/cuda/mps

## Interpreting Results

The code will print a summary like:

```
KEY FINDINGS
============================================================
Median R² improvement: 0.0667
Fraction improved: 0.7200
Fraction improved by >0.05: 0.4500

State+Token model median R²: 0.1523

✓ EVIDENCE OF PERSISTENT INTERNAL MODES
  Adding state significantly improves prediction across styles.
```

**Threshold for "persistent modes":**
- Median improvement > 0.05
- >30% of features improve by >0.05

## Next Steps

### For Local Testing
1. Run with minimal parameters to verify everything works:
   ```bash
   uv run feature-dynamics --device cpu --layer 10 --top-k 64 --num-prompts 10
   ```

### For Production
1. Transfer to cluster (see `CLUSTER_SETUP.md`)
2. Run full analysis on GPU
3. Download results for visualization

### For Experimentation
- Try different layers: `--layer 0`, `--layer 10`, `--layer 20`
- Try different regularization: `--alpha 1.0`, `--alpha 50.0`
- After first run, use `--skip-collection` to reuse cached data

## Important Notes

⚠️ **First run will be slow**: Data collection takes 2-4 hours on GPU for 200 prompts

💡 **Use caching**: After first run, use `--skip-collection` flag

🔬 **Start small**: Test with `--num-prompts 20 --top-k 128` first

🖥️ **GPU recommended**: Gemma 27B needs significant resources
   - CPU: Possible but very slow (~10x longer)
   - GPU: 40-80GB VRAM (use fp16 or model sharding)

## Files You Can Modify

- `src/feature_dynamics/config.py`: Change default layer, SAE selection, hyperparams
- `src/feature_dynamics/prompts.py`: Add new prompt styles, customize paraphrasing
- `test_setup.py`: Add more validation checks

## Getting Help

- **Full docs**: See `README.md`
- **Cluster setup**: See `CLUSTER_SETUP.md`
- **Code structure**: All modules have detailed docstrings

## References

- [SAE Lens Documentation](https://jbloomaus.github.io/SAELens/)
- [Gemma Scope 2 (Neuronpedia)](https://www.neuronpedia.org/gemma-scope-2)
- [Gemma Models (Hugging Face)](https://huggingface.co/google/gemma-2-27b-it)

## Sources

Based on research methodology for analyzing SAE feature dynamics in language models:
- [SAE Lens on PyPI](https://pypi.org/project/sae-lens/)
- [Gemma Scope 2 on Hugging Face](https://huggingface.co/google/gemma-scope-2-27b-it)
- [Google DeepMind Blog on Gemma Scope 2](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/)

---

Ready to start! Run `uv run feature-dynamics --help` for all options.
