# Cluster Setup Guide

This guide explains how to run the SAE feature dynamics analysis on a GPU cluster.

## Prerequisites

- Access to a cluster with GPU nodes (NVIDIA GPUs with CUDA support)
- At least 48GB GPU VRAM for Gemma 27B (or use model sharding)
- UV package manager installed on the cluster
- SLURM or similar job scheduler (examples use SLURM)

## Setup on Cluster

### 1. Clone and Setup Environment

```bash
# SSH into cluster
ssh your-cluster

# Navigate to your workspace
cd /path/to/your/workspace

# Clone the project (or transfer files)
git clone <your-repo-url> feature-dynamics
cd feature-dynamics

# Install UV if not available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv sync
```

### 2. Test Installation

```bash
# Run quick test
uv run python test_setup.py
```

## Running on SLURM

### Interactive Session (for testing)

```bash
# Request GPU node
srun --gres=gpu:1 --mem=64G --time=4:00:00 --pty bash

# Activate environment and run
cd feature-dynamics
uv run feature-dynamics \
    --device cuda \
    --layer 20 \
    --top-k 512 \
    --num-prompts 200 \
    --alpha 10.0
```

### Batch Job

Create a file `run_analysis.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=sae_dynamics
#SBATCH --output=logs/sae_dynamics_%j.out
#SBATCH --error=logs/sae_dynamics_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Load modules (adjust for your cluster)
module load cuda/12.1
module load python/3.12

# Navigate to project
cd /path/to/feature-dynamics

# Run analysis
uv run feature-dynamics \
    --device cuda \
    --layer 20 \
    --top-k 512 \
    --num-prompts 200 \
    --alpha 10.0 \
    --output-dir outputs/run_${SLURM_JOB_ID} \
    --cache-dir .cache/run_${SLURM_JOB_ID}

echo "Job completed at $(date)"
```

Submit the job:

```bash
mkdir -p logs
sbatch run_analysis.sh
```

### Parameter Sweep

Create `sweep_layers.sh` to test multiple layers:

```bash
#!/bin/bash
#SBATCH --job-name=sae_sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --array=0-25  # For layers 0-25

# Load modules
module load cuda/12.1

# Get layer from array index
LAYER=$SLURM_ARRAY_TASK_ID

# Run analysis for this layer
cd /path/to/feature-dynamics
uv run feature-dynamics \
    --device cuda \
    --layer $LAYER \
    --top-k 512 \
    --num-prompts 200 \
    --alpha 10.0 \
    --output-dir outputs/layer_${LAYER} \
    --cache-dir .cache/layer_${LAYER}

echo "Completed layer $LAYER at $(date)"
```

Submit array job:

```bash
sbatch sweep_layers.sh
```

## Memory Optimization

### For Limited GPU Memory

If you have <48GB VRAM, you can:

1. **Use smaller model**:
   ```bash
   # In config.py, change:
   model_name: str = "google/gemma-2-9b-it"
   sae_release: str = "gemma-scope-2-9b-it-resid_post"
   ```

2. **Use model sharding** (edit `data_collection.py`):
   ```python
   self.model = AutoModelForCausalLM.from_pretrained(
       config.model_name,
       torch_dtype=torch.float16,
       device_map="auto",  # Automatic sharding across GPUs/CPU
   )
   ```

3. **Reduce batch processing**:
   ```bash
   uv run feature-dynamics \
       --top-k 256 \      # Fewer features
       --num-prompts 100  # Fewer prompts
   ```

### For Multi-GPU

Modify the SLURM script:

```bash
#SBATCH --gres=gpu:2  # Request 2 GPUs
```

And use model parallelism in code.

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View output in real-time
tail -f logs/sae_dynamics_<job_id>.out

# Check GPU usage
ssh <gpu-node>
nvidia-smi
```

## Data Management

### Caching Strategy

Data collection can take hours. Use the cache to avoid re-running:

```bash
# First run: collect data
uv run feature-dynamics --device cuda --layer 20

# Subsequent runs: skip collection, try different regularization
uv run feature-dynamics --skip-collection --alpha 5.0
uv run feature-dynamics --skip-collection --alpha 20.0
```

### Transferring Results

```bash
# Download results from cluster
scp -r your-cluster:/path/to/feature-dynamics/outputs ./local-outputs

# Or use rsync for efficiency
rsync -avz your-cluster:/path/to/feature-dynamics/outputs/ ./local-outputs/
```

## Troubleshooting

### CUDA Out of Memory

```python
# Add to data_collection.py after model.generate():
torch.cuda.empty_cache()
```

### Module Not Found

```bash
# Re-sync environment
uv sync --reinstall
```

### Slow Data Collection

Check that:
1. GPU is actually being used: `nvidia-smi`
2. Model is on GPU: add `print(f"Model on {self.model.device}")` in code
3. Batch size isn't too small (but watch memory)

## Example Workflow

```bash
# 1. Test locally on CPU
uv run feature-dynamics --device cpu --layer 10 --top-k 128 --num-prompts 20

# 2. Test on cluster with small job
srun --gres=gpu:1 --mem=32G --time=1:00:00 uv run feature-dynamics \
    --device cuda --layer 10 --top-k 256 --num-prompts 50

# 3. Run full analysis
sbatch run_analysis.sh

# 4. Download and analyze results locally
scp -r cluster:/path/outputs ./
python -c "import pickle; print(pickle.load(open('outputs/results.pkl', 'rb')))"
```

## Cost Optimization

- Use preemptible/spot instances if available
- Cache data aggressively
- Run parameter sweeps only after validating one configuration
- Consider using smaller models (9B instead of 27B) for initial exploration

## Support

- Check cluster documentation for specific SLURM configurations
- Contact your cluster admin for GPU allocation policies
- See main README.md for code-level troubleshooting
