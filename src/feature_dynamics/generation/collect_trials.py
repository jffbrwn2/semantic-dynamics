"""Collect multiple trials (generations) per prompt for LFADS analysis.

This creates a dataset structure suitable for LFADS, where we have multiple
"trials" of the same "condition" (prompt), allowing the model to learn
consistent dynamics while capturing trial-to-trial variability.

Example usage:
    python -m feature_dynamics.collect_trials \
        --prompts "Explain quantum computing" "Write a poem about the ocean" \
        --trials-per-prompt 20 \
        --output-dir /path/to/outputs/trials_data
"""

import argparse
from pathlib import Path
import pickle
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional

from ..config import Config, _get_default_cache_dir, _get_default_output_dir
from ..data_collection import DataCollector


def collect_multi_trial_data(
    collector: DataCollector,
    prompts: List[str],
    trials_per_prompt: int = 20,
    max_tokens: int = 200,
    temperature: float = 0.7,
    output_path: Optional[Path] = None,
) -> Dict[str, List[Dict]]:
    """Collect multiple trials for each prompt.

    Args:
        collector: DataCollector instance
        prompts: List of prompts to generate from
        trials_per_prompt: Number of generations per prompt
        max_tokens: Max tokens per generation
        temperature: Sampling temperature
        output_path: Optional path to save results

    Returns:
        Dictionary mapping prompt_id -> list of trial data dicts
    """
    dataset = {
        'trials_by_prompt': {},  # prompt_id -> list of trials
        'prompts': prompts,
        'metadata': {
            'trials_per_prompt': trials_per_prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
        }
    }

    all_trials = []  # Flat list for easy loading

    for prompt_idx, prompt in enumerate(prompts):
        print(f"\nPrompt {prompt_idx + 1}/{len(prompts)}: {prompt[:50]}...")
        prompt_trials = []

        for trial in tqdm(range(trials_per_prompt), desc=f"  Trials"):
            data = collector.generate_and_collect(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if data is not None:
                data['prompt_id'] = prompt_idx
                data['trial_id'] = trial
                data['condition'] = prompt_idx  # For LFADS: condition = prompt
                prompt_trials.append(data)
                all_trials.append(data)

        dataset['trials_by_prompt'][prompt_idx] = prompt_trials
        print(f"  Collected {len(prompt_trials)} valid trials")

    dataset['all_trials'] = all_trials

    # Save if output path provided
    if output_path is not None:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"\nSaved dataset to {output_path}")
        print(f"  Total trials: {len(all_trials)}")
        print(f"  Prompts: {len(prompts)}")

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Collect multiple trials per prompt for LFADS"
    )

    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=[
            "Explain the concept of machine learning in simple terms.",
            "Write a short story about a robot learning to feel emotions.",
            "What are the main causes of climate change?",
            "Describe the process of photosynthesis step by step.",
            "Write a poem about the beauty of mathematics.",
        ],
        help="List of prompts to generate from",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Path to file with prompts (one per line)",
    )
    parser.add_argument(
        "--trials-per-prompt",
        type=int,
        default=20,
        help="Number of generations per prompt (default: 20)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Max tokens per generation (default: 200)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=31,
        help="Layer to extract SAE activations from (default: 31)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_get_default_output_dir() / "trials_data",
        help="Output directory",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_get_default_cache_dir(),
        help="Cache directory for models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()

    # Load prompts from file if provided
    if args.prompts_file is not None:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = args.prompts

    print("=" * 60)
    print("Multi-Trial Data Collection for LFADS")
    print("=" * 60)
    print(f"Prompts: {len(prompts)}")
    print(f"Trials per prompt: {args.trials_per_prompt}")
    print(f"Total trials: {len(prompts) * args.trials_per_prompt}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Layer: {args.layer}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Initialize config and collector
    config = Config(
        layer_idx=args.layer,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        min_tokens=50,  # Lower threshold for trials
    )

    print("\nInitializing data collector...")
    collector = DataCollector(config, device=args.device)

    # Collect data
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "trials_dataset.pkl"

    dataset = collect_multi_trial_data(
        collector,
        prompts,
        trials_per_prompt=args.trials_per_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        output_path=output_path,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for prompt_idx, prompt in enumerate(prompts):
        n_trials = len(dataset['trials_by_prompt'].get(prompt_idx, []))
        print(f"  Prompt {prompt_idx}: {n_trials} trials - \"{prompt[:40]}...\"")

    # Compute statistics
    all_trials = dataset['all_trials']
    if all_trials:
        seq_lens = [t['sae_acts'].shape[0] for t in all_trials]
        print(f"\nSequence lengths: {np.min(seq_lens)} - {np.max(seq_lens)} (mean: {np.mean(seq_lens):.1f})")

        # Sparsity
        all_acts = np.concatenate([t['sae_acts'] for t in all_trials], axis=0)
        sparsity = (all_acts == 0).mean()
        print(f"Activation sparsity: {sparsity:.1%}")

    print(f"\nDataset saved to: {output_path}")
    print("\nTo train LFADS on this data:")
    print(f"  python -m feature_dynamics.lfads \\")
    print(f"      --data-dir {args.output_dir} \\")
    print(f"      --output-dir {args.output_dir}/lfads \\")
    print(f"      --seq-len 64 --epochs 100")


if __name__ == "__main__":
    main()
