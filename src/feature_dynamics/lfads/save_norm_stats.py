"""Save normalization stats and feature indices for existing LFADS models.

This utility recomputes normalization stats from the original training data
and saves them to the model output directories.

Usage:
    python -m feature_dynamics.lfads.save_norm_stats \
        --data-dir /path/to/trials_data \
        --output-dir /path/to/lfads_small
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def save_normalization_stats(
    data_dir: Path,
    output_dir: Path,
    n_features: int = None,
    feature_selection: str = "all",
    use_pre_relu: bool = False,
):
    """Recompute and save normalization stats from training data.

    Args:
        data_dir: Directory containing trials_dataset.pkl
        output_dir: LFADS model output directory to save stats to
        n_features: Number of features to select (None = all)
        feature_selection: "variance" or "all"
        use_pre_relu: Whether to use pre-ReLU activations
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Load trials dataset
    trials_path = data_dir / "trials_dataset.pkl"
    if not trials_path.exists():
        raise FileNotFoundError(f"No trials_dataset.pkl found in {data_dir}")

    print(f"Loading data from {trials_path}...")
    with open(trials_path, 'rb') as f:
        data = pickle.load(f)

    # Handle different dataset formats
    if isinstance(data, dict) and 'all_trials' in data:
        trials = data['all_trials']
    elif isinstance(data, list):
        trials = data
    else:
        raise ValueError(f"Unknown dataset format: {type(data)}")

    # Extract activations
    act_key = 'pre_relu' if use_pre_relu else 'sae_acts'
    all_acts = np.concatenate([d[act_key] for d in trials], axis=0)
    print(f"Loaded activations: {all_acts.shape}")

    total_features = all_acts.shape[1]

    # Feature selection
    selected_indices = None
    if n_features is not None and n_features < total_features:
        if feature_selection == "variance":
            print(f"Selecting top {n_features} features by variance...")
            variances = np.var(all_acts, axis=0)
            selected_indices = np.argsort(variances)[-n_features:]
            selected_indices = np.sort(selected_indices)  # Keep sorted order
            all_acts = all_acts[:, selected_indices]
            print(f"Selected features shape: {all_acts.shape}")
        elif feature_selection == "random":
            np.random.seed(42)
            selected_indices = np.random.choice(total_features, n_features, replace=False)
            selected_indices = np.sort(selected_indices)
            all_acts = all_acts[:, selected_indices]
        else:
            selected_indices = np.arange(n_features)
            all_acts = all_acts[:, selected_indices]

    # Compute stats
    mean = all_acts.mean(axis=0)
    std = all_acts.std(axis=0)
    std[std == 0] = 1.0  # Prevent division by zero

    print(f"Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"Std range: [{std.min():.4f}, {std.max():.4f}]")

    # Save normalization stats
    norm_path = output_dir / "normalization_stats.npz"
    np.savez(norm_path, mean=mean, std=std)
    print(f"Saved normalization stats to {norm_path}")

    # Save feature indices if using subset
    if selected_indices is not None:
        indices_path = output_dir / "feature_indices.pkl"
        with open(indices_path, 'wb') as f:
            pickle.dump({'indices': selected_indices}, f)
        print(f"Saved feature indices to {indices_path}")

    return mean, std, selected_indices


def main():
    parser = argparse.ArgumentParser(
        description="Save normalization stats for existing LFADS models"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing trials_dataset.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="LFADS model output directory",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=None,
        help="Number of features (reads from args.json if not specified)",
    )
    parser.add_argument(
        "--feature-selection",
        type=str,
        default=None,
        help="Feature selection method (reads from args.json if not specified)",
    )
    parser.add_argument(
        "--use-pre-relu",
        action="store_true",
        help="Use pre-ReLU activations",
    )

    args = parser.parse_args()

    # Try to read settings from args.json if not specified
    args_json_path = args.output_dir / "args.json"
    if args_json_path.exists():
        with open(args_json_path) as f:
            train_args = json.load(f)

        if args.n_features is None:
            args.n_features = train_args.get('n_features')
        if args.feature_selection is None:
            args.feature_selection = train_args.get('feature_selection', 'all')
        if not args.use_pre_relu:
            args.use_pre_relu = train_args.get('use_pre_relu', False)

        print(f"Loaded settings from {args_json_path}:")
        print(f"  n_features: {args.n_features}")
        print(f"  feature_selection: {args.feature_selection}")
        print(f"  use_pre_relu: {args.use_pre_relu}")

    save_normalization_stats(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_features=args.n_features,
        feature_selection=args.feature_selection or "all",
        use_pre_relu=args.use_pre_relu,
    )


if __name__ == "__main__":
    main()
