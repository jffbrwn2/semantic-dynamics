"""Train RNN predictors for SAE feature dynamics."""

import argparse
import pickle
from pathlib import Path
import numpy as np
import torch

from predictors import RNNTokenOnlyPredictor, RNNStateTokenPredictor, save_predictor


def load_data(cache_dir: Path, use_pre_relu: bool = True, data_file: str = "train_data.pkl",
              apply_cached_subset: bool = True):
    """Load cached data.

    Args:
        cache_dir: Path to cache directory
        use_pre_relu: If True, use pre_relu activations
        data_file: Name of data file to load
        apply_cached_subset: If True, apply the cached feature_indices.pkl subset

    Returns:
        List of data dictionaries
    """
    with open(cache_dir / data_file, 'rb') as f:
        data = pickle.load(f)

    if apply_cached_subset and (cache_dir / "feature_indices.pkl").exists():
        with open(cache_dir / "feature_indices.pkl", 'rb') as f:
            feature_info = pickle.load(f)

        # Apply feature subset if needed
        indices = feature_info['indices']
        act_key = 'pre_relu' if use_pre_relu else 'sae_acts'

        n_features = data[0][act_key].shape[1]
        if n_features != len(indices):
            print(f"Subsetting features: {n_features} -> {len(indices)}")
            data = [
                {**d, act_key: d[act_key][:, indices]}
                for d in data
            ]

    return data


def select_top_k_features(data, k, use_pre_relu=True):
    """Select top-k features by variance.

    Args:
        data: List of data dictionaries
        k: Number of features to select
        use_pre_relu: Whether to use pre_relu activations

    Returns:
        Array of selected feature indices
    """
    act_key = 'pre_relu' if use_pre_relu else 'sae_acts'
    all_acts = np.concatenate([d[act_key] for d in data], axis=0)
    variances = np.var(all_acts, axis=0)
    top_k_indices = np.argsort(variances)[-k:]
    return top_k_indices


def subset_features(data, indices, use_pre_relu=True):
    """Create a subset of the data with only selected features.

    Args:
        data: List of data dictionaries
        indices: Feature indices to keep
        use_pre_relu: Whether to use pre_relu activations

    Returns:
        List of data dictionaries with subset features
    """
    result = []
    for d in data:
        new_d = {**d}
        # Subset both sae_acts and pre_relu to avoid mismatches
        new_d['sae_acts'] = d['sae_acts'][:, indices]
        if 'pre_relu' in d:
            new_d['pre_relu'] = d['pre_relu'][:, indices]
        result.append(new_d)
    return result


def main():
    parser = argparse.ArgumentParser(description="Train RNN predictors")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("/home/jffbrwn/orcd/pool/semantic-dynamics/.cache"),
                        help="Cache directory for data")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/home/jffbrwn/orcd/pool/semantic-dynamics/outputs"),
                        help="Output directory for trained models")
    parser.add_argument("--pre-relu", action="store_true", default=True,
                        help="Use pre-ReLU activations (default: True)")
    parser.add_argument("--no-pre-relu", action="store_false", dest="pre_relu",
                        help="Use post-ReLU SAE activations")
    parser.add_argument("--hidden-size", type=int, default=None,
                        help="RNN hidden size (default: n_features)")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="Number of RNN layers")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Device to train on")
    parser.add_argument("--n-features", type=int, default=None,
                        help="Number of features to use (default: all available)")
    parser.add_argument("--token-only", action="store_true",
                        help="Train only token-only RNN")
    parser.add_argument("--state-token-only", action="store_true",
                        help="Train only state+token RNN")

    args = parser.parse_args()

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = "cpu"

    print("="*60)
    print("RNN Predictor Training")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Hidden size: {args.hidden_size if args.hidden_size else 'n_features'}")
    print(f"Num layers: {args.num_layers}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("="*60 + "\n")

    # Load data
    print("Loading cached training data...")
    train_data_full = load_data(args.cache_dir, use_pre_relu=args.pre_relu,
                                 data_file="train_data_full.pkl", apply_cached_subset=False)
    print(f"Loaded {len(train_data_full)} training sequences")

    # Split into train/val (80/20 split)
    val_size = int(0.2 * len(train_data_full))
    train_data = train_data_full[val_size:]
    val_data = train_data_full[:val_size]
    print(f"Split into {len(train_data)} train / {len(val_data)} validation sequences")

    # Load embedding matrix
    embed_path = args.cache_dir / "embed_matrix.npy"
    if not embed_path.exists():
        print("Error: Embedding matrix not found. Run main.py first to generate it.")
        return
    embed_matrix = np.load(embed_path)
    print(f"Loaded embedding matrix: {embed_matrix.shape}")

    # Infer n_features
    act_key = 'pre_relu' if args.pre_relu else 'sae_acts'
    available_features = train_data_full[0][act_key].shape[1]
    print(f"Total features available: {available_features}")

    # Select subset of features if requested
    if args.n_features is not None and args.n_features < available_features:
        print(f"Selecting top {args.n_features} features by variance...")
        feature_indices = select_top_k_features(train_data_full, args.n_features, args.pre_relu)
        print(f"Selected feature indices: {feature_indices[:10]}... (showing first 10)")

        # Subset data
        train_data = subset_features(train_data, feature_indices, args.pre_relu)
        val_data = subset_features(val_data, feature_indices, args.pre_relu)
        n_features = args.n_features
    else:
        n_features = available_features

    print(f"Training with {n_features} features\n")

    # Create output directory
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # Decide which models to train
    train_token_only = not args.state_token_only
    train_state_token = not args.token_only

    # Train RNN Token-Only Predictor
    if train_token_only:
        print("Training RNN Token-Only Predictor...")
        print("-" * 60)
        rnn_token_predictor = RNNTokenOnlyPredictor(
            n_features=n_features,
            use_pre_relu=args.pre_relu,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
        rnn_token_predictor.fit(train_data, embed_matrix, val_dataset=val_data)

        suffix = "_prerelu" if args.pre_relu else ""
        output_path = args.output_dir / f"rnn_token_predictor{suffix}.pkl"
        save_predictor(rnn_token_predictor, output_path)
        print(f"\nSaved to {output_path}\n")

    # Train RNN State+Token Predictor
    if train_state_token:
        print("Training RNN State+Token Predictor...")
        print("-" * 60)
        rnn_state_token_predictor = RNNStateTokenPredictor(
            n_features=n_features,
            use_pre_relu=args.pre_relu,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
        rnn_state_token_predictor.fit(train_data, embed_matrix, val_dataset=val_data)

        suffix = "_prerelu" if args.pre_relu else ""
        output_path = args.output_dir / f"rnn_state_token_predictor{suffix}.pkl"
        save_predictor(rnn_state_token_predictor, output_path)
        print(f"\nSaved to {output_path}\n")

    print("="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
