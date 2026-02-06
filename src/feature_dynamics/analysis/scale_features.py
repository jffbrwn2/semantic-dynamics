"""Analyze how prediction performance scales with number of features."""

import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from predictors import TokenOnlyPredictor, StateTokenPredictor
from evaluation import compute_r2_per_feature


def load_data(cache_dir: Path, use_pre_relu: bool = True):
    """Load full cached data without feature subsetting.

    Args:
        cache_dir: Path to cache directory
        use_pre_relu: If True, use pre_relu activations

    Returns:
        train_data, test_data (with all features)
    """
    with open(cache_dir / "train_data_full.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open(cache_dir / "test_data_full.pkl", 'rb') as f:
        test_data = pickle.load(f)

    return train_data, test_data


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


def train_and_evaluate(train_data, test_data, embed_matrix, train_feature_indices,
                       eval_feature_indices, alpha, use_pre_relu=True):
    """Train models on subset of features and evaluate on fixed evaluation set.

    Args:
        train_data: Training data
        test_data: Test data
        embed_matrix: Token embedding matrix
        train_feature_indices: Feature indices to train on
        eval_feature_indices: Feature indices to evaluate on (fixed subset)
        alpha: Ridge regularization
        use_pre_relu: Whether to use pre_relu activations

    Returns:
        dict with token_r2 and state_token_r2 (median values)
    """
    act_key = 'pre_relu' if use_pre_relu else 'sae_acts'
    n_train_features = len(train_feature_indices)
    n_eval_features = len(eval_feature_indices)

    # Subset data for training
    train_subset = subset_features(train_data, train_feature_indices, use_pre_relu)

    # Subset test data to training features (for prediction)
    test_train_subset = subset_features(test_data, train_feature_indices, use_pre_relu)

    print(f"\nTraining with {n_train_features} features, evaluating on {n_eval_features} features...")

    # Train token-only predictor
    print("  Training TokenOnly...")
    token_predictor = TokenOnlyPredictor(alpha=alpha, use_pre_relu=use_pre_relu)
    token_predictor.fit(train_subset, embed_matrix)

    # Train state+token predictor
    print("  Training State+Token...")
    state_token_predictor = StateTokenPredictor(alpha=alpha, use_pre_relu=use_pre_relu)
    state_token_predictor.fit(train_subset, embed_matrix)

    # Predict on test set (with all training features)
    token_preds_full = token_predictor.predict(test_train_subset)
    state_token_preds_full = state_token_predictor.predict(test_train_subset)

    # Find which columns in training features correspond to eval features
    eval_indices_in_train = [np.where(train_feature_indices == idx)[0][0]
                             for idx in eval_feature_indices]

    # Extract predictions for only eval features
    token_preds = [pred[:, eval_indices_in_train] for pred in token_preds_full]
    state_token_preds = [pred[:, eval_indices_in_train] for pred in state_token_preds_full]

    # Ground truth for eval features
    test_eval_subset = subset_features(test_data, eval_feature_indices, use_pre_relu)
    y_true = [data[act_key][1:] for data in test_eval_subset]

    token_r2 = compute_r2_per_feature(y_true, token_preds)
    state_token_r2 = compute_r2_per_feature(y_true, state_token_preds)

    results = {
        'token_median_r2': float(np.median(token_r2)),
        'token_mean_r2': float(np.mean(token_r2)),
        'state_token_median_r2': float(np.median(state_token_r2)),
        'state_token_mean_r2': float(np.mean(state_token_r2)),
        'token_frac_positive': float(np.mean(token_r2 > 0)),
        'state_token_frac_positive': float(np.mean(state_token_r2 > 0)),
    }

    print(f"  Token-only median R²: {results['token_median_r2']:.4f}")
    print(f"  State+token median R²: {results['state_token_median_r2']:.4f}")

    return results


def plot_scaling(feature_counts, results_list, output_path):
    """Plot how R² scales with number of features.

    Args:
        feature_counts: List of feature counts
        results_list: List of result dicts
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract metrics
    token_median = [r['token_median_r2'] for r in results_list]
    token_mean = [r['token_mean_r2'] for r in results_list]
    state_median = [r['state_token_median_r2'] for r in results_list]
    state_mean = [r['state_token_mean_r2'] for r in results_list]

    # Plot 1: Median R²
    ax = axes[0]
    ax.plot(feature_counts, token_median, 'o-', label='Token-Only (B*u)', linewidth=2, markersize=8)
    ax.plot(feature_counts, state_median, 's-', label='State+Token (A*x + B*u)', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Training Features', fontsize=12)
    ax.set_ylabel(f'Median R² (on {min(feature_counts)} eval features)', fontsize=12)
    ax.set_title('Does Training on More Features Improve Core Predictions?', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Plot 2: Mean R²
    ax = axes[1]
    ax.plot(feature_counts, token_mean, 'o-', label='Token-Only (B*u)', linewidth=2, markersize=8)
    ax.plot(feature_counts, state_mean, 's-', label='State+Token (A*x + B*u)', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Training Features', fontsize=12)
    ax.set_ylabel(f'Mean R² (on {min(feature_counts)} eval features)', fontsize=12)
    ax.set_title('Mean Prediction Performance vs Training Size', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze feature scaling")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("/home/jffbrwn/orcd/pool/semantic-dynamics/.cache"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/home/jffbrwn/orcd/pool/semantic-dynamics/outputs"))
    parser.add_argument("--feature-counts", nargs='+', type=int,
                        default=[32, 64, 128, 256, 512],
                        help="List of feature counts to test")
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Ridge regularization parameter")
    parser.add_argument("--pre-relu", action="store_true", default=True)
    parser.add_argument("--no-pre-relu", action="store_false", dest="pre_relu")
    args = parser.parse_args()

    print("="*60)
    print("Feature Scaling Analysis")
    print("="*60)
    print(f"Feature counts: {args.feature_counts}")
    print(f"Ridge alpha: {args.alpha}")
    print(f"Using pre-ReLU: {args.pre_relu}")
    print("="*60)

    # Load data
    print("\nLoading full cached data...")
    train_data, test_data = load_data(args.cache_dir, use_pre_relu=args.pre_relu)
    print(f"Loaded {len(train_data)} train, {len(test_data)} test sequences")

    # Load embedding matrix
    embed_path = args.cache_dir / "embed_matrix.npy"
    if not embed_path.exists():
        print("Error: Embedding matrix not found. Run main.py first.")
        return
    embed_matrix = np.load(embed_path)
    print(f"Loaded embedding matrix: {embed_matrix.shape}")

    # Get max features available
    act_key = 'pre_relu' if args.pre_relu else 'sae_acts'
    max_features = train_data[0][act_key].shape[1]
    print(f"Total features available: {max_features}")

    # Filter feature counts that are too large
    feature_counts = [f for f in args.feature_counts if f <= max_features]
    if len(feature_counts) < len(args.feature_counts):
        print(f"Warning: Filtered feature counts to {feature_counts} (max available: {max_features})")

    # Select top features once (using max requested feature count)
    max_requested = max(feature_counts)
    print(f"\nSelecting top {max_requested} features by variance...")
    all_feature_indices = select_top_k_features(train_data, max_requested, args.pre_relu)
    print(f"Selected feature indices: {all_feature_indices[:10]}... (showing first 10)")

    # Use smallest feature set as evaluation target
    min_features = min(feature_counts)
    eval_feature_indices = all_feature_indices[-min_features:]
    print(f"\nWill evaluate all models on the same {min_features} features (highest variance)")
    print(f"Training on increasing feature counts to see if more features help predict the core set\n")

    # Train and evaluate for each feature count (using nested subsets)
    results_list = []
    for n_features in feature_counts:
        # Use top n_features (from end of sorted array, highest variance)
        train_feature_subset = all_feature_indices[-n_features:]
        results = train_and_evaluate(
            train_data, test_data, embed_matrix,
            train_feature_subset, eval_feature_indices,
            args.alpha, args.pre_relu
        )
        results_list.append(results)

    # Save results
    output_dir = args.output_dir / "feature_scaling"
    output_dir.mkdir(exist_ok=True, parents=True)

    results_path = output_dir / "scaling_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump({
            'feature_counts': feature_counts,
            'results': results_list,
            'alpha': args.alpha,
            'use_pre_relu': args.pre_relu,
        }, f)
    print(f"\nSaved results to {results_path}")

    # Plot
    plot_path = output_dir / "feature_scaling.png"
    plot_scaling(feature_counts, results_list, plot_path)

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY (All models evaluated on same {} core features)".format(min(feature_counts)))
    print("="*70)
    print(f"{'Train Features':<15} {'Token R²':<12} {'State+Token R²':<15} {'Improvement':<12}")
    print("-"*70)
    for n_feat, res in zip(feature_counts, results_list):
        improvement = res['state_token_median_r2'] - res['token_median_r2']
        print(f"{n_feat:<15} {res['token_median_r2']:<12.4f} {res['state_token_median_r2']:<15.4f} {improvement:<12.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
