"""Analyze feature statistics from SAE activation data."""

import argparse
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Analyze SAE feature statistics")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing trials_dataset.pkl or activations_*.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for figure (default: show interactively)",
    )
    parser.add_argument(
        "--use-pre-relu",
        action="store_true",
        help="Use pre-ReLU activations instead of post-ReLU",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log scale for histogram",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Highlight top-k features by variance (default: 200)",
    )
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    trials_path = args.data_dir / "trials_dataset.pkl"

    if trials_path.exists():
        with open(trials_path, "rb") as f:
            dataset = pickle.load(f)
        data = dataset.get("all_trials", [])
        print(f"Loaded trials dataset: {len(data)} trials")
    else:
        # Try loading individual activation files
        act_files = sorted(args.data_dir.glob("activations_*.pkl"))
        if not act_files:
            raise FileNotFoundError(f"No data found in {args.data_dir}")
        data = []
        for f in act_files:
            with open(f, "rb") as fp:
                data.append(pickle.load(fp))
        print(f"Loaded {len(data)} activation files")

    # Extract activations
    act_key = "pre_relu" if args.use_pre_relu else "sae_acts"
    all_acts = np.concatenate([d[act_key] for d in data], axis=0)
    print(f"Activation shape: {all_acts.shape} (timesteps x features)")

    # Compute statistics
    variances = np.var(all_acts, axis=0)
    means = np.mean(all_acts, axis=0)
    sparsity = (all_acts == 0).mean(axis=0)  # Fraction of zeros per feature
    max_vals = np.max(all_acts, axis=0)

    # Summary stats
    print(f"\n{'='*60}")
    print("Feature Statistics Summary")
    print(f"{'='*60}")
    print(f"Total features: {len(variances)}")
    print(f"Overall sparsity: {(all_acts == 0).mean():.1%}")
    print(f"\nVariance: min={variances.min():.6f}, max={variances.max():.2f}, "
          f"mean={variances.mean():.4f}, median={np.median(variances):.6f}")
    print(f"Mean activation: min={means.min():.6f}, max={means.max():.2f}, "
          f"mean={means.mean():.4f}")

    # Count features by activity level
    n_zero_var = (variances == 0).sum()
    n_low_var = ((variances > 0) & (variances < 0.01)).sum()
    n_med_var = ((variances >= 0.01) & (variances < 0.1)).sum()
    n_high_var = (variances >= 0.1).sum()

    print(f"\nFeatures by variance:")
    print(f"  Zero variance: {n_zero_var} ({n_zero_var/len(variances):.1%})")
    print(f"  Low (0, 0.01): {n_low_var} ({n_low_var/len(variances):.1%})")
    print(f"  Medium [0.01, 0.1): {n_med_var} ({n_med_var/len(variances):.1%})")
    print(f"  High [0.1+]: {n_high_var} ({n_high_var/len(variances):.1%})")

    # Top-k analysis
    top_k_idx = np.argsort(variances)[-args.top_k:]
    top_k_var_threshold = variances[top_k_idx[0]]
    top_k_total_var = variances[top_k_idx].sum()
    total_var = variances.sum()

    print(f"\nTop-{args.top_k} features:")
    print(f"  Variance threshold: {top_k_var_threshold:.4f}")
    print(f"  Variance explained: {top_k_total_var/total_var:.1%} of total")
    print(f"  Mean sparsity: {sparsity[top_k_idx].mean():.1%}")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Histogram of variances
    ax = axes[0, 0]
    var_nonzero = variances[variances > 0]
    if args.log_scale and len(var_nonzero) > 0:
        ax.hist(np.log10(var_nonzero), bins=100, color='steelblue', alpha=0.7)
        ax.axvline(np.log10(top_k_var_threshold), color='red', linestyle='--',
                   label=f'Top-{args.top_k} threshold')
        ax.set_xlabel('log10(Variance)')
    else:
        ax.hist(variances, bins=100, color='steelblue', alpha=0.7)
        ax.axvline(top_k_var_threshold, color='red', linestyle='--',
                   label=f'Top-{args.top_k} threshold')
        ax.set_xlabel('Variance')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.set_title('Distribution of Feature Variances')
    ax.legend()

    # 2. Histogram of mean activations
    ax = axes[0, 1]
    means_nonzero = means[means > 0]
    if args.log_scale and len(means_nonzero) > 0:
        ax.hist(np.log10(means_nonzero + 1e-10), bins=100, color='forestgreen', alpha=0.7)
        ax.set_xlabel('log10(Mean Activation)')
    else:
        ax.hist(means, bins=100, color='forestgreen', alpha=0.7)
        ax.set_xlabel('Mean Activation')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.set_title('Distribution of Feature Mean Activations')

    # 3. Sparsity histogram
    ax = axes[1, 0]
    ax.hist(sparsity, bins=100, color='coral', alpha=0.7)
    ax.set_xlabel('Sparsity (fraction of zeros)')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.set_title('Distribution of Feature Sparsity')

    # 4. Cumulative variance explained
    ax = axes[1, 1]
    sorted_var = np.sort(variances)[::-1]
    cumsum_var = np.cumsum(sorted_var) / total_var
    ax.plot(np.arange(1, len(cumsum_var) + 1), cumsum_var, color='purple')
    ax.axvline(args.top_k, color='red', linestyle='--', label=f'Top-{args.top_k}')
    ax.axhline(cumsum_var[args.top_k - 1], color='red', linestyle=':', alpha=0.5)
    ax.set_xlabel('Number of Features (sorted by variance)')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_title('Variance Explained by Top-K Features')
    ax.set_xlim(0, min(2000, len(variances)))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
