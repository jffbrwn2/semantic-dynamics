"""Visualize SAE activation dynamics from cached data."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_data(cache_dir: Path):
    """Load cached data."""
    with open(cache_dir / "train_data.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open(cache_dir / "feature_indices.pkl", 'rb') as f:
        feature_info = pickle.load(f)

    # Apply feature subset if needed
    indices = feature_info['indices']
    n_features = train_data[0]['sae_acts'].shape[1]
    if n_features != len(indices):
        print(f"Subsetting features: {n_features} -> {len(indices)}")
        train_data = [
            {**d, 'sae_acts': d['sae_acts'][:, indices]}
            for d in train_data
        ]

    return train_data


def plot_feature_timeseries(data, feature_indices=None, n_features=5, n_sequences=5, output_path=None):
    """Plot individual feature activations over time, one subplot per sequence."""
    if feature_indices is None:
        # Pick top features by variance
        all_acts = np.concatenate([d['sae_acts'] for d in data], axis=0)
        variances = np.var(all_acts, axis=0)
        feature_indices = np.argsort(variances)[-n_features:]

    # Wide figure to spread out tokens
    seq_len = len(data[0]['sae_acts'])
    fig, axes = plt.subplots(n_sequences, 1, figsize=(max(20, seq_len * 0.15), 4 * n_sequences), sharex=True)
    if n_sequences == 1:
        axes = [axes]

    for seq_idx, (ax, d) in enumerate(zip(axes, data[:n_sequences])):
        for feat in feature_indices:
            acts = d['sae_acts'][:, feat]
            ax.plot(acts, alpha=0.7, label=f'Feature {feat}' if seq_idx == 0 else None)

        ax.set_yscale('log')
        ax.set_ylabel(f"{d['style']}")
        ax.grid(True, alpha=0.3)

        # Add tokens along top of plot
        tokens = d['tokens']
        ylim = ax.get_ylim()
        for i, tok in enumerate(tokens):
            # Clean up token for display
            tok_display = tok.replace('\n', '\\n')
            ax.text(i, ylim[1], tok_display, fontsize=6, rotation=90,
                    ha='center', va='bottom', alpha=0.7)

    axes[-1].set_xlabel('Token position')
    axes[0].set_title(f'SAE feature activations over time ({len(feature_indices)} features)')
    axes[0].legend(loc='upper right', ncol=min(5, len(feature_indices)))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize SAE dynamics")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--output", type=Path, default=Path("outputs/feature_timeseries.png"))
    parser.add_argument("--n-sequences", type=int, default=5)
    parser.add_argument("--n-features", type=int, default=5)
    args = parser.parse_args()

    print("Loading cached data...")
    data = load_data(args.cache_dir)
    print(f"Loaded {len(data)} sequences")

    args.output.parent.mkdir(exist_ok=True, parents=True)
    plot_feature_timeseries(data, n_features=args.n_features,
                            n_sequences=args.n_sequences, output_path=args.output)


if __name__ == "__main__":
    main()
