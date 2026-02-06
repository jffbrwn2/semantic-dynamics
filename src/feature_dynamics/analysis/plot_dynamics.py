"""Visualize SAE activation dynamics from cached data."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from ..config import _get_default_cache_dir, _get_default_output_dir


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


def plot_feature_timeseries(data, feature_indices=None, n_features=5, n_sequences=5,
                            output_path=None, mode='spiky', use_pre_relu=False):
    """Plot individual feature activations over time, one subplot per sequence.

    Args:
        mode: 'spiky' (high variance), 'persistent' (high mean, low CV), or 'both'
        use_pre_relu: if True, plot pre-ReLU activations (can be negative)
    """
    act_key = 'pre_relu' if use_pre_relu else 'sae_acts'

    if use_pre_relu and 'pre_relu' not in data[0]:
        print("Warning: pre_relu not in data, need to re-collect. Using sae_acts.")
        act_key = 'sae_acts'

    if feature_indices is None:
        all_acts = np.concatenate([d[act_key] for d in data], axis=0)

        if mode == 'spiky':
            # High variance features
            variances = np.var(all_acts, axis=0)
            feature_indices = np.argsort(variances)[-n_features:]
        elif mode == 'persistent':
            # High mean, low coefficient of variation
            means = np.mean(all_acts, axis=0)
            stds = np.std(all_acts, axis=0)
            cv = stds / (means + 1e-8)  # coefficient of variation
            # Score: high mean, low cv
            score = means / (cv + 1e-8)
            feature_indices = np.argsort(score)[-n_features:]
        elif mode == 'both':
            # Half spiky, half persistent
            variances = np.var(all_acts, axis=0)
            spiky = np.argsort(variances)[-(n_features // 2):]

            means = np.mean(all_acts, axis=0)
            stds = np.std(all_acts, axis=0)
            cv = stds / (means + 1e-8)
            score = means / (cv + 1e-8)
            persistent = np.argsort(score)[-(n_features - n_features // 2):]

            feature_indices = np.concatenate([persistent, spiky])

    # Wide figure to spread out tokens
    seq_len = len(data[0]['sae_acts'])
    fig, axes = plt.subplots(n_sequences, 1, figsize=(max(20, seq_len * 0.15), 4 * n_sequences), sharex=True)
    if n_sequences == 1:
        axes = [axes]

    for seq_idx, (ax, d) in enumerate(zip(axes, data[:n_sequences])):
        for feat in feature_indices:
            acts = d[act_key][:, feat]
            ax.plot(acts, alpha=0.7, label=f'Feature {feat}' if seq_idx == 0 else None)

        if not use_pre_relu:
            ax.set_yscale('log')
        else:
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)  # zero line for pre_relu
        # Show prompt on left side
        prompt_text = d.get('prompt', d.get('style', f'Seq {seq_idx}'))
        # Truncate long prompts for display
        if len(prompt_text) > 50:
            prompt_text = prompt_text[:47] + '...'
        ax.set_ylabel(prompt_text, fontsize=8)
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
    parser.add_argument("--cache-dir", type=Path, default=_get_default_cache_dir())
    parser.add_argument("--output", type=Path, default=_get_default_output_dir() / "feature_timeseries")
    parser.add_argument("--n-sequences", type=int, default=5)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--mode", choices=['spiky', 'persistent', 'both'], default='spiky',
                        help="Feature selection: spiky (high variance), persistent (high mean, low CV), or both")
    parser.add_argument("--pre-relu", action="store_true", default=True,
                        help="Plot pre-ReLU activations (can be negative, default: True)")
    parser.add_argument("--no-pre-relu", action="store_false", dest="pre_relu",
                        help="Plot post-ReLU SAE activations")
    args = parser.parse_args()

    print("Loading cached data...")
    data = load_data(args.cache_dir)
    print(f"Loaded {len(data)} sequences")
    if args.pre_relu:
        args.output = args.output / f"mode_{args.mode}_pre_relu.png"
    else:
        args.output = args.output / f"mode_{args.mode}.png"

    args.output.parent.mkdir(exist_ok=True, parents=True)
    plot_feature_timeseries(data, n_features=args.n_features,
                            n_sequences=args.n_sequences, output_path=args.output,
                            mode=args.mode, use_pre_relu=args.pre_relu)


if __name__ == "__main__":
    main()
