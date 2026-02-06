"""Compare actual vs predicted dynamics from linear models."""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from predictors import load_predictor


def load_data(cache_dir: Path, use_pre_relu: bool = True, data_file: str = "train_data.pkl"):
    """Load cached data.

    Args:
        cache_dir: Path to cache directory
        use_pre_relu: If True, use pre_relu activations
        data_file: Name of data file to load (train_data.pkl or test_data.pkl)

    Returns:
        List of data dictionaries
    """
    with open(cache_dir / data_file, 'rb') as f:
        data = pickle.load(f)
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


def select_features(data, n_features=5, mode='spiky', use_pre_relu=True):
    """Select features to display based on mode.

    Args:
        data: List of data dictionaries
        n_features: Number of features to select
        mode: 'spiky' (high variance), 'persistent' (high mean, low CV), or 'both'
        use_pre_relu: If True, use pre_relu activations

    Returns:
        Array of feature indices
    """
    act_key = 'pre_relu' if use_pre_relu else 'sae_acts'
    all_acts = np.concatenate([d[act_key] for d in data], axis=0)

    if mode == 'spiky':
        variances = np.var(all_acts, axis=0)
        return np.argsort(variances)[-n_features:]
    elif mode == 'persistent':
        means = np.mean(all_acts, axis=0)
        stds = np.std(all_acts, axis=0)
        cv = stds / (means + 1e-8)
        score = means / (cv + 1e-8)
        return np.argsort(score)[-n_features:]
    elif mode == 'both':
        variances = np.var(all_acts, axis=0)
        spiky = np.argsort(variances)[-(n_features // 2):]

        means = np.mean(all_acts, axis=0)
        stds = np.std(all_acts, axis=0)
        cv = stds / (means + 1e-8)
        score = means / (cv + 1e-8)
        persistent = np.argsort(score)[-(n_features - n_features // 2):]

        return np.concatenate([persistent, spiky])
    else:
        raise ValueError(f"Unknown mode: {mode}")


def plot_dynamics_comparison(
    data_item,
    predictors,
    feature_indices,
    output_path=None,
    use_pre_relu=True,
):
    """Plot actual vs predicted dynamics for a single sequence.

    Shows panels: Actual dynamics | Model 1 | Model 2 | ... per feature

    Args:
        data_item: Single data dictionary
        predictors: List of (name, predictor) tuples
        feature_indices: Indices of features to plot
        output_path: Path to save figure
        use_pre_relu: If True, use pre_relu activations
    """
    act_key = 'pre_relu' if use_pre_relu else 'sae_acts'
    actual = data_item[act_key]
    tokens = data_item['tokens']
    seq_len = len(tokens)

    # Get simulated trajectories for each model
    trajectories = [(name, pred.simulate_trajectory(data_item)) for name, pred in predictors]

    n_features = len(feature_indices)
    n_cols = 1 + len(predictors)  # Actual + N models

    # Create figure with n_cols columns (actual + models) and n_features rows
    fig, axes = plt.subplots(
        n_features, n_cols,
        figsize=(max(20, seq_len * 0.15 * n_cols / 3), 3 * n_features),
        sharex=True
    )

    if n_features == 1:
        axes = axes.reshape(1, -1)

    colors = plt.cm.tab10(np.linspace(0, 1, n_features))

    for i, (feat_idx, color) in enumerate(zip(feature_indices, colors)):
        # Actual dynamics (first column)
        ax_actual = axes[i, 0]
        ax_actual.plot(actual[:, feat_idx], color=color, linewidth=1.5)
        ax_actual.set_ylabel(f'Feature {feat_idx}', fontsize=9)
        if i == 0:
            ax_actual.set_title('Actual Dynamics', fontsize=11, fontweight='bold')
        ax_actual.grid(True, alpha=0.3)
        if use_pre_relu:
            ax_actual.axhline(y=0, color='black', linestyle='--', alpha=0.3)

        # Predicted dynamics (remaining columns)
        for col_idx, (model_name, traj) in enumerate(trajectories, start=1):
            ax = axes[i, col_idx]
            ax.plot(actual[:, feat_idx], color='gray', alpha=0.4, linewidth=1, label='Actual')
            ax.plot(traj[:, feat_idx], color=color, linewidth=1.5, label='Predicted')
            if i == 0:
                ax.set_title(model_name, fontsize=11, fontweight='bold')
                ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            if use_pre_relu:
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Add tokens along bottom
    for col in range(n_cols):
        ax = axes[-1, col]
        ax.set_xlabel('Token position', fontsize=10)

    # Add token labels to top row (after all plotting so ylim is stable)
    for col in range(n_cols):
        ax = axes[0, col]
        ax.autoscale_view()
        ylim = ax.get_ylim()
        for t, tok in enumerate(tokens):
            tok_display = tok.replace('\n', '\\n')
            ax.text(t, ylim[1], tok_display, fontsize=5, rotation=90,
                    ha='center', va='bottom', alpha=0.6, clip_on=False)

    # Add prompt as title
    prompt = data_item.get('prompt', '')
    if len(prompt) > 100:
        prompt = prompt[:97] + '...'
    fig.suptitle(f'Prompt: {prompt}', fontsize=10, y=1.02)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")

    plt.show()


def plot_multi_sequence_comparison(
    data,
    predictors,
    feature_indices,
    n_sequences=3,
    output_path=None,
    use_pre_relu=True,
):
    """Plot dynamics comparison across multiple sequences for selected features.

    Creates a grid where:
    - Rows are sequences
    - Columns show: Actual | Model 1 | Model 2 | ...
    - Each cell shows all selected features overlaid

    Args:
        data: List of data dictionaries
        predictors: List of (name, predictor) tuples
        feature_indices: Indices of features to plot
        n_sequences: Number of sequences to show
        output_path: Path to save figure
        use_pre_relu: If True, use pre_relu activations
    """
    act_key = 'pre_relu' if use_pre_relu else 'sae_acts'
    n_seqs = min(n_sequences, len(data))
    n_cols = 1 + len(predictors)  # Actual + N models

    fig, axes = plt.subplots(n_seqs, n_cols, figsize=(6 * n_cols, 4 * n_seqs), sharex=False)

    if n_seqs == 1:
        axes = axes.reshape(1, -1)

    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_indices)))

    for seq_idx in range(n_seqs):
        data_item = data[seq_idx]
        actual = data_item[act_key]
        tokens = data_item['tokens']
        seq_len = len(tokens)

        # Get predictions for all models
        trajectories = [(name, pred.simulate_trajectory(data_item)) for name, pred in predictors]
        r2_scores = [(name, compute_trajectory_r2(actual, traj, feature_indices))
                     for name, traj in trajectories]

        # Actual dynamics (first column)
        ax_actual = axes[seq_idx, 0]
        for feat_idx, color in zip(feature_indices, colors):
            ax_actual.plot(actual[:, feat_idx], color=color, alpha=0.7, linewidth=1)

        prompt = data_item.get('prompt', f'Seq {seq_idx}')
        if len(prompt) > 40:
            prompt = prompt[:37] + '...'
        ax_actual.set_ylabel(prompt, fontsize=8)

        if seq_idx == 0:
            ax_actual.set_title('Actual Dynamics', fontsize=11, fontweight='bold')
        ax_actual.grid(True, alpha=0.3)
        if use_pre_relu:
            ax_actual.axhline(y=0, color='black', linestyle='--', alpha=0.3)

        # Predicted dynamics (remaining columns)
        for col_idx, ((model_name, traj), (_, r2)) in enumerate(zip(trajectories, r2_scores), start=1):
            ax = axes[seq_idx, col_idx]
            for feat_idx, color in zip(feature_indices, colors):
                ax.plot(actual[:, feat_idx], color='gray', alpha=0.3, linewidth=0.5)
                ax.plot(traj[:, feat_idx], color=color, alpha=0.7, linewidth=1)

            if seq_idx == 0:
                ax.set_title(model_name, fontsize=11, fontweight='bold')
            ax.text(0.02, 0.98, f'R²={r2:.3f}', transform=ax.transAxes,
                   fontsize=9, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.grid(True, alpha=0.3)
            if use_pre_relu:
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

        # Add tokens along top of each row
        for col in range(n_cols):
            ax = axes[seq_idx, col]
            ax.autoscale_view()
            ylim = ax.get_ylim()
            for t, tok in enumerate(tokens):
                tok_display = tok.replace('\n', '\\n')
                ax.text(t, ylim[1], tok_display, fontsize=5, rotation=90,
                        ha='center', va='bottom', alpha=0.6, clip_on=False)

        # Add x-axis label to bottom row
        if seq_idx == n_seqs - 1:
            for col in range(n_cols):
                axes[seq_idx, col].set_xlabel('Token position', fontsize=10)

    # Legend
    legend_elements = [plt.Line2D([0], [0], color=c, label=f'Feature {idx}')
                       for idx, c in zip(feature_indices, colors)]
    fig.legend(handles=legend_elements, loc='upper center', ncol=min(5, len(feature_indices)),
               bbox_to_anchor=(0.5, 1.02), fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")

    plt.show()


def compute_trajectory_r2(actual, predicted, feature_indices=None):
    """Compute R^2 for trajectory prediction on selected features.

    Args:
        actual: (seq_len, n_features) actual activations
        predicted: (seq_len, n_features) predicted activations
        feature_indices: Indices of features to use (None = all)

    Returns:
        R^2 score averaged over selected features
    """
    if feature_indices is not None:
        actual_sub = actual[:, feature_indices]
        predicted_sub = predicted[:, feature_indices]
    else:
        actual_sub = actual
        predicted_sub = predicted

    ss_res = np.sum((actual_sub - predicted_sub) ** 2)
    ss_tot = np.sum((actual_sub - np.mean(actual_sub, axis=0)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - ss_res / ss_tot


def compute_per_sequence_r2(data, predictors, use_pre_relu=True):
    """Compute R^2 for each sequence for all models.

    Args:
        data: List of data dictionaries
        predictors: List of (name, predictor) tuples
        use_pre_relu: Whether to use pre_relu activations

    Returns:
        Dict mapping model name to array of R^2 values (one per sequence)
    """
    act_key = 'pre_relu' if use_pre_relu else 'sae_acts'

    # Initialize dict with empty lists for each model
    r2_dict = {name: [] for name, _ in predictors}

    for data_item in data:
        actual = data_item[act_key]

        # Get simulated trajectories and compute R^2 for each model
        for name, predictor in predictors:
            traj = predictor.simulate_trajectory(data_item)
            r2 = compute_trajectory_r2(actual, traj)
            r2_dict[name].append(r2)

    # Convert lists to arrays
    return {name: np.array(r2s) for name, r2s in r2_dict.items()}


def plot_r2_scatter(data, predictors, use_pre_relu=True, output_path=None):
    """Create scatter plot comparing R^2 between first two models across sequences.

    Args:
        data: List of data dictionaries
        predictors: List of (name, predictor) tuples (uses first 2 models)
        use_pre_relu: Whether to use pre_relu activations
        output_path: Path to save figure
    """
    if len(predictors) < 2:
        print("Error: Need at least 2 models for scatter plot")
        return

    r2_dict = compute_per_sequence_r2(data, predictors, use_pre_relu)

    model_names = list(r2_dict.keys())
    model1_name, model2_name = model_names[0], model_names[1]
    r2s_1 = r2_dict[model1_name]
    r2s_2 = r2_dict[model2_name]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(r2s_1, r2s_2, alpha=0.6, s=40, edgecolors='white', linewidth=0.5)

    # Add diagonal line (y=x)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel(f'{model1_name} R²', fontsize=12)
    ax.set_ylabel(f'{model2_name} R²', fontsize=12)
    ax.set_title('Model Comparison: R² per Sequence', fontsize=14)

    # Add statistics
    improvement = r2s_2 - r2s_1
    stats_text = (
        f'n = {len(r2s_1)} sequences\n'
        f'{model1_name} median R²: {np.median(r2s_1):.3f}\n'
        f'{model2_name} median R²: {np.median(r2s_2):.3f}\n'
        f'Median improvement: {np.median(improvement):.3f}\n'
        f'Frac improved: {np.mean(improvement > 0):.1%}'
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")

    plt.show()

    return r2_dict


def main():
    parser = argparse.ArgumentParser(description="Compare actual vs predicted dynamics")
    parser.add_argument("--cache-dir", type=Path, default=Path("/home/jffbrwn/orcd/pool/semantic-dynamics/.cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/jffbrwn/orcd/pool/semantic-dynamics/outputs"))
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file path (default: outputs/dynamics_comparison/)")
    parser.add_argument("--models", nargs='+',
                        default=['linear_token', 'linear_state'],
                        choices=['linear_token', 'linear_state', 'rnn_token', 'rnn_state'],
                        help="Models to compare (default: linear_token linear_state)")
    parser.add_argument("--n-sequences", type=int, default=3)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--sequence-idx", type=int, default=None,
                        help="Plot a single specific sequence by index")
    parser.add_argument("--mode", choices=['spiky', 'persistent', 'both'], default='spiky',
                        help="Feature selection mode")
    parser.add_argument("--pre-relu", action="store_true", default=True,
                        help="Use pre-ReLU activations (default: True)")
    parser.add_argument("--no-pre-relu", action="store_false", dest="pre_relu",
                        help="Use post-ReLU SAE activations")
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Ridge regularization parameter (for linear models)")
    parser.add_argument("--scatter", action="store_true",
                        help="Plot R² scatter comparing first 2 models across all sequences")
    parser.add_argument("--test-data", action="store_true",
                        help="Use test data instead of train data")
    args = parser.parse_args()

    # Load data
    data_file = "test_data.pkl" if args.test_data else "train_data.pkl"
    print(f"Loading cached data from {data_file}...")
    data = load_data(args.cache_dir, use_pre_relu=args.pre_relu, data_file=data_file)
    print(f"Loaded {len(data)} sequences")

    # Load embedding matrix
    embed_path = args.cache_dir / "embed_matrix.npy"
    if not embed_path.exists():
        print("Error: Embedding matrix not found. Run main.py first to generate it.")
        return
    embed_matrix = np.load(embed_path)

    # Load requested predictors
    print(f"Loading models: {args.models}")
    predictors = []

    # Use _prerelu suffix if using pre-relu data
    suffix = "_prerelu" if args.pre_relu else ""

    model_paths = {
        'linear_token': args.output_dir / f"token_predictor{suffix}.pkl",
        'linear_state': args.output_dir / f"state_token_predictor{suffix}.pkl",
        'rnn_token': args.output_dir / f"rnn_token_predictor{suffix}.pkl",
        'rnn_state': args.output_dir / f"rnn_state_token_predictor{suffix}.pkl",
    }

    model_display_names = {
        'linear_token': 'Linear Token-Only (B*u)',
        'linear_state': 'Linear State+Token (A*x + B*u)',
        'rnn_token': 'RNN Token-Only',
        'rnn_state': 'RNN State+Token',
    }

    for model_name in args.models:
        model_path = model_paths[model_name]
        if not model_path.exists():
            print(f"Error: Model '{model_name}' not found at {model_path}")
            print(f"Please train the model first.")
            if 'linear' in model_name:
                print("Run main.py to train linear models.")
            else:
                print("Run train_rnn.py to train RNN models.")
            return

        print(f"  Loading {model_name} from {model_path}")
        predictor = load_predictor(model_path)
        display_name = model_display_names[model_name]
        predictors.append((display_name, predictor))

    # Output directory
    output_dir = args.output_dir / "dynamics_comparison"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Scatter plot mode
    if args.scatter:
        print("Computing R² for all sequences...")
        if args.output is None:
            models_str = "_".join([m.replace('_', '') for m in args.models[:2]])
            output_path = output_dir / f"r2_scatter_{models_str}_test.png" if args.test_data else output_dir / f"r2_scatter_{models_str}_train.png"
        else:
            output_path = args.output

        plot_r2_scatter(
            data,
            predictors,
            use_pre_relu=args.pre_relu,
            output_path=output_path,
        )
        return

    # Select features for trajectory plots
    feature_indices = select_features(data, n_features=args.n_features,
                                       mode=args.mode, use_pre_relu=args.pre_relu)
    print(f"Selected features: {feature_indices}")

    # Output path
    if args.output is None:
        models_str = "_".join([m.replace('_', '') for m in args.models])
        if args.sequence_idx is not None:
            output_path = output_dir / f"seq_{args.sequence_idx}_mode_{args.mode}_{models_str}.png"
        else:
            output_path = output_dir / f"multi_seq_mode_{args.mode}_{models_str}.png"
    else:
        output_path = args.output
        output_path.parent.mkdir(exist_ok=True, parents=True)

    # Plot
    if args.sequence_idx is not None:
        if args.sequence_idx >= len(data):
            print(f"Error: sequence_idx {args.sequence_idx} out of range (0-{len(data)-1})")
            return

        print(f"Plotting single sequence {args.sequence_idx}...")
        plot_dynamics_comparison(
            data[args.sequence_idx],
            predictors,
            feature_indices,
            output_path=output_path,
            use_pre_relu=args.pre_relu,
        )
    else:
        print(f"Plotting {args.n_sequences} sequences...")
        plot_multi_sequence_comparison(
            data,
            predictors,
            feature_indices,
            n_sequences=args.n_sequences,
            output_path=output_path,
            use_pre_relu=args.pre_relu,
        )


if __name__ == "__main__":
    main()
