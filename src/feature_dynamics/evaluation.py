"""Evaluation metrics for feature predictors."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import r2_score
import pandas as pd


def get_act_key(use_pre_relu: bool) -> str:
    """Get the activation key based on use_pre_relu flag."""
    return 'pre_relu' if use_pre_relu else 'sae_acts'


def compute_r2_per_feature(
    y_true: List[np.ndarray],
    y_pred: List[np.ndarray],
) -> np.ndarray:
    """Compute R^2 score per feature.

    Args:
        y_true: List of true target arrays (seq_len, n_features)
        y_pred: List of predicted arrays (seq_len, n_features)

    Returns:
        Array of R^2 scores per feature (n_features,)
    """
    # Concatenate all sequences
    y_true_cat = np.concatenate(y_true, axis=0)
    y_pred_cat = np.concatenate(y_pred, axis=0)

    n_features = y_true_cat.shape[1]
    r2_scores = np.zeros(n_features)

    for i in range(n_features):
        r2_scores[i] = r2_score(y_true_cat[:, i], y_pred_cat[:, i])

    return r2_scores


def compute_aggregate_metrics(r2_scores: np.ndarray) -> Dict[str, float]:
    """Compute aggregate metrics from per-feature R^2 scores.

    Args:
        r2_scores: Array of R^2 scores per feature

    Returns:
        Dictionary of aggregate metrics
    """
    return {
        'mean_r2': float(np.mean(r2_scores)),
        'median_r2': float(np.median(r2_scores)),
        'std_r2': float(np.std(r2_scores)),
        'min_r2': float(np.min(r2_scores)),
        'max_r2': float(np.max(r2_scores)),
        'frac_positive': float(np.mean(r2_scores > 0)),
        'frac_good': float(np.mean(r2_scores > 0.1)),  # > 10% variance explained
    }


def compare_predictors(
    token_r2: np.ndarray,
    state_token_r2: np.ndarray,
    epsilon: float = 0.05,
) -> Dict[str, float]:
    """Compare two predictors.

    Args:
        token_r2: R^2 scores for token-only model
        state_token_r2: R^2 scores for state+token model
        epsilon: Threshold for "meaningful improvement"

    Returns:
        Dictionary of comparison metrics
    """
    improvement = state_token_r2 - token_r2

    return {
        'mean_improvement': float(np.mean(improvement)),
        'median_improvement': float(np.median(improvement)),
        'frac_improved': float(np.mean(improvement > 0)),
        f'frac_improved_by_{epsilon}': float(np.mean(improvement > epsilon)),
        'max_improvement': float(np.max(improvement)),
        'min_improvement': float(np.min(improvement)),
    }


def evaluate_predictors(
    dataset: List[Dict],
    token_predictor,
    state_token_predictor,
    epsilon: float = 0.05,
    use_pre_relu: Optional[bool] = None,
) -> Dict:
    """Evaluate both predictors on a dataset.

    Args:
        dataset: Test dataset
        token_predictor: Fitted TokenOnlyPredictor
        state_token_predictor: Fitted StateTokenPredictor
        epsilon: Threshold for meaningful improvement
        use_pre_relu: If True, use pre_relu activations. If None, infer from predictor.

    Returns:
        Dictionary containing:
            - token_only_r2: Per-feature R^2 for token-only model
            - state_token_r2: Per-feature R^2 for state+token model
            - token_only_metrics: Aggregate metrics for token-only
            - state_token_metrics: Aggregate metrics for state+token
            - comparison: Comparison metrics
    """
    # Determine activation key
    if use_pre_relu is None:
        use_pre_relu = getattr(token_predictor, 'use_pre_relu', True)
    act_key = get_act_key(use_pre_relu)

    # Extract ground truth
    y_true = [data[act_key][1:] for data in dataset]

    # Token-only predictions
    token_preds = token_predictor.predict(dataset)
    token_r2 = compute_r2_per_feature(y_true, token_preds)
    token_metrics = compute_aggregate_metrics(token_r2)

    # State+token predictions
    state_token_preds = state_token_predictor.predict(dataset)
    state_token_r2 = compute_r2_per_feature(y_true, state_token_preds)
    state_token_metrics = compute_aggregate_metrics(state_token_r2)

    # Comparison
    comparison = compare_predictors(token_r2, state_token_r2, epsilon)

    return {
        'token_only_r2': token_r2,
        'state_token_r2': state_token_r2,
        'token_only_metrics': token_metrics,
        'state_token_metrics': state_token_metrics,
        'comparison': comparison,
    }


def print_evaluation_results(results: Dict):
    """Print evaluation results in a readable format.

    Args:
        results: Results dictionary from evaluate_predictors
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    print("\nToken-Only Model:")
    print("-" * 40)
    for key, value in results['token_only_metrics'].items():
        print(f"  {key:20s}: {value:.4f}")

    print("\nState+Token Model:")
    print("-" * 40)
    for key, value in results['state_token_metrics'].items():
        print(f"  {key:20s}: {value:.4f}")

    print("\nComparison (State+Token vs Token-Only):")
    print("-" * 40)
    for key, value in results['comparison'].items():
        print(f"  {key:30s}: {value:.4f}")

    print("\n" + "="*60)


def analyze_by_style(
    dataset: List[Dict],
    token_predictor,
    state_token_predictor,
    use_pre_relu: Optional[bool] = None,
) -> pd.DataFrame:
    """Analyze predictor performance by prompt style.

    Args:
        dataset: Test dataset
        token_predictor: Fitted TokenOnlyPredictor
        state_token_predictor: Fitted StateTokenPredictor
        use_pre_relu: If True, use pre_relu activations. If None, infer from predictor.

    Returns:
        DataFrame with per-style metrics
    """
    # Determine activation key
    if use_pre_relu is None:
        use_pre_relu = getattr(token_predictor, 'use_pre_relu', True)
    act_key = get_act_key(use_pre_relu)

    styles = set(d['style'] for d in dataset)
    results = []

    for style in styles:
        style_data = [d for d in dataset if d['style'] == style]

        if len(style_data) == 0:
            continue

        # Evaluate on this style
        y_true = [data[act_key][1:] for data in style_data]
        token_preds = token_predictor.predict(style_data)
        state_token_preds = state_token_predictor.predict(style_data)

        token_r2 = compute_r2_per_feature(y_true, token_preds)
        state_token_r2 = compute_r2_per_feature(y_true, state_token_preds)

        results.append({
            'style': style,
            'n_samples': len(style_data),
            'token_median_r2': np.median(token_r2),
            'state_token_median_r2': np.median(state_token_r2),
            'median_improvement': np.median(state_token_r2 - token_r2),
            'frac_improved': np.mean(state_token_r2 > token_r2),
        })

    return pd.DataFrame(results)


def plot_r2_comparison(
    token_r2: np.ndarray,
    state_token_r2: np.ndarray,
    output_path: str = None,
):
    """Plot R^2 comparison scatter plot.

    Args:
        token_r2: R^2 scores for token-only model
        state_token_r2: R^2 scores for state+token model
        output_path: Path to save figure (optional)
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(token_r2, state_token_r2, alpha=0.5, s=10)
        ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.3, label='y=x')

        ax.set_xlabel('Token-Only R²')
        ax.set_ylabel('State+Token R²')
        ax.set_title('Per-Feature R² Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Equal aspect ratio
        ax.set_aspect('equal')

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {output_path}")

        plt.close()

    except ImportError:
        print("matplotlib not available, skipping plot")
