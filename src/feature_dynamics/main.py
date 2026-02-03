"""Main entry point for SAE feature dynamics analysis."""

import argparse
import pickle
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModel

from config import Config
from prompts import PromptGenerator
from data_collection import DataCollector
from predictors import TokenOnlyPredictor, StateTokenPredictor, save_predictor, load_predictor


def load_embedding_matrix(model_name: str, cache_dir: Path, device: str = "cpu") -> np.ndarray:
    """Load token embedding matrix from model.

    Args:
        model_name: HuggingFace model name
        cache_dir: Directory to cache the embedding matrix
        device: Device for loading model

    Returns:
        (vocab_size, embed_dim) numpy array of token embeddings
    """
    embed_path = cache_dir / "embed_matrix.npy"

    if embed_path.exists():
        print(f"  Loading cached embeddings from {embed_path}")
        return np.load(embed_path)

    print(f"  Loading embeddings from {model_name}...")
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    embed_matrix = model.get_input_embeddings().weight.detach().float().numpy()
    del model  # Free memory

    # Cache for future use
    np.save(embed_path, embed_matrix)
    print(f"  Cached embeddings to {embed_path}")

    return embed_matrix


from evaluation import (
    evaluate_predictors,
    print_evaluation_results,
    analyze_by_style,
    plot_r2_comparison,
)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="SAE Feature Dynamics Analysis"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=31,
        help="Layer index to analyze (default: 20)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=512,
        help="Number of top features to select (default: 512)"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=200,
        help="Total number of prompts (default: 200)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="Ridge regularization parameter (default: 10.0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (cpu/cuda/mps, default: cpu)"
    )
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip data collection (use cached data)"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache"),
        help="Cache directory for data (default: .cache)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory (default: outputs)"
    )

    args = parser.parse_args()

    # Initialize config
    config = Config(
        layer_idx=args.layer,
        top_k_features=args.top_k,
        num_prompts=args.num_prompts,
        ridge_alpha=args.alpha,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )

    print("="*60)
    print("SAE Feature Dynamics Analysis")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Layer: {config.layer_idx}")
    print(f"Top-K features: {config.top_k_features}")
    print(f"Number of prompts: {config.num_prompts}")
    print(f"Device: {args.device}")
    print("="*60 + "\n")

    # Step 1: Generate prompts
    print("Step 1: Generating prompt corpus...")
    generator = PromptGenerator(seed=config.random_seed)
    corpus = generator.generate_corpus(
        num_prompts=config.num_prompts,
        num_paraphrases=config.num_paraphrases
    )

    # Split into train/test
    train_corpus, test_corpus = generator.split_corpus(
        corpus,
        test_size=config.test_size,
        seed=config.random_seed
    )

    # Count prompts
    train_count = sum(len(prompts) for prompts in train_corpus.values())
    test_count = sum(len(prompts) for prompts in test_corpus.values())
    print(f"  Train prompts: {train_count}")
    print(f"  Test prompts: {test_count}")

    # Data paths
    train_data_path = config.cache_dir / "train_data.pkl"
    test_data_path = config.cache_dir / "test_data.pkl"
    feature_indices_path = config.cache_dir / "feature_indices.pkl"

    # Step 2: Collect data (only generate what's missing)
    collector = None

    def apply_feature_subset(dataset, indices):
        """Apply feature subset to dataset if not already applied."""
        n_features = dataset[0]['sae_acts'].shape[1]
        if n_features == len(indices):
            return dataset  # Already subset
        print(f"  Subsetting features: {n_features} -> {len(indices)}")
        return [
            {**d, 'sae_acts': d['sae_acts'][:, indices]}
            for d in dataset
        ]

    # Load or collect training data
    if args.skip_collection and train_data_path.exists() and feature_indices_path.exists():
        print("\nStep 2a: Loading cached training data...")
        with open(train_data_path, 'rb') as f:
            train_dataset = pickle.load(f)
        with open(feature_indices_path, 'rb') as f:
            feature_info = pickle.load(f)
            top_k_indices = feature_info['indices']
        train_dataset = apply_feature_subset(train_dataset, top_k_indices)
        print(f"  Loaded {len(train_dataset)} training sequences")
        print(f"  Using {len(top_k_indices)} features")
    else:
        print("\nStep 2a: Collecting training data...")
        print("  (This may take a while...)")
        collector = DataCollector(config, device=args.device)

        train_dataset = collector.collect_corpus_data(
            train_corpus,
            output_path=train_data_path
        )

        print(f"\nSelecting top-{config.top_k_features} features by variance...")
        top_k_indices, feature_vars = collector.select_top_features(
            train_dataset,
            top_k=config.top_k_features
        )

        with open(feature_indices_path, 'wb') as f:
            pickle.dump({
                'indices': top_k_indices,
                'variances': feature_vars
            }, f)

        train_dataset = collector.prepare_feature_subset(train_dataset, top_k_indices)
        with open(train_data_path, 'wb') as f:
            pickle.dump(train_dataset, f)
        print(f"  Collected {len(train_dataset)} training sequences")

    # Load or collect test data
    if args.skip_collection and test_data_path.exists():
        print("\nStep 2b: Loading cached test data...")
        with open(test_data_path, 'rb') as f:
            test_dataset = pickle.load(f)
        test_dataset = apply_feature_subset(test_dataset, top_k_indices)
        print(f"  Loaded {len(test_dataset)} test sequences")
    else:
        print("\nStep 2b: Collecting test data...")
        if collector is None:
            collector = DataCollector(config, device=args.device)

        test_dataset = collector.collect_corpus_data(
            test_corpus,
            output_path=test_data_path
        )
        test_dataset = collector.prepare_feature_subset(test_dataset, top_k_indices)
        with open(test_data_path, 'wb') as f:
            pickle.dump(test_dataset, f)
        print(f"  Collected {len(test_dataset)} test sequences")

    # Step 3: Fit predictors
    print("\nStep 3: Fitting predictors...")

    # Load embedding matrix
    print("  Loading token embeddings...")
    embed_matrix = load_embedding_matrix(config.model_name, config.cache_dir, args.device)
    print(f"  Embedding matrix shape: {embed_matrix.shape}")

    # Token-only model
    print("  Fitting token-only baseline...")
    token_predictor = TokenOnlyPredictor(
        alpha=config.ridge_alpha,
        n_features=config.top_k_features,
    )

    token_predictor.fit(train_dataset, embed_matrix, per_feature=config.fit_per_feature)
    save_predictor(token_predictor, config.output_dir / "token_predictor.pkl")

    # State+token model
    print("  Fitting state+token model...")
    state_token_predictor = StateTokenPredictor(
        alpha=config.ridge_alpha,
        n_features=config.top_k_features,
    )
    state_token_predictor.fit(train_dataset, embed_matrix, per_feature=config.fit_per_feature)
    save_predictor(state_token_predictor, config.output_dir / "state_token_predictor.pkl")

    # Step 4: Evaluate
    print("\nStep 4: Evaluating on held-out test set...")
    results = evaluate_predictors(
        test_dataset,
        token_predictor,
        state_token_predictor,
        epsilon=0.05,
    )

    # Print results
    print_evaluation_results(results)

    # Save results
    results_path = config.output_dir / "results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {results_path}")

    # Analyze by style
    print("\nPer-style analysis:")
    style_df = analyze_by_style(test_dataset, token_predictor, state_token_predictor)
    print(style_df.to_string(index=False))

    style_csv_path = config.output_dir / "results_by_style.csv"
    style_df.to_csv(style_csv_path, index=False)
    print(f"\nSaved per-style results to {style_csv_path}")

    # Plot comparison
    plot_path = config.output_dir / "r2_comparison.png"
    plot_r2_comparison(
        results['token_only_r2'],
        results['state_token_r2'],
        output_path=plot_path,
    )

    # Summary
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    comparison = results['comparison']
    state_metrics = results['state_token_metrics']

    print(f"\nMedian R² improvement: {comparison['median_improvement']:.4f}")
    print(f"Fraction improved: {comparison['frac_improved']:.1%}")
    print(f"Fraction improved by >0.05: {comparison['frac_improved_by_0.05']:.1%}")

    print(f"\nState+Token model median R²: {state_metrics['median_r2']:.4f}")
    print(f"State+Token model features with R² > 0.1: {state_metrics['frac_good']:.1%}")

    if comparison['median_improvement'] > 0.05 and comparison['frac_improved_by_0.05'] > 0.3:
        print("\n✓ EVIDENCE OF PERSISTENT INTERNAL MODES")
        print("  Adding state significantly improves prediction across styles.")
    else:
        print("\n✗ LIMITED EVIDENCE OF PERSISTENT MODES")
        print("  Trajectories appear mostly driven by immediate token forcing.")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
