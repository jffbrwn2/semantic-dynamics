"""Train LFADS on SAE feature activation data.

Example usage:
    # Basic training (uses full sequences by default)
    python -m feature_dynamics.lfads.train \
        --data-dir /path/to/cache \
        --output-dir /path/to/outputs/lfads \
        --batch-size 32 \
        --epochs 100

    # With token embeddings as external inputs
    python -m feature_dynamics.lfads.train \
        --data-dir /path/to/cache \
        --output-dir /path/to/outputs/lfads_tokens \
        --use-token-embeddings \
        --embed-project-dim 64 \
        --epochs 100

    # With fixed-length windowing (optional)
    python -m feature_dynamics.lfads.train \
        --data-dir /path/to/cache \
        --output-dir /path/to/outputs/lfads \
        --seq-len 64 \
        --stride 32 \
        --epochs 100
"""

import argparse
from pathlib import Path
import torch
import numpy as np
import pickle
import json
from torch.utils.data import DataLoader, random_split

from feature_dynamics.config import _get_default_cache_dir, _get_default_output_dir, get_timestamped_dir
from feature_dynamics.lfads.model import LFADS, LFADSConfig
from feature_dynamics.lfads.training import (
    SAEActivationDataset,
    EmbeddingLookup,
    LFADSTrainer,
    TrainingConfig,
    load_sae_data,
    load_embedding_matrix,
    compute_reconstruction_metrics,
    extract_all_latents,
    collate_with_tokens,
    collate_variable_length,
)


def main():
    parser = argparse.ArgumentParser(description="Train LFADS on SAE activations")

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_get_default_cache_dir(),
        help="Directory containing SAE activation data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_get_default_output_dir() / "lfads",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--use-pre-relu",
        action="store_true",
        help="Use pre-ReLU activations instead of SAE activations",
    )

    # Token embedding arguments
    parser.add_argument(
        "--use-token-embeddings",
        action="store_true",
        help="Use token embeddings as external inputs to the generator",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/gemma-3-27b-it",
        help="Model name for loading embeddings (default: google/gemma-3-27b-it)",
    )
    parser.add_argument(
        "--embed-project-dim",
        type=int,
        default=None,
        help="Project embeddings to this dimension (default: use full embed dim)",
    )
    parser.add_argument(
        "--trainable-embed-proj",
        action="store_true",
        help="Make the embedding projection layer trainable",
    )

    # Data processing
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length for training (default: None = use full sequences)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride for extracting subsequences (only used if --seq-len is set)",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=None,
        help="Number of features to use (default: all)",
    )
    parser.add_argument(
        "--feature-selection",
        type=str,
        choices=["variance", "random", "all"],
        default="all",
        help="Feature selection method",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Z-score normalize features (recommended for Gaussian likelihood)",
    )
    parser.add_argument(
        "--log-transform",
        action="store_true",
        help="Apply log(1+x) transform to activations (helps compress large values)",
    )

    # Model arguments
    parser.add_argument("--enc-dim", type=int, default=128, help="Encoder hidden dim")
    parser.add_argument("--gen-dim", type=int, default=128, help="Generator hidden dim")
    parser.add_argument("--fac-dim", type=int, default=32, help="Factor dim")
    parser.add_argument("--ic-dim", type=int, default=64, help="Initial condition dim")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--likelihood",
        type=str,
        choices=["poisson", "gaussian", "zero_inflated_poisson"],
        default="poisson",
        help="Likelihood type for reconstruction",
    )
    parser.add_argument(
        "--use-controller",
        action="store_true",
        help="Use controller network for inferred inputs",
    )
    parser.add_argument("--con-dim", type=int, default=64, help="Controller hidden dim")
    parser.add_argument("--ci-dim", type=int, default=1, help="Controller input dim")
    parser.add_argument(
        "--substeps",
        type=int,
        default=1,
        help="Substeps per token for smoother dynamics (default: 1). "
             "Higher values let dynamics evolve smoothly between tokens.",
    )

    # Training arguments
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--kl-warmup", type=int, default=20, help="KL warmup epochs")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Gradient clipping")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping",
    )
    parser.add_argument("--l2-gen", type=float, default=0.0, help="L2 reg on generator")
    parser.add_argument("--kl-ic-weight", type=float, default=1.0, help="KL IC weight")

    # Other
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Don't create timestamped subfolder for outputs",
    )

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create timestamped output directory for this run
    if not args.no_timestamp:
        args.output_dir = get_timestamped_dir(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Save config
    with open(args.output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # =========================================================================
    # Load embeddings (if using token inputs)
    # =========================================================================
    embedding_lookup = None
    ext_input_dim = 0

    if args.use_token_embeddings:
        print("\n" + "=" * 60)
        print("Loading token embeddings...")
        print("=" * 60)

        embed_matrix = load_embedding_matrix(
            args.model_name,
            args.data_dir,
            device=args.device,
        )
        print(f"Embedding matrix shape: {embed_matrix.shape}")

        # Determine external input dimension
        if args.embed_project_dim is not None:
            ext_input_dim = args.embed_project_dim
            print(f"Projecting embeddings: {embed_matrix.shape[1]} -> {ext_input_dim}")
        else:
            ext_input_dim = embed_matrix.shape[1]

        embedding_lookup = EmbeddingLookup(
            embed_matrix,
            trainable=False,  # Embeddings themselves are frozen
            project_dim=args.embed_project_dim,
        )
        print(f"External input dim: {ext_input_dim}")

    # =========================================================================
    # Load data
    # =========================================================================
    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)

    train_data_path = args.data_dir / "train_data_full.pkl"
    test_data_path = args.data_dir / "test_data_full.pkl"
    feature_indices_path = args.data_dir / "feature_indices.pkl"

    # Try full data first, fall back to subset
    if not train_data_path.exists():
        train_data_path = args.data_dir / "train_data.pkl"
        test_data_path = args.data_dir / "test_data.pkl"

    # Fall back to trials dataset format (from collect_trials.py)
    if not train_data_path.exists():
        trials_path = args.data_dir / "trials_dataset.pkl"
        if trials_path.exists():
            train_data_path = trials_path
            test_data_path = trials_path  # Will split later
            print("Using trials dataset format (will split into train/test)")
        else:
            raise FileNotFoundError(f"No training data found in {args.data_dir}")

    train_data, feature_indices = load_sae_data(
        train_data_path,
        use_pre_relu=args.use_pre_relu,
        feature_indices_path=feature_indices_path,
    )
    test_data, _ = load_sae_data(
        test_data_path,
        use_pre_relu=args.use_pre_relu,
        feature_indices_path=feature_indices_path,
    )

    print(f"Loaded {len(train_data)} training sequences")
    print(f"Loaded {len(test_data)} test sequences")

    # Check if token_ids are available
    has_token_ids = 'token_ids' in train_data[0]
    if args.use_token_embeddings and not has_token_ids:
        raise ValueError("--use-token-embeddings requires token_ids in the dataset")

    # Determine number of features
    act_key = 'pre_relu' if args.use_pre_relu else 'sae_acts'
    total_features = train_data[0][act_key].shape[-1]
    print(f"Total features in data: {total_features}")

    # Feature selection
    if args.n_features is not None and args.n_features < total_features:
        if args.feature_selection == "variance":
            # Select by variance
            all_acts = np.concatenate([d[act_key] for d in train_data], axis=0)
            variances = np.var(all_acts, axis=0)
            selected_indices = np.argsort(variances)[-args.n_features:]
        elif args.feature_selection == "random":
            selected_indices = np.random.choice(total_features, args.n_features, replace=False)
        else:
            selected_indices = np.arange(args.n_features)
        n_features = args.n_features
        print(f"Selected {n_features} features using {args.feature_selection}")
    else:
        selected_indices = None
        n_features = total_features

    # Create datasets
    if args.seq_len is None:
        print("\nCreating datasets with full sequences (no windowing)...")
    else:
        stride = args.stride if args.stride is not None else args.seq_len // 2
        print(f"\nCreating datasets with seq_len={args.seq_len}, stride={stride}...")
    if args.normalize:
        print("  Normalizing features (z-score)")
    if args.log_transform:
        print("  Applying log(1+x) transform")

    # Compute stride (only relevant if seq_len is set)
    stride = args.stride if args.stride is not None else (args.seq_len // 2 if args.seq_len else 1)

    train_dataset = SAEActivationDataset(
        train_data,
        use_pre_relu=args.use_pre_relu,
        seq_len=args.seq_len,
        stride=stride,
        feature_indices=selected_indices,
        return_token_ids=args.use_token_embeddings,
        normalize=args.normalize,
        log_transform=args.log_transform,
    )

    # Use train normalization stats for test set
    norm_stats = train_dataset.get_normalization_stats()

    # Save normalization stats for inference
    if args.normalize and norm_stats is not None:
        np.savez(
            args.output_dir / "normalization_stats.npz",
            mean=norm_stats['mean'],
            std=norm_stats['std'],
        )
        print(f"Saved normalization stats to {args.output_dir / 'normalization_stats.npz'}")

    # Save feature indices if using subset selection
    if selected_indices is not None:
        with open(args.output_dir / "feature_indices.pkl", 'wb') as f:
            pickle.dump({'indices': selected_indices}, f)
        print(f"Saved feature indices to {args.output_dir / 'feature_indices.pkl'}")

    test_dataset = SAEActivationDataset(
        test_data,
        use_pre_relu=args.use_pre_relu,
        seq_len=args.seq_len,
        stride=stride,
        feature_indices=selected_indices,
        return_token_ids=args.use_token_embeddings,
        normalize=args.normalize,
        log_transform=args.log_transform,
        normalization_stats=norm_stats,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Features per sample: {train_dataset.n_features}")
    if args.use_token_embeddings:
        print(f"Token IDs included: {train_dataset.has_token_ids}")

    # Split training into train/val
    val_size = int(len(train_dataset) * args.val_split)
    train_size = len(train_dataset) - val_size

    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"Train/Val split: {train_size}/{val_size}")

    # Data loaders - use variable length collate if no seq_len specified
    if args.seq_len is None:
        collate_fn = collate_variable_length
    elif args.use_token_embeddings:
        collate_fn = collate_with_tokens
    else:
        collate_fn = None

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if args.device == "cuda" else False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # =========================================================================
    # Create model
    # =========================================================================
    print("\n" + "=" * 60)
    print("Creating LFADS model...")
    print("=" * 60)

    model_config = LFADSConfig(
        n_features=train_dataset.n_features,
        enc_dim=args.enc_dim,
        gen_dim=args.gen_dim,
        fac_dim=args.fac_dim,
        ic_dim=args.ic_dim,
        use_controller=args.use_controller,
        con_dim=args.con_dim,
        ci_dim=args.ci_dim,
        ext_input_dim=ext_input_dim,  # Token embeddings dimension
        substeps_per_token=args.substeps,
        dropout=args.dropout,
        likelihood=args.likelihood,
        kl_ic_weight=args.kl_ic_weight,
        l2_gen_scale=args.l2_gen,
    )

    model = LFADS(model_config)
    print(model)
    if args.substeps > 1:
        print(f"\nUsing {args.substeps} substeps per token for smoother dynamics")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")

    if embedding_lookup is not None and embedding_lookup.projection is not None:
        n_embed_params = sum(p.numel() for p in embedding_lookup.projection.parameters())
        print(f"Embedding projection parameters: {n_embed_params:,}")

    # Save model config
    with open(args.output_dir / "model_config.json", "w") as f:
        json.dump(vars(model_config), f, indent=2)

    # =========================================================================
    # Train
    # =========================================================================
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    train_config = TrainingConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        grad_clip=args.grad_clip,
        kl_warmup_epochs=args.kl_warmup,
        patience=args.patience,
        early_stopping=not args.no_early_stopping,
    )

    trainer = LFADSTrainer(
        model=model,
        train_config=train_config,
        device=device,
        output_dir=args.output_dir,
        embedding_lookup=embedding_lookup,
    )

    history = trainer.train(train_loader, val_loader)

    # =========================================================================
    # Evaluate
    # =========================================================================
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    # Load best model
    best_path = args.output_dir / "best.pt"
    if best_path.exists():
        trainer.load_checkpoint(best_path)
        print("Loaded best checkpoint")

    # Compute metrics
    metrics = compute_reconstruction_metrics(
        model, test_loader, device, embedding_lookup=embedding_lookup
    )

    print(f"\nTest set reconstruction metrics:")
    print(f"  Mean R²: {metrics['r2_per_feature'].mean():.4f}")
    print(f"  Median R²: {np.median(metrics['r2_per_feature']):.4f}")
    print(f"  Mean correlation: {metrics['correlation_per_feature'].mean():.4f}")
    print(f"  Fraction R² > 0.1: {(metrics['r2_per_feature'] > 0.1).mean():.2%}")

    # Save metrics
    np.savez(
        args.output_dir / "test_metrics.npz",
        r2_per_feature=metrics["r2_per_feature"],
        mse_per_feature=metrics["mse_per_feature"],
        correlation_per_feature=metrics["correlation_per_feature"],
    )

    # Extract latents
    print("\nExtracting latent factors...")
    latents = extract_all_latents(
        model, test_loader, device, embedding_lookup=embedding_lookup
    )

    save_dict = {
        "factors": latents["factors"],
        "ic": latents["ic"],
        "gen_states": latents["gen_states"],
    }
    if "substep_factors" in latents:
        save_dict["substep_factors"] = latents["substep_factors"]

    np.savez(args.output_dir / "test_latents.npz", **save_dict)

    print(f"\nLatent factor shape: {latents['factors'].shape}")
    print(f"Initial conditions shape: {latents['ic'].shape}")
    if "substep_factors" in latents:
        print(f"Substep factors shape: {latents['substep_factors'].shape}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print(f"\nModel saved to: {args.output_dir}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Final test R² (mean): {metrics['r2_per_feature'].mean():.4f}")

    if args.use_token_embeddings:
        print(f"\nUsed token embeddings as external inputs (dim={ext_input_dim})")

    # Compute explained variance ratio of latent factors
    factors_flat = latents["factors"].reshape(-1, latents["factors"].shape[-1])
    factor_vars = np.var(factors_flat, axis=0)
    factor_var_ratio = factor_vars / factor_vars.sum()

    print(f"\nLatent factor variance ratios (top 5):")
    sorted_idx = np.argsort(factor_var_ratio)[::-1]
    for i in range(min(5, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"  Factor {idx}: {factor_var_ratio[idx]:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
