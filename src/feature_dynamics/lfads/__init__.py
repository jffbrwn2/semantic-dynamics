"""LFADS (Latent Factor Analysis via Dynamical Systems) for SAE feature dynamics.

This package implements LFADS to model sparse feature activation time series from
Sparse Autoencoders as neural population dynamics.

Reference:
    Pandarinath et al. (2018) "Inferring single-trial neural population dynamics
    using sequential auto-encoders" Nature Methods.
"""

from feature_dynamics.lfads.model import (
    LFADS,
    LFADSConfig,
    GRUEncoder,
    Generator,
    Controller,
    extract_latent_trajectories,
)

from feature_dynamics.lfads.training import (
    SAEActivationDataset,
    EmbeddingLookup,
    LFADSTrainer,
    TrainingConfig,
    load_sae_data,
    load_embedding_matrix,
    compute_reconstruction_metrics,
    extract_all_latents,
    collate_variable_length,
    collate_with_tokens,
)

__all__ = [
    # Model
    "LFADS",
    "LFADSConfig",
    "GRUEncoder",
    "Generator",
    "Controller",
    "extract_latent_trajectories",
    # Training
    "SAEActivationDataset",
    "EmbeddingLookup",
    "LFADSTrainer",
    "TrainingConfig",
    "load_sae_data",
    "load_embedding_matrix",
    "compute_reconstruction_metrics",
    "extract_all_latents",
    "collate_variable_length",
    "collate_with_tokens",
]
