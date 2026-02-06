"""Training utilities for LFADS on SAE feature activation data."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
from pathlib import Path
import pickle
from tqdm import tqdm
from dataclasses import dataclass
import json

from feature_dynamics.lfads.model import LFADS, LFADSConfig


class SAEActivationDataset(Dataset):
    """PyTorch Dataset for SAE activation time series with optional token IDs."""

    def __init__(
        self,
        data: List[Dict],
        use_pre_relu: bool = False,
        seq_len: Optional[int] = None,
        stride: int = 1,
        feature_indices: Optional[np.ndarray] = None,
        return_token_ids: bool = False,
        normalize: bool = False,
        log_transform: bool = False,
        normalization_stats: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Args:
            data: List of dictionaries with 'sae_acts' (and optionally 'pre_relu', 'token_ids')
            use_pre_relu: If True, use pre_relu activations instead of sae_acts
            seq_len: If provided, extract fixed-length subsequences. If None, use full sequences.
            stride: Stride for extracting subsequences (only used if seq_len is provided)
            feature_indices: Optional subset of features to use
            return_token_ids: If True, also return token IDs for embedding lookup
            normalize: If True, z-score normalize each feature
            log_transform: If True, apply log(1 + x) transform before normalization
            normalization_stats: Optional pre-computed {'mean': ..., 'std': ...} for normalization
        """
        self.use_pre_relu = use_pre_relu
        self.seq_len = seq_len
        self.feature_indices = feature_indices
        self.return_token_ids = return_token_ids
        self.normalize = normalize
        self.log_transform = log_transform

        # Extract sequences
        self.sequences = []
        self.token_ids = []
        self.metadata = []

        act_key = 'pre_relu' if use_pre_relu else 'sae_acts'

        for d in data:
            acts = d[act_key].copy()  # Copy to avoid modifying original
            tokens = d.get('token_ids', None)

            # Apply feature subset if specified
            if feature_indices is not None:
                acts = acts[:, feature_indices]

            # Apply log transform
            if log_transform:
                acts = np.log1p(acts)  # log(1 + x)

            if seq_len is not None:
                # Extract fixed-length subsequences
                for start in range(0, len(acts) - seq_len + 1, stride):
                    self.sequences.append(acts[start:start + seq_len])
                    if tokens is not None and return_token_ids:
                        self.token_ids.append(tokens[start:start + seq_len])
                    self.metadata.append({
                        'style': d.get('style'),
                        'base_id': d.get('base_id'),
                        'start_idx': start,
                    })
            else:
                self.sequences.append(acts)
                if tokens is not None and return_token_ids:
                    self.token_ids.append(tokens)
                self.metadata.append({
                    'style': d.get('style'),
                    'base_id': d.get('base_id'),
                })

        self.n_features = self.sequences[0].shape[-1] if self.sequences else 0
        self.has_token_ids = len(self.token_ids) > 0

        # Compute or use provided normalization stats
        self.mean = None
        self.std = None
        if normalize:
            if normalization_stats is not None:
                self.mean = normalization_stats['mean']
                self.std = normalization_stats['std']
            else:
                # Compute stats from data
                all_acts = np.concatenate(self.sequences, axis=0)
                self.mean = all_acts.mean(axis=0)
                self.std = all_acts.std(axis=0) + 1e-8  # Avoid division by zero

            # Apply normalization
            self.sequences = [(s - self.mean) / self.std for s in self.sequences]

    def get_normalization_stats(self) -> Optional[Dict[str, np.ndarray]]:
        """Return normalization stats for use with test set."""
        if self.mean is not None:
            return {'mean': self.mean, 'std': self.std}
        return None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Returns (seq_len, n_features) tensor, optionally with token IDs."""
        acts = torch.tensor(self.sequences[idx], dtype=torch.float32)

        if self.return_token_ids and self.has_token_ids:
            tokens = torch.tensor(self.token_ids[idx], dtype=torch.long)
            return acts, tokens
        return acts


class EmbeddingLookup(nn.Module):
    """Wrapper for embedding matrix lookup, supports frozen or trainable embeddings."""

    def __init__(
        self,
        embed_matrix: np.ndarray,
        trainable: bool = False,
        project_dim: Optional[int] = None,
    ):
        """
        Args:
            embed_matrix: (vocab_size, embed_dim) numpy array of token embeddings
            trainable: If True, embeddings are trainable; if False, frozen
            project_dim: If provided, project embeddings to this dimension
        """
        super().__init__()
        vocab_size, embed_dim = embed_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embed_matrix, dtype=torch.float32),
            freeze=not trainable,
        )
        self.embed_dim = embed_dim
        self.output_dim = project_dim if project_dim is not None else embed_dim

        # Optional projection layer
        self.projection = None
        if project_dim is not None:
            self.projection = nn.Linear(embed_dim, project_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) token IDs

        Returns:
            embeddings: (batch, seq_len, output_dim) token embeddings
        """
        embeds = self.embedding(token_ids)  # (batch, seq_len, embed_dim)
        if self.projection is not None:
            embeds = self.projection(embeds)
        return embeds


def collate_with_tokens(
    batch: List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]
) -> Dict[str, torch.Tensor]:
    """Collate function for sequences with optional token IDs.

    Args:
        batch: List of tensors or (acts, token_ids) tuples

    Returns:
        Dictionary with 'acts' and optionally 'token_ids'
    """
    if isinstance(batch[0], tuple):
        acts, tokens = zip(*batch)
        return {
            'acts': torch.stack(acts, dim=0),
            'token_ids': torch.stack(tokens, dim=0),
        }
    else:
        return {
            'acts': torch.stack(batch, dim=0),
        }


def collate_variable_length(
    batch: List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]
) -> Dict[str, torch.Tensor]:
    """Collate function for variable-length sequences with optional token IDs.

    Args:
        batch: List of tensors or (acts, token_ids) tuples

    Returns:
        Dictionary with 'acts', 'lengths', and optionally 'token_ids'
    """
    if isinstance(batch[0], tuple):
        acts_list, tokens_list = zip(*batch)
    else:
        acts_list = batch
        tokens_list = None

    lengths = torch.tensor([x.size(0) for x in acts_list])
    max_len = lengths.max().item()
    n_features = acts_list[0].size(1)

    # Pad activations
    padded_acts = torch.zeros(len(acts_list), max_len, n_features)
    for i, x in enumerate(acts_list):
        padded_acts[i, :x.size(0), :] = x

    result = {
        'acts': padded_acts,
        'lengths': lengths,
    }

    # Pad token IDs if present
    if tokens_list is not None:
        padded_tokens = torch.zeros(len(tokens_list), max_len, dtype=torch.long)
        for i, t in enumerate(tokens_list):
            padded_tokens[i, :t.size(0)] = t
        result['token_ids'] = padded_tokens

    return result


@dataclass
class TrainingConfig:
    """Training configuration for LFADS."""

    # Optimization
    lr: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    grad_clip: float = 5.0

    # KL annealing
    kl_warmup_epochs: int = 20  # Epochs to anneal KL weight from 0 to 1
    kl_min: float = 0.0
    kl_max: float = 1.0

    # Learning rate schedule
    lr_decay: float = 0.95
    lr_decay_every: int = 10
    lr_min: float = 1e-5

    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4

    # Logging
    log_every: int = 10
    eval_every: int = 1
    save_every: int = 10


class LFADSTrainer:
    """Trainer for LFADS model with optional token embedding inputs."""

    def __init__(
        self,
        model: LFADS,
        train_config: TrainingConfig,
        device: torch.device,
        output_dir: Path,
        embedding_lookup: Optional[EmbeddingLookup] = None,
    ):
        """
        Args:
            model: LFADS model
            train_config: Training configuration
            device: torch device
            output_dir: Directory for checkpoints
            embedding_lookup: Optional EmbeddingLookup for token embeddings as external inputs
        """
        self.model = model.to(device)
        self.config = train_config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Embedding lookup for external inputs
        self.embedding_lookup = None
        if embedding_lookup is not None:
            self.embedding_lookup = embedding_lookup.to(device)

        # Optimizer - include embedding projection if trainable
        params = list(model.parameters())
        if self.embedding_lookup is not None and self.embedding_lookup.projection is not None:
            params.extend(self.embedding_lookup.projection.parameters())
        self.optimizer = optim.Adam(params, lr=train_config.lr)

        # LR scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=train_config.lr_decay_every,
            gamma=train_config.lr_decay,
        )

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0

    def get_kl_weight(self, epoch: int) -> float:
        """Compute KL weight with annealing."""
        if epoch < self.config.kl_warmup_epochs:
            progress = epoch / self.config.kl_warmup_epochs
            return self.config.kl_min + progress * (self.config.kl_max - self.config.kl_min)
        return self.config.kl_max

    def _unpack_batch(self, batch) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Unpack batch into activations and optional external inputs.

        Args:
            batch: Either a tensor, tuple, or dict from DataLoader

        Returns:
            x: (batch, seq_len, n_features) activations
            ext_inputs: (batch, seq_len, ext_input_dim) or None
        """
        ext_inputs = None

        if isinstance(batch, dict):
            x = batch['acts'].to(self.device)
            if 'token_ids' in batch and self.embedding_lookup is not None:
                token_ids = batch['token_ids'].to(self.device)
                ext_inputs = self.embedding_lookup(token_ids)
        elif isinstance(batch, tuple):
            if len(batch) == 2 and batch[1].dtype == torch.long:
                # (acts, token_ids) tuple
                x = batch[0].to(self.device)
                if self.embedding_lookup is not None:
                    token_ids = batch[1].to(self.device)
                    ext_inputs = self.embedding_lookup(token_ids)
            else:
                # Legacy (x, lengths) tuple
                x = batch[0].to(self.device)
        else:
            x = batch.to(self.device)

        return x, ext_inputs

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        if self.embedding_lookup is not None:
            self.embedding_lookup.train()

        kl_weight = self.get_kl_weight(epoch)

        total_loss = 0.0
        total_recon = 0.0
        total_kl_ic = 0.0
        total_kl_co = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            x, ext_inputs = self._unpack_batch(batch)

            self.optimizer.zero_grad()

            # Forward with optional external inputs
            output = self.model(x, ext_inputs=ext_inputs)
            losses = self.model.compute_loss(x, output, kl_weight=kl_weight)

            # Backward
            losses["loss"].backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )

            self.optimizer.step()

            # Track
            total_loss += losses["loss"].item()
            total_recon += losses["recon_loss"].item()
            total_kl_ic += losses["kl_ic"].item()
            total_kl_co += losses["kl_co"].item()
            n_batches += 1

            pbar.set_postfix({
                "loss": losses["loss"].item(),
                "recon": losses["recon_loss"].item(),
                "kl": losses["kl_ic"].item(),
            })

        return {
            "loss": total_loss / n_batches,
            "recon_loss": total_recon / n_batches,
            "kl_ic": total_kl_ic / n_batches,
            "kl_co": total_kl_co / n_batches,
            "kl_weight": kl_weight,
        }

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        if self.embedding_lookup is not None:
            self.embedding_lookup.eval()

        kl_weight = self.get_kl_weight(epoch)

        total_loss = 0.0
        total_recon = 0.0
        total_kl_ic = 0.0
        n_batches = 0

        # Also compute R² for reconstruction
        all_targets = []
        all_preds = []

        for batch in val_loader:
            x, ext_inputs = self._unpack_batch(batch)

            output = self.model(x, ext_inputs=ext_inputs)
            losses = self.model.compute_loss(x, output, kl_weight=kl_weight)

            total_loss += losses["loss"].item()
            total_recon += losses["recon_loss"].item()
            total_kl_ic += losses["kl_ic"].item()
            n_batches += 1

            # Collect for R² computation
            all_targets.append(x.cpu())
            if self.model.config.likelihood == "poisson":
                all_preds.append(output["recon_params"]["rates"].cpu())
            elif self.model.config.likelihood == "gaussian":
                all_preds.append(output["recon_params"]["mean"].cpu())
            else:
                all_preds.append(output["recon_params"]["rates"].cpu())

        # Compute R²
        targets = torch.cat(all_targets, dim=0).numpy()
        preds = torch.cat(all_preds, dim=0).numpy()

        # Flatten to (n_samples * seq_len, n_features)
        targets_flat = targets.reshape(-1, targets.shape[-1])
        preds_flat = preds.reshape(-1, preds.shape[-1])

        # Per-feature R²
        ss_res = ((targets_flat - preds_flat) ** 2).sum(axis=0)
        ss_tot = ((targets_flat - targets_flat.mean(axis=0)) ** 2).sum(axis=0)
        r2_per_feature = 1 - ss_res / (ss_tot + 1e-8)

        return {
            "loss": total_loss / n_batches,
            "recon_loss": total_recon / n_batches,
            "kl_ic": total_kl_ic / n_batches,
            "r2_mean": float(np.mean(r2_per_feature)),
            "r2_median": float(np.median(r2_per_feature)),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> Dict[str, List[float]]:
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            callback: Optional callback function(epoch, metrics)

        Returns:
            Dictionary with training history
        """
        history = {
            "train_loss": [],
            "train_recon": [],
            "train_kl_ic": [],
            "val_loss": [],
            "val_recon": [],
            "val_r2_mean": [],
            "lr": [],
            "kl_weight": [],
        }

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_metrics["loss"])
            history["train_recon"].append(train_metrics["recon_loss"])
            history["train_kl_ic"].append(train_metrics["kl_ic"])
            history["kl_weight"].append(train_metrics["kl_weight"])
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # Validate
            if val_loader is not None and epoch % self.config.eval_every == 0:
                val_metrics = self.evaluate(val_loader, epoch)
                history["val_loss"].append(val_metrics["loss"])
                history["val_recon"].append(val_metrics["recon_loss"])
                history["val_r2_mean"].append(val_metrics["r2_mean"])

                # Early stopping check
                if val_metrics["loss"] < self.best_val_loss - self.config.min_delta:
                    self.best_val_loss = val_metrics["loss"]
                    self.patience_counter = 0
                    self.save_checkpoint("best.pt")
                else:
                    self.patience_counter += 1

                if epoch % self.config.log_every == 0:
                    print(
                        f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
                        f"val_loss={val_metrics['loss']:.4f}, "
                        f"val_r2={val_metrics['r2_mean']:.4f}, "
                        f"kl_weight={train_metrics['kl_weight']:.3f}"
                    )
            else:
                if epoch % self.config.log_every == 0:
                    print(
                        f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
                        f"kl_weight={train_metrics['kl_weight']:.3f}"
                    )

            # LR schedule
            self.scheduler.step()

            # Enforce minimum LR
            for param_group in self.optimizer.param_groups:
                if param_group["lr"] < self.config.lr_min:
                    param_group["lr"] = self.config.lr_min

            # Periodic save
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt")

            # Callback
            if callback is not None:
                all_metrics = {**train_metrics}
                if val_loader is not None:
                    all_metrics.update(val_metrics)
                callback(epoch, all_metrics)

            # Early stopping
            if self.config.early_stopping and self.patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Save final
        self.save_checkpoint("final.pt")

        # Save history
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        return history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "model_config": self.model.config,
        }, self.output_dir / filename)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))


def load_sae_data(
    data_path: Path,
    use_pre_relu: bool = False,
    feature_indices_path: Optional[Path] = None,
) -> Tuple[List[Dict], Optional[np.ndarray]]:
    """Load SAE activation data from pickle file.

    Supports two formats:
    1. List of dicts (original format from main.py)
    2. Trials dataset dict with 'all_trials' key (from collect_trials.py)

    Args:
        data_path: Path to pickled dataset
        use_pre_relu: Whether to use pre-relu activations
        feature_indices_path: Optional path to feature indices

    Returns:
        Tuple of (data, feature_indices)
    """
    with open(data_path, "rb") as f:
        raw_data = pickle.load(f)

    # Handle trials dataset format
    if isinstance(raw_data, dict) and 'all_trials' in raw_data:
        data = raw_data['all_trials']
        print(f"  Loaded trials dataset: {len(data)} trials from {len(raw_data.get('prompts', []))} prompts")
    else:
        data = raw_data

    feature_indices = None
    if feature_indices_path is not None and feature_indices_path.exists():
        with open(feature_indices_path, "rb") as f:
            feature_info = pickle.load(f)
            feature_indices = feature_info["indices"]

    return data, feature_indices


def load_embedding_matrix(
    model_name: str,
    cache_dir: Path,
    device: str = "cpu",
) -> np.ndarray:
    """Load token embedding matrix from model (matches main.py implementation).

    Args:
        model_name: HuggingFace model name (e.g., "google/gemma-3-27b-it")
        cache_dir: Directory to cache the embedding matrix
        device: Device for loading model

    Returns:
        (vocab_size, embed_dim) numpy array of token embeddings
    """
    from transformers import AutoModel

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


def _unpack_batch_standalone(
    batch,
    device: torch.device,
    embedding_lookup: Optional[EmbeddingLookup] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Unpack batch into activations and optional external inputs (standalone version).

    Args:
        batch: Either a tensor, tuple, or dict from DataLoader
        device: torch device
        embedding_lookup: Optional embedding lookup for token IDs

    Returns:
        x: (batch, seq_len, n_features) activations
        ext_inputs: (batch, seq_len, ext_input_dim) or None
    """
    ext_inputs = None

    if isinstance(batch, dict):
        x = batch['acts'].to(device)
        if 'token_ids' in batch and embedding_lookup is not None:
            token_ids = batch['token_ids'].to(device)
            ext_inputs = embedding_lookup(token_ids)
    elif isinstance(batch, tuple):
        if len(batch) == 2 and batch[1].dtype == torch.long:
            # (acts, token_ids) tuple
            x = batch[0].to(device)
            if embedding_lookup is not None:
                token_ids = batch[1].to(device)
                ext_inputs = embedding_lookup(token_ids)
        else:
            # Legacy (x, lengths) tuple
            x = batch[0].to(device)
    else:
        x = batch.to(device)

    return x, ext_inputs


def compute_reconstruction_metrics(
    model: LFADS,
    data_loader: DataLoader,
    device: torch.device,
    embedding_lookup: Optional[EmbeddingLookup] = None,
) -> Dict[str, np.ndarray]:
    """Compute detailed reconstruction metrics.

    Args:
        model: Trained LFADS model
        data_loader: Data loader
        device: torch device
        embedding_lookup: Optional embedding lookup for token inputs

    Returns:
        Dictionary with:
            - r2_per_feature: R² for each feature
            - mse_per_feature: MSE for each feature
            - correlation_per_feature: Pearson correlation per feature
    """
    model.eval()
    if embedding_lookup is not None:
        embedding_lookup.eval()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            x, ext_inputs = _unpack_batch_standalone(batch, device, embedding_lookup)

            output = model(x, ext_inputs=ext_inputs)

            all_targets.append(x.cpu().numpy())
            if model.config.likelihood == "poisson":
                all_preds.append(output["recon_params"]["rates"].cpu().numpy())
            elif model.config.likelihood == "gaussian":
                all_preds.append(output["recon_params"]["mean"].cpu().numpy())
            else:
                all_preds.append(output["recon_params"]["rates"].cpu().numpy())

    targets = np.concatenate(all_targets, axis=0)  # (N, T, F)
    preds = np.concatenate(all_preds, axis=0)

    # Flatten time dimension
    n_samples, seq_len, n_features = targets.shape
    targets_flat = targets.reshape(-1, n_features)
    preds_flat = preds.reshape(-1, n_features)

    # Per-feature metrics
    ss_res = ((targets_flat - preds_flat) ** 2).sum(axis=0)
    ss_tot = ((targets_flat - targets_flat.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    mse = ((targets_flat - preds_flat) ** 2).mean(axis=0)

    # Correlation
    target_centered = targets_flat - targets_flat.mean(axis=0)
    pred_centered = preds_flat - preds_flat.mean(axis=0)
    correlation = (
        (target_centered * pred_centered).sum(axis=0)
        / (np.sqrt((target_centered ** 2).sum(axis=0)) * np.sqrt((pred_centered ** 2).sum(axis=0)) + 1e-8)
    )

    return {
        "r2_per_feature": r2,
        "mse_per_feature": mse,
        "correlation_per_feature": correlation,
    }


def extract_all_latents(
    model: LFADS,
    data_loader: DataLoader,
    device: torch.device,
    embedding_lookup: Optional[EmbeddingLookup] = None,
) -> Dict[str, np.ndarray]:
    """Extract latent factors for entire dataset.

    Args:
        model: Trained LFADS model
        data_loader: Data loader
        device: torch device
        embedding_lookup: Optional embedding lookup for token inputs

    Returns:
        Dictionary with:
            - factors: (N, T, fac_dim) latent factors at token boundaries
            - ic: (N, ic_dim) initial conditions
            - gen_states: (N, T, gen_dim) generator states
            - substep_factors: (N, T, substeps, fac_dim) if substeps > 1
    """
    model.eval()
    if embedding_lookup is not None:
        embedding_lookup.eval()

    all_factors = []
    all_ic = []
    all_gen_states = []
    all_substep_factors = []
    has_substeps = model.config.substeps_per_token > 1

    with torch.no_grad():
        for batch in data_loader:
            x, ext_inputs = _unpack_batch_standalone(batch, device, embedding_lookup)

            output = model(x, ext_inputs=ext_inputs)

            all_factors.append(output["factors"].cpu().numpy())
            all_ic.append(output["ic_mean"].cpu().numpy())
            all_gen_states.append(output["gen_states"].cpu().numpy())

            if has_substeps and "substep_factors" in output:
                all_substep_factors.append(output["substep_factors"].cpu().numpy())

    result = {
        "factors": np.concatenate(all_factors, axis=0),
        "ic": np.concatenate(all_ic, axis=0),
        "gen_states": np.concatenate(all_gen_states, axis=0),
    }

    if has_substeps and all_substep_factors:
        result["substep_factors"] = np.concatenate(all_substep_factors, axis=0)

    return result
