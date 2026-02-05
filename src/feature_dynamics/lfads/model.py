"""LFADS (Latent Factor Analysis via Dynamical Systems) model.

This module implements the LFADS model architecture to model sparse feature
activation time series from Sparse Autoencoders as neural population dynamics.

Reference:
    Pandarinath et al. (2018) "Inferring single-trial neural population dynamics
    using sequential auto-encoders" Nature Methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence, Poisson
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass


@dataclass
class LFADSConfig:
    """Configuration for LFADS model."""

    # Data dimensions
    n_features: int  # Number of SAE features (observed dimensionality)

    # Model dimensions
    enc_dim: int = 128  # Encoder GRU hidden size
    gen_dim: int = 128  # Generator GRU hidden size
    fac_dim: int = 32   # Factor (latent) dimensionality
    ic_dim: int = 64    # Initial condition dimensionality

    # Controller (optional - for inferring inputs)
    use_controller: bool = False
    con_dim: int = 64   # Controller GRU hidden size
    ci_dim: int = 1     # Controller input dimensionality (inferred inputs)

    # External inputs (optional - for known inputs like token embeddings)
    ext_input_dim: int = 0  # External input dimensionality

    # Architecture
    dropout: float = 0.1
    clip_val: float = 5.0  # Gradient clipping value

    # Likelihood
    likelihood: Literal["poisson", "gaussian", "zero_inflated_poisson"] = "poisson"

    # KL annealing
    kl_ic_weight: float = 1.0      # Weight for IC KL divergence
    kl_co_weight: float = 1.0      # Weight for controller KL divergence
    l2_gen_scale: float = 0.0      # L2 regularization on generator weights
    l2_con_scale: float = 0.0      # L2 regularization on controller weights


class GRUEncoder(nn.Module):
    """Bidirectional GRU encoder for LFADS."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,  # No dropout between layers (single layer)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim) input sequences

        Returns:
            outputs: (batch, seq_len, 2*hidden_dim) all hidden states
            h_final: (batch, 2*hidden_dim) final hidden state (fwd + bwd concatenated)
        """
        x = self.dropout(x)
        outputs, h_n = self.gru(x)  # h_n: (2, batch, hidden_dim)

        # Concatenate forward and backward final states
        h_final = torch.cat([h_n[0], h_n[1]], dim=-1)  # (batch, 2*hidden_dim)

        return outputs, h_final


class Generator(nn.Module):
    """Generator GRU that produces latent dynamics."""

    def __init__(
        self,
        gen_dim: int,
        fac_dim: int,
        input_dim: int = 0,  # Optional controller/external inputs
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gen_dim = gen_dim
        self.fac_dim = fac_dim
        self.input_dim = input_dim

        # GRU cell for step-by-step generation
        gru_input_dim = input_dim if input_dim > 0 else 1  # Need at least 1 for dummy input
        self.gru_cell = nn.GRUCell(gru_input_dim, gen_dim)

        # Factor readout from generator state
        self.fac_linear = nn.Linear(gen_dim, fac_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        g0: torch.Tensor,
        seq_len: int,
        inputs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            g0: (batch, gen_dim) initial generator state
            seq_len: number of timesteps to generate
            inputs: (batch, seq_len, input_dim) optional inputs at each timestep

        Returns:
            gen_states: (batch, seq_len, gen_dim) generator hidden states
            factors: (batch, seq_len, fac_dim) latent factors
        """
        batch_size = g0.size(0)
        device = g0.device

        gen_states = []
        factors = []

        g = g0
        for t in range(seq_len):
            if inputs is not None and self.input_dim > 0:
                inp = inputs[:, t, :]
            else:
                # Dummy input of zeros
                inp = torch.zeros(batch_size, 1, device=device)

            g = self.gru_cell(inp, g)
            gen_states.append(g)

            # Compute factors
            f = self.fac_linear(self.dropout(g))
            factors.append(f)

        gen_states = torch.stack(gen_states, dim=1)  # (batch, seq_len, gen_dim)
        factors = torch.stack(factors, dim=1)  # (batch, seq_len, fac_dim)

        return gen_states, factors


class Controller(nn.Module):
    """Controller network that infers inputs from encoder states."""

    def __init__(
        self,
        enc_dim: int,  # Encoder output dim (2*enc_hidden for bidirectional)
        fac_dim: int,
        con_dim: int,
        ci_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.con_dim = con_dim
        self.ci_dim = ci_dim

        # Controller GRU takes encoder output + current factors
        self.gru_cell = nn.GRUCell(enc_dim + fac_dim, con_dim)

        # Output posterior params for inferred inputs
        self.ci_mean = nn.Linear(con_dim, ci_dim)
        self.ci_logvar = nn.Linear(con_dim, ci_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        enc_outputs: torch.Tensor,
        factors: torch.Tensor,
        c0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            enc_outputs: (batch, seq_len, enc_dim) encoder outputs
            factors: (batch, seq_len, fac_dim) current factors (from generator)
            c0: (batch, con_dim) initial controller state

        Returns:
            ci_means: (batch, seq_len, ci_dim) posterior means
            ci_logvars: (batch, seq_len, ci_dim) posterior log-variances
            ci_samples: (batch, seq_len, ci_dim) sampled inferred inputs
        """
        batch_size, seq_len, _ = enc_outputs.size()

        ci_means = []
        ci_logvars = []
        ci_samples = []

        c = c0
        for t in range(seq_len):
            # Concatenate encoder output and factors
            inp = torch.cat([enc_outputs[:, t, :], factors[:, t, :]], dim=-1)
            inp = self.dropout(inp)

            c = self.gru_cell(inp, c)

            # Posterior params
            mean = self.ci_mean(c)
            logvar = self.ci_logvar(c)

            # Reparameterized sample
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            sample = mean + eps * std

            ci_means.append(mean)
            ci_logvars.append(logvar)
            ci_samples.append(sample)

        ci_means = torch.stack(ci_means, dim=1)
        ci_logvars = torch.stack(ci_logvars, dim=1)
        ci_samples = torch.stack(ci_samples, dim=1)

        return ci_means, ci_logvars, ci_samples


class LFADS(nn.Module):
    """Full LFADS model for sparse feature activation dynamics."""

    def __init__(self, config: LFADSConfig):
        super().__init__()
        self.config = config

        # Encoder
        self.encoder = GRUEncoder(
            input_dim=config.n_features,
            hidden_dim=config.enc_dim,
            dropout=config.dropout,
        )

        # Initial condition posterior from encoder final state
        enc_output_dim = 2 * config.enc_dim  # Bidirectional
        self.ic_mean = nn.Linear(enc_output_dim, config.ic_dim)
        self.ic_logvar = nn.Linear(enc_output_dim, config.ic_dim)

        # Map IC sample to generator initial state
        self.ic_to_g0 = nn.Linear(config.ic_dim, config.gen_dim)

        # Generator
        gen_input_dim = config.ci_dim if config.use_controller else 0
        gen_input_dim += config.ext_input_dim

        self.generator = Generator(
            gen_dim=config.gen_dim,
            fac_dim=config.fac_dim,
            input_dim=gen_input_dim,
            dropout=config.dropout,
        )

        # Controller (optional)
        self.controller = None
        if config.use_controller:
            self.controller = Controller(
                enc_dim=enc_output_dim,
                fac_dim=config.fac_dim,
                con_dim=config.con_dim,
                ci_dim=config.ci_dim,
                dropout=config.dropout,
            )
            # Initial controller state
            self.c0_linear = nn.Linear(enc_output_dim, config.con_dim)

        # Decoder: factors -> rates/observations
        if config.likelihood == "poisson":
            # Output log-rates for Poisson
            self.decoder = nn.Sequential(
                nn.Linear(config.fac_dim, config.n_features),
                nn.Softplus(),  # Ensure positive rates
            )
        elif config.likelihood == "gaussian":
            # Output mean and log-variance
            self.dec_mean = nn.Linear(config.fac_dim, config.n_features)
            self.dec_logvar = nn.Linear(config.fac_dim, config.n_features)
        elif config.likelihood == "zero_inflated_poisson":
            # Output log-rates and logit for zero-inflation
            self.dec_rate = nn.Sequential(
                nn.Linear(config.fac_dim, config.n_features),
                nn.Softplus(),
            )
            self.dec_gate = nn.Linear(config.fac_dim, config.n_features)  # Logit for P(spike)

    def encode(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input sequence to initial condition posterior.

        Args:
            x: (batch, seq_len, n_features) input sequences

        Returns:
            enc_outputs: (batch, seq_len, enc_dim*2) encoder hidden states
            ic_mean: (batch, ic_dim) IC posterior mean
            ic_logvar: (batch, ic_dim) IC posterior log-variance
            ic_sample: (batch, ic_dim) sampled IC
        """
        enc_outputs, h_final = self.encoder(x)

        # IC posterior
        ic_mean = self.ic_mean(h_final)
        ic_logvar = self.ic_logvar(h_final)

        # Reparameterized sample
        std = torch.exp(0.5 * ic_logvar)
        eps = torch.randn_like(std)
        ic_sample = ic_mean + eps * std

        return enc_outputs, ic_mean, ic_logvar, ic_sample

    def decode(
        self,
        factors: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Decode factors to observation distribution parameters.

        Args:
            factors: (batch, seq_len, fac_dim) latent factors

        Returns:
            Dictionary with distribution parameters based on likelihood type
        """
        if self.config.likelihood == "poisson":
            rates = self.decoder(factors)  # (batch, seq_len, n_features)
            return {"rates": rates}

        elif self.config.likelihood == "gaussian":
            mean = self.dec_mean(factors)
            logvar = self.dec_logvar(factors)
            return {"mean": mean, "logvar": logvar}

        elif self.config.likelihood == "zero_inflated_poisson":
            rates = self.dec_rate(factors)
            gate_logits = self.dec_gate(factors)
            return {"rates": rates, "gate_logits": gate_logits}

    def forward(
        self,
        x: torch.Tensor,
        ext_inputs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass through LFADS.

        Args:
            x: (batch, seq_len, n_features) input sequences
            ext_inputs: (batch, seq_len, ext_input_dim) optional external inputs

        Returns:
            Dictionary containing:
                - recon_params: decoder output parameters
                - ic_mean, ic_logvar: IC posterior
                - ci_mean, ci_logvar (if controller): inferred input posteriors
                - factors: (batch, seq_len, fac_dim) latent factors
                - gen_states: (batch, seq_len, gen_dim) generator states
        """
        batch_size, seq_len, _ = x.size()

        # Encode
        enc_outputs, ic_mean, ic_logvar, ic_sample = self.encode(x)

        # Map IC to generator initial state
        g0 = self.ic_to_g0(ic_sample)

        # Prepare generator inputs
        gen_inputs = None
        ci_mean = ci_logvar = ci_samples = None

        if self.config.use_controller or self.config.ext_input_dim > 0:
            inputs_list = []

            if self.config.use_controller:
                # Need to run generator and controller in interleaved fashion
                # For simplicity, we do a two-pass approach:
                # 1. Run generator without controller to get initial factors
                # 2. Run controller to get inferred inputs
                # 3. Re-run generator with inferred inputs

                # First pass: no inputs
                _, init_factors = self.generator(g0, seq_len, inputs=None)

                # Controller pass
                c0 = self.c0_linear(enc_outputs[:, -1, :])
                ci_mean, ci_logvar, ci_samples = self.controller(
                    enc_outputs, init_factors, c0
                )
                inputs_list.append(ci_samples)

            if ext_inputs is not None:
                inputs_list.append(ext_inputs)

            if inputs_list:
                gen_inputs = torch.cat(inputs_list, dim=-1)

        # Generate
        gen_states, factors = self.generator(g0, seq_len, inputs=gen_inputs)

        # Decode
        recon_params = self.decode(factors)

        output = {
            "recon_params": recon_params,
            "ic_mean": ic_mean,
            "ic_logvar": ic_logvar,
            "factors": factors,
            "gen_states": gen_states,
        }

        if self.config.use_controller:
            output["ci_mean"] = ci_mean
            output["ci_logvar"] = ci_logvar

        return output

    def compute_loss(
        self,
        x: torch.Tensor,
        output: Dict[str, torch.Tensor],
        kl_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute LFADS loss.

        Args:
            x: (batch, seq_len, n_features) target sequences
            output: forward pass output dictionary
            kl_weight: weight for KL terms (for annealing)

        Returns:
            Dictionary with loss components and total loss
        """
        recon_params = output["recon_params"]

        # Reconstruction loss
        if self.config.likelihood == "poisson":
            rates = recon_params["rates"]
            # Poisson negative log-likelihood
            # For sparse activations, we treat them as pseudo-counts
            # Clamp x to avoid log(0)
            recon_loss = (rates - x * torch.log(rates + 1e-8)).mean()

        elif self.config.likelihood == "gaussian":
            mean = recon_params["mean"]
            logvar = recon_params["logvar"]
            # Gaussian NLL with logvar clamped to prevent tiny variance
            logvar = torch.clamp(logvar, min=-4.0, max=4.0)  # var in [~0.018, ~55]
            var = torch.exp(logvar)
            recon_loss = 0.5 * (logvar + (x - mean)**2 / var).mean()

        elif self.config.likelihood == "zero_inflated_poisson":
            rates = recon_params["rates"]
            gate_logits = recon_params["gate_logits"]

            # Zero-inflated Poisson
            # P(x=0) = sigmoid(-gate) + sigmoid(gate) * exp(-rate)
            # P(x>0) = sigmoid(gate) * Poisson(x; rate)

            is_zero = (x == 0).float()

            # Log probability for zeros
            log_p_zero = torch.logsumexp(
                torch.stack([
                    F.logsigmoid(-gate_logits),
                    F.logsigmoid(gate_logits) - rates,
                ], dim=0),
                dim=0,
            )

            # Log probability for non-zeros (Poisson)
            log_p_nonzero = (
                F.logsigmoid(gate_logits)
                + x * torch.log(rates + 1e-8) - rates
                - torch.lgamma(x + 1)
            )

            log_p = is_zero * log_p_zero + (1 - is_zero) * log_p_nonzero
            recon_loss = -log_p.mean()

        # KL divergence for initial conditions
        ic_mean = output["ic_mean"]
        ic_logvar = output["ic_logvar"]

        # KL(q(z|x) || N(0,1))
        kl_ic = -0.5 * (1 + ic_logvar - ic_mean**2 - torch.exp(ic_logvar))
        kl_ic = kl_ic.sum(dim=-1).mean()  # Sum over dims, mean over batch

        # KL for controller (if used)
        kl_co = torch.tensor(0.0, device=x.device)
        if self.config.use_controller and "ci_mean" in output:
            ci_mean = output["ci_mean"]
            ci_logvar = output["ci_logvar"]
            kl_co = -0.5 * (1 + ci_logvar - ci_mean**2 - torch.exp(ci_logvar))
            kl_co = kl_co.sum(dim=-1).mean(dim=-1).mean()  # Sum dims, mean seq & batch

        # L2 regularization
        l2_gen = torch.tensor(0.0, device=x.device)
        if self.config.l2_gen_scale > 0:
            for p in self.generator.parameters():
                l2_gen = l2_gen + p.pow(2).sum()
            l2_gen = l2_gen * self.config.l2_gen_scale

        l2_con = torch.tensor(0.0, device=x.device)
        if self.config.l2_con_scale > 0 and self.controller is not None:
            for p in self.controller.parameters():
                l2_con = l2_con + p.pow(2).sum()
            l2_con = l2_con * self.config.l2_con_scale

        # Total loss
        total_loss = (
            recon_loss
            + kl_weight * self.config.kl_ic_weight * kl_ic
            + kl_weight * self.config.kl_co_weight * kl_co
            + l2_gen
            + l2_con
        )

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_ic": kl_ic,
            "kl_co": kl_co,
            "l2_gen": l2_gen,
            "l2_con": l2_con,
        }

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        ext_inputs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample from the generative model (prior).

        Args:
            batch_size: number of samples
            seq_len: sequence length
            device: torch device
            ext_inputs: optional external inputs

        Returns:
            Dictionary with samples and intermediate states
        """
        # Sample IC from prior
        ic_sample = torch.randn(batch_size, self.config.ic_dim, device=device)

        # Map to generator state
        g0 = self.ic_to_g0(ic_sample)

        # Prepare inputs
        gen_inputs = None
        if self.config.use_controller:
            # Sample inferred inputs from prior
            ci_samples = torch.randn(
                batch_size, seq_len, self.config.ci_dim, device=device
            )
            gen_inputs = ci_samples

        if ext_inputs is not None:
            if gen_inputs is not None:
                gen_inputs = torch.cat([gen_inputs, ext_inputs], dim=-1)
            else:
                gen_inputs = ext_inputs

        # Generate
        gen_states, factors = self.generator(g0, seq_len, inputs=gen_inputs)

        # Decode
        recon_params = self.decode(factors)

        # Sample observations
        if self.config.likelihood == "poisson":
            rates = recon_params["rates"]
            samples = torch.poisson(rates)
        elif self.config.likelihood == "gaussian":
            mean = recon_params["mean"]
            logvar = recon_params["logvar"]
            std = torch.exp(0.5 * logvar)
            samples = mean + std * torch.randn_like(mean)
        elif self.config.likelihood == "zero_inflated_poisson":
            rates = recon_params["rates"]
            gate_logits = recon_params["gate_logits"]
            # Sample spike/no-spike
            spike = torch.sigmoid(gate_logits) > torch.rand_like(gate_logits)
            # Sample count where spike
            counts = torch.poisson(rates)
            samples = spike.float() * counts

        return {
            "samples": samples,
            "factors": factors,
            "gen_states": gen_states,
            "recon_params": recon_params,
        }


def extract_latent_trajectories(
    model: LFADS,
    x: torch.Tensor,
    ext_inputs: Optional[torch.Tensor] = None,
) -> Dict[str, np.ndarray]:
    """Extract latent trajectories from trained model.

    Args:
        model: trained LFADS model
        x: (batch, seq_len, n_features) input sequences
        ext_inputs: optional external inputs

    Returns:
        Dictionary with numpy arrays:
            - factors: (batch, seq_len, fac_dim)
            - gen_states: (batch, seq_len, gen_dim)
            - ic: (batch, ic_dim) inferred initial conditions
    """
    model.eval()
    with torch.no_grad():
        output = model(x, ext_inputs)

    return {
        "factors": output["factors"].cpu().numpy(),
        "gen_states": output["gen_states"].cpu().numpy(),
        "ic": output["ic_mean"].cpu().numpy(),
    }
