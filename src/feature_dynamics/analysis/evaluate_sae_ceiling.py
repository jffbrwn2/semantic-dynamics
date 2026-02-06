"""Evaluate ceiling performance of SAE reconstruction for text generation.

This script measures the maximal potential performance when using SAE reconstructions
at a target layer. It runs layers 0-31 normally, reconstructs through SAE, then
propagates through layers 32-61. This establishes the ceiling for predictor-driven
generation - how well we could do if our predictor was perfect.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask
)

from ..prompts import PromptGenerator


class SAEModel:
    """Wrapper for SAE encoder/decoder with JumpReLU activation."""

    def __init__(self, sae_path: Path):
        """Load SAE model from path."""
        with open(sae_path, 'rb') as f:
            sae_data = pickle.load(f)

        self.encoder_weight = sae_data['encoder_weight']  # (d_model, n_features)
        self.encoder_bias = sae_data['encoder_bias']  # (n_features,)
        self.decoder_weight = sae_data['decoder_weight']  # (n_features, d_model)
        self.decoder_bias = sae_data.get('decoder_bias', None)  # (d_model,) or None

        # JumpReLU parameters - required for Gemma Scope SAEs
        self.activation_fn = sae_data.get('activation_fn')
        self.threshold = sae_data.get('threshold')

        if self.activation_fn is None:
            raise ValueError(
                "SAE weights file missing 'activation_fn'. "
                "Re-run save_sae_weights.py to extract with threshold parameter."
            )

        if self.activation_fn == 'jumprelu' and self.threshold is None:
            raise ValueError(
                f"SAE uses {self.activation_fn} but threshold is missing. "
                "Re-run save_sae_weights.py to extract threshold."
            )

        self.n_features = self.encoder_weight.shape[1]
        self.d_model = self.encoder_weight.shape[0]

    def encode(self, residual: np.ndarray) -> np.ndarray:
        """Encode residual stream to SAE features."""
        hidden_pre = residual @ self.encoder_weight + self.encoder_bias

        if self.activation_fn == 'jumprelu':
            # JumpReLU: relu(x) * (x > threshold)
            base_acts = np.maximum(0, hidden_pre)
            jump_mask = (hidden_pre > self.threshold).astype(base_acts.dtype)
            return base_acts * jump_mask
        elif self.activation_fn == 'relu':
            return np.maximum(0, hidden_pre)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_fn}")

    def decode(self, features: np.ndarray) -> np.ndarray:
        """Decode SAE features back to residual stream."""
        residual = features @ self.decoder_weight
        if self.decoder_bias is not None:
            residual = residual + self.decoder_bias
        return residual

    def reconstruct(
        self,
        residual: np.ndarray,
        feature_indices: Optional[np.ndarray] = None,
        binarize: bool = False
    ) -> np.ndarray:
        """Encode then decode, optionally using only a subset of features.

        Args:
            residual: (d_model,) residual stream vector
            feature_indices: Optional indices to select subset of features
            binarize: If True, set all positive feature activations to 1.0

        Returns:
            Reconstructed residual stream (d_model,)
        """
        features = self.encode(residual)

        if feature_indices is not None:
            # Zero out non-selected features
            mask = np.zeros(self.n_features)
            mask[feature_indices] = 1.0
            features = features * mask

        if binarize:
            # Set all positive activations to 1.0 (test effect of magnitude)
            features = (features > 0).astype(features.dtype)

        return self.decode(features)


class SAECeilingGenerator:
    """Text generator using SAE reconstruction (ceiling performance)."""

    def __init__(
        self,
        sae_model: SAEModel,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        feature_indices: Optional[np.ndarray] = None,
        target_layer: int = 31,
        device: str = "cpu",
        binarize_features: bool = False,
        rescale_reconstruction: bool = False
    ):
        """Initialize generator.

        Args:
            sae_model: SAE encoder/decoder
            tokenizer: Gemma tokenizer
            model: Gemma model
            feature_indices: Optional indices to select subset of SAE features
            target_layer: Layer where we intervene with SAE reconstruction
            device: Device to run on
            binarize_features: If True, set all positive feature activations to 1.0
            rescale_reconstruction: If True, rescale reconstruction to match original RMS norm
        """
        self.sae_model = sae_model
        self.tokenizer = tokenizer
        self.model = model
        self.feature_indices = feature_indices
        self.target_layer = target_layer
        self.device = device
        self.binarize_features = binarize_features
        self.rescale_reconstruction = rescale_reconstruction

        self.model.to(device)
        self.model.eval()

        # Quick access to model components
        self.lm = self.model.model.language_model
        self.embed_tokens = self.lm.embed_tokens
        self.layers = self.lm.layers
        self.norm = self.lm.norm
        self.lm_head = self.model.lm_head

        # Get number of layers
        self.num_layers = len(self.layers)

        # Determine attention types for position embeddings
        self.config = self.model.config
        if hasattr(self.config, 'text_config'):
            self.text_config = self.config.text_config
            self.layer_types = self.text_config.layer_types if hasattr(self.text_config, 'layer_types') else None
        else:
            self.text_config = self.config
            self.layer_types = self.config.layer_types if hasattr(self.config, 'layer_types') else None

    def _create_causal_mask_mapping(
        self,
        input_embeds: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[DynamicCache] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Create attention masks for different layer types."""
        mask_kwargs = {
            "config": self.text_config,
            "input_embeds": input_embeds,
            "attention_mask": None,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }

        causal_mask_mapping = {}

        if self.layer_types and "full_attention" in self.layer_types:
            causal_mask_mapping["full_attention"] = create_causal_mask(**mask_kwargs)

        if self.layer_types and "sliding_attention" in self.layer_types:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        return causal_mask_mapping

    def _run_lower_layers(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run layers 0-31 on input to get hidden state at target layer.

        Args:
            input_ids: (1, seq_len) token IDs

        Returns:
            hidden_states at target layer: (1, seq_len, d_model)
        """
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(0, seq_len, device=self.device).unsqueeze(0)
        cache_position = torch.arange(0, seq_len, device=self.device)

        hidden_states = self.embed_tokens(input_ids)

        causal_mask_mapping = self._create_causal_mask_mapping(
            input_embeds=hidden_states,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        position_embeddings_global = self.lm.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.lm.rotary_emb_local(hidden_states, position_ids)

        for layer_idx in range(self.target_layer + 1):
            layer = self.layers[layer_idx]

            layer_mask = None
            if self.layer_types and layer_idx < len(self.layer_types):
                layer_type = self.layer_types[layer_idx]
                layer_mask = causal_mask_mapping.get(layer_type)

            layer_outputs = layer(
                hidden_states,
                attention_mask=layer_mask,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                position_ids=position_ids,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]

        return hidden_states

    def _run_upper_layers(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run layers 32-61 on hidden states and return logits.

        Args:
            hidden_states: (1, seq_len, d_model) hidden states at layer 31

        Returns:
            logits: (1, seq_len, vocab_size)
        """
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(0, seq_len, device=self.device).unsqueeze(0)
        cache_position = torch.arange(0, seq_len, device=self.device)

        causal_mask_mapping = self._create_causal_mask_mapping(
            input_embeds=hidden_states,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        position_embeddings_global = self.lm.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.lm.rotary_emb_local(hidden_states, position_ids)

        for layer_idx in range(self.target_layer + 1, self.num_layers):
            layer = self.layers[layer_idx]

            layer_mask = None
            if self.layer_types and layer_idx < len(self.layer_types):
                layer_type = self.layer_types[layer_idx]
                layer_mask = causal_mask_mapping.get(layer_type)

            layer_outputs = layer(
                hidden_states,
                attention_mask=layer_mask,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                position_ids=position_ids,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    def _reconstruct_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply SAE reconstruction to the last token's hidden state.

        Args:
            hidden_states: (1, seq_len, d_model) hidden states at target layer

        Returns:
            Reconstructed hidden states (1, seq_len, d_model)
        """
        # Only reconstruct the last token (for autoregressive generation)
        last_hidden = hidden_states[0, -1].float().cpu().numpy()

        # Reconstruct through SAE (with optional feature selection and binarization)
        reconstructed = self.sae_model.reconstruct(
            last_hidden,
            self.feature_indices,
            binarize=self.binarize_features
        )

        # Rescale to match original RMS norm if requested
        if self.rescale_reconstruction:
            original_rms = np.sqrt(np.mean(last_hidden ** 2) + 1e-6)
            reconstructed_rms = np.sqrt(np.mean(reconstructed ** 2) + 1e-6)
            reconstructed = reconstructed * (original_rms / reconstructed_rms)

        # Replace the last position with reconstructed version
        result = hidden_states.clone()
        result[0, -1] = torch.tensor(
            reconstructed,
            dtype=self.model.dtype,
            device=self.device
        )

        return result

    def _get_baseline_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get logits from baseline model (no SAE intervention).

        Args:
            input_ids: (1, seq_len) token IDs

        Returns:
            logits: (1, seq_len, vocab_size)
        """
        with torch.no_grad():
            outputs = self.model(input_ids)
            return outputs.logits

    def _get_reconstructed_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get logits after SAE reconstruction at target layer.

        Args:
            input_ids: (1, seq_len) token IDs

        Returns:
            logits: (1, seq_len, vocab_size)
        """
        hidden_states = self._run_lower_layers(input_ids)
        hidden_states = self._reconstruct_hidden_states(hidden_states)
        return self._run_upper_layers(hidden_states)

    def compute_reconstruction_metrics(
        self,
        input_ids: torch.Tensor,
        next_token: Optional[int] = None
    ) -> Dict[str, float]:
        """Compute reconstruction metrics comparing baseline vs SAE-reconstructed.

        Args:
            input_ids: (1, seq_len) token IDs
            next_token: Optional actual next token for cross-entropy calculation

        Returns:
            Dictionary of metrics
        """
        with torch.no_grad():
            # Get logits from both paths
            baseline_logits = self._get_baseline_logits(input_ids)
            reconstructed_logits = self._get_reconstructed_logits(input_ids)

            # Get last position logits
            baseline_last = baseline_logits[0, -1, :]
            reconstructed_last = reconstructed_logits[0, -1, :]

            # Convert to probabilities
            baseline_probs = F.softmax(baseline_last, dim=-1)
            reconstructed_probs = F.softmax(reconstructed_last, dim=-1)

            # KL divergence: KL(baseline || reconstructed)
            # Using log_softmax for numerical stability
            baseline_log_probs = F.log_softmax(baseline_last, dim=-1)
            reconstructed_log_probs = F.log_softmax(reconstructed_last, dim=-1)
            kl_div = F.kl_div(
                reconstructed_log_probs,
                baseline_probs,
                reduction='sum'
            ).item()

            # Top-1 agreement
            baseline_top1 = baseline_last.argmax().item()
            reconstructed_top1 = reconstructed_last.argmax().item()
            top1_agreement = float(baseline_top1 == reconstructed_top1)

            # Top-5 overlap
            baseline_top5 = set(baseline_last.topk(5).indices.tolist())
            reconstructed_top5 = set(reconstructed_last.topk(5).indices.tolist())
            top5_overlap = len(baseline_top5 & reconstructed_top5) / 5

            # Top-10 overlap
            baseline_top10 = set(baseline_last.topk(10).indices.tolist())
            reconstructed_top10 = set(reconstructed_last.topk(10).indices.tolist())
            top10_overlap = len(baseline_top10 & reconstructed_top10) / 10

            # Cosine similarity of logits
            cosine_sim = F.cosine_similarity(
                baseline_last.unsqueeze(0),
                reconstructed_last.unsqueeze(0)
            ).item()

            metrics = {
                'kl_divergence': kl_div,
                'top1_agreement': top1_agreement,
                'top5_overlap': top5_overlap,
                'top10_overlap': top10_overlap,
                'cosine_similarity': cosine_sim,
                'baseline_top1': baseline_top1,
                'reconstructed_top1': reconstructed_top1,
            }

            # Cross-entropy if next token provided
            if next_token is not None:
                ce_baseline = -baseline_log_probs[next_token].item()
                ce_reconstructed = -reconstructed_log_probs[next_token].item()
                metrics['ce_baseline'] = ce_baseline
                metrics['ce_reconstructed'] = ce_reconstructed
                metrics['ce_increase'] = ce_reconstructed - ce_baseline

            return metrics

    def evaluate_reconstruction(
        self,
        prompt: str,
        max_positions: int = 50
    ) -> Dict[str, Any]:
        """Evaluate reconstruction quality over multiple positions.

        Generates a sequence with the baseline model, then evaluates
        reconstruction metrics at each position.

        Args:
            prompt: Input prompt
            max_positions: Maximum number of positions to evaluate

        Returns:
            Dictionary with aggregated metrics and per-position details
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.shape[1]

        metrics_list = []

        with torch.no_grad():
            # Generate sequence with baseline model (greedy for reproducibility)
            baseline_output = self.model.generate(
                input_ids,
                max_new_tokens=max_positions,
                do_sample=False
            )

            # Evaluate reconstruction at each generated position
            for pos in range(prompt_len, min(len(baseline_output[0]), prompt_len + max_positions)):
                current_input = baseline_output[:, :pos]
                next_token = baseline_output[0, pos].item()

                metrics = self.compute_reconstruction_metrics(current_input, next_token)
                metrics['position'] = pos - prompt_len  # 0-indexed from start of generation
                metrics_list.append(metrics)

        # Aggregate metrics
        if not metrics_list:
            return {'error': 'No positions to evaluate'}

        aggregated = {
            'num_positions': len(metrics_list),
            'prompt': prompt,
            'mean_kl_divergence': np.mean([m['kl_divergence'] for m in metrics_list]),
            'mean_top1_agreement': np.mean([m['top1_agreement'] for m in metrics_list]),
            'mean_top5_overlap': np.mean([m['top5_overlap'] for m in metrics_list]),
            'mean_top10_overlap': np.mean([m['top10_overlap'] for m in metrics_list]),
            'mean_cosine_similarity': np.mean([m['cosine_similarity'] for m in metrics_list]),
            'mean_ce_baseline': np.mean([m['ce_baseline'] for m in metrics_list]),
            'mean_ce_reconstructed': np.mean([m['ce_reconstructed'] for m in metrics_list]),
            'mean_ce_increase': np.mean([m['ce_increase'] for m in metrics_list]),
            'per_position': metrics_list,
        }

        return aggregated

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> str:
        """Generate text using SAE reconstruction at target layer.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            Generated text
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_tokens = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Run layers 0-31
                hidden_states = self._run_lower_layers(input_ids)

                # Reconstruct through SAE
                hidden_states = self._reconstruct_hidden_states(hidden_states)

                # Run layers 32-61
                logits = self._run_upper_layers(hidden_states)

                # Get logits for last position
                next_logits = logits[0, -1, :]

                # Sample next token
                if temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    if top_k > 0:
                        top_k_probs, top_k_indices = torch.topk(probs, top_k)
                        top_k_probs = top_k_probs / top_k_probs.sum()
                        next_token = top_k_indices[torch.multinomial(top_k_probs, 1)].item()
                    else:
                        next_token = torch.multinomial(probs, 1).item()
                else:
                    next_token = next_logits.argmax().item()

                # Check for EOS
                if next_token == self.tokenizer.eos_token_id:
                    break

                generated_tokens.append(next_token)

                # Append to input for next iteration
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[next_token]], device=self.device)
                ], dim=1)

        generated_text = self.tokenizer.decode(generated_tokens)
        return generated_text

    def generate_baseline(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> str:
        """Generate text without SAE intervention (baseline)."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=True
            )

        baseline_text = self.tokenizer.decode(
            output[0][len(input_ids[0]):],
            skip_special_tokens=True
        )
        return baseline_text


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SAE reconstruction ceiling performance for text generation"
    )
    parser.add_argument(
        "--sae-model",
        type=Path,
        required=True,
        help="Path to SAE model weights"
    )
    parser.add_argument(
        "--feature-indices",
        type=Path,
        default=None,
        help="Path to feature indices pickle file (for top-k feature selection)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/gemma-3-27b-it",
        help="Gemma model name or path"
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        default=31,
        help="Layer to apply SAE reconstruction (default: 31)"
    )

    # Prompt options
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single input prompt for generation"
    )
    prompt_group.add_argument(
        "--use-corpus",
        action="store_true",
        help="Use prompts from PromptGenerator corpus"
    )

    # Corpus options
    parser.add_argument(
        "--styles",
        type=str,
        nargs="+",
        default=["qa", "story", "policy"],
        choices=["qa", "story", "policy", "paraphrase"],
        help="Prompt styles to use from corpus (default: qa story policy)"
    )
    parser.add_argument(
        "--prompts-per-style",
        type=int,
        default=5,
        help="Number of prompts to use per style (default: 5)"
    )

    # Generation options
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate per prompt"
    )

    # Other options
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on"
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also generate baseline text without SAE intervention"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save results as JSON (optional)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt generation"
    )
    parser.add_argument(
        "--eval-metrics",
        action="store_true",
        help="Evaluate reconstruction metrics (KL divergence, top-k agreement, etc.) instead of generating text"
    )
    parser.add_argument(
        "--binarize-features",
        action="store_true",
        help="Binarize feature activations (set positive values to 1.0) to test effect of magnitude"
    )
    parser.add_argument(
        "--rescale-reconstruction",
        action="store_true",
        help="Rescale reconstruction to match original hidden state's RMS norm (useful with --binarize-features)"
    )

    args = parser.parse_args()

    # Default to single prompt if neither specified
    if not args.use_corpus and args.prompt is None:
        args.prompt = "The capital of France is"

    print("="*80)
    print("SAE Reconstruction Ceiling Evaluation")
    print("="*80)
    print(f"SAE model: {args.sae_model}")
    print(f"Gemma model: {args.model_name}")
    print(f"Target layer: {args.target_layer}")
    if args.use_corpus:
        print(f"Using corpus prompts: {args.styles}")
        print(f"Prompts per style: {args.prompts_per_style}")
    else:
        print(f"Prompt: {args.prompt}")
    print(f"Samples per prompt: {args.num_samples}")
    print(f"Device: {args.device}")
    print(f"Mode: {'metrics evaluation' if args.eval_metrics else 'text generation'}")
    print(f"Binarize features: {args.binarize_features}")
    print(f"Rescale reconstruction: {args.rescale_reconstruction}")
    print("="*80 + "\n")

    # Load SAE model
    print("Loading SAE model...")
    sae_model = SAEModel(args.sae_model)
    print(f"SAE: {sae_model.n_features} features, {sae_model.d_model} dimensions")

    # Load feature indices if provided
    feature_indices = None
    if args.feature_indices:
        print(f"Loading feature indices from {args.feature_indices}...")
        with open(args.feature_indices, 'rb') as f:
            feature_data = pickle.load(f)
            feature_indices = feature_data['indices']
        print(f"Using {len(feature_indices)} selected features (zeroing others)")
    else:
        print("Using all SAE features")

    # Load Gemma model
    print(f"Loading Gemma model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device_map=args.device if args.device == "cuda" else None,
    )

    if args.device == "cpu":
        model = model.to(args.device)

    print("Model loaded!\n")

    # Create generator
    generator = SAECeilingGenerator(
        sae_model=sae_model,
        tokenizer=tokenizer,
        model=model,
        feature_indices=feature_indices,
        target_layer=args.target_layer,
        device=args.device,
        binarize_features=args.binarize_features,
        rescale_reconstruction=args.rescale_reconstruction
    )

    # Build prompt list
    if args.use_corpus:
        prompt_generator = PromptGenerator(seed=args.seed)
        # Generate enough prompts: need prompts_per_style for each of 3 base styles
        # Formula in generate_corpus: base_prompts_per_style = num_prompts // (3 * (1 + num_paraphrases))
        # So we need: num_prompts = desired_per_style * 3 * (1 + num_paraphrases)
        corpus = prompt_generator.generate_corpus(
            num_prompts=args.prompts_per_style * 3 * 4,  # 3 styles, (1 + 3 paraphrases)
            num_paraphrases=3
        )

        prompts = []
        for style in args.styles:
            style_prompts = corpus.get(style, [])[:args.prompts_per_style]
            for prompt_text, prompt_id, family_id in style_prompts:
                prompts.append({
                    "text": prompt_text,
                    "style": style,
                    "id": prompt_id,
                    "family_id": family_id
                })
        print(f"Using {len(prompts)} prompts from corpus\n")
    else:
        prompts = [{"text": args.prompt, "style": "custom", "id": "custom_0", "family_id": 0}]

    # Branch based on evaluation mode
    if args.eval_metrics:
        # ========================================
        # METRICS EVALUATION MODE (with generations)
        # ========================================
        print("Running reconstruction metrics evaluation...\n")

        all_metrics = []
        generation_results = []
        pbar = tqdm(prompts, desc="Evaluating prompts")

        for prompt_info in pbar:
            prompt_text = prompt_info["text"]
            pbar.set_postfix_str(f"{prompt_text[:30]}...")

            metrics = generator.evaluate_reconstruction(
                prompt_text,
                max_positions=args.max_tokens
            )
            metrics["style"] = prompt_info["style"]
            metrics["id"] = prompt_info["id"]
            metrics["family_id"] = prompt_info["family_id"]
            all_metrics.append(metrics)

            # Also generate text samples for display
            prompt_results = {
                "prompt": prompt_text,
                "style": prompt_info["style"],
                "id": prompt_info["id"],
                "family_id": prompt_info["family_id"],
                "ceiling_samples": [],
                "baseline_samples": []
            }

            for sample_idx in range(args.num_samples):
                ceiling_text = generator.generate(
                    prompt_text,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k
                )
                prompt_results["ceiling_samples"].append(ceiling_text)

            if args.compare_baseline:
                for sample_idx in range(args.num_samples):
                    baseline_text = generator.generate_baseline(
                        prompt_text,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k
                    )
                    prompt_results["baseline_samples"].append(baseline_text)

            generation_results.append(prompt_results)

        pbar.close()

        # Aggregate across all prompts
        print("\n" + "="*80)
        print("RECONSTRUCTION METRICS")
        print("="*80)

        # Overall aggregates
        overall = {
            'mean_kl_divergence': np.mean([m['mean_kl_divergence'] for m in all_metrics]),
            'mean_top1_agreement': np.mean([m['mean_top1_agreement'] for m in all_metrics]),
            'mean_top5_overlap': np.mean([m['mean_top5_overlap'] for m in all_metrics]),
            'mean_top10_overlap': np.mean([m['mean_top10_overlap'] for m in all_metrics]),
            'mean_cosine_similarity': np.mean([m['mean_cosine_similarity'] for m in all_metrics]),
            'mean_ce_increase': np.mean([m['mean_ce_increase'] for m in all_metrics]),
        }

        print(f"\nOverall (across {len(all_metrics)} prompts, {sum(m['num_positions'] for m in all_metrics)} positions):")
        print(f"  KL Divergence:      {overall['mean_kl_divergence']:.4f}")
        print(f"  Top-1 Agreement:    {overall['mean_top1_agreement']:.1%}")
        print(f"  Top-5 Overlap:      {overall['mean_top5_overlap']:.1%}")
        print(f"  Top-10 Overlap:     {overall['mean_top10_overlap']:.1%}")
        print(f"  Cosine Similarity:  {overall['mean_cosine_similarity']:.4f}")
        print(f"  CE Increase:        {overall['mean_ce_increase']:.4f} nats")

        # Per-style breakdown if using corpus
        if args.use_corpus:
            print("\nPer-style breakdown:")
            styles = set(m['style'] for m in all_metrics)
            for style in sorted(styles):
                style_metrics = [m for m in all_metrics if m['style'] == style]
                print(f"\n  [{style.upper()}] ({len(style_metrics)} prompts)")
                print(f"    KL Divergence:   {np.mean([m['mean_kl_divergence'] for m in style_metrics]):.4f}")
                print(f"    Top-1 Agreement: {np.mean([m['mean_top1_agreement'] for m in style_metrics]):.1%}")
                print(f"    Top-5 Overlap:   {np.mean([m['mean_top5_overlap'] for m in style_metrics]):.1%}")
                print(f"    CE Increase:     {np.mean([m['mean_ce_increase'] for m in style_metrics]):.4f} nats")

        # Print generations
        print("\n" + "="*80)
        print("GENERATIONS")
        print("="*80)

        for result in generation_results:
            print(f"\n[{result['style'].upper()}] {result['prompt']}")
            print("-" * 60)

            for i, ceiling in enumerate(result["ceiling_samples"]):
                print(f"  Ceiling #{i+1}: {ceiling[:200]}{'...' if len(ceiling) > 200 else ''}")

            if result["baseline_samples"]:
                print()
                for i, baseline in enumerate(result["baseline_samples"]):
                    print(f"  Baseline #{i+1}: {baseline[:200]}{'...' if len(baseline) > 200 else ''}")

        # Save results if output path provided
        if args.output:
            output_data = {
                "config": {
                    "sae_model": str(args.sae_model),
                    "model_name": args.model_name,
                    "target_layer": args.target_layer,
                    "feature_indices": str(args.feature_indices) if args.feature_indices else None,
                    "num_features": len(feature_indices) if feature_indices is not None else sae_model.n_features,
                    "max_tokens": args.max_tokens,
                    "binarize_features": args.binarize_features,
                    "rescale_reconstruction": args.rescale_reconstruction,
                    "mode": "eval_metrics",
                    "timestamp": datetime.now().isoformat()
                },
                "overall": overall,
                "per_prompt": all_metrics,
                "generations": generation_results
            }

            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {args.output}")

    else:
        # ========================================
        # GENERATION MODE
        # ========================================
        results = []

        total_generations = len(prompts) * args.num_samples
        if args.compare_baseline:
            total_generations *= 2

        pbar = tqdm(total=total_generations, desc="Generating")

        for prompt_info in prompts:
            prompt_text = prompt_info["text"]
            prompt_results = {
                "prompt": prompt_text,
                "style": prompt_info["style"],
                "id": prompt_info["id"],
                "family_id": prompt_info["family_id"],
                "ceiling_samples": [],
                "baseline_samples": []
            }

            # Generate SAE ceiling samples
            for sample_idx in range(args.num_samples):
                ceiling_text = generator.generate(
                    prompt_text,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k
                )
                prompt_results["ceiling_samples"].append(ceiling_text)
                pbar.update(1)

            # Generate baseline samples
            if args.compare_baseline:
                for sample_idx in range(args.num_samples):
                    baseline_text = generator.generate_baseline(
                        prompt_text,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k
                    )
                    prompt_results["baseline_samples"].append(baseline_text)
                    pbar.update(1)

            results.append(prompt_results)

        pbar.close()

        # Print results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)

        for result in results:
            print(f"\n[{result['style'].upper()}] {result['prompt']}")
            print("-" * 60)

            for i, ceiling in enumerate(result["ceiling_samples"]):
                print(f"  Ceiling #{i+1}: {ceiling[:200]}{'...' if len(ceiling) > 200 else ''}")

            if result["baseline_samples"]:
                print()
                for i, baseline in enumerate(result["baseline_samples"]):
                    print(f"  Baseline #{i+1}: {baseline[:200]}{'...' if len(baseline) > 200 else ''}")

        # Save results if output path provided
        if args.output:
            output_data = {
                "config": {
                    "sae_model": str(args.sae_model),
                    "model_name": args.model_name,
                    "target_layer": args.target_layer,
                    "feature_indices": str(args.feature_indices) if args.feature_indices else None,
                    "num_features": len(feature_indices) if feature_indices is not None else sae_model.n_features,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "num_samples": args.num_samples,
                    "binarize_features": args.binarize_features,
                    "rescale_reconstruction": args.rescale_reconstruction,
                    "mode": "generation",
                    "timestamp": datetime.now().isoformat()
                },
                "results": results
            }

            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
