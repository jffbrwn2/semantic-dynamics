"""Text generation using predictor-driven SAE dynamics.

This script demonstrates "hacking" Gemma by using learned dynamics models
to predict SAE features at layer 31, then decoding them back to the residual
stream and running the remaining layers (32-61) to generate text.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask
)

from feature_dynamics.predictors import load_predictor


class SAEModel:
    """Wrapper for SAE encoder/decoder with JumpReLU activation."""

    def __init__(self, sae_path: Path):
        """Load SAE model from path.

        Args:
            sae_path: Path to SAE model weights
        """
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
        """Encode residual stream to SAE features.

        Args:
            residual: (d_model,) residual stream vector

        Returns:
            SAE features (n_features,) after activation
        """
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
        """Decode SAE features back to residual stream.

        Args:
            features: (n_features,) SAE feature activations

        Returns:
            Reconstructed residual stream (d_model,)
        """
        residual = features @ self.decoder_weight
        if self.decoder_bias is not None:
            residual = residual + self.decoder_bias
        return residual


class PredictorGenerator:
    """Text generator using predictor-driven dynamics."""

    def __init__(
        self,
        predictor,
        sae_model: SAEModel,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        feature_indices: Optional[np.ndarray] = None,
        target_layer: int = 31,
        device: str = "cpu"
    ):
        """Initialize generator.

        Args:
            predictor: Trained predictor (linear or RNN)
            sae_model: SAE encoder/decoder
            tokenizer: Gemma tokenizer
            model: Gemma model
            feature_indices: Optional indices to select subset of SAE features
            target_layer: Layer where we intervene with predicted features
            device: Device to run on
        """
        self.predictor = predictor
        self.sae_model = sae_model
        self.tokenizer = tokenizer
        self.model = model
        self.feature_indices = feature_indices
        self.target_layer = target_layer
        self.device = device

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
        # For multimodal Gemma3, layer_types and other settings are in text_config
        # We need to use text_config for mask creation as well
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
        """Create attention masks for different layer types.

        Args:
            input_embeds: Input embeddings (batch, seq_len, d_model)
            cache_position: Current cache positions
            past_key_values: Optional KV cache
            position_ids: Optional position IDs

        Returns:
            Dictionary mapping attention type to mask tensor
        """
        # Base mask arguments - use text_config for multimodal models
        mask_kwargs = {
            "config": self.text_config,
            "input_embeds": input_embeds,
            "attention_mask": None,  # We use default causal masking
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }

        causal_mask_mapping = {}

        # Create full attention mask
        if self.layer_types and "full_attention" in self.layer_types:
            causal_mask_mapping["full_attention"] = create_causal_mask(**mask_kwargs)

        # Create sliding window attention mask
        if self.layer_types and "sliding_attention" in self.layer_types:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
                **mask_kwargs
            )

        return causal_mask_mapping

    def _run_prefix(self, input_ids: torch.Tensor) -> Dict:
        """Run model on prefix to get initial state and build KV cache.

        Args:
            input_ids: (seq_len,) token IDs

        Returns:
            Dictionary with:
                - sae_features: encoded SAE features for last token at layer 31
                - past_key_values: KV cache for layers 32-61
                - prefix_len: length of prefix
        """
        with torch.no_grad():
            input_ids = input_ids.unsqueeze(0).to(self.device)  # (1, seq_len)
            prefix_len = input_ids.shape[1]

            # Prepare position inputs
            position_ids = torch.arange(0, prefix_len, device=self.device).unsqueeze(0)
            cache_position = torch.arange(0, prefix_len, device=self.device)

            # Compute embeddings
            hidden_states = self.embed_tokens(input_ids)  # (1, prefix_len, d_model)

            # Create attention masks for prefix
            causal_mask_mapping = self._create_causal_mask_mapping(
                input_embeds=hidden_states,
                cache_position=cache_position,
                past_key_values=None,
                position_ids=position_ids,
            )

            # Compute position embeddings for both attention types
            position_embeddings_global = self.lm.rotary_emb(hidden_states, position_ids)
            position_embeddings_local = self.lm.rotary_emb_local(hidden_states, position_ids)

            # Run through layers 0 to target_layer (0-31)
            for layer_idx in range(self.target_layer + 1):
                layer = self.layers[layer_idx]

                # Get mask for this layer's attention type
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

            # hidden_states now contains layer 31 output: (1, prefix_len, d_model)

            # Extract SAE features from ALL positions
            all_hidden = hidden_states[0].float().cpu().numpy()  # (prefix_len, d_model)
            all_sae_features = np.array([
                self.sae_model.encode(all_hidden[t])
                for t in range(prefix_len)
            ])  # (prefix_len, n_features)

            # Last token features (for backward compatibility with non-LFADS predictors)
            last_sae_features = all_sae_features[-1]

            # Apply feature selection if indices provided
            if self.feature_indices is not None:
                all_sae_features = all_sae_features[:, self.feature_indices]
                last_sae_features = last_sae_features[self.feature_indices]

            # Now build KV cache for layers 32-61
            past_key_values = DynamicCache()

            for layer_idx in range(self.target_layer + 1, self.num_layers):
                layer = self.layers[layer_idx]

                # Get mask for this layer
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
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                hidden_states = layer_outputs[0]

        return {
            'all_sae_features': all_sae_features,  # (prefix_len, n_features)
            'sae_features': last_sae_features,     # (n_features,) - backward compat
            'past_key_values': past_key_values,
            'prefix_len': prefix_len
        }

    def _run_upper_layers(
        self,
        features: np.ndarray,
        past_key_values: DynamicCache,
        current_position: int
    ) -> tuple[torch.Tensor, DynamicCache]:
        """Run layers 32-61 on reconstructed residual stream with KV cache.

        Args:
            features: (n_features,) SAE feature activations (possibly subset)
            past_key_values: KV cache from previous tokens
            current_position: Current position in sequence

        Returns:
            Tuple of (logits, updated_cache)
                - logits: (vocab_size,) logits for next token
                - updated_cache: Updated KV cache
        """
        with torch.no_grad():
            # Expand features back to full space if using subset
            if self.feature_indices is not None:
                full_features = np.zeros(self.sae_model.n_features)
                full_features[self.feature_indices] = features
                features = full_features

            # Decode features to residual stream and convert to tensor
            residual = self.sae_model.decode(features)
            hidden_states = torch.tensor(residual, dtype=self.model.dtype, device=self.device).unsqueeze(0).unsqueeze(0)
            # Shape: (1, 1, d_model) - batch=1, seq=1

            # Prepare position inputs for single new token
            position_ids = torch.tensor([[current_position]], device=self.device)
            cache_position = torch.tensor([current_position], device=self.device)

            # Create attention masks for generation (single new token)
            causal_mask_mapping = self._create_causal_mask_mapping(
                input_embeds=hidden_states,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

            # Compute position embeddings for new position
            position_embeddings_global = self.lm.rotary_emb(hidden_states, position_ids)
            position_embeddings_local = self.lm.rotary_emb_local(hidden_states, position_ids)

            # Run through layers 32-61 with cache
            for layer_idx in range(self.target_layer + 1, self.num_layers):
                layer = self.layers[layer_idx]

                # Get mask for this layer
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
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                hidden_states = layer_outputs[0]

            # Apply final norm
            hidden_states = self.norm(hidden_states)

            # Get logits
            logits = self.lm_head(hidden_states)  # (1, 1, vocab_size)
            logits = logits[0, 0, :]  # (vocab_size,)

        return logits, past_key_values

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> str:
        """Generate text using predictor-driven dynamics.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            Generated text
        """
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0]

        # Run prefix through model to build initial state and KV cache
        prefix_output = self._run_prefix(input_ids)

        # Initialize predictor with prefix features
        # LFADS predictors need full prefix sequence, others need only last token
        if hasattr(self.predictor, 'needs_full_prefix') and self.predictor.needs_full_prefix:
            self.predictor.reset(prefix_output['all_sae_features'])
        else:
            self.predictor.reset(prefix_output['sae_features'])

        # Get last token embedding for prediction
        last_token_id = input_ids[-1].item()
        last_token_embed = self.predictor.embed_matrix[last_token_id]

        # Initialize generation state
        generated_tokens = []
        current_features = prefix_output['sae_features']
        current_token_embed = last_token_embed
        past_key_values = prefix_output['past_key_values']
        current_position = prefix_output['prefix_len']

        for _ in range(max_new_tokens):
            # Predict next SAE features using dynamics model
            predicted_features = self.predictor.predict_next(
                current_token_embed,
                previous_features=current_features
            )

            # Run upper layers (32-61) to get logits
            # _run_upper_layers will expand features and decode to residual
            logits, past_key_values = self._run_upper_layers(
                predicted_features,
                past_key_values,
                current_position
            )

            # Sample next token
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                if top_k > 0:
                    # Top-k filtering
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    top_k_probs = top_k_probs / top_k_probs.sum()
                    next_token = top_k_indices[torch.multinomial(top_k_probs, 1)].item()
                else:
                    next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = logits.argmax().item()

            # Check for EOS
            if next_token == self.tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token)

            # Update state for next iteration
            current_features = predicted_features
            current_token_embed = self.predictor.embed_matrix[next_token]
            current_position += 1

        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_tokens)

        return generated_text


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using predictor-driven SAE dynamics"
    )
    parser.add_argument(
        "--predictor",
        type=Path,
        required=True,
        help="Path to trained predictor (.pkl file)"
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
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Input prompt for generation"
    )
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
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on"
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also generate baseline text with normal Gemma for comparison"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save output (JSON format with prompt, generated text, etc.)"
    )

    args = parser.parse_args()

    print("="*80)
    print("Predictor-Driven Text Generation")
    print("="*80)
    print(f"Predictor: {args.predictor}")
    print(f"SAE model: {args.sae_model}")
    print(f"Gemma model: {args.model_name}")
    print(f"Prompt: {args.prompt}")
    print(f"Device: {args.device}")
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
        print(f"Using {len(feature_indices)} selected features")

    # Load Gemma model (needed before loading LFADS predictors for embed_matrix)
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

    # Get embedding matrix for LFADS predictors
    embed_matrix = model.model.language_model.embed_tokens.weight.detach().cpu().numpy()

    # Load predictor (LFADS needs embed_matrix)
    print("Loading predictor...")
    predictor = load_predictor(args.predictor, embed_matrix=embed_matrix, device=args.device)
    print(f"Loaded predictor: {type(predictor).__name__}")

    # Create generator
    generator = PredictorGenerator(
        predictor=predictor,
        sae_model=sae_model,
        tokenizer=tokenizer,
        model=model,
        feature_indices=feature_indices,
        device=args.device
    )

    # Generate with predictor
    print("Generating with predictor-driven dynamics...")
    print("-" * 80)
    generated_text = generator.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )
    print(f"Prompt: {args.prompt}")
    print(f"Generated: {generated_text}")
    print("-" * 80 + "\n")

    # Optional baseline comparison
    baseline_text = None
    if args.compare_baseline:
        print("Generating baseline (normal Gemma)...")
        print("-" * 80)
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                do_sample=True
            )
        baseline_text = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
        print(f"Prompt: {args.prompt}")
        print(f"Generated: {baseline_text}")
        print("-" * 80 + "\n")

    # Save output if requested
    if args.output:
        output_data = {
            "prompt": args.prompt,
            "generated": generated_text,
            "predictor": str(args.predictor),
            "predictor_type": type(predictor).__name__,
            "model": args.model_name,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
        }
        if baseline_text is not None:
            output_data["baseline"] = baseline_text

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
