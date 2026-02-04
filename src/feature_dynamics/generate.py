"""Text generation using predictor-driven SAE dynamics.

This script demonstrates "hacking" Gemma by using learned dynamics models
to predict SAE features at layer 31, then decoding them back to the residual
stream and running the remaining layers (32-61) to generate text.
"""

import argparse
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

from predictors import load_predictor


class SAEModel:
    """Wrapper for SAE encoder/decoder."""

    def __init__(self, sae_path: Path):
        """Load SAE model from path.

        Args:
            sae_path: Path to SAE model weights
        """
        with open(sae_path, 'rb') as f:
            sae_data = pickle.load(f)

        # Extract encoder and decoder
        # Format depends on how SAE is saved
        self.encoder_weight = sae_data['encoder_weight']  # (d_model, n_features)
        self.encoder_bias = sae_data['encoder_bias']  # (n_features,)
        self.decoder_weight = sae_data['decoder_weight']  # (n_features, d_model)
        self.decoder_bias = sae_data.get('decoder_bias', None)  # (d_model,) or None

        self.n_features = self.encoder_weight.shape[1]
        self.d_model = self.encoder_weight.shape[0]

    def encode(self, residual: np.ndarray) -> np.ndarray:
        """Encode residual stream to SAE features.

        Args:
            residual: (d_model,) residual stream vector

        Returns:
            SAE features (n_features,) after ReLU
        """
        pre_relu = residual @ self.encoder_weight + self.encoder_bias
        return np.maximum(0, pre_relu)

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
        target_layer: int = 31,
        device: str = "cpu"
    ):
        """Initialize generator.

        Args:
            predictor: Trained predictor (linear or RNN)
            sae_model: SAE encoder/decoder
            tokenizer: Gemma tokenizer
            model: Gemma model
            target_layer: Layer where we intervene with predicted features
            device: Device to run on
        """
        self.predictor = predictor
        self.sae_model = sae_model
        self.tokenizer = tokenizer
        self.model = model
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
        # For multimodal Gemma3, layer_types is in text_config
        if hasattr(self.config, 'text_config') and hasattr(self.config.text_config, 'layer_types'):
            self.layer_types = self.config.text_config.layer_types
        elif hasattr(self.config, 'layer_types'):
            self.layer_types = self.config.layer_types
        else:
            self.layer_types = None

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
        # Base mask arguments
        mask_kwargs = {
            "config": self.config,
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
            position_embeddings = {}
            if self.layer_types:
                for layer_type in set(self.layer_types):
                    position_embeddings[layer_type] = self.lm.rotary_emb(
                        hidden_states, position_ids, layer_type
                    )

            # Run through layers 0 to target_layer (0-31)
            for layer_idx in range(self.target_layer + 1):
                layer = self.layers[layer_idx]

                # Get position embedding and mask for this layer's attention type
                layer_pos_emb = None
                layer_mask = None
                if self.layer_types and layer_idx < len(self.layer_types):
                    layer_type = self.layer_types[layer_idx]
                    layer_pos_emb = position_embeddings.get(layer_type)
                    layer_mask = causal_mask_mapping.get(layer_type)

                layer_outputs = layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_embeddings=layer_pos_emb,
                    position_ids=position_ids,
                    cache_position=cache_position,
                )
                hidden_states = layer_outputs[0]

            # hidden_states now contains layer 31 output: (1, prefix_len, d_model)

            # Extract SAE features from last token
            last_hidden = hidden_states[0, -1].cpu().numpy()  # (d_model,)
            sae_features = self.sae_model.encode(last_hidden)

            # Now build KV cache for layers 32-61
            past_key_values = DynamicCache()

            for layer_idx in range(self.target_layer + 1, self.num_layers):
                layer = self.layers[layer_idx]

                # Get position embedding and mask for this layer
                layer_pos_emb = None
                layer_mask = None
                if self.layer_types and layer_idx < len(self.layer_types):
                    layer_type = self.layer_types[layer_idx]
                    layer_pos_emb = position_embeddings.get(layer_type)
                    layer_mask = causal_mask_mapping.get(layer_type)

                layer_outputs = layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_embeddings=layer_pos_emb,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    past_key_value=past_key_values,
                    use_cache=True,
                )
                hidden_states = layer_outputs[0]

        return {
            'sae_features': sae_features,
            'past_key_values': past_key_values,
            'prefix_len': prefix_len
        }

    def _run_upper_layers(
        self,
        residual: np.ndarray,
        past_key_values: DynamicCache,
        current_position: int
    ) -> tuple[torch.Tensor, DynamicCache]:
        """Run layers 32-61 on reconstructed residual stream with KV cache.

        Args:
            residual: (d_model,) residual stream vector at layer 31
            past_key_values: KV cache from previous tokens
            current_position: Current position in sequence

        Returns:
            Tuple of (logits, updated_cache)
                - logits: (vocab_size,) logits for next token
                - updated_cache: Updated KV cache
        """
        with torch.no_grad():
            # Convert residual to tensor
            hidden_states = torch.FloatTensor(residual).unsqueeze(0).unsqueeze(0).to(self.device)
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
            position_embeddings = {}
            if self.layer_types:
                for layer_type in set(self.layer_types):
                    position_embeddings[layer_type] = self.lm.rotary_emb(
                        hidden_states, position_ids, layer_type
                    )

            # Run through layers 32-61 with cache
            for layer_idx in range(self.target_layer + 1, self.num_layers):
                layer = self.layers[layer_idx]

                # Get position embedding and mask for this layer
                layer_pos_emb = None
                layer_mask = None
                if self.layer_types and layer_idx < len(self.layer_types):
                    layer_type = self.layer_types[layer_idx]
                    layer_pos_emb = position_embeddings.get(layer_type)
                    layer_mask = causal_mask_mapping.get(layer_type)

                layer_outputs = layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_embeddings=layer_pos_emb,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    past_key_value=past_key_values,
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

            # Decode features back to residual stream at layer 31
            reconstructed_residual = self.sae_model.decode(predicted_features)

            # Run upper layers (32-61) to get logits
            logits, past_key_values = self._run_upper_layers(
                reconstructed_residual,
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

    # Load predictor
    print("Loading predictor...")
    predictor = load_predictor(args.predictor)
    print(f"Loaded predictor: {type(predictor).__name__}")

    # Load SAE model
    print("Loading SAE model...")
    sae_model = SAEModel(args.sae_model)
    print(f"SAE: {sae_model.n_features} features, {sae_model.d_model} dimensions")

    # Load Gemma model
    print(f"Loading Gemma model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map=args.device
    )
    print("Model loaded!\n")

    # Create generator
    generator = PredictorGenerator(
        predictor=predictor,
        sae_model=sae_model,
        tokenizer=tokenizer,
        model=model,
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


if __name__ == "__main__":
    main()
