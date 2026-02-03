"""Collect SAE activations and token data from model generations."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE 

from config import Config


class DataCollector:
    """Collect SAE activations from model generations."""

    def __init__(self, config: Config, device: str = "cpu"):
        """Initialize data collector.

        Args:
            config: Configuration object
            device: Device to run model on
        """
        self.config = config
        self.device = device

        print(f"Loading model {config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Ensure pad_token is set (required for Gemma models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None,
        )

        if device == "cpu":
            self.model = self.model.to(device)

        self.model.eval()

        print(f"Loading SAE for layer {config.layer_idx}...")
        self.sae = self._load_sae()

    def _load_sae(self) -> SAE:
        """Load the SAE from Gemma Scope.

        Returns:
            Loaded SAE model
        """
        # Load SAE from Gemma Scope 2
        # Format: release="gemma-scope-2-27b-it-resid_post"
        #         sae_id="layer_20/width_16k/average_l0_71"
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=self.config.sae_release,
            sae_id=self.config.sae_id,
            device=str(self.device)
        )
        sae.eval()
        return sae

    def generate_and_collect(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Optional[Dict]:
        """Generate text and collect SAE activations.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Dictionary containing:
                - sae_acts: (seq_len, n_features) SAE encoder activations
                - token_ids: (seq_len,) generated token IDs
                - tokens: (seq_len,) generated tokens (strings)
                - metadata: per-token metadata (position, is_newline, is_punct, etc.)
        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        # Storage for activations
        all_sae_acts = []
        all_token_ids = []
        all_metadata = []

        # Generate with hooks to collect activations
        with torch.no_grad():
            try:
                generated = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )
            except RuntimeError as e:
                print(f"Error during generation: {e}")
                print(f"Prompt: {prompt[:100]}...")
                return None

            # Get generated token IDs (excluding prompt)
            generated_ids = generated.sequences[0, prompt_len:]

            # Extract hidden states for the layer of interest
            # hidden_states is a tuple of length (num_generated_tokens,)
            # Each element is tuple of (num_layers + 1,) tensors
            for step_idx in range(len(generated.hidden_states)):
                # Get hidden state at our layer
                # hidden_states[step][layer_idx] has shape (batch=1, seq=1, d_model)
                hidden = generated.hidden_states[step_idx][self.config.layer_idx + 1]
                hidden = hidden[0, -1, :]  # (d_model,)

                # Pass through SAE encoder
                sae_acts = self.sae.encode(hidden.unsqueeze(0))  # (1, n_features)
                sae_acts = sae_acts.squeeze(0).cpu().numpy()  # (n_features,)

                # Get token info
                token_id = generated_ids[step_idx].item()
                token = self.tokenizer.decode([token_id])

                # Metadata
                metadata = {
                    'position': step_idx,
                    'is_newline': '\n' in token,
                    'is_punct': token.strip() in '.,!?;:',
                    'is_space': token.strip() == '',
                }

                all_sae_acts.append(sae_acts)
                all_token_ids.append(token_id)
                all_metadata.append(metadata)

        # Check if generation is long enough
        if len(all_token_ids) < self.config.min_tokens:
            print(f"Warning: Generation too short ({len(all_token_ids)} tokens), skipping")
            return None

        return {
            'sae_acts': np.array(all_sae_acts),  # (seq_len, n_features)
            'token_ids': np.array(all_token_ids),  # (seq_len,)
            'tokens': [self.tokenizer.decode([tid]) for tid in all_token_ids],
            'metadata': all_metadata,
            'prompt': prompt,
        }

    def select_top_features(
        self,
        dataset: List[Dict],
        top_k: int = None
    ) -> Tuple[np.ndarray, List[int]]:
        """Select top-K features by variance across dataset.

        Args:
            dataset: List of data dictionaries from generate_and_collect
            top_k: Number of features to select (uses config if None)

        Returns:
            (indices of top-K features, feature variances)
        """
        if top_k is None:
            top_k = self.config.top_k_features

        # Concatenate all SAE activations
        all_acts = np.concatenate([d['sae_acts'] for d in dataset], axis=0)

        # Compute variance per feature
        feature_vars = np.var(all_acts, axis=0)

        # Select top-K
        top_k_indices = np.argsort(feature_vars)[-top_k:]

        return top_k_indices, feature_vars

    def collect_corpus_data(
        self,
        corpus: Dict[str, List[Tuple[str, str, int]]],
        output_path: Path = None,
    ) -> List[Dict]:
        """Collect data for entire corpus.

        Args:
            corpus: Corpus dictionary from PromptGenerator
            output_path: Path to save collected data (optional)

        Returns:
            List of data dictionaries with additional fields:
                - style: prompt style
                - base_id: base prompt ID
                - family_id: prompt family ID
        """
        dataset = []

        # Flatten corpus
        all_prompts = []
        for style, prompts in corpus.items():
            for prompt, base_id, family_id in prompts:
                all_prompts.append((style, prompt, base_id, family_id))

        # Collect data for each prompt
        for style, prompt, base_id, family_id in tqdm(all_prompts, desc="Collecting data"):
            data = self.generate_and_collect(prompt)

            if data is not None:
                data['style'] = style
                data['base_id'] = base_id
                data['family_id'] = family_id
                dataset.append(data)

        # Save if output path provided
        if output_path is not None:
            output_path.parent.mkdir(exist_ok=True, parents=True)
            with open(output_path, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"Saved dataset to {output_path}")

        return dataset

    def prepare_feature_subset(
        self,
        dataset: List[Dict],
        top_k_indices: np.ndarray,
    ) -> List[Dict]:
        """Restrict dataset to top-K features.

        Args:
            dataset: Full dataset
            top_k_indices: Indices of features to keep

        Returns:
            Dataset with sae_acts restricted to top-K features
        """
        subset_dataset = []
        for data in dataset:
            subset_data = data.copy()
            subset_data['sae_acts'] = data['sae_acts'][:, top_k_indices]
            subset_dataset.append(subset_data)

        return subset_dataset
