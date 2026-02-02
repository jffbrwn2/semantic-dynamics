"""Configuration for SAE feature dynamics analysis."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration for the feature dynamics experiment."""

    # Model configuration
    model_name: str = "google/gemma-2-27b-it"
    layer_idx: int = 20  # Layer to extract SAE activations from

    # SAE configuration
    # Gemma Scope 2 SAE hook point format: "blocks.{layer}.hook_resid_post"
    sae_release: str = "gemma-scope-2-27b-it-resid_post"  # Gemma Scope 2 release name
    sae_id: str = None  # Will be set based on layer_idx (format: "layer_X/width_16k/average_l0_71")
    top_k_features: int = 512  # Restrict to top-K features by variance

    # Data generation
    num_prompts: int = 200  # Total prompts across all styles
    num_tokens_per_prompt: int = 300  # Target generation length
    max_tokens: int = 400  # Max tokens to generate
    min_tokens: int = 200  # Min tokens to generate

    # Prompt styles
    prompt_styles: list = None
    num_paraphrases: int = 3  # Number of paraphrases per base prompt

    # Train/test split
    test_size: float = 0.25  # Hold out 25% of prompt families
    random_seed: int = 42

    # Model fitting
    ridge_alpha: float = 10.0  # Ridge regression regularization
    fit_per_feature: bool = True  # Fit separate model per feature vs joint

    # Output
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path(".cache")

    def __post_init__(self):
        """Initialize derived configuration."""
        if self.prompt_styles is None:
            self.prompt_styles = [
                "qa",  # Plain Q&A
                "story",  # Creative writing
                "policy",  # Policy/safety questions
                "paraphrase"  # Paraphrased versions
            ]

        if self.sae_id is None:
            # Format: "layer_20/width_16k/average_l0_71"
            # Using 16k width and medium sparsity as default
            self.sae_id = f"layer_{self.layer_idx}/width_16k/average_l0_71"

        # Create directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    @property
    def hook_point(self) -> str:
        """Get the hook point name for the SAE."""
        return f"blocks.{self.layer_idx}.hook_resid_post"
