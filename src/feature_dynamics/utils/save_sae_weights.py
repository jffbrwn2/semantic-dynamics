"""Extract and save SAE weights from sae_lens to pickle format.

This utility loads an SAE from Gemma Scope and saves the encoder/decoder
weights to a pickle file for use in generation.
"""

import argparse
import pickle
from pathlib import Path

import torch
from sae_lens import SAE

from ..config import Config


def extract_sae_weights(sae: SAE) -> dict:
    """Extract weights from SAE object.

    Args:
        sae: Loaded SAE from sae_lens

    Returns:
        Dictionary with encoder/decoder weights and JumpReLU threshold
    """
    # Detect activation function from config class name or threshold presence
    config_class_name = type(sae.cfg).__name__
    if 'JumpReLU' in config_class_name or hasattr(sae, 'threshold'):
        activation_fn = 'jumprelu'
    else:
        activation_fn = 'relu'

    weights = {
        'encoder_weight': sae.W_enc.detach().cpu().numpy(),  # (d_model, n_features)
        'encoder_bias': sae.b_enc.detach().cpu().numpy(),     # (n_features,)
        'decoder_weight': sae.W_dec.detach().cpu().numpy(),  # (n_features, d_model)
        'decoder_bias': sae.b_dec.detach().cpu().numpy() if hasattr(sae, 'b_dec') else None,  # (d_model,) or None
        'threshold': sae.threshold.detach().cpu().numpy() if hasattr(sae, 'threshold') else None,  # (n_features,) for JumpReLU
        'activation_fn': activation_fn,
    }

    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save SAE weights from Gemma Scope"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/jffbrwn/orcd/pool/semantic-dynamics/outputs/sae_weights.pkl"),
        help="Output path for SAE weights pickle file"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=31,
        help="Layer index for SAE (default: 31)"
    )
    parser.add_argument(
        "--sae-release",
        type=str,
        default="gemma-scope-2-27b-it-res",
        help="SAE release name from Gemma Scope"
    )
    parser.add_argument(
        "--sae-id",
        type=str,
        default=None,
        help="SAE ID (e.g., 'layer_31_width_16k_l0_medium'). If not specified, uses layer index to construct ID."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to load SAE on"
    )

    args = parser.parse_args()

    # Construct SAE ID if not provided
    if args.sae_id is None:
        args.sae_id = f"layer_{args.layer}_width_16k_l0_medium"

    print("="*80)
    print("SAE Weight Extraction")
    print("="*80)
    print(f"Release: {args.sae_release}")
    print(f"SAE ID: {args.sae_id}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    print("="*80 + "\n")

    # Load SAE from Gemma Scope
    print("Loading SAE from Gemma Scope...")
    try:
        # Try new API first (returns only SAE object)
        sae = SAE.from_pretrained(
            release=args.sae_release,
            sae_id=args.sae_id,
            device=args.device
        )
    except TypeError:
        # Fall back to old API if needed
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=args.sae_release,
            sae_id=args.sae_id,
            device=args.device
        )
    print(f"SAE loaded: {sae.cfg.d_in} input dims, {sae.cfg.d_sae} features\n")

    # Extract weights
    print("Extracting weights...")
    weights = extract_sae_weights(sae)

    print(f"Encoder weight shape: {weights['encoder_weight'].shape}")
    print(f"Encoder bias shape: {weights['encoder_bias'].shape}")
    print(f"Decoder weight shape: {weights['decoder_weight'].shape}")
    if weights['decoder_bias'] is not None:
        print(f"Decoder bias shape: {weights['decoder_bias'].shape}")
    else:
        print("Decoder bias: None")
    print(f"Activation function: {weights['activation_fn']}")
    if weights['threshold'] is not None:
        print(f"Threshold shape: {weights['threshold'].shape}")
    else:
        print("Threshold: None (using standard ReLU)")
    print()

    # Save to pickle
    print(f"Saving to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(weights, f)

    # Get file size
    file_size = args.output.stat().st_size / (1024 ** 2)  # MB
    print(f"Saved! File size: {file_size:.1f} MB")
    print("\nYou can now use this file with generate.py:")
    print(f"  python generate.py --sae-model {args.output} --predictor <predictor.pkl> --prompt 'Your prompt'")


if __name__ == "__main__":
    main()
