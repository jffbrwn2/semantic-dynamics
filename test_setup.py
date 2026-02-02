"""Quick test script to verify setup works."""

import sys
from pathlib import Path

# Test imports
print("Testing imports...")
try:
    from feature_dynamics import Config, PromptGenerator
    print("✓ Core imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test configuration
print("\nTesting configuration...")
try:
    config = Config(
        layer_idx=10,
        top_k_features=128,
        num_prompts=20,
    )
    print(f"✓ Config created: layer={config.layer_idx}, top_k={config.top_k_features}")
except Exception as e:
    print(f"✗ Config error: {e}")
    sys.exit(1)

# Test prompt generation
print("\nTesting prompt generation...")
try:
    generator = PromptGenerator(seed=42)
    corpus = generator.generate_corpus(num_prompts=20, num_paraphrases=2)

    total_prompts = sum(len(prompts) for prompts in corpus.values())
    print(f"✓ Generated {total_prompts} prompts across {len(corpus)} styles")

    for style, prompts in corpus.items():
        print(f"  - {style}: {len(prompts)} prompts")

except Exception as e:
    print(f"✗ Prompt generation error: {e}")
    sys.exit(1)

# Test train/test split
print("\nTesting train/test split...")
try:
    train_corpus, test_corpus = generator.split_corpus(corpus, test_size=0.25)

    train_count = sum(len(prompts) for prompts in train_corpus.values())
    test_count = sum(len(prompts) for prompts in test_corpus.values())

    print(f"✓ Split successful: {train_count} train, {test_count} test")

except Exception as e:
    print(f"✗ Split error: {e}")
    sys.exit(1)

# Test directory creation
print("\nTesting directory setup...")
try:
    config.output_dir.mkdir(exist_ok=True, parents=True)
    config.cache_dir.mkdir(exist_ok=True, parents=True)
    print(f"✓ Directories created:")
    print(f"  - {config.output_dir}")
    print(f"  - {config.cache_dir}")
except Exception as e:
    print(f"✗ Directory error: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ All tests passed! Setup is working correctly.")
print("="*60)
print("\nNext steps:")
print("1. For a full test run on CPU (may take a while):")
print("   uv run feature-dynamics --device cpu --layer 10 --top-k 128 --num-prompts 20")
print("\n2. For GPU-based production run:")
print("   uv run feature-dynamics --device cuda --layer 20 --top-k 512 --num-prompts 200")
print("\n3. To skip data collection after first run:")
print("   uv run feature-dynamics --skip-collection")
print("="*60)
