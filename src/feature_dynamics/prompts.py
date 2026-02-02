"""Generate corpus of prompts across different styles."""

import random
from typing import List, Tuple, Dict


class PromptGenerator:
    """Generate prompts across different styles for testing."""

    def __init__(self, seed: int = 42):
        """Initialize prompt generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)

        # Base prompts for each style
        self.qa_prompts = [
            "What is the capital of France?",
            "Explain how photosynthesis works.",
            "What are the main causes of climate change?",
            "How does a computer processor work?",
            "What is the difference between DNA and RNA?",
            "Explain the theory of relativity in simple terms.",
            "What are the primary functions of the human liver?",
            "How do neural networks learn?",
            "What is the water cycle?",
            "Explain how vaccines work.",
            "What causes ocean tides?",
            "How does gravity work?",
            "What is machine learning?",
            "Explain the process of evolution.",
            "What are antibiotics and how do they work?",
            "How does the internet work?",
            "What is quantum computing?",
            "Explain the greenhouse effect.",
            "What is artificial intelligence?",
            "How do rockets work?",
        ]

        self.story_prompts = [
            "Write a story about a robot learning to feel emotions.",
            "Tell me a story about a time traveler who gets stuck in the past.",
            "Write a short story about a detective solving a mysterious case.",
            "Create a story about an astronaut discovering a new planet.",
            "Write a story about a chef who can taste memories.",
            "Tell a story about a painter who brings their art to life.",
            "Write about a librarian who discovers books that predict the future.",
            "Create a story about a musician who can control emotions with music.",
            "Write a story about a gardener who grows impossible plants.",
            "Tell a story about an engineer building a bridge between worlds.",
            "Write about a dancer whose movements change reality.",
            "Create a story about a teacher who can see students' potential.",
            "Write a story about a photographer capturing moments from the future.",
            "Tell about a baker whose creations bring people together.",
            "Write a story about a weaver creating tapestries of time.",
        ]

        self.policy_prompts = [
            "What are your content policies regarding medical advice?",
            "How do you handle requests for harmful information?",
            "What guidelines do you follow for privacy and data protection?",
            "Can you explain your approach to bias and fairness?",
            "What are the limits of what you can help with?",
            "How do you handle controversial topics?",
            "What safety measures are in place for your responses?",
            "Can you help with illegal activities?",
            "What is your policy on generating misinformation?",
            "How do you ensure user safety in conversations?",
            "What are your ethical guidelines?",
            "How do you handle requests that might cause harm?",
            "What content do you refuse to generate?",
            "How do you approach potentially sensitive topics?",
            "What are your limitations as an AI assistant?",
        ]

    def generate_paraphrases(self, prompt: str, num_paraphrases: int = 3) -> List[str]:
        """Generate paraphrases of a prompt.

        This is a simple heuristic approach. For better paraphrases,
        you could use a paraphrasing model.

        Args:
            prompt: Original prompt
            num_paraphrases: Number of paraphrases to generate

        Returns:
            List of paraphrased prompts
        """
        paraphrases = []

        # Simple paraphrasing strategies
        strategies = [
            lambda p: p.replace("What is", "Can you explain what"),
            lambda p: p.replace("Explain", "Could you tell me about"),
            lambda p: p.replace("How does", "What is the mechanism behind how"),
            lambda p: f"I'm curious about: {p.lower()}",
            lambda p: f"Could you help me understand {p.lower()}",
            lambda p: p.replace("Write a story", "Create a narrative"),
            lambda p: p.replace("Tell me", "Share with me"),
            lambda p: f"Please provide information on: {p.lower()}",
        ]

        # Apply random strategies
        for _ in range(num_paraphrases):
            strategy = random.choice(strategies)
            try:
                paraphrase = strategy(prompt)
                if paraphrase != prompt:
                    paraphrases.append(paraphrase)
            except:
                # Fallback: add a prefix
                paraphrases.append(f"Here's a question: {prompt}")

        # Ensure we have enough paraphrases
        while len(paraphrases) < num_paraphrases:
            paraphrases.append(f"Variant: {prompt}")

        return paraphrases[:num_paraphrases]

    def generate_corpus(
        self,
        num_prompts: int = 200,
        num_paraphrases: int = 3
    ) -> Dict[str, List[Tuple[str, str, int]]]:
        """Generate corpus of prompts across styles.

        Args:
            num_prompts: Total number of prompts to generate
            num_paraphrases: Number of paraphrases per base prompt

        Returns:
            Dictionary mapping style to list of (prompt, base_id, family_id) tuples
        """
        corpus = {
            "qa": [],
            "story": [],
            "policy": [],
            "paraphrase": []
        }

        # Calculate prompts per style
        base_prompts_per_style = num_prompts // (3 * (1 + num_paraphrases))

        # QA prompts
        qa_base = self.qa_prompts[:base_prompts_per_style]
        for i, prompt in enumerate(qa_base):
            corpus["qa"].append((prompt, f"qa_{i}", i))

            # Generate paraphrases
            for j, para in enumerate(self.generate_paraphrases(prompt, num_paraphrases)):
                corpus["paraphrase"].append((para, f"qa_{i}_para_{j}", i))

        # Story prompts
        story_base = self.story_prompts[:base_prompts_per_style]
        for i, prompt in enumerate(story_base):
            corpus["story"].append((prompt, f"story_{i}", i + len(qa_base)))

            # Generate paraphrases
            for j, para in enumerate(self.generate_paraphrases(prompt, num_paraphrases)):
                corpus["paraphrase"].append((para, f"story_{i}_para_{j}", i + len(qa_base)))

        # Policy prompts
        policy_base = self.policy_prompts[:base_prompts_per_style]
        for i, prompt in enumerate(policy_base):
            corpus["policy"].append((prompt, f"policy_{i}", i + len(qa_base) + len(story_base)))

            # Generate paraphrases
            for j, para in enumerate(self.generate_paraphrases(prompt, num_paraphrases)):
                corpus["paraphrase"].append((para, f"policy_{i}_para_{j}", i + len(qa_base) + len(story_base)))

        return corpus

    def split_corpus(
        self,
        corpus: Dict[str, List[Tuple[str, str, int]]],
        test_size: float = 0.25,
        seed: int = None
    ) -> Tuple[Dict, Dict]:
        """Split corpus into train and test by prompt families.

        Args:
            corpus: Corpus dictionary from generate_corpus
            test_size: Fraction of families to hold out for test
            seed: Random seed (uses self.seed if None)

        Returns:
            (train_corpus, test_corpus) dictionaries
        """
        if seed is None:
            seed = self.seed

        random.seed(seed)

        # Get all unique family IDs
        all_families = set()
        for prompts in corpus.values():
            for _, _, family_id in prompts:
                all_families.add(family_id)

        # Split families
        families = sorted(list(all_families))
        random.shuffle(families)
        split_idx = int(len(families) * (1 - test_size))
        train_families = set(families[:split_idx])
        test_families = set(families[split_idx:])

        # Split corpus by family
        train_corpus = {style: [] for style in corpus.keys()}
        test_corpus = {style: [] for style in corpus.keys()}

        for style, prompts in corpus.items():
            for prompt, base_id, family_id in prompts:
                if family_id in train_families:
                    train_corpus[style].append((prompt, base_id, family_id))
                else:
                    test_corpus[style].append((prompt, base_id, family_id))

        return train_corpus, test_corpus
