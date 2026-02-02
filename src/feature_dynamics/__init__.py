"""SAE Feature Dynamics Analysis.

Analyze whether SAE feature trajectories show persistent internal modes
beyond immediate token forcing.
"""

from .main import main
from .config import Config
from .prompts import PromptGenerator
from .data_collection import DataCollector
from .predictors import TokenOnlyPredictor, StateTokenPredictor
from .evaluation import evaluate_predictors

__version__ = "0.1.0"

__all__ = [
    "main",
    "Config",
    "PromptGenerator",
    "DataCollector",
    "TokenOnlyPredictor",
    "StateTokenPredictor",
    "evaluate_predictors",
]
