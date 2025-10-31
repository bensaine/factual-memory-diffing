"""
Data generation module for factual memory diffing research.

Contains tools for generating synthetic triplets and verbalizing them
into natural language sentences for training and evaluation.
"""

from src.triplet_generator import TripletGenerator, Triplet
from src.verbalization import (
    Verbalizer,
    VerbalizationResult,
    PromptTemplate,
    InferenceTemplate,
)

__all__ = [
    "TripletGenerator",
    "Triplet",
    "Verbalizer",
    "VerbalizationResult",
    "PromptTemplate",
    "InferenceTemplate",
]
