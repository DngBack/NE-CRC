"""Baseline systems module."""

from .heuristic import (
    HeuristicThreshold,
    create_heuristic_baseline,
)
from .trivial import (
    AlwaysAnswer,
    AlwaysAbstain,
    create_always_answer,
    create_always_abstain,
)
from .unicr import (
    UniCRBaseline,
    create_unicr_baseline,
)

__all__ = [
    # Heuristic
    "HeuristicThreshold",
    "create_heuristic_baseline",
    # Trivial
    "AlwaysAnswer",
    "AlwaysAbstain",
    "create_always_answer",
    "create_always_abstain",
    # UniCR
    "UniCRBaseline",
    "create_unicr_baseline",
]

