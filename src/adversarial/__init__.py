"""Constrained search for realistic, difficult-to-hedge price paths."""

from src.adversarial.search import (
    AdversarialConfig,
    AdversarialResult,
    search_adversarial_paths,
)

__all__ = ["AdversarialConfig", "AdversarialResult", "search_adversarial_paths"]
