"""Stable public core facade for TAC-Transformer.

This package defines the reviewer-facing import boundary for the core TAC model.
It intentionally re-exports the existing implementation instead of moving files,
so older research scripts keep working.
"""

from tac_transformer.model import (
    IdentityState,
    IdentityFieldOutput,
    TACAuxiliaryOutput,
    TACConfig,
    TACOutput,
    TACTransformerBlock,
    TACTransformerLM,
    VanillaTransformerBlock,
    VanillaTransformerLM,
)

__all__ = [
    "IdentityState",
    "IdentityFieldOutput",
    "TACAuxiliaryOutput",
    "TACConfig",
    "TACOutput",
    "TACTransformerBlock",
    "TACTransformerLM",
    "VanillaTransformerBlock",
    "VanillaTransformerLM",
]
