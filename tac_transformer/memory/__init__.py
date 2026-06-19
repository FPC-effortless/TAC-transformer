"""Stable memory facade for TAC-Transformer.

This package re-exports promoted memory-related modules without moving the
underlying research files.
"""

try:
    from tac_transformer.structure_memory import StructureMemoryModule
except Exception:  # pragma: no cover - optional research module compatibility
    StructureMemoryModule = None

try:
    from tac_transformer.procedural_memory import ProceduralMemory
except Exception:  # pragma: no cover
    ProceduralMemory = None

__all__ = [
    "StructureMemoryModule",
    "ProceduralMemory",
]
