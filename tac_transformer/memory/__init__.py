"""Stable memory facade for TAC-Transformer.

This package re-exports promoted memory-related modules without moving the
underlying research files.
"""

from tac_transformer.structure_memory import (
    StructureMemoryModule,
    StructureMemoryRead,
    StructureMemoryState,
    StructureMemoryWrite,
)
from tac_transformer.procedural_memory import (
    ProceduralMemoryRead,
    ProceduralMemoryRecord,
    ProceduralMemoryStore,
    ProceduralStep,
)

__all__ = [
    "StructureMemoryModule",
    "StructureMemoryRead",
    "StructureMemoryState",
    "StructureMemoryWrite",
    "ProceduralMemoryRead",
    "ProceduralMemoryRecord",
    "ProceduralMemoryStore",
    "ProceduralStep",
]
