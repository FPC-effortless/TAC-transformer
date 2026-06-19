"""Stable routing facade for TAC-Transformer.

This package provides a reviewer-facing boundary for promoted routing modules.
It does not move or delete existing research files.
"""

try:
    from tac_transformer.structure_slots import StructureSlotPool
except Exception:  # pragma: no cover - optional research module compatibility
    StructureSlotPool = None

try:
    from tac_transformer.structure_bridge import LinearStructureBridge, MLPStructureBridge, GatedResidualStructureBridge
except Exception:  # pragma: no cover
    LinearStructureBridge = None
    MLPStructureBridge = None
    GatedResidualStructureBridge = None

__all__ = [
    "StructureSlotPool",
    "LinearStructureBridge",
    "MLPStructureBridge",
    "GatedResidualStructureBridge",
]
