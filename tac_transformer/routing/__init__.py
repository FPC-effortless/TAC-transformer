"""Stable routing and structure-bridge facade for TAC-Transformer.

This package provides a reviewer-facing boundary for promoted routing and
structure-conditioning modules. It does not move or delete existing research
files.
"""

from tac_transformer.structure_routing import (
    SpecialistRouter,
    StructureFamilyRouter,
    TwoLevelStructureRoute,
    TwoLevelStructureRouter,
)
from tac_transformer.structure_slots import (
    SlotConditionedProgramBottleneck,
    SlotExecutionOutput,
    StructureSlotState,
)
from tac_transformer.structure_bridge import (
    GatedResidualStructureBridge,
    LinearStructureBridge,
    MLPStructureBridge,
    OracleStructureBridge,
    StructureBridgeOutput,
    build_structure_bridge,
)

__all__ = [
    "SpecialistRouter",
    "StructureFamilyRouter",
    "TwoLevelStructureRoute",
    "TwoLevelStructureRouter",
    "SlotConditionedProgramBottleneck",
    "SlotExecutionOutput",
    "StructureSlotState",
    "GatedResidualStructureBridge",
    "LinearStructureBridge",
    "MLPStructureBridge",
    "OracleStructureBridge",
    "StructureBridgeOutput",
    "build_structure_bridge",
]
