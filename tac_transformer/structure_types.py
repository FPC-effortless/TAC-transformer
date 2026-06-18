from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from torch import Tensor


class LifecyclePhase(Enum):
    appear = "appear"
    survive = "survive"
    strengthen = "strengthen"
    specialize = "specialize"
    merge = "merge"
    decay = "decay"
    retire = "retire"


@dataclass
class StructureObject:
    structure_id: int
    family_id: Optional[int] = None
    slot_id: Optional[int] = None
    key_vector: Optional[Tensor] = None
    value_vector: Optional[Tensor] = None
    procedure_trace: Optional[Tensor] = None
    specialist_id: Optional[int] = None
    usage_count: int = 0
    success_score: float = 0.0
    transfer_score: float = 0.0
    survival_score: float = 0.0
    last_used_step: int = 0


@dataclass
class StructureFamily:
    family_id: int
    name: str = ""
    member_ids: List[int] = field(default_factory=list)
    centroid_vector: Optional[Tensor] = None


@dataclass
class StructureLifecycleStats:
    usage_count: int = 0
    success_rate: float = 0.0
    transfer_gain: float = 0.0
    reset_sensitivity: float = 0.0
    shuffle_sensitivity: float = 0.0
    attack_recovery: float = 0.0
    shift_retention: float = 0.0
    survival_score: float = 0.0
