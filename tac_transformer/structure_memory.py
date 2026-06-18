from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .structure_types import LifecyclePhase, StructureLifecycleStats, StructureObject


@dataclass
class StructureMemoryState:
    structures: List[StructureObject] = field(default_factory=list)
    lifecycle_stats: Dict[int, StructureLifecycleStats] = field(default_factory=dict)


@dataclass
class StructureMemoryRead:
    structure: Optional[StructureObject]
    similarity_score: float
    read_gate: float


@dataclass
class StructureMemoryWrite:
    structure_id: int
    update_vector: Tensor
    write_gate: float


_SURVIVAL_USAGE_WEIGHT = 0.4
_SURVIVAL_SUCCESS_WEIGHT = 0.4
_SURVIVAL_TRANSFER_WEIGHT = 0.2
_SURVIVAL_THRESHOLD = 0.1


class StructureMemoryModule(nn.Module):
    def __init__(self, d_model: int, n_structure_slots: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_structure_slots = n_structure_slots

        self.key_bank = nn.Parameter(torch.empty(n_structure_slots, d_model))
        self.value_bank = nn.Parameter(torch.empty(n_structure_slots, d_model))
        nn.init.normal_(self.key_bank, mean=0.0, std=0.02)
        nn.init.normal_(self.value_bank, mean=0.0, std=0.02)

        self.read_gate = nn.Linear(d_model, 1)
        self.novelty_gate = nn.Linear(d_model * 2, 1)

        self.retired: List[bool] = [False] * n_structure_slots
        self._lifecycle_stats: Dict[int, StructureLifecycleStats] = {
            i: StructureLifecycleStats() for i in range(n_structure_slots)
        }

    def read(self, query_vector: Tensor) -> StructureMemoryRead:
        keys = F.normalize(self.key_bank, dim=-1)
        q = F.normalize(query_vector.unsqueeze(0) if query_vector.dim() == 1 else query_vector, dim=-1)
        sims = (q @ keys.T).squeeze(0)

        for slot_id in range(self.n_structure_slots):
            if self.retired[slot_id]:
                sims[..., slot_id] = float("-inf")

        best_idx = int(sims.argmax(dim=-1).item())
        best_sim = float(sims[..., best_idx].item())

        gate_val = float(torch.sigmoid(self.read_gate(query_vector.mean(0) if query_vector.dim() > 1 else query_vector)).item())

        obj = StructureObject(
            structure_id=best_idx,
            slot_id=best_idx,
            key_vector=self.key_bank[best_idx].detach(),
            value_vector=self.value_bank[best_idx].detach(),
            usage_count=self._lifecycle_stats[best_idx].usage_count,
            success_score=self._lifecycle_stats[best_idx].success_rate,
            survival_score=self._lifecycle_stats[best_idx].survival_score,
        )
        return StructureMemoryRead(structure=obj, similarity_score=best_sim, read_gate=gate_val)

    def write(self, candidate: StructureMemoryWrite) -> None:
        slot = candidate.structure_id % self.n_structure_slots
        gate = candidate.write_gate

        with torch.no_grad():
            self.key_bank[slot].mul_(1 - gate).add_(candidate.update_vector * gate)
            self.value_bank[slot].mul_(1 - gate).add_(candidate.update_vector * gate)

    def novelty_write_gate(self, query_vector: Tensor, candidate_vector: Tensor) -> float:
        q = query_vector.mean(0) if query_vector.dim() > 1 else query_vector
        c = candidate_vector.mean(0) if candidate_vector.dim() > 1 else candidate_vector
        combined = torch.cat([q, c], dim=-1)
        return float(torch.sigmoid(self.novelty_gate(combined)).item())

    def update_lifecycle(self, structure_id: int, success: bool, transfer_gain: float = 0.0) -> None:
        slot = structure_id % self.n_structure_slots
        stats = self._lifecycle_stats[slot]
        stats.usage_count += 1
        n = stats.usage_count
        stats.success_rate = stats.success_rate + (float(success) - stats.success_rate) / n
        stats.transfer_gain = stats.transfer_gain + (transfer_gain - stats.transfer_gain) / n
        stats.survival_score = self._compute_survival(stats)

    def _compute_survival(self, stats: StructureLifecycleStats) -> float:
        usage_norm = min(1.0, stats.usage_count / max(1, 100))
        return (
            _SURVIVAL_USAGE_WEIGHT * usage_norm
            + _SURVIVAL_SUCCESS_WEIGHT * stats.success_rate
            + _SURVIVAL_TRANSFER_WEIGHT * stats.transfer_gain
        )

    def score_survival(self) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        for slot_id, stats in self._lifecycle_stats.items():
            scores[slot_id] = self._compute_survival(stats)
        return scores

    def decay_retired(self, threshold: float = _SURVIVAL_THRESHOLD) -> List[int]:
        retired_ids: List[int] = []
        scores = self.score_survival()
        for slot_id, score in scores.items():
            if not self.retired[slot_id] and score < threshold:
                self.retired[slot_id] = True
                retired_ids.append(slot_id)
        return retired_ids

    def get_state(self) -> StructureMemoryState:
        structures: List[StructureObject] = []
        for slot_id in range(self.n_structure_slots):
            stats = self._lifecycle_stats[slot_id]
            obj = StructureObject(
                structure_id=slot_id,
                slot_id=slot_id,
                key_vector=self.key_bank[slot_id].detach(),
                value_vector=self.value_bank[slot_id].detach(),
                usage_count=stats.usage_count,
                success_score=stats.success_rate,
                survival_score=stats.survival_score,
            )
            structures.append(obj)
        return StructureMemoryState(
            structures=structures,
            lifecycle_stats=dict(self._lifecycle_stats),
        )

    def forward(self, query_vector: Tensor) -> Tensor:
        keys = F.normalize(self.key_bank, dim=-1)
        q = F.normalize(query_vector, dim=-1)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        sims = q @ keys.T
        weights = torch.softmax(sims, dim=-1)
        out = weights @ self.value_bank
        gate = torch.sigmoid(self.read_gate(query_vector if query_vector.dim() > 1 else query_vector.unsqueeze(0)))
        return out * gate
