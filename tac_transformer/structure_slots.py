from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .structure_routing import TwoLevelStructureRoute


@dataclass
class StructureSlotState:
    slot_embeddings: Tensor
    slot_weights: Tensor
    slot_id: Tensor
    specialist_id: Tensor


@dataclass
class SlotExecutionOutput:
    hidden: Tensor
    slot_state: StructureSlotState
    specialist_probs: Tensor
    specialist_context: Tensor
    gate: Tensor
    auxiliary_loss: Tensor


class SlotConditionedProgramBottleneck(nn.Module):
    """Reusable structure-slot bottleneck before behavior projection.

    This module is intentionally separate from TACTransformerLM.  It gives the
    TAC-SCM lane a narrow, trainable structure path without allowing structures
    to write directly to logits.
    """

    def __init__(
        self,
        d_model: int,
        n_structure_slots: int,
        n_programs: int,
        load_balance_weight: float = 0.0,
    ):
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        if n_structure_slots < 1:
            raise ValueError("n_structure_slots must be at least 1")
        if n_programs < 1:
            raise ValueError("n_programs must be at least 1")
        if load_balance_weight < 0.0:
            raise ValueError("load_balance_weight must be non-negative")

        self.d_model = d_model
        self.n_structure_slots = n_structure_slots
        self.n_programs = n_programs
        self.load_balance_weight = load_balance_weight

        self.slot_bank = nn.Parameter(torch.empty(n_structure_slots, d_model))
        self.specialist_bank = nn.Parameter(torch.empty(n_programs, d_model))
        nn.init.normal_(self.slot_bank, mean=0.0, std=0.02)
        nn.init.normal_(self.specialist_bank, mean=0.0, std=0.02)

        self.specialist_head = nn.Linear(d_model, n_programs, bias=False)
        self.execution_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.execution_gate = nn.Linear(d_model * 3, d_model)

    def forward(
        self,
        hidden: Tensor,
        *,
        specialist_probs: Optional[Tensor] = None,
        route: Optional[TwoLevelStructureRoute] = None,
    ) -> SlotExecutionOutput:
        if hidden.size(-1) != self.d_model:
            raise ValueError("hidden last dimension must match d_model")
        if route is not None:
            specialist_probs = route.specialist_probs
        if specialist_probs is not None and specialist_probs.shape != (
            *hidden.shape[:-1],
            self.n_programs,
        ):
            raise ValueError("specialist_probs must match hidden leading shape")

        slot_logits = hidden @ self.slot_bank.T / sqrt(self.d_model)
        slot_weights = F.softmax(slot_logits, dim=-1)
        slot_context = slot_weights @ self.slot_bank
        slot_id = slot_weights.argmax(dim=-1)

        if specialist_probs is None:
            specialist_probs = F.softmax(self.specialist_head(hidden), dim=-1)
        specialist_id = specialist_probs.argmax(dim=-1)
        specialist_context = specialist_probs @ self.specialist_bank

        combined = torch.cat([hidden, slot_context, specialist_context], dim=-1)
        delta = self.execution_mlp(combined)
        gate = torch.sigmoid(self.execution_gate(combined))
        bottleneck_hidden = hidden + gate * delta

        auxiliary_loss = self._load_balance_loss(slot_weights, self.n_structure_slots)
        auxiliary_loss = auxiliary_loss + self._load_balance_loss(
            specialist_probs,
            self.n_programs,
        )

        return SlotExecutionOutput(
            hidden=bottleneck_hidden,
            slot_state=StructureSlotState(
                slot_embeddings=slot_context,
                slot_weights=slot_weights,
                slot_id=slot_id,
                specialist_id=specialist_id,
            ),
            specialist_probs=specialist_probs,
            specialist_context=specialist_context,
            gate=gate,
            auxiliary_loss=auxiliary_loss,
        )

    def _load_balance_loss(self, probs: Tensor, n_items: int) -> Tensor:
        if self.load_balance_weight == 0.0:
            return probs.new_zeros(())
        mean_probs = probs.mean(dim=tuple(range(probs.dim() - 1)))
        target = torch.full_like(mean_probs, 1.0 / n_items)
        return self.load_balance_weight * F.mse_loss(mean_probs, target)
