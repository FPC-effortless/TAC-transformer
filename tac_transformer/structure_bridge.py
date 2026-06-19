from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class StructureBridgeOutput:
    hidden: Tensor
    bridge_delta: Tensor
    gate: Optional[Tensor]
    structure_vector: Tensor


class LinearStructureBridge(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        _validate_d_model(d_model)
        self.d_model = d_model
        self.projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden: Tensor, structure_vector: Tensor) -> StructureBridgeOutput:
        structure_vector = _align_structure_vector(hidden, structure_vector, self.d_model)
        delta = self.projection(structure_vector)
        return StructureBridgeOutput(
            hidden=hidden + delta,
            bridge_delta=delta,
            gate=None,
            structure_vector=structure_vector,
        )


class MLPStructureBridge(nn.Module):
    def __init__(self, d_model: int, hidden_multiplier: int = 2):
        super().__init__()
        _validate_d_model(d_model)
        if hidden_multiplier < 1:
            raise ValueError("hidden_multiplier must be at least 1")
        self.d_model = d_model
        hidden_dim = d_model * hidden_multiplier
        self.projection = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, hidden: Tensor, structure_vector: Tensor) -> StructureBridgeOutput:
        structure_vector = _align_structure_vector(hidden, structure_vector, self.d_model)
        delta = self.projection(structure_vector)
        return StructureBridgeOutput(
            hidden=hidden + delta,
            bridge_delta=delta,
            gate=None,
            structure_vector=structure_vector,
        )


class GatedResidualStructureBridge(nn.Module):
    def __init__(self, d_model: int, hidden_multiplier: int = 2):
        super().__init__()
        _validate_d_model(d_model)
        if hidden_multiplier < 1:
            raise ValueError("hidden_multiplier must be at least 1")
        self.d_model = d_model
        hidden_dim = d_model * hidden_multiplier
        self.projection = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, hidden: Tensor, structure_vector: Tensor) -> StructureBridgeOutput:
        structure_vector = _align_structure_vector(hidden, structure_vector, self.d_model)
        gate = torch.sigmoid(self.gate(torch.cat([hidden, structure_vector], dim=-1)))
        delta = self.projection(structure_vector)
        return StructureBridgeOutput(
            hidden=hidden + gate * delta,
            bridge_delta=delta,
            gate=gate,
            structure_vector=structure_vector,
        )


class OracleStructureBridge(nn.Module):
    """Oracle-label bridge for REAL003 ablations.

    The oracle path maps known structure labels to hidden-state residuals.  It is
    deliberately still a hidden adapter, so the LM head remains the only logit
    producer in the normal architecture path.
    """

    def __init__(self, d_model: int, n_oracle_structures: int):
        super().__init__()
        _validate_d_model(d_model)
        if n_oracle_structures < 1:
            raise ValueError("n_oracle_structures must be at least 1")
        self.d_model = d_model
        self.n_oracle_structures = n_oracle_structures
        self.oracle_embedding = nn.Embedding(n_oracle_structures, d_model)
        self.projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden: Tensor, oracle_structure_id: Tensor) -> StructureBridgeOutput:
        structure_vector = self.oracle_embedding(oracle_structure_id)
        structure_vector = _align_structure_vector(hidden, structure_vector, self.d_model)
        delta = self.projection(structure_vector)
        return StructureBridgeOutput(
            hidden=hidden + delta,
            bridge_delta=delta,
            gate=None,
            structure_vector=structure_vector,
        )


def build_structure_bridge(
    bridge_type: str,
    d_model: int,
    *,
    hidden_multiplier: int = 2,
    n_oracle_structures: int = 1,
) -> nn.Module:
    if bridge_type == "linear":
        return LinearStructureBridge(d_model)
    if bridge_type == "mlp":
        return MLPStructureBridge(d_model, hidden_multiplier=hidden_multiplier)
    if bridge_type == "gated_residual":
        return GatedResidualStructureBridge(
            d_model,
            hidden_multiplier=hidden_multiplier,
        )
    if bridge_type == "oracle":
        return OracleStructureBridge(d_model, n_oracle_structures=n_oracle_structures)
    raise ValueError(
        "bridge_type must be 'linear', 'mlp', 'gated_residual', or 'oracle'"
    )


def _validate_d_model(d_model: int) -> None:
    if d_model < 1:
        raise ValueError("d_model must be positive")


def _align_structure_vector(hidden: Tensor, structure_vector: Tensor, d_model: int) -> Tensor:
    if hidden.size(-1) != d_model:
        raise ValueError("hidden last dimension must match d_model")
    if structure_vector.size(-1) != d_model:
        raise ValueError("structure_vector last dimension must match d_model")
    if structure_vector.shape == hidden.shape:
        return structure_vector
    if hidden.dim() == 3 and structure_vector.dim() == 2:
        if structure_vector.shape[0] != hidden.shape[0]:
            raise ValueError("batched structure_vector must match hidden batch size")
        return structure_vector.unsqueeze(1).expand(-1, hidden.shape[1], -1)
    if hidden.dim() == 2 and structure_vector.dim() == 1:
        return structure_vector.unsqueeze(0).expand(hidden.shape[0], -1)
    raise ValueError("structure_vector shape must match or broadcast to hidden")
