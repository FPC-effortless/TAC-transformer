from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ConceptVolumeOutput:
    concept_embedding: Tensor
    family_logits: Tensor
    family_probs: Tensor
    family_id: Tensor


class ConceptVolumeEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_structure_families: int,
        use_rms_norm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_structure_families = n_structure_families

        self.projection = nn.Linear(d_model, d_model, bias=False)

        if use_rms_norm:
            self.norm: Optional[nn.Module] = _RMSNorm(d_model)
        else:
            self.norm = None

        self.family_head = nn.Linear(d_model, n_structure_families, bias=False)

        self.family_centroid_bank = nn.Parameter(
            torch.empty(n_structure_families, d_model)
        )
        nn.init.normal_(self.family_centroid_bank, mean=0.0, std=0.02)

    def forward(self, hidden: Tensor) -> ConceptVolumeOutput:
        concept = self.projection(hidden)
        if self.norm is not None:
            concept = self.norm(concept)

        family_logits = self.family_head(concept)
        family_probs = F.softmax(family_logits, dim=-1)
        family_id = family_probs.argmax(dim=-1)

        return ConceptVolumeOutput(
            concept_embedding=concept,
            family_logits=family_logits,
            family_probs=family_probs,
            family_id=family_id,
        )


class _RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / norm
