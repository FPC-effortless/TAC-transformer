from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .concept_volumes import ConceptVolumeOutput


@dataclass
class TwoLevelStructureRoute:
    family_id: Tensor
    family_probs: Tensor
    specialist_id: Tensor
    specialist_probs: Tensor
    route_loss: Tensor


class StructureFamilyRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_structure_families: int,
    ):
        super().__init__()
        self.n_structure_families = n_structure_families
        self.family_head = nn.Linear(d_model * 2, n_structure_families, bias=False)

    def forward(
        self,
        hidden: Tensor,
        concept_embedding: Tensor,
        load_balance_weight: float = 0.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        combined = torch.cat([hidden, concept_embedding], dim=-1)
        family_logits = self.family_head(combined)
        family_probs = F.softmax(family_logits, dim=-1)
        family_id = family_probs.argmax(dim=-1)

        if load_balance_weight > 0.0:
            mean_probs = family_probs.mean(dim=list(range(family_probs.dim() - 1)))
            n = self.n_structure_families
            target = torch.full_like(mean_probs, 1.0 / n)
            load_loss = load_balance_weight * F.mse_loss(mean_probs, target)
        else:
            load_loss = hidden.new_zeros(())

        return family_id, family_probs, load_loss


class SpecialistRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_programs: int,
        n_structure_families: int,
    ):
        super().__init__()
        self.n_programs = n_programs
        self.n_structure_families = n_structure_families

        self.specialist_head = nn.Linear(d_model, n_programs, bias=False)
        self.family_bias = nn.Embedding(n_structure_families, n_programs)

    def forward(
        self,
        hidden: Tensor,
        family_embedding: Tensor,
        family_id: Tensor,
        load_balance_weight: float = 0.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        base_logits = self.specialist_head(hidden + family_embedding)
        family_bias = self.family_bias(family_id)
        specialist_logits = base_logits + family_bias
        specialist_probs = F.softmax(specialist_logits, dim=-1)
        specialist_id = specialist_probs.argmax(dim=-1)

        if load_balance_weight > 0.0:
            mean_probs = specialist_probs.mean(dim=list(range(specialist_probs.dim() - 1)))
            n = self.n_programs
            target = torch.full_like(mean_probs, 1.0 / n)
            load_loss = load_balance_weight * F.mse_loss(mean_probs, target)
        else:
            load_loss = hidden.new_zeros(())

        return specialist_id, specialist_probs, load_loss


class TwoLevelStructureRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_programs: int,
        n_structure_families: int,
        family_route_loss_weight: float = 0.0,
        specialist_route_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_programs = n_programs
        self.n_structure_families = n_structure_families
        self.family_route_loss_weight = family_route_loss_weight
        self.specialist_route_loss_weight = specialist_route_loss_weight

        self.family_router = StructureFamilyRouter(d_model, n_structure_families)
        self.specialist_router = SpecialistRouter(d_model, n_programs, n_structure_families)

        self.family_embedding = nn.Embedding(n_structure_families, d_model)

    def forward(
        self,
        hidden: Tensor,
        concept_output: ConceptVolumeOutput,
    ) -> TwoLevelStructureRoute:
        family_id, family_probs, family_load_loss = self.family_router(
            hidden,
            concept_output.concept_embedding,
            load_balance_weight=self.family_route_loss_weight,
        )

        fam_emb = self.family_embedding(family_id)

        specialist_id, specialist_probs, specialist_load_loss = self.specialist_router(
            hidden,
            fam_emb,
            family_id,
            load_balance_weight=self.specialist_route_loss_weight,
        )

        route_loss = family_load_loss + specialist_load_loss

        return TwoLevelStructureRoute(
            family_id=family_id,
            family_probs=family_probs,
            specialist_id=specialist_id,
            specialist_probs=specialist_probs,
            route_loss=route_loss,
        )
