from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .agentic_rl_math import (
    ScratchpadItem,
    SimulationBranch,
    bounded_scratchpad_update,
    commit_verified_scratchpad_items,
    process_trace_distillation_loss,
    value_prediction_loss,
)


@dataclass(frozen=True)
class AgenticPolicyControllerConfig:
    scratchpad_feature_dim: int = 8
    simulation_feature_dim: int = 5
    context_feature_dim: int = 4
    hidden_dim: int = 32
    process_steps: int = 4
    process_classes: int = 4


@dataclass(frozen=True)
class AgenticPolicyFeatures:
    scratchpad_features: Tensor
    simulation_features: Tensor
    context_features: Tensor


@dataclass(frozen=True)
class AgenticScratchpadState:
    items: tuple[ScratchpadItem, ...]
    budget: int
    step: int = 0

    @classmethod
    def empty(cls, *, budget: int) -> "AgenticScratchpadState":
        if budget < 0:
            raise ValueError("budget must be non-negative")
        return cls(items=(), budget=budget, step=0)

    def item_ids(self) -> tuple[str, ...]:
        return tuple(item.item_id for item in self.items)


class AgenticPolicyController(nn.Module):
    """Trainable policy heads for TAC-Agent-RL internal control actions."""

    def __init__(self, config: AgenticPolicyControllerConfig | None = None):
        super().__init__()
        self.config = config or AgenticPolicyControllerConfig()
        self.scratchpad_head = _scalar_head(
            self.config.scratchpad_feature_dim,
            self.config.hidden_dim,
        )
        self.simulation_head = _scalar_head(
            self.config.simulation_feature_dim,
            self.config.hidden_dim,
        )
        self.process_head = nn.Sequential(
            nn.Linear(self.config.context_feature_dim, self.config.hidden_dim),
            nn.Tanh(),
            nn.Linear(
                self.config.hidden_dim,
                self.config.process_steps * self.config.process_classes,
            ),
        )
        self.value_head = _scalar_head(
            self.config.context_feature_dim,
            self.config.hidden_dim,
        )

    def forward(
        self,
        *,
        scratchpad_features: Tensor,
        simulation_features: Tensor,
        context_features: Tensor,
    ) -> dict[str, Tensor]:
        if scratchpad_features.shape[-1] != self.config.scratchpad_feature_dim:
            raise ValueError("scratchpad_features last dimension does not match config")
        if simulation_features.shape[-1] != self.config.simulation_feature_dim:
            raise ValueError("simulation_features last dimension does not match config")
        if context_features.shape[-1] != self.config.context_feature_dim:
            raise ValueError("context_features last dimension does not match config")
        process_logits = self.process_head(context_features).reshape(
            -1,
            self.config.process_steps,
            self.config.process_classes,
        )
        return {
            "scratchpad_logits": self.scratchpad_head(scratchpad_features).squeeze(-1),
            "simulation_logits": self.simulation_head(simulation_features).squeeze(-1),
            "process_logits": process_logits,
            "value": self.value_head(context_features).squeeze(-1),
        }

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)


def agentic_controller_supervised_loss(
    outputs: dict[str, Tensor],
    *,
    scratchpad_targets: Tensor,
    simulation_targets: Tensor,
    process_targets: Tensor,
    verifier_scores: Tensor | None = None,
    value_targets: Tensor | None = None,
    value_mask: Tensor | None = None,
    scratchpad_weight: float = 1.0,
    simulation_weight: float = 1.0,
    process_weight: float = 1.0,
    value_weight: float = 1.0,
) -> dict[str, Tensor]:
    scratchpad_loss = F.binary_cross_entropy_with_logits(
        outputs["scratchpad_logits"],
        scratchpad_targets.to(dtype=outputs["scratchpad_logits"].dtype),
    )
    simulation_loss = F.cross_entropy(
        outputs["simulation_logits"],
        simulation_targets,
    )
    process_loss = process_trace_distillation_loss(
        outputs["process_logits"],
        process_targets,
        verifier_scores=verifier_scores,
    )
    total = (
        scratchpad_weight * scratchpad_loss
        + simulation_weight * simulation_loss
        + process_weight * process_loss
    )
    result = {
        "loss": total,
        "scratchpad_loss": scratchpad_loss.detach(),
        "simulation_loss": simulation_loss.detach(),
        "process_loss": process_loss.detach(),
    }
    if value_targets is not None:
        value_loss = value_prediction_loss(
            outputs["value"],
            value_targets,
            mask=value_mask,
        )
        result["loss"] = result["loss"] + value_weight * value_loss
        result["value_loss"] = value_loss.detach()
    return result


def apply_agentic_scratchpad_transition(
    state: AgenticScratchpadState,
    candidates: Sequence[ScratchpadItem],
    *,
    commit_logits: Tensor,
    verifier_supported_ids: Iterable[str],
    min_commit_probability: float = 0.5,
    min_confidence: float = 0.5,
) -> tuple[AgenticScratchpadState, dict[str, Any]]:
    if not 0.0 <= min_commit_probability <= 1.0:
        raise ValueError("min_commit_probability must be in [0, 1]")
    if commit_logits.ndim != 1:
        raise ValueError("commit_logits must have shape [candidates]")
    if commit_logits.shape[0] != len(candidates):
        raise ValueError("commit_logits length must match candidates")

    probabilities = torch.sigmoid(commit_logits.detach()).cpu()
    selected = [
        item
        for item, probability in zip(candidates, probabilities.tolist())
        if probability >= min_commit_probability
    ]
    verified = [
        replace(item, verified=True)
        for item in commit_verified_scratchpad_items(
            selected,
            verifier_supported_ids=verifier_supported_ids,
            min_confidence=min_confidence,
        )
    ]
    updated_items = bounded_scratchpad_update(
        state.items,
        verified,
        budget=state.budget,
    )
    committed_ids = {item.item_id for item in verified}
    rejected = [item for item in selected if item.item_id not in committed_ids]
    contaminated_count = sum(1 for item in updated_items if item.imagined and not item.verified)
    next_state = AgenticScratchpadState(
        items=tuple(updated_items),
        budget=state.budget,
        step=state.step + 1,
    )
    return next_state, {
        "schema": "agentic_scratchpad_transition.v1",
        "step": next_state.step,
        "budget": state.budget,
        "previous_count": len(state.items),
        "candidate_count": len(candidates),
        "selected_ids": [item.item_id for item in selected],
        "committed_ids": [item.item_id for item in verified],
        "rejected_ids": [item.item_id for item in rejected],
        "state_item_ids": list(next_state.item_ids()),
        "commit_probabilities": [float(value) for value in probabilities.tolist()],
        "hypothesis_contamination_rate": contaminated_count / max(len(next_state.items), 1),
    }


def build_agentic_policy_features_from_tac_output(
    tac_output: Any,
    *,
    branches: Sequence[SimulationBranch] | None = None,
    scratchpad_slots: int = 3,
    eps: float = 1e-6,
) -> AgenticPolicyFeatures:
    if scratchpad_slots <= 0:
        raise ValueError("scratchpad_slots must be positive")
    hidden = tac_output.hidden_states
    if hidden is None or hidden.ndim != 3:
        raise ValueError("tac_output.hidden_states must have shape [batch, tokens, d_model]")
    batch_size, _, hidden_dim = hidden.shape
    final_hidden = hidden[:, -1, :]
    dtype = final_hidden.dtype
    device = final_hidden.device
    hidden_scale = max(hidden_dim, 1) ** 0.5
    hidden_norm = final_hidden.norm(dim=-1) / hidden_scale
    hidden_mean = final_hidden.mean(dim=-1)

    aux = tac_output.aux
    identity_state = tac_output.identity_states[-1] if tac_output.identity_states else None
    route_fraction = _mean_last_or_global(
        aux.token_selected_program_mask,
        aux.selected_program_mask,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
    )
    activation_mean = _mean_last_or_global(
        aux.token_program_activations,
        aux.program_activations,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
    )
    stability_mean = (
        identity_state.stability.to(dtype=dtype, device=device).mean(dim=-1)
        if identity_state is not None
        else final_hidden.new_zeros(batch_size)
    )
    if identity_state is not None:
        memory = identity_state.program_memory.to(dtype=dtype, device=device)
        memory_norm = memory.norm(dim=-1).mean(dim=-1) / (memory.shape[-1] ** 0.5)
    else:
        memory_norm = final_hidden.new_zeros(batch_size)
    coherence = aux.coherence
    if coherence is not None and coherence.ndim == 3:
        coherence_last = coherence.to(dtype=dtype, device=device)[:, -1, :].mean(dim=-1)
    else:
        coherence_last = final_hidden.new_zeros(batch_size)

    live = torch.stack(
        [
            hidden_norm,
            hidden_mean,
            route_fraction,
            activation_mean,
            stability_mean,
            memory_norm,
            coherence_last,
        ],
        dim=-1,
    )
    slot_ids = torch.linspace(
        0.0,
        1.0,
        scratchpad_slots,
        device=device,
        dtype=dtype,
    ).view(1, scratchpad_slots, 1)
    scratchpad_features = torch.cat(
        [
            slot_ids.expand(batch_size, -1, -1),
            live[:, None, :].expand(-1, scratchpad_slots, -1),
        ],
        dim=-1,
    )

    branch_list = tuple(branches) if branches is not None else _default_simulation_branches()
    if not branch_list:
        raise ValueError("branches must not be empty")
    branch_constants = torch.tensor(
        [
            [
                branch.predicted_reward,
                branch.cost,
                branch.risk,
                branch.confidence,
            ]
            for branch in branch_list
        ],
        dtype=dtype,
        device=device,
    )
    live_context = torch.sigmoid(hidden_norm + activation_mean - memory_norm + eps)
    simulation_features = torch.cat(
        [
            branch_constants.unsqueeze(0).expand(batch_size, -1, -1),
            live_context[:, None, None].expand(batch_size, len(branch_list), 1),
        ],
        dim=-1,
    )
    context_features = torch.stack(
        [
            hidden_norm,
            route_fraction,
            activation_mean,
            memory_norm,
        ],
        dim=-1,
    )
    return AgenticPolicyFeatures(
        scratchpad_features=scratchpad_features,
        simulation_features=simulation_features,
        context_features=context_features,
    )


def run_agentic_policy_controller_from_tac_output(
    controller: AgenticPolicyController,
    tac_output: Any,
    *,
    branches: Sequence[SimulationBranch] | None = None,
    scratchpad_slots: int = 3,
) -> dict[str, Tensor]:
    features = build_agentic_policy_features_from_tac_output(
        tac_output,
        branches=branches,
        scratchpad_slots=scratchpad_slots,
    )
    return controller(
        scratchpad_features=features.scratchpad_features,
        simulation_features=features.simulation_features,
        context_features=features.context_features,
    )


def _scalar_head(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1),
    )


def _mean_last_or_global(
    token_tensor: Tensor | None,
    global_tensor: Tensor | None,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if token_tensor is not None:
        token_tensor = token_tensor.to(dtype=dtype, device=device)
        if token_tensor.ndim == 3:
            return token_tensor[:, -1, :].mean(dim=-1)
        if token_tensor.ndim == 2:
            return token_tensor[:, -1]
    if global_tensor is not None:
        global_tensor = global_tensor.to(dtype=dtype, device=device)
        if global_tensor.ndim >= 2:
            return global_tensor.reshape(batch_size, -1).mean(dim=-1)
    return torch.zeros(batch_size, dtype=dtype, device=device)


def _default_simulation_branches() -> tuple[SimulationBranch, ...]:
    return (
        SimulationBranch(
            "safe",
            ("read_scratchpad", "answer"),
            predicted_reward=0.75,
            cost=0.1,
            risk=0.0,
            confidence=0.9,
        ),
        SimulationBranch(
            "deep",
            ("simulate", "verify", "answer"),
            predicted_reward=0.9,
            cost=0.6,
            risk=0.1,
            confidence=0.8,
        ),
        SimulationBranch(
            "risky",
            ("guess",),
            predicted_reward=0.99,
            cost=0.1,
            risk=0.9,
            confidence=0.35,
        ),
    )
