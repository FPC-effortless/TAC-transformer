from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class TACOptimizerConfig:
    """Configuration for TAC-aware AdamW parameter groups."""

    learning_rate: float
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    core_lr_mult: float = 1.0
    identity_lr_mult: float = 1.0
    router_lr_mult: float = 1.0
    memory_lr_mult: float = 1.0
    head_lr_mult: float = 1.0
    no_decay_weight_decay: float = 0.0
    foreach: bool | None = None


_CATEGORY_ORDER = ("core", "identity", "router", "memory", "head")
_LR_MULTIPLIER_FIELDS = {
    "core": "core_lr_mult",
    "identity": "identity_lr_mult",
    "router": "router_lr_mult",
    "memory": "memory_lr_mult",
    "head": "head_lr_mult",
}
_HEAD_KEYWORDS = (
    "lm_head",
    "multi_token_heads",
    "action_head",
    "memory_action_head",
    "planner_head",
    "agent_policy_heads",
    "agent_critic",
    "next_observation_head",
    "reward_head",
    "reflection_head",
)
_ROUTER_KEYWORDS = (
    "authority_router",
    "authority_program_head",
    "authority_mode_head",
    "authority_halt_head",
    "stability_gate",
    "raw_energy_costs",
    "cognitive_gate",
    "recurrent_gate",
)
_MEMORY_KEYWORDS = (
    "memory_gate",
    "memory_novelty_gate",
    "memory_lookup",
    "memory_reconsolidate_gate",
    "content_read",
    "content_stream_gate",
    "identity_stream_gate",
    "memory_adapter",
    "memory_store",
    "identity_key_value",
)
_IDENTITY_KEYWORDS = (
    "identity_field",
    "program_embeddings",
    "program_expert",
    "program_update",
    "program_projection",
)


def build_tac_optimizer(
    model: nn.Module,
    config: TACOptimizerConfig,
) -> torch.optim.AdamW:
    """Build AdamW with TAC-specific parameter grouping.

    The optimizer remains a standard PyTorch AdamW instance, so checkpoints,
    AMP scaling, DDP, and existing training loops keep working. The TAC-specific
    part is the grouping: identity, routing, memory, heads, and core parameters
    can use separate learning-rate multipliers and no-decay handling.
    """

    kwargs: dict[str, object] = {
        "lr": config.learning_rate,
        "betas": config.betas,
        "eps": config.eps,
    }
    if config.foreach is not None:
        kwargs["foreach"] = config.foreach
    return torch.optim.AdamW(tac_optimizer_param_groups(model, config), **kwargs)


def tac_optimizer_param_groups(
    model: nn.Module,
    config: TACOptimizerConfig,
) -> list[dict[str, object]]:
    buckets: dict[tuple[str, bool], list[tuple[str, nn.Parameter]]] = {
        (category, decay): []
        for category in _CATEGORY_ORDER
        for decay in (True, False)
    }
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        category = classify_tac_parameter(name)
        decay = should_decay_parameter(name, parameter)
        buckets[(category, decay)].append((name, parameter))

    groups: list[dict[str, object]] = []
    for category in _CATEGORY_ORDER:
        lr_mult = float(getattr(config, _LR_MULTIPLIER_FIELDS[category]))
        for decay in (True, False):
            named_parameters = buckets[(category, decay)]
            if not named_parameters:
                continue
            group_name = f"{category}_{'decay' if decay else 'no_decay'}"
            weight_decay = (
                config.weight_decay if decay else config.no_decay_weight_decay
            )
            groups.append(
                {
                    "params": [parameter for _, parameter in named_parameters],
                    "lr": config.learning_rate * lr_mult,
                    "base_lr": config.learning_rate,
                    "tac_lr_mult": lr_mult,
                    "weight_decay": weight_decay,
                    "tac_group": group_name,
                    "tac_category": category,
                    "tac_param_names": [name for name, _ in named_parameters],
                }
            )
    if not groups:
        raise ValueError("cannot build TAC optimizer for a model with no trainable parameters")
    return groups


def classify_tac_parameter(name: str) -> str:
    normalized = _normalize_parameter_name(name)
    if _contains_any(normalized, _HEAD_KEYWORDS):
        return "head"
    if _contains_any(normalized, _ROUTER_KEYWORDS):
        return "router"
    if _contains_any(normalized, _MEMORY_KEYWORDS):
        return "memory"
    if _contains_any(normalized, _IDENTITY_KEYWORDS):
        return "identity"
    return "core"


def should_decay_parameter(name: str, parameter: nn.Parameter) -> bool:
    normalized = _normalize_parameter_name(name)
    if parameter.ndim <= 1:
        return False
    if normalized.endswith(".bias"):
        return False
    if ".norm_" in normalized or normalized.endswith("_norm.weight"):
        return False
    if "norm." in normalized or ".norm" in normalized:
        return False
    return True


def _normalize_parameter_name(name: str) -> str:
    normalized = name
    prefixes = ("module.", "backbone.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                changed = True
    return normalized


def _contains_any(value: str, keywords: Iterable[str]) -> bool:
    return any(keyword in value for keyword in keywords)
