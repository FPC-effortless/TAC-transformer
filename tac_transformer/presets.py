from __future__ import annotations

from dataclasses import MISSING, fields
from typing import Any

from .model import TACConfig


BEST_TAC_ARCHITECTURE: dict[str, Any] = {
    "norm_type": "rmsnorm",
    "mlp_type": "swiglu",
    "position_type": "rope",
    "program_compute_type": "linear_expert",
    "routing_type": "base",
    "state_update_type": "gated",
    "memory_write_type": "novelty_gated",
    "memory_tier_type": "flat",
    "memory_lookup_type": "none",
    "memory_read_type": "content_addressed",
    "content_store_size": 8,
    "content_read_steps": 2,
    "content_read_gate_type": "synthesis",
    "memory_adapter_type": "gated_residual",
    "identity_attention_type": "identity_first",
    "residual_stream_type": "single",
    "sequence_mixer_type": "attention",
    "state_mixer_kernel_size": 4,
    "n_sink_programs": 0,
    "n_prediction_heads": 1,
    "multi_token_loss_weight": 0.0,
    "memory_separation_weight": 0.01,
    "content_cue_separation_weight": 0.005,
    "content_gate_entropy_weight": 0.005,
    "content_reconsolidate": True,
    "content_reconsolidate_rate": 0.1,
    "detach_identity_state": False,
}


BEST_TAC_CHUNKED_MEMORY_TRAINING: dict[str, float] = {
    "value_loss_weight": 3.0,
    "memory_read_loss_weight": 3.0,
    "memory_injection_weight": 0.0,
    "memory_adapter_weight": 6.0,
}

RUN5_CAPABILITY_ARCHITECTURE: dict[str, Any] = {
    **BEST_TAC_ARCHITECTURE,
    "routing_type": "base_semantic",
    "routing_top_k": 2,
    "routing_load_balance_weight": 0.05,
    "n_programs": 12,
}

RUN5_CAPABILITY_TRAINING: dict[str, float | str | int] = {
    "category_route_weight": 0.1,
    "category_route_objective": "mi",
    "warmup_steps": 2000,
}

RUN5B_CAPABILITY_ARCHITECTURE: dict[str, Any] = {
    **RUN5_CAPABILITY_ARCHITECTURE,
}

RUN5B_CAPABILITY_TRAINING: dict[str, float | str | int] = {
    **RUN5_CAPABILITY_TRAINING,
    "precision": "fp32",
    "min_healthy_gradient_norm": 1e-12,
    "fail_on_unhealthy_optimization": 1,
}

KAGGLE_FAST_TAC_ARCHITECTURE: dict[str, Any] = {
    **RUN5B_CAPABILITY_ARCHITECTURE,
    "content_read_query_top_k": 8,
    "attention_window_size": 128,
}

KAGGLE_FAST_TAC_TRAINING: dict[str, float | str | int] = {
    **RUN5B_CAPABILITY_TRAINING,
    "category_route_weight": 0.5,
    "category_route_objective": "selected_mi",
}

CPU_RESEARCH_TAC_ARCHITECTURE: dict[str, Any] = {
    **KAGGLE_FAST_TAC_ARCHITECTURE,
    "n_programs": 8,
    "routing_top_k": 1,
    "routing_load_balance_weight": 0.0,
    "content_store_size": 4,
    "content_read_steps": 1,
    "content_read_query_top_k": 4,
    "attention_window_size": 64,
    "memory_adapter_type": "residual",
}

CPU_RESEARCH_TAC_TRAINING: dict[str, float | str | int] = {
    **RUN5B_CAPABILITY_TRAINING,
    "category_route_weight": 0.0,
    "category_route_objective": "selected_mi",
    "warmup_steps": 500,
    "aux_loss_cadence": 4,
    "torch_threads": 1,
    "torch_interop_threads": 1,
}

MEMORY_ADVANTAGE_ARCHITECTURE: dict[str, Any] = {
    **BEST_TAC_ARCHITECTURE,
    "routing_type": "base_semantic",
    "routing_top_k": 2,
    "routing_load_balance_weight": 0.05,
    "n_programs": 24,
    "program_memory_update_type": "program_conditioned",
    "memory_allocation_type": "creb",
    "memory_allocation_k": 6,
    "memory_separation_weight": 0.1,
    "content_store_size": 16,
    "content_read_query_top_k": 8,
    "coalition_context_type": "program_memory_graph",
    "coalition_context_scale": 0.1,
}

MEMORY_ADVANTAGE_TRAINING: dict[str, float | str | int] = {
    **BEST_TAC_CHUNKED_MEMORY_TRAINING,
    "category_route_weight": 0.5,
    "category_route_objective": "selected_mi",
    "warmup_steps": 2000,
    "precision": "fp32",
    "min_healthy_gradient_norm": 1e-12,
    "fail_on_unhealthy_optimization": 1,
}

RUN5B_BEST_CAPABILITY_FAST_ARCHITECTURE: dict[str, Any] = {
    **MEMORY_ADVANTAGE_ARCHITECTURE,
    "content_read_gate_type": "cue_match",
    "attention_window_size": 128,
}

RUN5B_BEST_CAPABILITY_FAST_TRAINING: dict[str, float | str | int] = {
    **RUN5B_CAPABILITY_TRAINING,
    "category_route_weight": 0.1,
    "category_route_objective": "selected_mi",
    "aux_loss_cadence": 4,
}


def best_tac_config(*, vocab_size: int, **overrides: Any) -> TACConfig:
    """Build the strongest TAC config found by the harder 2026-05-31 matrix."""

    values = _default_config_values(vocab_size)
    values.update(BEST_TAC_ARCHITECTURE)
    values.update(overrides)
    if values["n_kv_heads"] is None:
        values["n_kv_heads"] = max(1, values["n_heads"] // 2)
    return TACConfig(**values)


def run5_capability_config(*, vocab_size: int, **overrides: Any) -> TACConfig:
    """Build the Run 5 candidate after Run 4's high-weight capability failure."""

    values = _default_config_values(vocab_size)
    values.update(RUN5_CAPABILITY_ARCHITECTURE)
    values.update(overrides)
    if values["n_kv_heads"] is None:
        values["n_kv_heads"] = max(1, values["n_heads"] // 2)
    return TACConfig(**values)


def run5_capability_training_kwargs(
    **overrides: float | str | int,
) -> dict[str, float | str | int]:
    values = dict(RUN5_CAPABILITY_TRAINING)
    values.update(overrides)
    return values


def run5b_capability_config(*, vocab_size: int, **overrides: Any) -> TACConfig:
    """Build the Run 5B candidate with Run 5 architecture and safer optimization defaults."""

    values = _default_config_values(vocab_size)
    values.update(RUN5B_CAPABILITY_ARCHITECTURE)
    values.update(overrides)
    if values["n_kv_heads"] is None:
        values["n_kv_heads"] = max(1, values["n_heads"] // 2)
    return TACConfig(**values)


def run5b_capability_training_kwargs(
    **overrides: float | str | int,
) -> dict[str, float | str | int]:
    values = dict(RUN5B_CAPABILITY_TRAINING)
    values.update(overrides)
    return values


def kaggle_fast_tac_config(*, vocab_size: int, **overrides: Any) -> TACConfig:
    """Build the opt-in Kaggle TAC profile that reduces avoidable training work."""

    values = _default_config_values(vocab_size)
    values.update(KAGGLE_FAST_TAC_ARCHITECTURE)
    values.update(overrides)
    if values["n_kv_heads"] is None:
        values["n_kv_heads"] = max(1, values["n_heads"] // 2)
    return TACConfig(**values)


def kaggle_fast_tac_training_kwargs(
    **overrides: float | str | int,
) -> dict[str, float | str | int]:
    values = dict(KAGGLE_FAST_TAC_TRAINING)
    values.update(overrides)
    return values


def cpu_research_tac_config(*, vocab_size: int, **overrides: Any) -> TACConfig:
    """Build the opt-in CPU research TAC version for local efficiency probes."""

    values = _default_config_values(vocab_size)
    values.update(CPU_RESEARCH_TAC_ARCHITECTURE)
    values.update(overrides)
    if values["n_kv_heads"] is None:
        values["n_kv_heads"] = max(1, values["n_heads"] // 2)
    return TACConfig(**values)


def cpu_research_tac_training_kwargs(
    **overrides: float | str | int,
) -> dict[str, float | str | int]:
    values = dict(CPU_RESEARCH_TAC_TRAINING)
    values.update(overrides)
    return values


def memory_advantage_config(*, vocab_size: int, **overrides: Any) -> TACConfig:
    """Build the opt-in TAC-188 long-horizon memory advantage candidate."""

    values = _default_config_values(vocab_size)
    values.update(MEMORY_ADVANTAGE_ARCHITECTURE)
    values.update(overrides)
    if values["n_kv_heads"] is None:
        values["n_kv_heads"] = max(1, values["n_heads"] // 2)
    return TACConfig(**values)


def memory_advantage_training_kwargs(
    **overrides: float | str | int,
) -> dict[str, float | str | int]:
    values = dict(MEMORY_ADVANTAGE_TRAINING)
    values.update(overrides)
    return values


def run5b_best_capability_fast_config(
    *, vocab_size: int, **overrides: Any
) -> TACConfig:
    """Build the Run 5B capability launch candidate with TAC-188/TAC-169 and speed defaults."""

    values = _default_config_values(vocab_size)
    values.update(RUN5B_BEST_CAPABILITY_FAST_ARCHITECTURE)
    values.update(overrides)
    if values["n_kv_heads"] is None:
        values["n_kv_heads"] = max(1, values["n_heads"] // 4)
    return TACConfig(**values)


def run5b_best_capability_fast_training_kwargs(
    **overrides: float | str | int,
) -> dict[str, float | str | int]:
    values = dict(RUN5B_BEST_CAPABILITY_FAST_TRAINING)
    values.update(overrides)
    return values


def best_chunked_memory_training_kwargs(**overrides: float) -> dict[str, float]:
    values = dict(BEST_TAC_CHUNKED_MEMORY_TRAINING)
    values.update(overrides)
    return values


def _default_config_values(vocab_size: int) -> dict[str, Any]:
    values: dict[str, Any] = {"vocab_size": vocab_size}
    for field in fields(TACConfig):
        if field.name == "vocab_size":
            continue
        if field.default is MISSING:
            raise TypeError(f"TACConfig field {field.name} has no default")
        values[field.name] = field.default
    return values
