from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import TACConfig
from tac_transformer.training import (
    estimate_tac_parameter_count,
    estimate_vanilla_parameter_count,
)


VOCAB_SIZE = 8192
D_MODEL = 384
N_LAYERS = 6
N_HEADS = 8
MAX_SEQ_LEN = 1024


def tac_v02_50m_config() -> TACConfig:
    """Return the 30M-50M real-LM pilot TAC configuration."""

    return TACConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_kv_heads=N_HEADS,
        n_programs=24,
        max_seq_len=MAX_SEQ_LEN,
        beta=1.5,
        energy_budget=6.0,
        mlp_ratio=5,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        position_type="rope",
        program_compute_type="low_rank_linear_expert",
        program_expert_rank=96,
        lm_readout_type="slot_conditioned_program_bottleneck",
        routing_type="base",
        routing_top_k=4,
        state_update_type="gated",
        memory_write_type="novelty_gated",
        memory_read_type="program_memory",
        memory_adapter_type="gated_residual",
        identity_attention_type="identity_first",
        memory_separation_weight=0.01,
        content_cue_separation_weight=0.005,
        content_gate_entropy_weight=0.005,
        routing_load_balance_weight=0.05,
        detach_identity_state=False,
    )


def transformer_v02_50m_config() -> TACConfig:
    """Return the matched transformer baseline for the 30M-50M pilot."""

    return TACConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_kv_heads=N_HEADS,
        n_programs=1,
        max_seq_len=MAX_SEQ_LEN,
        mlp_ratio=12,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        position_type="rope",
    )


def tac_v02_50m_late_bottleneck_config() -> TACConfig:
    """TAC-281 variant: keep early layers transformer-like, use TAC late."""

    return replace(
        tac_v02_50m_config(),
        tac_active_layer_start=N_LAYERS // 2,
        program_residual_scale=0.5,
        routing_load_balance_weight=0.02,
    )


def tac_v02_50m_small_adapter_config() -> TACConfig:
    """TAC-281 variant: small memory/procedure adapter around a normal LM path."""

    return replace(
        tac_v02_50m_config(),
        n_programs=12,
        program_expert_rank=64,
        routing_top_k=2,
        tac_active_layer_start=4,
        lm_readout_type="hidden",
        identity_attention_type="none",
        memory_adapter_type="gated_residual",
        program_residual_scale=0.25,
        memory_separation_weight=0.005,
        content_cue_separation_weight=0.0025,
        content_gate_entropy_weight=0.0025,
        routing_load_balance_weight=0.01,
    )


def tac_v02_50m_auxiliary_mechanism_config() -> TACConfig:
    """TAC-281 variant: keep LM dominant but sharpen mechanism rows."""

    return replace(
        tac_v02_50m_config(),
        memory_separation_weight=0.02,
        content_cue_separation_weight=0.01,
        content_gate_entropy_weight=0.01,
        routing_load_balance_weight=0.08,
        decision_continuity_loss_weight=0.10,
    )


TAC_V02_50M_CONFIG = tac_v02_50m_config()
TAC_V02_50M_LATE_BOTTLENECK_CONFIG = tac_v02_50m_late_bottleneck_config()
TAC_V02_50M_SMALL_ADAPTER_CONFIG = tac_v02_50m_small_adapter_config()
TAC_V02_50M_AUXILIARY_MECHANISM_CONFIG = tac_v02_50m_auxiliary_mechanism_config()
TRANSFORMER_V02_50M_CONFIG = transformer_v02_50m_config()
TAC_V02_50M_PARAMS = estimate_tac_parameter_count(TAC_V02_50M_CONFIG)
TAC_V02_50M_LATE_BOTTLENECK_PARAMS = estimate_tac_parameter_count(
    TAC_V02_50M_LATE_BOTTLENECK_CONFIG
)
TAC_V02_50M_SMALL_ADAPTER_PARAMS = estimate_tac_parameter_count(
    TAC_V02_50M_SMALL_ADAPTER_CONFIG
)
TAC_V02_50M_AUXILIARY_MECHANISM_PARAMS = estimate_tac_parameter_count(
    TAC_V02_50M_AUXILIARY_MECHANISM_CONFIG
)
TRANSFORMER_V02_50M_PARAMS = estimate_vanilla_parameter_count(
    TRANSFORMER_V02_50M_CONFIG
)


def config_summary() -> dict:
    return {
        "schema": "tac_v02_50m_config.v1",
        "tac_estimated_parameters": TAC_V02_50M_PARAMS,
        "tac_late_bottleneck_estimated_parameters": TAC_V02_50M_LATE_BOTTLENECK_PARAMS,
        "tac_small_adapter_estimated_parameters": TAC_V02_50M_SMALL_ADAPTER_PARAMS,
        "tac_auxiliary_mechanism_estimated_parameters": TAC_V02_50M_AUXILIARY_MECHANISM_PARAMS,
        "transformer_estimated_parameters": TRANSFORMER_V02_50M_PARAMS,
        "relative_gap": abs(TAC_V02_50M_PARAMS - TRANSFORMER_V02_50M_PARAMS)
        / TAC_V02_50M_PARAMS,
        "tac_config": asdict(TAC_V02_50M_CONFIG),
        "tac_late_bottleneck_config": asdict(TAC_V02_50M_LATE_BOTTLENECK_CONFIG),
        "tac_small_adapter_config": asdict(TAC_V02_50M_SMALL_ADAPTER_CONFIG),
        "tac_auxiliary_mechanism_config": asdict(TAC_V02_50M_AUXILIARY_MECHANISM_CONFIG),
        "transformer_config": asdict(TRANSFORMER_V02_50M_CONFIG),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(config_summary(), indent=2, sort_keys=True))
