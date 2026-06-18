from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import TACConfig
from tac_transformer.training import estimate_tac_parameter_count


VOCAB_SIZE = 8192
D_MODEL = 512
N_LAYERS = 8
N_HEADS = 8
MAX_SEQ_LEN = 2048

TAC_V02_TARGET_PARAMS = 112_000_000
TAC_V02_TOLERANCE = 0.01


def tac_v02_112m_config() -> TACConfig:
    """Return the locked TAC v0.2 scaling configuration.

    The user-specified core dimensions are fixed. Low-rank program experts keep
    TAC near the 112M target without changing vocab/model/layer/head dimensions.
    """

    return TACConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_kv_heads=N_HEADS,
        n_programs=32,
        max_seq_len=MAX_SEQ_LEN,
        beta=1.5,
        energy_budget=8.0,
        mlp_ratio=7,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        position_type="rope",
        program_compute_type="low_rank_linear_expert",
        program_expert_rank=128,
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


TAC_V02_112M_CONFIG = tac_v02_112m_config()
TAC_V02_112M_PARAMS = estimate_tac_parameter_count(TAC_V02_112M_CONFIG)


def config_summary() -> dict:
    config = TAC_V02_112M_CONFIG
    return {
        "schema": "tac_v02_112m_config.v1",
        "target_parameters": TAC_V02_TARGET_PARAMS,
        "parameter_tolerance": TAC_V02_TOLERANCE,
        "estimated_parameters": TAC_V02_112M_PARAMS,
        "relative_error": abs(TAC_V02_112M_PARAMS - TAC_V02_TARGET_PARAMS)
        / TAC_V02_TARGET_PARAMS,
        "locked_dimensions": {
            "vocab_size": config.vocab_size,
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
        },
        "config": asdict(config),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(config_summary(), indent=2, sort_keys=True))
