from __future__ import annotations

from dataclasses import asdict

from configs.tac_v02_112m import (
    D_MODEL,
    MAX_SEQ_LEN,
    N_HEADS,
    N_LAYERS,
    TAC_V02_112M_PARAMS,
    VOCAB_SIZE,
)
from tac_transformer import TACConfig
from tac_transformer.training import estimate_vanilla_parameter_count


TRANSFORMER_V02_PARAM_TOLERANCE = 0.01


def transformer_112m_config() -> TACConfig:
    """Return the matched v0.2 transformer baseline configuration.

    The baseline keeps the same vocab, d_model, layer count, head count, context
    length, RoPE, and normalization choices as TAC. It widens only the MLP ratio
    to match TAC's parameter budget without adding identity-field mechanisms.
    """

    return TACConfig(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_kv_heads=N_HEADS,
        n_programs=1,
        max_seq_len=MAX_SEQ_LEN,
        mlp_ratio=15,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        position_type="rope",
    )


TRANSFORMER_V02_112M_CONFIG = transformer_112m_config()
TRANSFORMER_V02_112M_PARAMS = estimate_vanilla_parameter_count(
    TRANSFORMER_V02_112M_CONFIG
)


def config_summary() -> dict:
    config = TRANSFORMER_V02_112M_CONFIG
    return {
        "schema": "transformer_v02_112m_config.v1",
        "matched_to_tac_parameters": TAC_V02_112M_PARAMS,
        "estimated_parameters": TRANSFORMER_V02_112M_PARAMS,
        "relative_gap_to_tac": abs(
            TRANSFORMER_V02_112M_PARAMS - TAC_V02_112M_PARAMS
        )
        / TAC_V02_112M_PARAMS,
        "parameter_tolerance": TRANSFORMER_V02_PARAM_TOLERANCE,
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

