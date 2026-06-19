from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.structure_bridge import (
    GatedResidualStructureBridge,
    LinearStructureBridge,
    MLPStructureBridge,
    OracleStructureBridge,
)


def run_real003_structure_to_behavior_smoke(
    *,
    seed: int = 0,
    d_model: int = 16,
    batch_size: int = 2,
    seq_len: int = 3,
    n_oracle_structures: int = 4,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    hidden = torch.randn(batch_size, seq_len, d_model)
    structure = torch.randn(batch_size, seq_len, d_model)
    oracle_ids = torch.randint(0, n_oracle_structures, (batch_size, seq_len))

    variants = {
        "frozen_structure_encoder_linear_bridge": LinearStructureBridge(d_model),
        "frozen_structure_encoder_mlp_bridge": MLPStructureBridge(d_model),
        "end_to_end_gated_residual_bridge": GatedResidualStructureBridge(d_model),
        "oracle_structure_label_bridge": OracleStructureBridge(
            d_model,
            n_oracle_structures=n_oracle_structures,
        ),
    }

    results: dict[str, dict[str, Any]] = {}
    for name, bridge in variants.items():
        if name == "oracle_structure_label_bridge":
            out = bridge(hidden, oracle_ids)
        else:
            out = bridge(hidden, structure)
        shape_ok = tuple(out.hidden.shape) == tuple(hidden.shape)
        delta_norm = float(out.bridge_delta.norm().detach().item())
        gate_mean = None if out.gate is None else float(out.gate.mean().detach().item())
        results[name] = {
            "shape_ok": shape_ok,
            "delta_norm": delta_norm,
            "gate_mean": gate_mean,
            "uses_hidden_adapter": True,
            "direct_logit_injection": False,
        }

    passed = all(item["shape_ok"] for item in results.values())
    return {
        "benchmark": "TAC-SCM-REAL003 structure-to-behavior bridge smoke",
        "status": "passed" if passed else "failed",
        "seed": seed,
        "d_model": d_model,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "variants": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = run_real003_structure_to_behavior_smoke(
        seed=args.seed,
        d_model=args.d_model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
