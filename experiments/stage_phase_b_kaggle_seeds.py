from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import build_phase_b_kaggle_kernel


DEFAULT_OUTPUT_DIR = Path("runs/kaggle_tac_control_v1_phase_b_2026_06_04")
DEFAULT_CODE_DATASET = "jeffkolo/tac-run5b-capability-code-2026-06-04"
DEFAULT_DATA_DATASET = "jeffkolo/tac-run5b-capability-data-2026-06-03"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage TAC-Control-v1 Phase B seed kernels for Kaggle."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--code-dataset", default=DEFAULT_CODE_DATASET)
    parser.add_argument("--data-dataset", default=DEFAULT_DATA_DATASET)
    parser.add_argument("--owner", default="jeffkolo")
    args = parser.parse_args()

    staged = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        kernel = build_phase_b_kaggle_kernel(
            seed=seed,
            code_dataset=args.code_dataset,
            data_dataset=args.data_dataset,
            owner=args.owner,
        )
        kernel_dir = args.output_dir / f"seed_{seed}"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        _write_json(kernel_dir / "kernel-metadata.json", kernel["metadata"])
        (kernel_dir / kernel["code_file"]).write_text(
            kernel["script"],
            encoding="utf-8",
        )
        staged.append(
            {
                "seed": seed,
                "kernel_dir": str(kernel_dir),
                "kernel_id": kernel["metadata"]["id"],
                "code_file": kernel["code_file"],
            }
        )

    manifest = {
        "schema": "tac_control_v1_phase_b_kaggle_staging.v1",
        "code_dataset": args.code_dataset,
        "data_dataset": args.data_dataset,
        "staged": staged,
    }
    _write_json(args.output_dir / "phase_b_kaggle_staging.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
