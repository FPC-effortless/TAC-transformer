from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.phase_d_benchmarks import stage_phase_d_benchmark_suite


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac_control_v1_phase_d_suite_2026_06_04")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage deterministic TAC-Control-v1 Phase D benchmark tasks."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--examples-per-task", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=4096)
    args = parser.parse_args()

    manifest = stage_phase_d_benchmark_suite(
        output_dir=args.output_dir,
        seeds=args.seeds,
        examples_per_task=args.examples_per_task,
        context_length=args.context_length,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
