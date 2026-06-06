from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import stage_ats_transfer_training_corpus


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/ats_transfer_training_corpus_2026_06_05")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage an ATS transfer supervised corpus for TAC and vanilla trainers."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--examples-per-domain", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = stage_ats_transfer_training_corpus(
        output_dir=args.output_dir,
        seed=args.seed,
        examples_per_domain=args.examples_per_domain,
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
