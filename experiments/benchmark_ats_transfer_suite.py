from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    aggregate_ats_transfer_results,
    build_ats_oracle_predictions,
    build_ats_surface_baseline_predictions,
    build_ats_transfer_suite,
    score_ats_transfer_predictions,
    write_ats_transfer_artifacts,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/ats_transfer_suite_2026_06_05")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate an OOD multi-step / ATS transfer benchmark suite."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--examples-per-domain", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_ats_transfer_suite_probe(
        seed=args.seed,
        examples_per_domain=args.examples_per_domain,
    )
    write_ats_transfer_artifacts(
        output_dir=args.output_dir,
        suite=report["suite_payload"],
        predictions=report["predictions"],
        score_rows=report["score_rows"],
        aggregate=report["aggregate"],
    )
    public_report = {
        key: value
        for key, value in report.items()
        if key not in {"suite_payload", "predictions", "score_rows", "aggregate"}
    }
    (args.output_dir / "ats_transfer_probe.json").write_text(
        json.dumps(public_report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(public_report, indent=2), flush=True)


def run_ats_transfer_suite_probe(
    *,
    seed: int = 37,
    examples_per_domain: int = 4,
) -> dict[str, Any]:
    suite = build_ats_transfer_suite(
        seed=seed,
        examples_per_domain=examples_per_domain,
    )
    oracle_predictions = build_ats_oracle_predictions(
        suite["examples"],
        control_id="identity_oracle",
    )
    baseline_predictions = build_ats_surface_baseline_predictions(
        suite["examples"],
        control_id="surface_baseline",
    )
    predictions = oracle_predictions + baseline_predictions
    score_rows = score_ats_transfer_predictions(suite["examples"], predictions)
    aggregate = aggregate_ats_transfer_results(score_rows)
    oracle_test = aggregate["controls"]["identity_oracle"]["splits"]["test"][
        "mean_score"
    ]
    baseline_test = aggregate["controls"]["surface_baseline"]["splits"]["test"][
        "mean_score"
    ]
    return {
        "schema": "ats_transfer_probe.v1",
        "date": "2026-06-05",
        "suite": {
            "seed": seed,
            "task_ids": suite["task_ids"],
            "train_domains": suite["train_domains"],
            "test_domains": suite["test_domains"],
            "examples_per_domain": examples_per_domain,
            "example_count": suite["example_count"],
        },
        "scores": {
            "identity_oracle_test_score": oracle_test,
            "surface_baseline_test_score": baseline_test,
            "oracle_test_advantage": oracle_test - baseline_test,
            "surface_baseline_train_score": aggregate["controls"]["surface_baseline"][
                "splits"
            ]["train"]["mean_score"],
        },
        "decision": aggregate["decision"],
        "suite_payload": suite,
        "predictions": predictions,
        "score_rows": score_rows,
        "aggregate": aggregate,
    }


if __name__ == "__main__":
    main()
