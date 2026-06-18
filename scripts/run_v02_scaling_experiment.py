from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.tac_v02_112m import config_summary as tac_config_summary
from tac_transformer.v02_logging import normalize_v02_metrics, write_v02_metrics
from transformer_112m import config_summary as transformer_config_summary


def dry_run_records() -> list[dict[str, Any]]:
    return [
        normalize_v02_metrics(
            model_name="transformer_112m",
            step=0,
            train_metrics={"loss": None},
            eval_metrics={},
            extra={"status": "planned", "purpose": "baseline first"},
        ),
        normalize_v02_metrics(
            model_name="tac_112m",
            step=0,
            train_metrics={"loss": None},
            eval_metrics={},
            extra={"status": "planned", "purpose": "same data/tokens/compute"},
        ),
    ]


def write_stability_report(path: Path, metrics_path: Path, *, dry_run: bool) -> None:
    status = "not_started" if dry_run else "requires_training_outputs"
    path.write_text(
        "\n".join(
            [
                "# TAC v0.2 Stability Report",
                "",
                f"- status: {status}",
                f"- metrics: {metrics_path}",
                "- divergence: pending real 112M training",
                "- routing collapse: pending real 112M TAC metrics",
                "- state collapse: pending real 112M TAC metrics",
                "",
                "Decision rule:",
                "",
                "- continue scaling only if TAC finishes the matched-token run without divergence, routing collapse, or state collapse.",
                "- stop scaling if the matched transformer wins while TAC loses persistent-state, repair, and compression gates.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/v02_scaling"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": "tac_v02_scaling_experiment.v1",
        "question": "Do persistent state, repair planning, and compression advantages survive at about 112M parameters?",
        "run_order": ["transformer_112m", "tac_112m"],
        "tac_config": tac_config_summary(),
        "transformer_config": transformer_config_summary(),
        "required_same": ["dataset", "tokens", "optimizer budget", "evaluation holdouts"],
        "dry_run": bool(args.dry_run),
    }
    (args.output_dir / "scaling_manifest.v02.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_path = args.output_dir / "metrics_v02.json"
    if args.dry_run:
        write_v02_metrics(metrics_path, dry_run_records())
    else:
        raise SystemExit(
            "Real training orchestration is intentionally delegated to the existing Kaggle/local trainer. "
            "Run transformer_112m first, TAC second, then normalize final summaries into metrics_v02.json."
        )
    write_stability_report(
        args.output_dir / "v02_stability_report.md",
        metrics_path,
        dry_run=args.dry_run,
    )
    print(json.dumps({"manifest": str(args.output_dir / "scaling_manifest.v02.json"), "metrics": str(metrics_path)}, indent=2))


if __name__ == "__main__":
    main()
