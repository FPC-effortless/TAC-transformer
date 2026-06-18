from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


LM50A_TRANSFORMER_BEST_LOSS = 1.0611
LM50A_TAC_BEST_LOSS = 1.4999
LM50A_TRANSFORMER_RUNTIME_SECONDS = 1757.34
LM50A_TAC_RUNTIME_SECONDS = 24932.19


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_runtime_seconds(summary: dict[str, Any]) -> float | None:
    latest = summary.get("latest_metrics") or {}
    value = latest.get("elapsed_seconds")
    return float(value) if value is not None else None


def best_loss(summary: dict[str, Any]) -> float | None:
    value = summary.get("best_eval_loss")
    return float(value) if value is not None else None


def variant_decision(
    *,
    transformer_summary: dict[str, Any],
    variant_summary: dict[str, Any],
    retest: dict[str, Any],
    min_gap_shrink: float,
    original_transformer_best_loss: float,
    original_tac_best_loss: float,
    original_transformer_runtime: float,
    original_tac_runtime: float,
) -> dict[str, Any]:
    tr_loss = best_loss(transformer_summary)
    variant_loss = best_loss(variant_summary)
    tr_runtime = latest_runtime_seconds(transformer_summary)
    variant_runtime = latest_runtime_seconds(variant_summary)
    original_gap = original_tac_best_loss - original_transformer_best_loss
    current_gap = None
    gap_shrink = None
    if tr_loss is not None and variant_loss is not None:
        current_gap = variant_loss - tr_loss
        gap_shrink = 1.0 - current_gap / max(original_gap, 1e-9)
    original_speed_penalty = original_tac_runtime / max(original_transformer_runtime, 1e-9)
    current_speed_penalty = None
    if tr_runtime is not None and variant_runtime is not None:
        current_speed_penalty = variant_runtime / max(tr_runtime, 1e-9)

    decision = retest["comparison"]["decision"]
    overall = retest["comparison"]["overall"]
    mechanism_wins = int(decision["tac_win_families"])
    carry_advantage = float(overall["tac_carry_advantage"])
    knockout_delta = float(overall["bottleneck_knockout_delta"])
    checks = {
        "mechanism_wins_ge_3_of_4": mechanism_wins >= 3,
        "carry_advantage_positive": carry_advantage > 0.0,
        "knockout_delta_positive": knockout_delta > 0.01,
        "lm_gap_shrinks_enough": gap_shrink is not None and gap_shrink >= min_gap_shrink,
        "speed_penalty_reduced": (
            current_speed_penalty is not None
            and current_speed_penalty < original_speed_penalty
        ),
    }
    status = "scale_ready" if all(checks.values()) else "not_scale_ready"
    return {
        "model": variant_summary.get("model"),
        "status": status,
        "checks": checks,
        "lm": {
            "transformer_best_eval_loss": tr_loss,
            "variant_best_eval_loss": variant_loss,
            "original_lm50a_gap": original_gap,
            "current_gap": current_gap,
            "gap_shrink_fraction": gap_shrink,
            "required_gap_shrink_fraction": min_gap_shrink,
        },
        "speed": {
            "transformer_runtime_seconds": tr_runtime,
            "variant_runtime_seconds": variant_runtime,
            "original_speed_penalty": original_speed_penalty,
            "current_speed_penalty": current_speed_penalty,
        },
        "mechanisms": {
            "status": decision["status"],
            "tac_win_families": mechanism_wins,
            "carry_positive_families": decision["carry_positive_families"],
            "tac_carry_advantage": carry_advantage,
            "bottleneck_knockout_delta": knockout_delta,
        },
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    transformer_summary = load_json(args.transformer_summary)
    variants = []
    for spec in args.variant:
        name, summary_path, retest_path = spec.split("=", 2)
        decision = variant_decision(
            transformer_summary=transformer_summary,
            variant_summary=load_json(Path(summary_path)),
            retest=load_json(Path(retest_path)),
            min_gap_shrink=args.min_gap_shrink,
            original_transformer_best_loss=args.original_transformer_best_loss,
            original_tac_best_loss=args.original_tac_best_loss,
            original_transformer_runtime=args.original_transformer_runtime,
            original_tac_runtime=args.original_tac_runtime,
        )
        decision["variant"] = name
        variants.append(decision)
    scale_ready = [row for row in variants if row["status"] == "scale_ready"]
    result = {
        "schema": "tac_v02_tac281_variant_decision.v1",
        "decision": {
            "status": "scale_to_112m" if scale_ready else "do_not_scale_yet",
            "scale_ready_variants": [row["variant"] for row in scale_ready],
            "variant_count": len(variants),
        },
        "baseline": {
            "original_transformer_best_loss": args.original_transformer_best_loss,
            "original_tac_best_loss": args.original_tac_best_loss,
            "original_transformer_runtime": args.original_transformer_runtime,
            "original_tac_runtime": args.original_tac_runtime,
        },
        "variants": variants,
        "boundary": (
            "TAC-281 is a mechanism-sharpening gate before 112M. "
            "It does not claim TAC is a better plain language model."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "tac281_variant_decision.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-summary", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="name=summary_json=retest_json; pass once per TAC-281 variant.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-gap-shrink", type=float, default=0.30)
    parser.add_argument("--original-transformer-best-loss", type=float, default=LM50A_TRANSFORMER_BEST_LOSS)
    parser.add_argument("--original-tac-best-loss", type=float, default=LM50A_TAC_BEST_LOSS)
    parser.add_argument("--original-transformer-runtime", type=float, default=LM50A_TRANSFORMER_RUNTIME_SECONDS)
    parser.add_argument("--original-tac-runtime", type=float, default=LM50A_TAC_RUNTIME_SECONDS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    result = summarize(parse_args(argv))
    print(json.dumps({"decision": result["decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
