from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tac236_240_common import write_artifact


REQUIRED_VARIANTS = ("late_bottleneck", "small_adapter", "auxiliary_mechanism")


def build_tac281_protocol() -> dict[str, Any]:
    """Return the TAC-281 efficiency gate protocol."""

    return {
        "schema": "tac281_efficiency_gate_protocol.v1",
        "gate": "TAC-281",
        "question": "Can TAC keep its mechanism while becoming a better language model?",
        "risk": "lm_efficiency_penalty",
        "experiment_type": "efficiency",
        "required_variants": list(REQUIRED_VARIANTS),
        "input_artifact": "tac281_variant_decision.json",
        "success_criteria": [
            "all required variants are accounted for",
            "at least one variant is scale_ready",
            "mechanism wins >= 3 of 4",
            "carry advantage remains positive",
            "bottleneck knockout delta remains positive",
            "LM loss gap shrinks by at least 30%",
            "speed penalty reduced",
        ],
        "boundary": (
            "This gate wraps TAC-281 variant summaries. It does not train models "
            "or claim 112M success."
        ),
    }


def evaluate_tac281_efficiency_gate(summary: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a TAC-281 variant-decision artifact as a four-gate result."""

    variants = {
        str(row.get("variant")): _normalize_variant(row)
        for row in summary.get("variants", [])
        if isinstance(row, dict) and row.get("variant") is not None
    }
    missing = [variant for variant in REQUIRED_VARIANTS if variant not in variants]
    scale_ready = [
        variant
        for variant in REQUIRED_VARIANTS
        if variants.get(variant, {}).get("status") == "scale_ready"
    ]
    if missing:
        status = "blocked"
        reason = "TAC-281 is missing one or more required variant results."
    elif scale_ready:
        status = "validated"
        reason = "At least one complete TAC-281 variant satisfies the scale-ready checks."
    else:
        status = "not_validated"
        reason = "All required variants completed, but none passed the TAC-281 scale-ready checks."

    result = {
        "schema": "tac281_efficiency_gate.v1",
        "protocol": build_tac281_protocol(),
        "source_schema": summary.get("schema"),
        "variants": variants,
        "metrics": {
            "variant_count": len(variants),
            "required_variant_count": len(REQUIRED_VARIANTS),
            "scale_ready_variant_count": len(scale_ready),
            "max_lm_gap_shrink_fraction": _max_metric(
                variants.values(),
                "lm_gap_shrink_fraction",
            ),
            "min_current_speed_penalty": _min_metric(
                variants.values(),
                "current_speed_penalty",
            ),
            "max_bottleneck_knockout_delta": _max_metric(
                variants.values(),
                "bottleneck_knockout_delta",
            ),
        },
        "decision": {
            "status": status,
            "passes_efficiency_gate": status == "validated",
            "reason": reason,
            "missing_variants": missing,
            "scale_ready_variants": scale_ready,
            "next_step": "launch_112m_pilot" if status == "validated" else "continue_tac281_or_redesign",
        },
    }
    return result


def run_tac281_efficiency_gate(
    *,
    output_dir: Path,
    decision_path: Path,
) -> dict[str, Any]:
    """Load a TAC-281 variant decision and write gate artifacts."""

    summary = json.loads(Path(decision_path).read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("decision_path must contain a TAC-281 JSON object.")
    result = evaluate_tac281_efficiency_gate(summary)
    result = write_artifact(output_dir, "tac281_efficiency_gate.json", result)
    markdown_path = output_dir / "TAC281_GATE.md"
    markdown_path.write_text(format_tac281_gate_markdown(result), encoding="utf-8")
    result["markdown_path"] = str(markdown_path)
    artifact_path = Path(result["artifact_path"])
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def format_tac281_gate_markdown(result: dict[str, Any]) -> str:
    """Format a reader-facing TAC-281 gate report."""

    decision = result["decision"]
    metrics = result["metrics"]
    lines = [
        "# TAC-281 Efficiency Gate",
        "",
        f"Decision: `{decision['status']}`",
        "",
        f"- Reason: {decision['reason']}",
        f"- Passes efficiency gate: `{decision['passes_efficiency_gate']}`",
        f"- Scale-ready variants: `{decision['scale_ready_variants']}`",
        f"- Max LM gap shrink: `{_format_metric(metrics['max_lm_gap_shrink_fraction'])}`",
        f"- Min current speed penalty: `{_format_metric(metrics['min_current_speed_penalty'])}`",
        f"- Max bottleneck knockout delta: `{_format_metric(metrics['max_bottleneck_knockout_delta'])}`",
        "",
        "| Variant | Status | Gap Shrink | Speed Penalty | Mechanism Wins | Carry Advantage | Knockout Delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in REQUIRED_VARIANTS:
        row = result["variants"].get(variant, {})
        lines.append(
            "| {variant} | `{status}` | {gap} | {speed} | {wins} | {carry} | {knockout} |".format(
                variant=variant,
                status=row.get("status", "missing"),
                gap=_format_metric(row.get("lm_gap_shrink_fraction")),
                speed=_format_metric(row.get("current_speed_penalty")),
                wins=_format_metric(row.get("tac_win_families")),
                carry=_format_metric(row.get("tac_carry_advantage")),
                knockout=_format_metric(row.get("bottleneck_knockout_delta")),
            )
        )
    if decision["missing_variants"]:
        lines.extend(["", "## Missing Variants", ""])
        for variant in decision["missing_variants"]:
            lines.append(f"- `{variant}`")
    return "\n".join(lines) + "\n"


def _normalize_variant(row: dict[str, Any]) -> dict[str, Any]:
    lm = row.get("lm") if isinstance(row.get("lm"), dict) else {}
    speed = row.get("speed") if isinstance(row.get("speed"), dict) else {}
    mechanisms = row.get("mechanisms") if isinstance(row.get("mechanisms"), dict) else {}
    checks = row.get("checks") if isinstance(row.get("checks"), dict) else {}
    return {
        "variant": str(row.get("variant")),
        "model": row.get("model"),
        "status": str(row.get("status", "unknown")),
        "checks": checks,
        "lm_gap_shrink_fraction": _number_or_none(lm.get("gap_shrink_fraction")),
        "required_gap_shrink_fraction": _number_or_none(lm.get("required_gap_shrink_fraction")),
        "original_speed_penalty": _number_or_none(speed.get("original_speed_penalty")),
        "current_speed_penalty": _number_or_none(speed.get("current_speed_penalty")),
        "tac_win_families": _number_or_none(mechanisms.get("tac_win_families")),
        "tac_carry_advantage": _number_or_none(mechanisms.get("tac_carry_advantage")),
        "bottleneck_knockout_delta": _number_or_none(
            mechanisms.get("bottleneck_knockout_delta")
        ),
    }


def _max_metric(rows: Any, key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return max(values) if values else None


def _min_metric(rows: Any, key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return min(values) if values else None


def _number_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_metric(value: Any) -> str:
    number = _number_or_none(value)
    if number is None:
        return "n/a"
    return f"{number:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/tac281_efficiency_gate"),
    )
    args = parser.parse_args()
    result = run_tac281_efficiency_gate(
        output_dir=args.output_dir,
        decision_path=args.decision_path,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])
    print(result["markdown_path"])


if __name__ == "__main__":
    main()
