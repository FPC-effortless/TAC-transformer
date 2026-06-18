from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tac236_240_common import write_artifact


EFFECTS = ("structure_memory", "procedural_memory", "identity_carry")
REQUIRED_CONTROLS = (
    "carried_identity",
    "reset_identity",
    "shuffled_identity",
    "identity_knockout",
)
MECHANISMS = ("IdentityState", "IdentityField")


def build_id001_protocol() -> dict[str, Any]:
    """Return the frozen identity-carry validation protocol."""

    return {
        "schema": "id001_identity_carry_protocol.v1",
        "gate": "ID001",
        "question": "Are structures and procedures better when carried by persistent identities?",
        "risk": "identity_carry_value",
        "experiment_type": "architecture",
        "mechanisms": list(MECHANISMS),
        "required_effects": list(EFFECTS),
        "required_controls": list(REQUIRED_CONTROLS),
        "primary_metric": "primary_score",
        "success_criteria": [
            "carried > reset",
            "carried > shuffled",
            "knockout hurts",
            "all required effect families pass",
        ],
        "boundary": (
            "This protocol evaluates identity-carry rows. It does not claim a "
            "trained checkpoint result unless rows come from trained checkpoint probes."
        ),
    }


def evaluate_id001_identity_carry_validation(
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate ID001 rows into blocked/not_validated/validated."""

    row_list = [_normalize_row(row) for row in rows]
    effect_results = {
        effect: _effect_result(effect, row_list)
        for effect in EFFECTS
    }
    missing_effects = [
        effect for effect, result in effect_results.items() if not result["present"]
    ]
    incomplete_effects = [
        effect
        for effect, result in effect_results.items()
        if result["present"] and result["missing_controls"]
    ]
    missing_mechanism_rows = [
        row["task_id"]
        for row in row_list
        if not all(mechanism in row["mechanisms"] for mechanism in MECHANISMS)
    ]

    ready = bool(row_list) and not missing_effects and not incomplete_effects
    mechanisms_ready = not missing_mechanism_rows
    validated = ready and mechanisms_ready and all(
        result["passes"] for result in effect_results.values()
    )
    if not ready:
        status = "blocked"
        reason = "Required ID001 effect/control evidence is missing."
    elif not mechanisms_ready:
        status = "blocked"
        reason = "One or more rows do not record both IdentityState and IdentityField."
    elif validated:
        status = "validated"
        reason = "Carried identity beats reset and shuffled controls, and knockout hurts, across all required effects."
    else:
        status = "not_validated"
        reason = "Identity carry did not beat all controls across every required effect."

    carry_reset = [
        result["carry_reset_delta"]
        for result in effect_results.values()
        if result["carry_reset_delta"] is not None
    ]
    carry_shuffled = [
        result["carry_shuffled_delta"]
        for result in effect_results.values()
        if result["carry_shuffled_delta"] is not None
    ]
    knockout_drop = [
        result["knockout_drop"]
        for result in effect_results.values()
        if result["knockout_drop"] is not None
    ]
    return {
        "schema": "id001_identity_carry_validation.v1",
        "protocol": build_id001_protocol(),
        "effects": effect_results,
        "metrics": {
            "task_row_count": len(row_list),
            "effect_count": sum(1 for result in effect_results.values() if result["present"]),
            "mean_carry_reset_delta": _mean_or_none(carry_reset),
            "mean_carry_shuffled_delta": _mean_or_none(carry_shuffled),
            "mean_knockout_drop": _mean_or_none(knockout_drop),
        },
        "decision": {
            "status": status,
            "passes_identity_gate": validated,
            "reason": reason,
            "missing_effects": missing_effects,
            "incomplete_effects": incomplete_effects,
            "missing_mechanism_rows": missing_mechanism_rows,
        },
        "rows": row_list,
    }


def run_id001_identity_carry_validation(
    *,
    output_dir: Path,
    results_path: Path,
) -> dict[str, Any]:
    """Load identity-carry rows, evaluate ID001, and write artifacts."""

    rows = json.loads(Path(results_path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("ID001 results_path must contain a JSON list of rows.")
    result = evaluate_id001_identity_carry_validation(rows)
    result = write_artifact(output_dir, "id001_identity_carry_validation.json", result)
    markdown_path = output_dir / "ID001_RESULTS.md"
    markdown_path.write_text(format_id001_markdown(result), encoding="utf-8")
    result["markdown_path"] = str(markdown_path)
    artifact_path = Path(result["artifact_path"])
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def format_id001_markdown(result: dict[str, Any]) -> str:
    """Format a reader-facing ID001 report."""

    decision = result["decision"]
    metrics = result["metrics"]
    lines = [
        "# ID001 Identity Carry Validation",
        "",
        f"Decision: `{decision['status']}`",
        "",
        f"- Reason: {decision['reason']}",
        f"- Passes identity gate: `{decision['passes_identity_gate']}`",
        f"- Task rows: `{metrics['task_row_count']}`",
        f"- Mean carry-reset delta: `{_format_metric(metrics['mean_carry_reset_delta'])}`",
        f"- Mean carry-shuffled delta: `{_format_metric(metrics['mean_carry_shuffled_delta'])}`",
        f"- Mean knockout drop: `{_format_metric(metrics['mean_knockout_drop'])}`",
        "",
        "| Effect | Carried | Reset | Shuffled | Knockout | Carry-Reset | Carry-Shuffled | Knockout Drop | Passes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for effect, row in result["effects"].items():
        scores = row["scores"]
        lines.append(
            "| {effect} | {carried} | {reset} | {shuffled} | {knockout} | {cr} | {cs} | {kd} | `{passes}` |".format(
                effect=effect,
                carried=_format_metric(scores.get("carried_identity")),
                reset=_format_metric(scores.get("reset_identity")),
                shuffled=_format_metric(scores.get("shuffled_identity")),
                knockout=_format_metric(scores.get("identity_knockout")),
                cr=_format_metric(row["carry_reset_delta"]),
                cs=_format_metric(row["carry_shuffled_delta"]),
                kd=_format_metric(row["knockout_drop"]),
                passes=row["passes"],
            )
        )
    if decision["missing_effects"] or decision["incomplete_effects"]:
        lines.extend(["", "## Missing Evidence", ""])
        for effect in decision["missing_effects"]:
            lines.append(f"- Missing effect: `{effect}`")
        for effect in decision["incomplete_effects"]:
            missing = result["effects"][effect]["missing_controls"]
            lines.append(f"- Incomplete effect `{effect}` missing controls: `{missing}`")
    return "\n".join(lines) + "\n"


def _effect_result(effect: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    effect_rows = [row for row in rows if row["effect"] == effect]
    scores = {
        control: _control_score(effect_rows, control)
        for control in REQUIRED_CONTROLS
    }
    missing_controls = [
        control for control, score in scores.items() if score is None
    ]
    carried = scores["carried_identity"]
    reset = scores["reset_identity"]
    shuffled = scores["shuffled_identity"]
    knockout = scores["identity_knockout"]
    carry_reset = carried - reset if carried is not None and reset is not None else None
    carry_shuffled = (
        carried - shuffled if carried is not None and shuffled is not None else None
    )
    knockout_drop = carried - knockout if carried is not None and knockout is not None else None
    passes = (
        not missing_controls
        and carry_reset is not None
        and carry_reset > 0.0
        and carry_shuffled is not None
        and carry_shuffled > 0.0
        and knockout_drop is not None
        and knockout_drop > 0.0
    )
    return {
        "present": bool(effect_rows),
        "row_count": len(effect_rows),
        "scores": scores,
        "missing_controls": missing_controls,
        "carry_reset_delta": carry_reset,
        "carry_shuffled_delta": carry_shuffled,
        "knockout_drop": knockout_drop,
        "passes": passes,
    }


def _control_score(rows: list[dict[str, Any]], control: str) -> float | None:
    scores = [
        row["primary_score"]
        for row in rows
        if row["control"] == control and row["primary_score"] is not None
    ]
    return mean(scores) if scores else None


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    mechanisms = row.get("mechanisms")
    if not isinstance(mechanisms, list):
        mechanisms = []
    return {
        "task_id": str(row.get("task_id", "")),
        "effect": str(row.get("effect", "")),
        "control": str(row.get("control", "")),
        "primary_score": _number_or_none(
            row.get("primary_score", row.get("score", row.get("accuracy")))
        ),
        "mechanisms": [str(mechanism) for mechanism in mechanisms],
    }


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


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
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/id001_identity_carry_validation"),
    )
    args = parser.parse_args()
    result = run_id001_identity_carry_validation(
        output_dir=args.output_dir,
        results_path=args.results_path,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])
    print(result["markdown_path"])


if __name__ == "__main__":
    main()
