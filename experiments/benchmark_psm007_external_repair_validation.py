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


TASK_SOURCES = ("real_github_bugs", "swe_bench_lite", "human_written_repair_tasks")
REQUIRED_CONTROLS = ("frozen_tac", "matched_transformer", "reset_memory_tac")
REQUIRED_CONSTRAINTS = ("no_redesign", "no_retuning", "no_metric_changes")


def build_psm007_protocol() -> dict[str, Any]:
    """Return the frozen external-repair protocol for PSM-007."""

    return {
        "schema": "psm007_external_repair_protocol.v1",
        "gate": "PSM-007",
        "question": "Does TAC work on repair problems it did not design?",
        "risk": "benchmark_artifact",
        "experiment_type": "credibility",
        "task_sources": list(TASK_SOURCES),
        "required_controls": list(REQUIRED_CONTROLS),
        "constraints": list(REQUIRED_CONSTRAINTS),
        "primary_metric": "primary_score",
        "success_criteria": [
            "all required task sources are present",
            "all required controls are present for every source",
            "frozen_tac beats matched_transformer for every source",
            "frozen_tac beats reset_memory_tac for every source",
            "rows record no redesign, no retuning, and no metric changes",
        ],
        "boundary": (
            "This protocol scores external repair rows. It does not generate "
            "or alter the external tasks, TAC architecture, or metrics."
        ),
    }


def evaluate_psm007_external_repair_validation(
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate external repair rows into the PSM-007 gate decision."""

    row_list = [_normalize_row(row) for row in rows]
    source_results = {
        source: _source_result(source, row_list)
        for source in TASK_SOURCES
    }
    missing_sources = [
        source
        for source, result in source_results.items()
        if not result["present"]
    ]
    incomplete_sources = [
        source
        for source, result in source_results.items()
        if result["present"] and result["missing_controls"]
    ]
    constraint_violations = [
        row["task_id"]
        for row in row_list
        if not all(row["constraints"].get(name) is True for name in REQUIRED_CONSTRAINTS)
    ]
    per_source_passes = [result["passes"] for result in source_results.values()]
    all_sources_ready = not missing_sources and not incomplete_sources
    all_constraints_ok = not constraint_violations and bool(row_list)
    validated = all_sources_ready and all_constraints_ok and all(per_source_passes)

    if missing_sources or incomplete_sources or not row_list:
        status = "blocked"
        reason = "Required external source/control evidence is missing."
    elif constraint_violations:
        status = "blocked"
        reason = "One or more rows violate the frozen PSM-007 constraints."
    elif validated:
        status = "validated"
        reason = "Frozen TAC has positive external repair advantage across all required sources."
    else:
        status = "not_validated"
        reason = "Frozen TAC did not beat controls on every required external source."

    tac_vs_transformer = [
        result["tac_vs_transformer_advantage"]
        for result in source_results.values()
        if result["tac_vs_transformer_advantage"] is not None
    ]
    tac_vs_reset = [
        result["tac_vs_reset_advantage"]
        for result in source_results.values()
        if result["tac_vs_reset_advantage"] is not None
    ]
    result = {
        "schema": "psm007_external_repair_validation.v1",
        "protocol": build_psm007_protocol(),
        "sources": source_results,
        "metrics": {
            "task_row_count": len(row_list),
            "source_count": sum(1 for source in source_results.values() if source["present"]),
            "mean_tac_vs_transformer_advantage": _mean_or_none(tac_vs_transformer),
            "mean_tac_vs_reset_advantage": _mean_or_none(tac_vs_reset),
        },
        "decision": {
            "status": status,
            "passes_external_gate": validated,
            "reason": reason,
            "missing_sources": missing_sources,
            "incomplete_sources": incomplete_sources,
            "constraint_violations": constraint_violations,
        },
        "rows": row_list,
    }
    return result


def run_psm007_external_repair_validation(
    *,
    output_dir: Path,
    results_path: Path,
) -> dict[str, Any]:
    """Load external rows, evaluate PSM-007, and write JSON/Markdown artifacts."""

    rows = json.loads(Path(results_path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("PSM-007 results_path must contain a JSON list of rows.")
    result = evaluate_psm007_external_repair_validation(rows)
    result = write_artifact(output_dir, "psm007_external_repair_validation.json", result)
    markdown_path = output_dir / "PSM007_RESULTS.md"
    markdown_path.write_text(format_psm007_markdown(result), encoding="utf-8")
    result["markdown_path"] = str(markdown_path)
    artifact_path = Path(result["artifact_path"])
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def format_psm007_markdown(result: dict[str, Any]) -> str:
    """Format a reader-facing PSM-007 report."""

    decision = result["decision"]
    metrics = result["metrics"]
    lines = [
        "# PSM-007 External Repair Validation",
        "",
        f"Decision: `{decision['status']}`",
        "",
        f"- Reason: {decision['reason']}",
        f"- Passes external gate: `{decision['passes_external_gate']}`",
        f"- Task rows: `{metrics['task_row_count']}`",
        f"- Mean TAC vs transformer advantage: `{_format_metric(metrics['mean_tac_vs_transformer_advantage'])}`",
        f"- Mean TAC vs reset advantage: `{_format_metric(metrics['mean_tac_vs_reset_advantage'])}`",
        "",
        "| Source | TAC | Transformer | Reset TAC | TAC vs Transformer | TAC vs Reset | Passes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for source, row in result["sources"].items():
        lines.append(
            "| {source} | {tac} | {transformer} | {reset} | {adv_tr} | {adv_reset} | `{passes}` |".format(
                source=source,
                tac=_format_metric(row["scores"].get("frozen_tac")),
                transformer=_format_metric(row["scores"].get("matched_transformer")),
                reset=_format_metric(row["scores"].get("reset_memory_tac")),
                adv_tr=_format_metric(row["tac_vs_transformer_advantage"]),
                adv_reset=_format_metric(row["tac_vs_reset_advantage"]),
                passes=row["passes"],
            )
        )
    if decision["missing_sources"] or decision["incomplete_sources"]:
        lines.extend(["", "## Missing Evidence", ""])
        for source in decision["missing_sources"]:
            lines.append(f"- Missing source: `{source}`")
        for source in decision["incomplete_sources"]:
            missing = result["sources"][source]["missing_controls"]
            lines.append(f"- Incomplete source `{source}` missing controls: `{missing}`")
    return "\n".join(lines) + "\n"


def _source_result(source: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_rows = [row for row in rows if row["source"] == source]
    scores = {
        control: _control_score(source_rows, control)
        for control in REQUIRED_CONTROLS
    }
    missing_controls = [
        control for control, score in scores.items() if score is None
    ]
    tac_score = scores["frozen_tac"]
    transformer_score = scores["matched_transformer"]
    reset_score = scores["reset_memory_tac"]
    tac_vs_transformer = (
        tac_score - transformer_score
        if tac_score is not None and transformer_score is not None
        else None
    )
    tac_vs_reset = (
        tac_score - reset_score
        if tac_score is not None and reset_score is not None
        else None
    )
    passes = (
        not missing_controls
        and tac_vs_transformer is not None
        and tac_vs_transformer > 0.0
        and tac_vs_reset is not None
        and tac_vs_reset > 0.0
    )
    return {
        "present": bool(source_rows),
        "row_count": len(source_rows),
        "scores": scores,
        "missing_controls": missing_controls,
        "tac_vs_transformer_advantage": tac_vs_transformer,
        "tac_vs_reset_advantage": tac_vs_reset,
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
    return {
        "task_id": str(row.get("task_id", "")),
        "source": str(row.get("source", "")),
        "control": str(row.get("control", "")),
        "primary_score": _number_or_none(
            row.get("primary_score", row.get("resolved", row.get("score")))
        ),
        "resolved": _number_or_none(row.get("resolved")),
        "constraints": row.get("constraints") if isinstance(row.get("constraints"), dict) else {},
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
        default=Path("runs/benchmarks/psm007_external_repair_validation"),
    )
    args = parser.parse_args()
    result = run_psm007_external_repair_validation(
        output_dir=args.output_dir,
        results_path=args.results_path,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])
    print(result["markdown_path"])


if __name__ == "__main__":
    main()
