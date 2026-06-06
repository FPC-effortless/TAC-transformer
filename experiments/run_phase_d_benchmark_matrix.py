from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.phase_d_benchmarks import (
    load_jsonl,
    run_phase_d_checkpoint_predictions,
    score_phase_d_predictions,
)


DEFAULT_SUITE_DIR = Path("runs/benchmarks/tac_control_v1_phase_d_suite_2026_06_04")
DEFAULT_PHASE_B_DIR = Path("runs/kaggle_results/tac_control_v1_phase_b_2026_06_04")
DEFAULT_PHASE_B_RESULTS = Path(
    "runs/benchmarks/tac_control_v1_phase_b_2026_06_04/phase_b_seed_results.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "runs/benchmarks/tac_control_v1_phase_d_predictions_2026_06_04"
)
DEFAULT_VANILLA_CHECKPOINT = Path(
    "runs/kaggle_results/tac_vanilla_run5b_parameter_matched_20k_2026_06_03"
) / "vanilla_run5b_parameter_matched" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TAC-Control-v1 Phase D predictions for seed checkpoints and vanilla control."
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--phase-b-dir", type=Path, default=DEFAULT_PHASE_B_DIR)
    parser.add_argument("--phase-b-results", type=Path, default=DEFAULT_PHASE_B_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vanilla-checkpoint", type=Path, default=DEFAULT_VANILLA_CHECKPOINT)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--tac-control-prefix", default="tac_control_v1")
    parser.add_argument("--vanilla-control-id", default="parameter_matched_vanilla")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--answer-extraction",
        choices=["raw", "first_line", "first_token"],
        default="first_token",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    payload = run_phase_d_benchmark_matrix(
        suite_dir=args.suite_dir,
        phase_b_dir=args.phase_b_dir,
        phase_b_results=args.phase_b_results,
        output_dir=args.output_dir,
        vanilla_checkpoint=args.vanilla_checkpoint,
        seeds=args.seeds,
        tac_control_prefix=args.tac_control_prefix,
        vanilla_control_id=args.vanilla_control_id,
        device=_select_device(args.device),
        precision=args.precision,
        max_new_tokens=args.max_new_tokens,
        answer_extraction=args.answer_extraction,
        force=args.force,
    )
    print(json.dumps(payload["decision"], indent=2), flush=True)


def run_phase_d_benchmark_matrix(
    *,
    suite_dir: str | Path = DEFAULT_SUITE_DIR,
    phase_b_dir: str | Path = DEFAULT_PHASE_B_DIR,
    phase_b_results: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    vanilla_checkpoint: str | Path | None = DEFAULT_VANILLA_CHECKPOINT,
    seeds: Iterable[int] | None = None,
    tac_control_prefix: str = "tac_control_v1",
    vanilla_control_id: str = "parameter_matched_vanilla",
    device: str | torch.device = "cpu",
    precision: str = "fp32",
    max_new_tokens: int = 32,
    answer_extraction: str = "first_token",
    force: bool = False,
) -> dict[str, Any]:
    suite_root = Path(suite_dir)
    phase_b_root = Path(phase_b_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    seed_list = list(seeds) if seeds is not None else discover_phase_d_suite_seeds(suite_root)
    missing: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []

    phase_b_payload = _load_phase_b_results(phase_b_results)
    if phase_b_payload is not None:
        phase_b_decision = phase_b_payload.get("decision", {})
        if not phase_b_decision.get("ready_for_phase_d", False):
            rows_path = output_root / "phase_d_benchmark_rows.jsonl"
            _write_jsonl(rows_path, combined_rows)
            payload = {
                "phase": "D",
                "schema": "tac_control_v1_phase_d_benchmark_matrix.v1",
                "suite_dir": str(suite_root),
                "phase_b_dir": str(phase_b_root),
                "phase_b_results": str(Path(phase_b_results)),
                "output_dir": str(output_root),
                "vanilla_checkpoint": None
                if vanilla_checkpoint is None
                else str(Path(vanilla_checkpoint)),
                "seeds": [int(seed) for seed in seed_list],
                "seed_count": len(seed_list),
                "run_count": 0,
                "row_count": 0,
                "rows_jsonl": str(rows_path),
                "missing": [],
                "runs": [],
                "phase_b_decision": phase_b_decision,
                "decision": {
                    "status": "blocked_by_phase_b",
                    "reason": "Phase D benchmark matrix skipped because Phase B is not ready for Phase D.",
                    "ready_for_phase_d": False,
                    "phase_b_status": phase_b_decision.get("status"),
                    "missing_count": 0,
                    "row_count": 0,
                },
            }
            _write_matrix_artifacts(output_root, payload)
            return payload

    vanilla_path = Path(vanilla_checkpoint) if vanilla_checkpoint is not None else None
    vanilla_available = vanilla_path is not None and vanilla_path.exists()
    if vanilla_path is not None and not vanilla_available:
        missing.append(
            {
                "component": "vanilla_checkpoint",
                "checkpoint": str(vanilla_path),
            }
        )

    for seed in seed_list:
        tasks_path = suite_root / f"seed_{int(seed)}" / "tasks.jsonl"
        if not tasks_path.exists():
            missing.append(
                {
                    "component": "tasks_jsonl",
                    "seed": int(seed),
                    "path": str(tasks_path),
                }
            )
            continue
        tac_checkpoint = discover_phase_d_seed_checkpoint(phase_b_root, seed=int(seed))
        if tac_checkpoint is None:
            missing.append(
                {
                    "component": "tac_checkpoint",
                    "seed": int(seed),
                    "phase_b_dir": str(phase_b_root),
                }
            )
            continue
        else:
            run = _run_one_control(
                tasks_path=tasks_path,
                checkpoint=tac_checkpoint,
                output_root=output_root,
                seed=int(seed),
                control_id=f"{tac_control_prefix}_seed_{int(seed)}",
                model_type="tac",
                device=device,
                precision=precision,
                max_new_tokens=max_new_tokens,
                answer_extraction=answer_extraction,
                force=force,
            )
            runs.append(run)
            combined_rows.extend(run["score_rows"])

        if vanilla_available and vanilla_path is not None:
            run = _run_one_control(
                tasks_path=tasks_path,
                checkpoint=vanilla_path,
                output_root=output_root,
                seed=int(seed),
                control_id=vanilla_control_id,
                model_type="vanilla",
                device=device,
                precision=precision,
                max_new_tokens=max_new_tokens,
                answer_extraction=answer_extraction,
                force=force,
            )
            runs.append(run)
            combined_rows.extend(run["score_rows"])

    rows_path = output_root / "phase_d_benchmark_rows.jsonl"
    _write_jsonl(rows_path, combined_rows)
    decision = _matrix_decision(missing, combined_rows)
    payload = {
        "phase": "D",
        "schema": "tac_control_v1_phase_d_benchmark_matrix.v1",
        "suite_dir": str(suite_root),
        "phase_b_dir": str(phase_b_root),
        "output_dir": str(output_root),
        "vanilla_checkpoint": None if vanilla_path is None else str(vanilla_path),
        "phase_b_results": None
        if phase_b_results is None
        else str(Path(phase_b_results)),
        "seeds": [int(seed) for seed in seed_list],
        "seed_count": len(seed_list),
        "run_count": len(runs),
        "row_count": len(combined_rows),
        "rows_jsonl": str(rows_path),
        "missing": missing,
        "runs": [_public_run_metadata(run) for run in runs],
        "decision": decision,
    }
    _write_matrix_artifacts(output_root, payload)
    return payload


def _load_phase_b_results(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(path)
    if not source.exists():
        return None
    return json.loads(source.read_text(encoding="utf-8"))


def _write_matrix_artifacts(output_root: Path, payload: dict[str, Any]) -> None:
    (output_root / "phase_d_prediction_matrix.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (output_root / "RESULTS.md").write_text(
        _format_matrix_markdown(payload),
        encoding="utf-8",
    )


def discover_phase_d_suite_seeds(suite_dir: str | Path) -> list[int]:
    root = Path(suite_dir)
    seeds = []
    for path in sorted(root.glob("seed_*/tasks.jsonl")):
        try:
            seeds.append(int(path.parent.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return seeds


def discover_phase_d_seed_checkpoint(
    phase_b_dir: str | Path,
    *,
    seed: int,
) -> Path | None:
    root = Path(phase_b_dir)
    seed_roots = [
        path
        for path in sorted(root.glob(f"seed_{int(seed)}*"))
        if path.is_dir()
    ]
    search_roots = sorted(
        seed_roots,
        key=_phase_d_seed_root_score,
        reverse=True,
    ) or [root]
    patterns = [
        "**/specialization_checkpoints/step_020000/checkpoint.pt",
        "**/specialization_checkpoints/step_20000/checkpoint.pt",
        "**/step_020000/checkpoint.pt",
        "**/step_20000/checkpoint.pt",
        "**/specialization_checkpoints/step_010000/checkpoint.pt",
        "**/specialization_checkpoints/step_10000/checkpoint.pt",
        "**/step_010000/checkpoint.pt",
        "**/step_10000/checkpoint.pt",
        "**/last.pt",
        "**/best.pt",
    ]
    for pattern in patterns:
        for search_root in search_roots:
            matches = sorted(search_root.rglob(pattern))
            if matches:
                return matches[0]
    return None


def _phase_d_seed_root_score(path: Path) -> tuple[int, int, str]:
    final_summaries = sorted(path.rglob("final_summary.json"))
    if not final_summaries:
        return (-1, 0, str(path))
    try:
        with final_summaries[0].open("r", encoding="utf-8-sig") as handle:
            summary = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return (0, 0, str(path))
    completed_steps = _optional_int(summary.get("completed_steps")) or 0
    target_steps = _optional_int(summary.get("target_steps")) or 0
    completed_target = int(target_steps > 0 and completed_steps >= target_steps)
    return (completed_steps, completed_target, str(path))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _run_one_control(
    *,
    tasks_path: Path,
    checkpoint: Path,
    output_root: Path,
    seed: int,
    control_id: str,
    model_type: str,
    device: str | torch.device,
    precision: str,
    max_new_tokens: int,
    answer_extraction: str,
    force: bool,
) -> dict[str, Any]:
    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    safe_control_id = _safe_name(control_id)
    predictions_jsonl = seed_dir / f"{safe_control_id}_predictions.jsonl"
    predictions_json = seed_dir / f"{safe_control_id}_predictions.json"
    score_json = seed_dir / f"{safe_control_id}_score.json"
    score_jsonl = seed_dir / f"{safe_control_id}_score.jsonl"
    if not force and score_jsonl.exists():
        return {
            "seed": seed,
            "control_id": control_id,
            "model_type": model_type,
            "checkpoint": str(checkpoint),
            "tasks_jsonl": str(tasks_path),
            "predictions_json": str(predictions_json),
            "predictions_jsonl": str(predictions_jsonl),
            "score_json": str(score_json),
            "score_jsonl": str(score_jsonl),
            "prediction_count": len(load_jsonl(predictions_jsonl))
            if predictions_jsonl.exists()
            else None,
            "skipped": "existing_score",
            "score_rows": load_jsonl(score_jsonl),
        }
    predictions_payload = run_phase_d_checkpoint_predictions(
        tasks_jsonl=tasks_path,
        checkpoint_path=checkpoint,
        control_id=control_id,
        seed=seed,
        output_jsonl=predictions_jsonl,
        model_type=model_type,
        device=device,
        precision=precision,
        max_new_tokens=max_new_tokens,
        answer_extraction=answer_extraction,
    )
    predictions_json.write_text(
        json.dumps(predictions_payload, indent=2),
        encoding="utf-8",
    )
    score_rows = score_phase_d_predictions(
        load_jsonl(tasks_path),
        predictions_payload["rows"],
        control_id=control_id,
        seed=seed,
    )
    score_payload = {
        "phase": "D",
        "schema": "tac_control_v1_phase_d_scored_predictions.v1",
        "tasks_jsonl": str(tasks_path),
        "predictions_jsonl": str(predictions_jsonl),
        "checkpoint": str(checkpoint),
        "control_id": control_id,
        "seed": seed,
        "rows": score_rows,
    }
    score_json.write_text(json.dumps(score_payload, indent=2), encoding="utf-8")
    _write_jsonl(score_jsonl, score_rows)
    return {
        "seed": seed,
        "control_id": control_id,
        "model_type": model_type,
        "checkpoint": str(checkpoint),
        "tasks_jsonl": str(tasks_path),
        "predictions_json": str(predictions_json),
        "predictions_jsonl": str(predictions_jsonl),
        "score_json": str(score_json),
        "score_jsonl": str(score_jsonl),
        "prediction_count": predictions_payload["prediction_count"],
        "score_rows": score_rows,
    }


def _matrix_decision(
    missing: list[dict[str, Any]],
    combined_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if missing:
        return {
            "status": "pending",
            "reason": "Phase D benchmark matrix is waiting for required task or checkpoint artifacts.",
            "missing_count": len(missing),
            "row_count": len(combined_rows),
        }
    if not combined_rows:
        return {
            "status": "pending",
            "reason": "No Phase D benchmark rows were produced.",
            "missing_count": 0,
            "row_count": 0,
        }
    return {
        "status": "completed",
        "reason": "Phase D benchmark predictions and scored rows were produced.",
        "missing_count": 0,
        "row_count": len(combined_rows),
    }


def _public_run_metadata(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key != "score_rows"}


def _format_matrix_markdown(payload: dict[str, Any]) -> str:
    decision = payload.get("decision", {})
    lines = [
        "# TAC-Control-v1 Phase D Prediction Matrix",
        "",
        f"- Status: `{decision.get('status')}`",
        f"- Reason: {decision.get('reason')}",
        f"- Seeds: {', '.join(str(seed) for seed in payload.get('seeds', []))}",
        f"- Runs: {payload.get('run_count')}",
        f"- Scored rows: {payload.get('row_count')}",
        f"- Combined rows JSONL: `{payload.get('rows_jsonl')}`",
        "",
        "| Seed | Control | Model | Checkpoint | Score JSONL |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for run in payload.get("runs", []):
        lines.append(
            "| {seed} | `{control}` | `{model}` | `{checkpoint}` | `{score}` |".format(
                seed=run.get("seed"),
                control=run.get("control_id"),
                model=run.get("model_type"),
                checkpoint=run.get("checkpoint"),
                score=run.get("score_jsonl"),
            )
        )
    if payload.get("missing"):
        lines.extend(["", "Missing artifacts:"])
        for row in payload["missing"]:
            lines.append(f"- `{row.get('component')}`: {json.dumps(row, sort_keys=True)}")
    return "\n".join(lines)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
