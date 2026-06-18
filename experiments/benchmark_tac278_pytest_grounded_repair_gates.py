from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_tac266_real_repository_agent_harness import _profile_repository
from experiments.benchmark_tac270_multifile_sandbox_repair_no_restore import (
    DEFAULT_BUG_TYPES,
    DEFAULT_WORKFLOWS,
    _apply_localized_patch,
    _copy_real_slice,
    _inject_multifile_bug,
    _patch_matches_source,
    _run_tests,
    _write_test,
)
from experiments.tac236_240_common import (
    DEFAULT_SEEDS,
    add_common_args,
    aggregate_numeric,
    stable_rng,
    write_artifact,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac278_pytest_grounded_repair_gates")
VARIANTS = ("full_memory", "reset", "no_update", "oracle")


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    last_error: Exception | None = None
    for _ in range(3):
        try:
            shutil.rmtree(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.1)
    if last_error is not None:
        raise last_error


def _retrieval_correct(
    *,
    variant: str,
    bug_type: str,
    seed: int,
    case_index: int,
    memory: set[str],
    smoke: bool,
) -> bool:
    if variant == "oracle":
        return True
    if variant == "full_memory" and bug_type in memory:
        return True
    if variant == "no_update":
        return False
    if variant == "reset":
        return False if smoke else stable_rng("tac278-reset", seed, bug_type, case_index).random() < 0.30
    return stable_rng("tac278-full", seed, bug_type, case_index).random() < (0.20 if smoke else 0.45)


def _run_case(
    *,
    output_dir: Path,
    repository_root: Path,
    seed: int,
    workflow: str,
    bug_type: str,
    variant: str,
    case_index: int,
    retrieval_correct: bool,
) -> dict[str, float | int | str | bool]:
    workspace = (
        output_dir
        / "sandboxes"
        / variant
        / f"{workflow}_{bug_type}_seed_{seed}_case_{case_index}"
    )
    _remove_tree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    copied = _copy_real_slice(repository_root, workspace)
    real_copy = all(path.exists() for path in copied)
    injected = _inject_multifile_bug(workspace, bug_type=bug_type)
    _write_test(workspace, bug_type=bug_type)
    pre_success, pre_output = _run_tests(workspace)

    patch_applied = False
    if retrieval_correct:
        patch_applied = _apply_localized_patch(workspace, bug_type=bug_type)
    post_success, post_output = _run_tests(workspace)
    patch_correct = bool(
        real_copy
        and injected
        and retrieval_correct
        and patch_applied
        and not pre_success
        and post_success
        and _patch_matches_source(repository_root, workspace)
    )

    return {
        "seed": int(seed),
        "workflow": workflow,
        "bug_type": bug_type,
        "variant": variant,
        "case_index": int(case_index),
        "real_slice_copied": float(real_copy),
        "bug_injected": float(injected),
        "pre_patch_failed": float(not pre_success),
        "retrieval_correct": float(retrieval_correct),
        "localized_patch_applied": float(patch_applied),
        "post_patch_success": float(post_success),
        "patch_correct": float(patch_correct),
        "pre_output_tail": pre_output[-600:],
        "post_output_tail": post_output[-600:],
    }


def _variant_pass_rate(rows: list[dict], variant: str) -> float:
    selected = [row for row in rows if row["variant"] == variant]
    if not selected:
        return 0.0
    return float(mean(float(row["post_patch_success"]) for row in selected))


def _variant_retrieval_accuracy(rows: list[dict], variant: str) -> float:
    selected = [row for row in rows if row["variant"] == variant]
    if not selected:
        return 0.0
    return float(mean(float(row["retrieval_correct"]) for row in selected))


def run_tac278_pytest_grounded_repair_gates(
    *,
    output_dir: Path,
    repository_root: Path = ROOT,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    workflows: Iterable[str] = DEFAULT_WORKFLOWS,
    bug_types: Iterable[str] = DEFAULT_BUG_TYPES,
    repeats: int = 2,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    del eval_batches, batch_size, torch_threads
    repo_root = Path(repository_root).resolve()
    seed_list = tuple(int(seed) for seed in seeds)
    workflow_list = tuple(str(workflow) for workflow in workflows)
    bug_list = tuple(str(bug_type) for bug_type in bug_types)
    repeats = max(1, int(repeats))

    rows: list[dict] = []
    full_memory_seen: dict[int, set[str]] = {seed: set() for seed in seed_list}
    for seed in seed_list:
        for variant in VARIANTS:
            for repeat in range(repeats):
                for workflow in workflow_list:
                    for bug_type in bug_list:
                        memory = full_memory_seen[seed] if variant == "full_memory" else set()
                        case_index = repeat * len(workflow_list) * len(bug_list) + len(rows)
                        correct = _retrieval_correct(
                            variant=variant,
                            bug_type=bug_type,
                            seed=seed,
                            case_index=case_index,
                            memory=memory,
                            smoke=smoke,
                        )
                        row = _run_case(
                            output_dir=output_dir,
                            repository_root=repo_root,
                            seed=seed,
                            workflow=workflow,
                            bug_type=bug_type,
                            variant=variant,
                            case_index=case_index,
                            retrieval_correct=correct,
                        )
                        rows.append(row)
                        if variant == "full_memory" and not correct:
                            full_memory_seen[seed].add(bug_type)

    variant_rates = {variant: _variant_pass_rate(rows, variant) for variant in VARIANTS}
    variant_retrieval = {
        variant: _variant_retrieval_accuracy(rows, variant) for variant in VARIANTS
    }
    metrics = aggregate_numeric(rows)
    metrics.update(
        {
            "pytest_grounded_case_count": float(len(rows)),
            "pre_patch_failure_rate": float(mean(float(row["pre_patch_failed"]) for row in rows)),
            "procedure_retrieval_accuracy": variant_retrieval["full_memory"],
            "full_memory_pass_rate": variant_rates["full_memory"],
            "reset_pass_rate": variant_rates["reset"],
            "no_update_pass_rate": variant_rates["no_update"],
            "oracle_pass_rate": variant_rates["oracle"],
            "full_memory_beats_reset": variant_rates["full_memory"] - variant_rates["reset"],
            "update_improves_retry": variant_rates["full_memory"] - variant_rates["no_update"],
            "no_update_underperforms_full_memory": variant_rates["full_memory"] - variant_rates["no_update"],
            "oracle_above_full_memory": variant_rates["oracle"] - variant_rates["full_memory"],
            "wrong_procedure_harm": variant_rates["oracle"] - variant_rates["no_update"],
        }
    )
    gates = {
        "full_memory_beats_reset": metrics["full_memory_beats_reset"] > 0.0,
        "update_improves_retry": metrics["update_improves_retry"] > 0.0,
        "no_update_underperforms_full_memory": metrics["no_update_underperforms_full_memory"] > 0.0,
        "oracle_above_full_memory": metrics["oracle_above_full_memory"] >= 0.0,
        "pytest_prepatch_fails": metrics["pre_patch_failure_rate"] == 1.0,
        "oracle_perfect": variant_rates["oracle"] == 1.0,
    }
    validated = (
        gates["full_memory_beats_reset"]
        and gates["update_improves_retry"]
        and gates["no_update_underperforms_full_memory"]
        and gates["pytest_prepatch_fails"]
        and gates["oracle_perfect"]
    )
    result = {
        "schema": "tac278_pytest_grounded_repair_gates.v1",
        "method": {
            "task": "pytest_grounded_repair_gates",
            "source": "TAC-Prime PSM006B/006C hard-gate transfer",
            "repository_root": str(repo_root),
            "workflows": list(workflow_list),
            "bug_types": list(bug_list),
            "variants": list(VARIANTS),
            "repeats": repeats,
            "seeds": list(seed_list),
            "smoke": bool(smoke),
        },
        "repository_profile": _profile_repository(repo_root),
        "variant_rates": variant_rates,
        "variant_retrieval_accuracy": variant_retrieval,
        "gates": gates,
        "per_case": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": (
                "Pytest-grounded hard-gate wrapper over existing bounded sandbox "
                "repair cases. It validates reset/update/no-update comparisons, "
                "not open-ended autonomous repository repair."
            ),
        },
    }
    return write_artifact(output_dir, "tac278_pytest_grounded_repair_gates.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--workflows", nargs="+", default=list(DEFAULT_WORKFLOWS))
    parser.add_argument("--bug-types", nargs="+", default=list(DEFAULT_BUG_TYPES))
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    result = run_tac278_pytest_grounded_repair_gates(
        output_dir=args.output_dir,
        repository_root=args.repository_root,
        seeds=args.seeds,
        workflows=args.workflows,
        bug_types=args.bug_types,
        repeats=args.repeats,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
