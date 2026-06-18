from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tac236_240_common import write_artifact
from tac_transformer.research_directions import (
    ProceduralMemoryRecord,
    ProceduralMemoryStore,
    adapt_procedural_memory_after_feedback,
    two_level_structure_route,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tac279_volume_procedural_memory_improvements")


def run_tac279_volume_procedural_memory_improvements(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    embeddings = torch.tensor([[0.10, 0.05], [4.20, 0.10]])
    family_means = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
    family_log_vars = torch.zeros_like(family_means)
    specialist_means = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 2.0]],
            [[4.0, 0.0], [4.0, 2.0]],
        ]
    )
    route = two_level_structure_route(
        embeddings,
        family_means,
        family_log_vars,
        specialist_means,
    )
    expected_families = torch.tensor([0, 1])
    expected_specialists = torch.tensor([0, 0])
    family_route_accuracy = float((route.family_ids == expected_families).float().mean())
    specialist_route_accuracy = float(
        (route.specialist_ids == expected_specialists).float().mean()
    )

    top_k_guard = 0.0
    try:
        two_level_structure_route(
            embeddings[:1],
            family_means,
            family_log_vars,
            specialist_means,
            top_k=3,
        )
    except ValueError as exc:
        top_k_guard = float("top_k must be <= specialists per family" in str(exc))

    store = ProceduralMemoryStore()
    store.write(
        ProceduralMemoryRecord(
            procedure_id="schema_patch_old",
            family_id="schema",
            task_descriptor="repair schema drift",
            steps=("edit schema",),
            embedding=torch.tensor([0.95, 0.05]),
            success_rate=0.95,
        )
    )
    store.write(
        ProceduralMemoryRecord(
            procedure_id="routing_patch",
            family_id="routing",
            task_descriptor="repair routing drift",
            steps=("edit route",),
            embedding=torch.tensor([0.70, 0.20]),
            success_rate=0.50,
        )
    )
    query = torch.tensor([1.0, 0.0])
    before_top = store.retrieve(query, top_k=1)[0].procedure_id
    update_summary = adapt_procedural_memory_after_feedback(
        store,
        selected_procedure_id="schema_patch_old",
        task_embedding=query,
        success=False,
        expected_family_id="routing",
        learning_rate=0.8,
    )
    after_top = store.retrieve(query, top_k=1)[0].procedure_id
    failed_procedure_recovery = float(
        before_top == "schema_patch_old" and after_top == "routing_patch"
    )
    retrieval_margin_improvement = float(
        update_summary["retrieval_margin_after"]
        - update_summary["retrieval_margin_before"]
    )

    metrics = {
        "family_route_accuracy": family_route_accuracy,
        "specialist_route_accuracy": specialist_route_accuracy,
        "mean_family_confidence": float(route.family_confidence.mean()),
        "mean_specialist_confidence": float(route.specialist_confidence.mean()),
        "top_k_guard": top_k_guard,
        "failed_procedure_recovery": failed_procedure_recovery,
        "retrieval_margin_before": float(update_summary["retrieval_margin_before"]),
        "retrieval_margin_after": float(update_summary["retrieval_margin_after"]),
        "retrieval_margin_improvement": retrieval_margin_improvement,
        "procedural_records": float(len(store)),
    }
    gates = {
        "family_route_accuracy": metrics["family_route_accuracy"] >= 1.0,
        "specialist_route_accuracy": metrics["specialist_route_accuracy"] >= 1.0,
        "top_k_guard": metrics["top_k_guard"] >= 1.0,
        "failed_procedure_recovery": metrics["failed_procedure_recovery"] >= 1.0,
        "retrieval_margin_improvement": metrics["retrieval_margin_improvement"] > 0.0,
    }
    status = "validated" if all(gates.values()) else "not_validated"
    result = {
        "schema": "tac279_volume_procedural_memory_improvements.v1",
        "method": {
            "task": "volume_procedural_memory_improvements",
            "routing": "adaptive_volume_family_then_family_local_specialist",
            "procedural_memory": "feedback_updates_push_failed_procedure_and_pull_expected_family",
        },
        "cases": [
            {
                "case": "two_level_structure_route",
                "family_ids": route.family_ids.tolist(),
                "specialist_ids": route.specialist_ids.tolist(),
            },
            {
                "case": "procedural_feedback_retrieval",
                "before_top": before_top,
                "after_top": after_top,
            },
        ],
        "metrics": metrics,
        "gates": gates,
        "decision": {
            "status": status,
            "boundary": (
                "Validates deterministic local routing/update primitives; it is not a trained "
                "checkpoint or open-ended repository-repair result."
            ),
        },
    }
    return write_artifact(output_dir, "tac279_volume_procedural_memory_improvements.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_tac279_volume_procedural_memory_improvements(output_dir=args.output_dir)
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
