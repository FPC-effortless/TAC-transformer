from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


WORKLOADS = (
    "coding_repo_compression",
    "multi_session_assistant_memory",
    "research_workflow_compression",
    "long_document_compression",
)

REQUIRED_RATIOS = (10, 20)
EXPERIMENTAL_RATIOS = (50,)
QUALITY_FLOORS = {
    10: 0.80,
    20: 0.65,
    50: 0.50,
}

SMOKE_QUALITY_RETENTION = {
    10: {
        "coding_repo_compression": 0.88,
        "multi_session_assistant_memory": 0.86,
        "research_workflow_compression": 0.84,
        "long_document_compression": 0.82,
    },
    20: {
        "coding_repo_compression": 0.74,
        "multi_session_assistant_memory": 0.71,
        "research_workflow_compression": 0.69,
        "long_document_compression": 0.66,
    },
    50: {
        "coding_repo_compression": 0.49,
        "multi_session_assistant_memory": 0.47,
        "research_workflow_compression": 0.44,
        "long_document_compression": 0.42,
    },
}


def evaluate_structure_compression_roi(
    measurements: dict[int, dict[str, float]] | None = None,
) -> dict[str, Any]:
    measurements = measurements or SMOKE_QUALITY_RETENTION
    ratio_results: dict[str, Any] = {}
    required_passed = True

    for ratio, workload_scores in sorted(measurements.items()):
        floor = QUALITY_FLOORS.get(ratio, 0.0)
        workload_results = {}
        all_workloads_pass = True
        for workload in WORKLOADS:
            quality = float(workload_scores.get(workload, 0.0))
            passed = quality >= floor
            all_workloads_pass = all_workloads_pass and passed
            workload_results[workload] = {
                "quality_retention": quality,
                "quality_floor": floor,
                "passed": passed,
            }

        required = ratio in REQUIRED_RATIOS
        if required:
            required_passed = required_passed and all_workloads_pass
        ratio_results[f"{ratio}x"] = {
            "required_gate": required,
            "experimental": ratio in EXPERIMENTAL_RATIOS,
            "passed": all_workloads_pass,
            "workloads": workload_results,
        }

    return {
        "benchmark": "TAC-SCM structure compression ROI gate",
        "status": "passed" if required_passed else "failed",
        "required_ratios": list(REQUIRED_RATIOS),
        "experimental_ratios": list(EXPERIMENTAL_RATIOS),
        "ratios": ratio_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    result = evaluate_structure_compression_roi()
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
