from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.procedural_memory import ProceduralMemoryStore, ProceduralStep
from tac_transformer.repair_controller import (
    VerificationResult,
    VerifierGuidedRepairController,
)


def run_procedural_repair_memory_smoke() -> dict[str, Any]:
    memory = ProceduralMemoryStore()
    memory.write(
        task_key="marker_repair",
        procedure_trace=[
            ProceduralStep(
                action="append fixed marker after verifier failure",
                observation="missing fixed marker",
                success=True,
                repair_delta=" fixed",
            )
        ],
        success_score=1.0,
    )
    controller = VerifierGuidedRepairController(memory=memory, max_attempts=3)

    def verifier(output: str) -> VerificationResult:
        return VerificationResult(
            passed="fixed" in output,
            feedback="missing fixed marker",
        )

    def repair(output: str, instruction: str) -> str:
        if "append fixed marker" in instruction:
            return output + " fixed"
        return output

    result = controller.run(
        task_key="marker_repair",
        initial_output="broken",
        verifier=verifier,
        repair=repair,
    )
    return {
        "benchmark": "TAC-SCM procedural repair memory smoke",
        "status": "passed" if result.passed else "failed",
        "attempts": len(result.attempts),
        "final_output": result.final_output,
        "memory_records": len(controller.memory.records),
        "external_to_base_lm": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    result = run_procedural_repair_memory_smoke()
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
