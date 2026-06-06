from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    AgenticScratchpadState,
    ScratchpadItem,
    apply_agentic_scratchpad_transition,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/agentic_scratchpad_state_2026_06_04")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove verifier-gated first-class TAC-Agent-RL scratchpad state."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_agentic_scratchpad_state_probe()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "agentic_scratchpad_state.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def run_agentic_scratchpad_state_probe() -> dict[str, Any]:
    state = AgenticScratchpadState.empty(budget=2)
    candidates = (
        ScratchpadItem(
            "left",
            "observation",
            "2",
            utility=0.8,
            confidence=0.95,
        ),
        ScratchpadItem(
            "right",
            "observation",
            "7",
            utility=0.8,
            confidence=0.95,
        ),
        ScratchpadItem(
            "imagined",
            "simulation",
            "9",
            utility=1.0,
            confidence=0.99,
            imagined=True,
        ),
    )
    next_state, transition = apply_agentic_scratchpad_transition(
        state,
        candidates,
        commit_logits=torch.tensor([8.0, 8.0, 8.0]),
        verifier_supported_ids={"left", "right"},
    )
    checks = {
        "commits_verified_observations": transition["committed_ids"] == ["left", "right"],
        "rejects_unverified_imagined_item": transition["rejected_ids"] == ["imagined"],
        "state_respects_budget": len(next_state.items) <= next_state.budget,
        "state_advances_step": next_state.step == 1,
        "hypothesis_contamination_blocked": transition["hypothesis_contamination_rate"] == 0.0,
    }
    return {
        "schema": "agentic_scratchpad_state.v1",
        "date": "2026-06-04",
        "initial_state": {
            "budget": state.budget,
            "step": state.step,
            "item_ids": list(state.item_ids()),
        },
        "transition": transition,
        "next_state": {
            "budget": next_state.budget,
            "step": next_state.step,
            "item_ids": list(next_state.item_ids()),
            "items": [
                {
                    "item_id": item.item_id,
                    "kind": item.kind,
                    "payload": item.payload,
                    "imagined": item.imagined,
                    "verified": item.verified,
                }
                for item in next_state.items
            ],
        },
        "decision": {
            "status": "scratchpad_state_verified" if all(checks.values()) else "blocked",
            "checks": checks,
            "scope": (
                "This proves a first-class scratchpad state transition can consume "
                "controller commit logits, require verifier support before budgeted "
                "state update, and block unverified imagined hypotheses. It does "
                "not yet attach scratchpad state to autoregressive decoding or "
                "Phase B/D task execution."
            ),
        },
    }


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Agentic Scratchpad State",
        "",
        f"Decision: `{report['decision']['status']}`",
        "",
        "## Transition",
        "",
        f"Selected: `{report['transition']['selected_ids']}`",
        f"Committed: `{report['transition']['committed_ids']}`",
        f"Rejected: `{report['transition']['rejected_ids']}`",
        f"Next state: `{report['next_state']['item_ids']}`",
        "",
        "## Safety",
        "",
        (
            "Hypothesis contamination rate: "
            f"`{report['transition']['hypothesis_contamination_rate']:.4f}`"
        ),
        "",
        "## Scope",
        "",
        report["decision"]["scope"],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
