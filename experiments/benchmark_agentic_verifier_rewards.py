from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import VerifierCase, build_authority_report
from tac_transformer.agentic_rl_math import verifier_reward_from_authority_report


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/agentic_verifier_rewards_2026_06_04")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove authority-report-based verifier reward shaping."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_agentic_verifier_reward_probe()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "agentic_verifier_rewards.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def run_agentic_verifier_reward_probe() -> dict[str, Any]:
    clean_report = build_authority_report(
        run_id="clean",
        events=[
            VerifierCase(
                "clean",
                "math",
                expected="4",
                observed="4",
                authority_mode="verified_execution",
            ).verify()
        ],
    )
    contaminated_report = build_authority_report(
        run_id="contaminated",
        events=[
            VerifierCase(
                "ok",
                "math",
                expected="4",
                observed="4",
                authority_mode="verified_execution",
            ).verify(),
            VerifierCase(
                "false",
                "math",
                expected="4",
                observed="5",
                authority_mode="verified_execution",
            ).verify(),
            VerifierCase(
                "cross",
                "math",
                expected="7",
                observed="7",
                authority_mode="retrieved_evidence",
                source_domain="history",
            ).verify(),
        ],
    )
    clean_reward = verifier_reward_from_authority_report(clean_report, base_reward=1.0)
    contaminated_reward = verifier_reward_from_authority_report(
        contaminated_report,
        base_reward=1.0,
        trusted_correct_bonus=0.25,
        false_authority_penalty=1.0,
        cross_domain_penalty=0.5,
    )
    checks = {
        "clean_reward_positive": clean_reward["verifier_reward"] > 1.0,
        "false_authority_penalized": contaminated_reward["false_authority_count"] == 1,
        "cross_domain_penalized": contaminated_reward[
            "cross_domain_authority_violation_count"
        ]
        == 1,
        "contaminated_reward_lower": contaminated_reward["verifier_reward"]
        < clean_reward["verifier_reward"],
        "false_authority_rate_reported": contaminated_reward["false_authority_rate"]
        > 0.0,
    }
    return {
        "schema": "agentic_verifier_rewards_probe.v1",
        "date": "2026-06-04",
        "clean_reward": clean_reward,
        "contaminated_reward": contaminated_reward,
        "decision": {
            "status": "verifier_rewards_proved" if all(checks.values()) else "blocked",
            "checks": checks,
            "scope": (
                "This proves reward shaping can consume the existing AuthorityReport "
                "manifest contract, reward clean trusted verification, and penalize "
                "false authority plus cross-domain authority contamination. It does "
                "not yet apply the shaped rewards inside a group-relative policy "
                "training objective."
            ),
        },
    }


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Agentic Verifier Rewards",
        "",
        f"Decision: `{report['decision']['status']}`",
        "",
        "## Rewards",
        "",
        f"Clean verifier reward: `{report['clean_reward']['verifier_reward']:.4f}`",
        (
            "Contaminated verifier reward: "
            f"`{report['contaminated_reward']['verifier_reward']:.4f}`"
        ),
        (
            "False authority rate: "
            f"`{report['contaminated_reward']['false_authority_rate']:.4f}`"
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
