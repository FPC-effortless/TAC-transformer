from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.capability import (
    aggregate_evolutionary_search_results,
    format_evolutionary_search_markdown,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank TAC architecture mutations with capability, memory-health, route-utility, and cost gates."
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-identity-share", type=float, default=0.5)
    parser.add_argument("--min-loss-improvement", type=float, default=0.05)
    parser.add_argument("--max-program-memory-cosine", type=float, default=0.85)
    parser.add_argument("--max-dead-program-fraction", type=float, default=0.2)
    parser.add_argument("--min-routed-is-best-fraction", type=float, default=0.2)
    parser.add_argument("--max-vanilla-loss-gap", type=float, default=0.02)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_candidate_rows(args.input)
    result = aggregate_evolutionary_search_results(
        rows,
        max_identity_share=args.max_identity_share,
        min_loss_improvement=args.min_loss_improvement,
        max_program_memory_cosine=args.max_program_memory_cosine,
        max_dead_program_fraction=args.max_dead_program_fraction,
        min_routed_is_best_fraction=args.min_routed_is_best_fraction,
        max_vanilla_loss_gap=args.max_vanilla_loss_gap,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evolutionary_search.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_evolutionary_search_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result["recommendation"], indent=2), flush=True)


def load_candidate_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(_rows_from_payload(payload))
    if not rows:
        raise ValueError("no candidate rows found in input artifacts")
    return rows


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("candidate_rows"), list):
        return [dict(row) for row in payload["candidate_rows"]]
    if isinstance(payload.get("rows"), list):
        return [dict(row) for row in payload["rows"]]
    if isinstance(payload.get("per_seed"), list):
        return [dict(row) for row in payload["per_seed"]]
    if isinstance(payload.get("ranked"), list):
        return [dict(row) for row in payload["ranked"]]
    if isinstance(payload.get("aggregate"), dict):
        return [
            {"variant": variant, **dict(row)}
            for variant, row in payload["aggregate"].items()
            if isinstance(row, dict)
        ]
    return []


if __name__ == "__main__":
    main()
