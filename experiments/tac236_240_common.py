from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence


DEFAULT_SEEDS = (7, 19, 31, 43, 59, 71, 83, 97, 109, 127)
DECISION_STATUSES = {"validated", "not_validated", "blocked"}


def stable_rng(*parts: object) -> random.Random:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return random.Random(int(digest[:16], 16))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def aggregate_numeric(rows: Sequence[dict]) -> dict[str, float]:
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    return {key: float(mean(float(row[key]) for row in rows if key in row)) for key in keys}


def write_artifact(output_dir: Path, filename: str, result: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / filename
    result["artifact_path"] = str(artifact_path)
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def training_strength(*steps: int, smoke: bool = False) -> float:
    if smoke:
        return 0.05
    total = max(0, sum(int(step) for step in steps))
    return clamp(total / 600.0, 0.05, 1.0)


def entropy(probabilities: Iterable[float]) -> float:
    return float(
        -sum(p * math.log(max(p, 1e-8)) for p in probabilities if p > 0.0)
    )


def add_common_args(parser: argparse.ArgumentParser, *, default_output_dir: Path) -> None:
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")


def blocked_or_status(
    *,
    tac236_validated: bool,
    validated: bool,
    boundary: str,
) -> dict[str, object]:
    if not tac236_validated:
        return {
            "status": "blocked",
            "reason": "TAC-236 reproduction/scaling validation has not been supplied as validated.",
            "boundary": boundary,
        }
    return {
        "status": "validated" if validated else "not_validated",
        "boundary": boundary,
    }

