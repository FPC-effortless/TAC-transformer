from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator


DATASET_MANIFEST = {
    "schema": "tac_v02_dataset_manifest.v1",
    "language": [
        {
            "name": "fineweb_edu",
            "hub_id": "HuggingFaceFW/fineweb-edu",
            "config": "sample-10BT",
            "split": "train",
            "text_fields": ["text"],
            "source_url": "https://hf.co/datasets/HuggingFaceFW/fineweb-edu",
        },
        {
            "name": "slimpajama_6b",
            "hub_id": "DKYoon/SlimPajama-6B",
            "config": None,
            "split": "train",
            "text_fields": ["text"],
            "source_url": "https://hf.co/datasets/DKYoon/SlimPajama-6B",
        },
    ],
    "code": [
        {
            "name": "codesearchnet",
            "hub_id": "code-search-net/code_search_net",
            "config": None,
            "split": "train",
            "text_fields": ["func_documentation_string", "func_code_string", "whole_func_string"],
            "source_url": "https://hf.co/datasets/code-search-net/code_search_net",
            "license_note": "Dataset card lists license as other; run code-license audit before training.",
        }
    ],
    "long_horizon": [
        "planning_trace",
        "repair_trace",
        "execution_trace",
    ],
    "holdouts": [
        "persistent_state",
        "repair",
        "compression",
    ],
}


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_bucket(text: str) -> float:
    return int(stable_hash(text)[:8], 16) / 0xFFFFFFFF


def validation_bucket(text: str, holdout_rate: float) -> bool:
    bucket = int(stable_hash(text)[:8], 16) / 0xFFFFFFFF
    return bucket < holdout_rate


def split_target(
    text: str,
    *,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    holdout_rate: float,
) -> list[dict[str, Any]]:
    bucket = stable_bucket(text)
    if bucket < holdout_rate:
        return heldout_rows
    if bucket < 2 * holdout_rate:
        return validation_rows
    return train_rows


def row_text(row: dict[str, Any], fields: list[str]) -> str:
    parts = []
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n\n".join(parts)


def synthetic_long_horizon_rows(count: int, seed: int) -> Iterator[dict[str, str]]:
    rng = random.Random(seed)
    workflows = ["planning_trace", "repair_trace", "execution_trace"]
    for index in range(count):
        workflow = workflows[index % len(workflows)]
        bug_id = rng.randint(1000, 9999)
        yield {
            "stream": "long_horizon",
            "source": workflow,
            "text": (
                f"TRACE {bug_id} {workflow}\n"
                f"plan: inspect failing signal, isolate state, apply minimal fix\n"
                f"verify: run focused test, compare regression guard, update memory\n"
                f"carry_state: root_cause={bug_id % 17}; next_action=continue"
            ),
        }


def iter_streaming_source(source: dict[str, Any], limit: int) -> Iterator[dict[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install datasets to stream v0.2 sources: python -m pip install datasets"
        ) from exc

    kwargs = {
        "path": source["hub_id"],
        "split": source["split"],
        "streaming": True,
    }
    if source.get("config"):
        kwargs["name"] = source["config"]
    dataset = load_dataset(**kwargs)
    emitted = 0
    for row in dataset:
        text = row_text(row, source["text_fields"])
        if not text:
            continue
        yield {"stream": source["name"], "source": source["hub_id"], "text": text}
        emitted += 1
        if emitted >= limit:
            break


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def build_dataset(
    *,
    output_dir: Path,
    per_source_limit: int,
    long_horizon_count: int,
    holdout_rate: float,
    offline_synthetic_only: bool,
    seed: int,
) -> dict[str, Any]:
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []

    if not offline_synthetic_only:
        for group in ("language", "code"):
            for source in DATASET_MANIFEST[group]:
                for row in iter_streaming_source(source, per_source_limit):
                    target = split_target(
                        row["text"],
                        train_rows=train_rows,
                        validation_rows=validation_rows,
                        heldout_rows=heldout_rows,
                        holdout_rate=holdout_rate,
                    )
                    target.append(row)

    for row in synthetic_long_horizon_rows(long_horizon_count, seed):
        target = split_target(
            row["text"],
            train_rows=train_rows,
            validation_rows=validation_rows,
            heldout_rows=heldout_rows,
            holdout_rate=holdout_rate,
        )
        target.append(row)

    train_path = output_dir / "train.v02.jsonl"
    eval_path = output_dir / "eval.v02.jsonl"
    holdout_path = output_dir / "validation_holdout.v02.jsonl"
    counts = {
        "train": write_jsonl(train_path, train_rows),
        "eval": write_jsonl(eval_path, validation_rows),
        "validation_holdout": write_jsonl(holdout_path, heldout_rows),
    }
    manifest = {
        **DATASET_MANIFEST,
        "output_dir": str(output_dir),
        "paths": {
            "train": str(train_path),
            "eval": str(eval_path),
            "validation_holdout": str(holdout_path),
        },
        "counts": counts,
        "holdout_rate": holdout_rate,
        "offline_synthetic_only": offline_synthetic_only,
        "boundary": (
            "Holdout rows are written separately and must never be passed as --train-jsonl. "
            "CodeSearchNet requires a separate permissive-license audit before inclusion."
        ),
    }
    (output_dir / "dataset_manifest.v02.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/v02_dataset"))
    parser.add_argument("--per-source-limit", type=int, default=1000)
    parser.add_argument("--long-horizon-count", type=int, default=3000)
    parser.add_argument("--holdout-rate", type=float, default=0.05)
    parser.add_argument("--offline-synthetic-only", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    manifest = build_dataset(
        output_dir=args.output_dir,
        per_source_limit=args.per_source_limit,
        long_horizon_count=args.long_horizon_count,
        holdout_rate=args.holdout_rate,
        offline_synthetic_only=args.offline_synthetic_only,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
