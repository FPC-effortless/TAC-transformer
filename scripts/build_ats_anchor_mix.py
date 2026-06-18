from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc


def reservoir_sample(path: Path, *, k: int, seed: int) -> list[dict[str, Any]]:
    if k <= 0:
        return []
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    seen = 0
    for row in iter_jsonl(path):
        seen += 1
        if len(sample) < k:
            sample.append(row)
            continue
        index = rng.randrange(seen)
        if index < k:
            sample[index] = row
    return sample


def normalize_ats_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    domain = str(row.get("domain", "unknown"))
    task_id = str(row.get("task_id", "unknown"))
    out["domain"] = f"ats_transfer_supervised:{domain}:{task_id}"
    out["stream"] = "ats_transfer"
    out["source"] = row.get("source", "ats_transfer_supervised")
    if "text" not in out:
        out["text"] = f"{out.get('prompt', '')}{out.get('answer', '')}\n"
    return out


def normalize_anchor_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.setdefault("stream", "anchor")
    out.setdefault("domain", "anchor")
    if "text" not in out:
        out["text"] = f"{out.get('prompt', '')}\n<|end|>\n{out.get('answer', '')}"
    return out


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an ATS transfer fine-tune JSONL with a small anchor sample."
    )
    parser.add_argument("--ats-train-jsonl", type=Path, required=True)
    parser.add_argument("--ats-eval-jsonl", type=Path, required=True)
    parser.add_argument("--anchor-train-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-records", type=int, default=None)
    parser.add_argument("--anchor-fraction-of-ats", type=float, default=0.10)
    parser.add_argument("--ats-weight", type=float, default=30.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.ats_weight <= 0.0:
        raise ValueError("--ats-weight must be positive")
    if args.anchor_weight <= 0.0:
        raise ValueError("--anchor-weight must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ats_train_rows = [normalize_ats_row(row) for row in iter_jsonl(args.ats_train_jsonl)]
    ats_eval_rows = [normalize_ats_row(row) for row in iter_jsonl(args.ats_eval_jsonl)]
    anchor_records = args.anchor_records
    if anchor_records is None:
        anchor_records = round(len(ats_train_rows) * float(args.anchor_fraction_of_ats))
    anchor_rows = [
        normalize_anchor_row(row)
        for row in reservoir_sample(args.anchor_train_jsonl, k=anchor_records, seed=args.seed)
    ]

    train_path = args.output_dir / "train.completions.jsonl"
    eval_path = args.output_dir / "eval.completions.jsonl"
    train_count = write_jsonl(train_path, [*ats_train_rows, *anchor_rows])
    eval_count = write_jsonl(eval_path, ats_eval_rows)
    weights = {
        "*": float(args.anchor_weight),
        "ats_transfer_supervised": float(args.ats_weight),
    }
    weights_path = args.output_dir / "sampling_weights.json"
    weights_path.write_text(json.dumps(weights, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema": "ats_anchor_mix.v1",
        "train_jsonl": str(train_path),
        "eval_jsonl": str(eval_path),
        "sampling_weights_json": str(weights_path),
        "ats_train_records": len(ats_train_rows),
        "ats_eval_records": len(ats_eval_rows),
        "anchor_train_records": len(anchor_rows),
        "train_records": train_count,
        "eval_records": eval_count,
        "ats_weight": float(args.ats_weight),
        "anchor_weight": float(args.anchor_weight),
        "notes": [
            "ATS rows keep prompt/answer fields for answer-only supervision.",
            "ATS rows are not physically duplicated; sampling_weights.json supplies the effective upsampling.",
            "Eval is ATS-only so transfer performance is not diluted by anchor examples.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
