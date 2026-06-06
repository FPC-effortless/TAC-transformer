from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.data import stable_text_hash
from tac_transformer.distillation_datasets import (
    DIFFICULTY_TIERS,
    STREAMS,
    DistillationRecord,
    estimate_tokens,
    generate_distillation_records,
    preference_pair_row,
    prepared_row,
    record_to_jsonl,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build coding, agentic, knowledge-work, DPO, repair, and curriculum distillation datasets."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/distillation_datasets"))
    parser.add_argument("--train-records-per-stream", type=int, default=1_000)
    parser.add_argument("--eval-records-per-stream", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--exact-cap", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_stats = _write_split(
        args.output_dir,
        split="train",
        records_per_stream=args.train_records_per_stream,
        seed=args.seed,
        exact_cap=args.exact_cap,
    )
    eval_stats = _write_split(
        args.output_dir,
        split="eval",
        records_per_stream=args.eval_records_per_stream,
        seed=args.seed + 10_000,
        exact_cap=args.exact_cap,
    )

    train_path = args.output_dir / "train.prepared.jsonl"
    eval_path = args.output_dir / "eval.prepared.jsonl"
    _concat(
        [args.output_dir / f"{stream}.train.prepared.jsonl" for stream in STREAMS],
        train_path,
    )
    _concat(
        [args.output_dir / f"{stream}.eval.prepared.jsonl" for stream in STREAMS],
        eval_path,
    )

    train_file_stats = _prepared_file_stats(train_path)
    eval_file_stats = _prepared_file_stats(eval_path)
    manifest = {
        "purpose": "production-grade KD dataset family for coding, agentic trajectories, repairs, knowledge synthesis, DPO, and curriculum sampling",
        "generator": "tac_transformer.distillation_datasets",
        "streams": list(STREAMS),
        "difficulty_tiers": DIFFICULTY_TIERS,
        "train_records_per_stream": args.train_records_per_stream,
        "eval_records_per_stream": args.eval_records_per_stream,
        "seed": args.seed,
        "exact_cap": args.exact_cap,
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "train_records": train_file_stats["records"],
        "eval_records": eval_file_stats["records"],
        "train_approx_tokens": train_file_stats["approx_tokens_chars_div_4"],
        "eval_approx_tokens": eval_file_stats["approx_tokens_chars_div_4"],
        "parts": {**train_stats, **eval_stats},
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def _write_split(
    output_dir: Path,
    *,
    split: str,
    records_per_stream: int,
    seed: int,
    exact_cap: int,
) -> dict[str, dict]:
    raw_handles = {
        stream: (output_dir / f"{stream}.{split}.raw.jsonl").open("w", encoding="utf-8", newline="\n")
        for stream in STREAMS
    }
    prepared_handles = {
        stream: (output_dir / f"{stream}.{split}.prepared.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        )
        for stream in STREAMS
    }
    preference_path = output_dir / f"preference_pairs.{split}.jsonl"
    preference_handle = preference_path.open("w", encoding="utf-8", newline="\n")
    counts = Counter()
    chars = Counter()
    duplicates = Counter()
    seen: dict[str, Counter[str]] = {stream: Counter() for stream in STREAMS}

    try:
        for record in generate_distillation_records(seed=seed, split=split):
            stream = record.domain
            if counts[stream] >= records_per_stream:
                if all(counts[s] >= records_per_stream for s in STREAMS):
                    break
                continue

            exact_hash = stable_text_hash(record.text)
            if seen[stream][exact_hash] >= exact_cap:
                duplicates[stream] += 1
                continue
            seen[stream][exact_hash] += 1

            raw_handles[stream].write(record_to_jsonl(record))
            raw_handles[stream].write("\n")
            prepared_handles[stream].write(json.dumps(prepared_row(record), ensure_ascii=False))
            prepared_handles[stream].write("\n")
            pair = preference_pair_row(record)
            if pair is not None:
                preference_handle.write(json.dumps(pair, ensure_ascii=False))
                preference_handle.write("\n")

            counts[stream] += 1
            chars[stream] += len(record.text)
    finally:
        for handle in raw_handles.values():
            handle.close()
        for handle in prepared_handles.values():
            handle.close()
        preference_handle.close()

    stats = {}
    for stream in STREAMS:
        stats[f"{stream}_{split}"] = {
            "path": str(output_dir / f"{stream}.{split}.raw.jsonl"),
            "prepared_path": str(output_dir / f"{stream}.{split}.prepared.jsonl"),
            "stats": {
                "written": counts[stream],
                "serialized_chars": chars[stream],
                "approx_tokens_chars_div_4": round(chars[stream] / 4),
                "duplicate_skipped": duplicates[stream],
            },
        }
    preference_records = _count_jsonl(preference_path)
    stats[f"preference_pair_{split}"] = {
        "path": str(preference_path),
        "stats": {
            "written": preference_records,
            "approx_tokens_chars_div_4": _estimate_jsonl_tokens(preference_path),
        },
    }
    return stats


def _concat(parts: list[Path], output: Path) -> None:
    with output.open("wb") as destination:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, destination)


def _prepared_file_stats(path: Path) -> dict[str, int]:
    records = 0
    chars = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records += 1
            chars += len(str(row.get("text", "")))
    return {
        "records": records,
        "serialized_chars": chars,
        "approx_tokens_chars_div_4": round(chars / 4),
    }


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _estimate_jsonl_tokens(path: Path) -> int:
    tokens = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tokens += estimate_tokens(line)
    return tokens


if __name__ == "__main__":
    main()
