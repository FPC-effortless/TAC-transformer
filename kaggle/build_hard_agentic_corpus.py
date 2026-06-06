from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.data import dedupe_prepared_jsonl
from tac_transformer.data import stable_text_hash
from tac_transformer.hard_agentic_data import (
    estimate_tokens,
    generate_hard_agentic_records,
    hard_record_to_jsonl,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a harder, deduplicated TAC agentic corpus."
    )
    parser.add_argument("--base-dir", type=Path, default=Path("runs/prepared_corpus"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/prepared_corpus_agentic_hard"))
    parser.add_argument("--template-cap", type=int, default=3)
    parser.add_argument("--exact-cap", type=int, default=1)
    parser.add_argument("--max-base-records", type=int, default=None)
    parser.add_argument("--hard-train-records", type=int, default=120_000)
    parser.add_argument("--hard-eval-records", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=101)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_train = args.base_dir / "train.prepared.jsonl"
    base_eval = args.base_dir / "eval.prepared.jsonl"
    deduped_base_train = args.output_dir / "base_train.deduped.jsonl"
    train_hard = args.output_dir / "hard_agentic_train.generated.jsonl"
    eval_hard = args.output_dir / "hard_agentic_eval.generated.jsonl"
    train_path = args.output_dir / "train.prepared.jsonl"
    eval_path = args.output_dir / "eval.prepared.jsonl"

    base_stats = dedupe_prepared_jsonl(
        base_train,
        deduped_base_train,
        exact_cap=args.exact_cap,
        template_cap=args.template_cap,
        max_records=args.max_base_records,
    )
    hard_train_stats = _write_generated(
        train_hard,
        records=args.hard_train_records,
        seed=args.seed,
        split="train",
    )
    hard_eval_stats = _write_generated(
        eval_hard,
        records=args.hard_eval_records,
        seed=args.seed + 10_000,
        split="eval",
    )

    _concat([deduped_base_train, train_hard], train_path)
    _concat([base_eval, eval_hard], eval_path)
    train_stats = _prepared_file_stats(train_path)
    eval_stats = _prepared_file_stats(eval_path)

    manifest = {
        "purpose": "deduplicated harder agentic corpus",
        "base_dir": str(args.base_dir),
        "template_cap": args.template_cap,
        "exact_cap": args.exact_cap,
        "max_base_records": args.max_base_records,
        "parts": {
            "deduped_base_train": {
                "path": str(deduped_base_train),
                "stats": base_stats,
            },
            "hard_agentic_train": {
                "path": str(train_hard),
                "stats": hard_train_stats,
            },
            "base_eval": {
                "path": str(base_eval),
                "stats": _prepared_file_stats(base_eval),
            },
            "hard_agentic_eval": {
                "path": str(eval_hard),
                "stats": hard_eval_stats,
            },
        },
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "train_records": train_stats["records"],
        "eval_records": eval_stats["records"],
        "train_approx_tokens": train_stats["approx_tokens_chars_div_4"],
        "eval_approx_tokens": eval_stats["approx_tokens_chars_div_4"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


def _write_generated(
    path: Path,
    *,
    records: int,
    seed: int,
    split: str,
) -> dict[str, int]:
    stats = {
        "written": 0,
        "serialized_chars": 0,
        "approx_tokens_chars_div_4": 0,
        "duplicate_skipped": 0,
        "attempts": 0,
    }
    seen_hashes: set[str] = set()
    max_attempts = max(records * 20, records + 1000)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for index, record in enumerate(generate_hard_agentic_records(seed=seed, split=split)):
            if stats["written"] >= records:
                break
            if stats["attempts"] >= max_attempts:
                break
            stats["attempts"] += 1
            text_hash = stable_text_hash(record.text)
            if text_hash in seen_hashes:
                stats["duplicate_skipped"] += 1
                continue
            seen_hashes.add(text_hash)
            line = hard_record_to_jsonl(record)
            output.write(line)
            output.write("\n")
            stats["written"] += 1
            stats["serialized_chars"] += len(record.text)
            stats["approx_tokens_chars_div_4"] += estimate_tokens(record.text)
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


if __name__ == "__main__":
    main()
