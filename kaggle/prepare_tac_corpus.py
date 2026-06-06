from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.data import prepare_jsonl_dataset


DEFAULT_USEF_FILES = [
    "arc_curriculum_dataset.json",
    "l0_primitive_transformations.json",
    "l1_compositional_procedures.json",
    "l2_verification_trajectories.json",
    "l3_procedural_memory.json",
    "l4_long_horizon_workflows.json",
    "l7_open_ended_growth.json",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a prepared TAC training corpus.")
    parser.add_argument("--agent-data-root", type=Path, required=True)
    parser.add_argument("--usef-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/prepared_corpus"))
    parser.add_argument("--duplicate-cap", type=int, default=3)
    parser.add_argument("--max-records-per-file", type=int, default=None)
    parser.add_argument("--include-usef", action="store_true", default=True)
    parser.add_argument("--no-usef", action="store_false", dest="include_usef")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "agent_data_root": str(args.agent_data_root),
        "usef_root": str(args.usef_root),
        "duplicate_cap": args.duplicate_cap,
        "max_records_per_file": args.max_records_per_file,
        "train_parts": [],
        "eval_parts": [],
    }

    master_train = args.agent_data_root / "master_500k" / "master_train.jsonl"
    master_eval = args.agent_data_root / "master_500k" / "master_eval.jsonl"
    train_parts = []
    eval_parts = []

    train_parts.append(
        _prepare_part(
            master_train,
            args.output_dir / "agent_master_500k_train.prepared.jsonl",
            duplicate_cap=args.duplicate_cap,
            max_records=args.max_records_per_file,
        )
    )
    eval_parts.append(
        _prepare_part(
            master_eval,
            args.output_dir / "agent_master_500k_eval.prepared.jsonl",
            duplicate_cap=args.duplicate_cap,
            max_records=args.max_records_per_file,
        )
    )

    if args.include_usef:
        for filename in DEFAULT_USEF_FILES:
            source = args.usef_root / filename
            if not source.exists():
                continue
            train_parts.append(
                _prepare_part(
                    source,
                    args.output_dir / f"{source.stem}.prepared.jsonl",
                    duplicate_cap=args.duplicate_cap,
                    max_records=args.max_records_per_file,
                )
            )

    train_path = args.output_dir / "train.prepared.jsonl"
    eval_path = args.output_dir / "eval.prepared.jsonl"
    _concat_parts([part["path"] for part in train_parts], train_path)
    _concat_parts([part["path"] for part in eval_parts], eval_path)

    manifest["train_parts"] = train_parts
    manifest["eval_parts"] = eval_parts
    manifest["train_path"] = str(train_path)
    manifest["eval_path"] = str(eval_path)
    manifest["train_records"] = sum(part["stats"].get("written", 0) for part in train_parts)
    manifest["eval_records"] = sum(part["stats"].get("written", 0) for part in eval_parts)
    manifest["train_approx_tokens"] = sum(
        part["stats"].get("approx_tokens_chars_div_4", 0) for part in train_parts
    )
    manifest["eval_approx_tokens"] = sum(
        part["stats"].get("approx_tokens_chars_div_4", 0) for part in eval_parts
    )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def _prepare_part(
    source: Path,
    output: Path,
    *,
    duplicate_cap: int,
    max_records: int | None,
) -> dict:
    stats = prepare_jsonl_dataset(
        source,
        output,
        duplicate_cap=duplicate_cap,
        max_records=max_records,
        sanitize=True,
    )
    return {
        "source": str(source),
        "path": str(output),
        "stats": stats,
    }


def _concat_parts(parts: list[str], output: Path) -> None:
    with output.open("wb") as destination:
        for part in parts:
            with Path(part).open("rb") as source:
                shutil.copyfileobj(source, destination)


if __name__ == "__main__":
    main()
