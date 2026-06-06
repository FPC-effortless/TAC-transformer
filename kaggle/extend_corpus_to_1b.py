from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.knowledge_work import (
    estimate_tokens,
    generate_knowledge_work_records,
    record_to_jsonl,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extend a prepared TAC corpus to a target token budget.")
    parser.add_argument("--base-dir", type=Path, default=Path("runs/prepared_corpus"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/prepared_corpus_1b"))
    parser.add_argument("--target-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--chunk-token-report", type=int, default=25_000_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = json.loads((args.base_dir / "manifest.json").read_text(encoding="utf-8"))
    base_train = args.base_dir / "train.prepared.jsonl"
    base_eval = args.base_dir / "eval.prepared.jsonl"
    output_train = args.output_dir / "train.prepared.jsonl"
    output_eval = args.output_dir / "eval.prepared.jsonl"
    generated_path = args.output_dir / "knowledge_work_generated.prepared.jsonl"

    shutil.copyfile(base_train, output_train)
    shutil.copyfile(base_eval, output_eval)

    base_tokens = int(base_manifest.get("train_approx_tokens", 0))
    needed_tokens = max(args.target_tokens - base_tokens, 0)
    generated_tokens = 0
    generated_records = 0
    next_report = args.chunk_token_report

    with generated_path.open("w", encoding="utf-8", newline="\n") as generated, output_train.open(
        "a", encoding="utf-8", newline="\n"
    ) as train:
        for record in generate_knowledge_work_records(seed=args.seed):
            if generated_tokens >= needed_tokens:
                break
            line = record_to_jsonl(record)
            generated.write(line)
            generated.write("\n")
            train.write(line)
            train.write("\n")
            generated_tokens += estimate_tokens(record.text)
            generated_records += 1
            if args.chunk_token_report and generated_tokens >= next_report:
                print(json.dumps({"generated_records": generated_records, "generated_tokens": generated_tokens}))
                next_report += args.chunk_token_report

    manifest = {
        "base_manifest": str(args.base_dir / "manifest.json"),
        "target_tokens": args.target_tokens,
        "base_train_tokens": base_tokens,
        "generated_tokens": generated_tokens,
        "train_approx_tokens": base_tokens + generated_tokens,
        "generated_records": generated_records,
        "train_path": str(output_train),
        "eval_path": str(output_eval),
        "generated_path": str(generated_path),
        "purpose": [
            "rag_multi_hop",
            "agentic_tool_use",
            "knowledge_work_synthesis",
            "coding_testing",
            "spreadsheet_analysis",
            "research_brief",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
