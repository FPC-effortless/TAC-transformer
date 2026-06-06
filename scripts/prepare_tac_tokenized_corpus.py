from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer.training import build_tokenized_memmap_from_jsonl


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build TAC-native tokenized memmap files from prepared JSONL splits. "
            "This preserves the byte-level TAC tokenizer instead of forcing GPT-2 BPE."
        )
    )
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--valid-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("tokenized"))
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--label-field", default="domain")
    parser.add_argument("--dtype", choices=["uint16", "uint32"], default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = args.output_dir / "train"
    valid_dir = args.output_dir / "valid"
    train_manifest = build_tokenized_memmap_from_jsonl(
        args.train_jsonl,
        train_dir,
        vocab_size=args.vocab_size,
        text_field=args.text_field,
        label_field=args.label_field,
        dtype=args.dtype,
    )
    valid_manifest = build_tokenized_memmap_from_jsonl(
        args.valid_jsonl,
        valid_dir,
        vocab_size=args.vocab_size,
        text_field=args.text_field,
        label_field=args.label_field,
        dtype=args.dtype,
    )
    manifest = {
        "schema": "tac_tokenized_corpus.v1",
        "tokenizer": "tac_byte",
        "vocab_size": args.vocab_size,
        "train_manifest": str(train_dir / "manifest.json"),
        "valid_manifest": str(valid_dir / "manifest.json"),
        "train_records": int(train_manifest["records"]),
        "valid_records": int(valid_manifest["records"]),
        "train_tokens": int(train_manifest["tokens"]),
        "valid_tokens": int(valid_manifest["tokens"]),
        "notes": (
            "TAC currently trains on UTF-8 byte tokens offset by 4, with EOS token 3. "
            "These memmaps remove JSON parsing from the hot training path."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
