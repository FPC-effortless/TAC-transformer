from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import requests

try:
    from scripts.build_private_reasoning_final_answer_dataset import byte_len, reject_reasons
except ModuleNotFoundError:  # pragma: no cover - direct script execution path.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts.build_private_reasoning_final_answer_dataset import byte_len, reject_reasons


DATASET_SERVER = "https://datasets-server.huggingface.co"
PAGE_SIZE = 100
EOS_TOKEN_ID = 3


@dataclass(frozen=True)
class RawTextSource:
    key: str
    dataset: str
    config: str
    split: str
    text_field: str
    domain: str
    stream: str
    license: str
    train_target_bytes_arg: str
    eval_target_bytes_arg: str
    train_max_pages_arg: str
    eval_max_pages_arg: str
    eval_start_offset: int


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def clean_text(text: Any) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\x00", " ")
    value = value.replace("<|endoftext|>", " ")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def safe_text(text: str) -> bool:
    text = clean_text(text)
    if byte_len(text) < 128:
        return False
    if len(text.split()) < 25:
        return False
    if reject_reasons(text):
        return False
    return True


def byte_chunks(text: str, *, chunk_bytes: int, min_bytes: int) -> list[str]:
    text = clean_text(text)
    chunks: list[str] = []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    current = ""
    for paragraph in paragraphs or [text]:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if byte_len(candidate) <= chunk_bytes:
            current = candidate
            continue
        if byte_len(current) >= min_bytes:
            chunks.append(current)
        current = ""
        words = paragraph.split()
        buf: list[str] = []
        for word in words:
            candidate_word = " ".join([*buf, word])
            if byte_len(candidate_word) > chunk_bytes:
                piece = " ".join(buf).strip()
                if byte_len(piece) >= min_bytes:
                    chunks.append(piece)
                buf = [word]
            else:
                buf.append(word)
        current = " ".join(buf).strip()
    if byte_len(current) >= min_bytes:
        chunks.append(current)
    return chunks


def hf_rows(
    dataset: str,
    config: str,
    split: str,
    *,
    start_offset: int,
    max_pages: int,
    timeout: int = 60,
):
    session = requests.Session()
    headers = {"User-Agent": "tac-20m-pretrain-builder/1.0"}
    delay = float(os.environ.get("TAC_HF_REQUEST_DELAY_SECONDS", "0.6"))
    max_attempts = int(os.environ.get("TAC_HF_MAX_ATTEMPTS", "20"))
    offset = start_offset
    for _ in range(max_pages):
        params = {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": PAGE_SIZE,
        }
        url = f"{DATASET_SERVER}/rows?{urlencode(params)}"
        last_error: Exception | None = None
        response: requests.Response | None = None
        for attempt in range(max_attempts):
            try:
                response = session.get(url, headers=headers, timeout=timeout)
                if response.status_code == 429:
                    last_error = RuntimeError(f"HTTP 429 for {url}")
                    time.sleep(min(120.0, max(20.0, 6.0 * (attempt + 1))))
                    continue
                if response.status_code in {500, 502, 503, 504}:
                    last_error = RuntimeError(f"HTTP {response.status_code} for {url}")
                    time.sleep(min(60.0, 2**attempt))
                    continue
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(min(60.0, 2**attempt))
        else:
            if last_error is not None:
                raise last_error
            raise RuntimeError(f"failed to fetch {url}")
        if response is None:
            raise RuntimeError(f"failed to fetch {url}")
        payload = response.json()
        rows = payload.get("rows") or []
        if not rows:
            return
        for wrapped in rows:
            row = dict(wrapped.get("row") or {})
            row["__row_idx"] = wrapped.get("row_idx")
            yield row
        if len(rows) < PAGE_SIZE:
            return
        if delay > 0:
            time.sleep(delay)
        offset += PAGE_SIZE


def raw_sources() -> list[RawTextSource]:
    return [
        RawTextSource(
            key="fineweb_edu_100bt",
            dataset="HuggingFaceFW/fineweb_edu_100BT-shuffled",
            config="default",
            split="train",
            text_field="text",
            domain="pretrain_english:fineweb_edu_100bt",
            stream="pretrain_english",
            license="odc-by",
            train_target_bytes_arg="fineweb_train_bytes",
            eval_target_bytes_arg="fineweb_eval_bytes",
            train_max_pages_arg="fineweb_train_max_pages",
            eval_max_pages_arg="fineweb_eval_max_pages",
            eval_start_offset=120000,
        ),
        RawTextSource(
            key="cosmopedia_web",
            dataset="HuggingFaceTB/cosmopedia",
            config="web_samples_v1",
            split="train",
            text_field="text",
            domain="pretrain_textbook:cosmopedia_web_samples",
            stream="pretrain_textbook",
            license="apache-2.0",
            train_target_bytes_arg="cosmopedia_train_bytes",
            eval_target_bytes_arg="cosmopedia_eval_bytes",
            train_max_pages_arg="cosmopedia_train_max_pages",
            eval_max_pages_arg="cosmopedia_eval_max_pages",
            eval_start_offset=120000,
        ),
    ]


def write_record(handle, row: dict[str, Any]) -> int:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return byte_len(str(row.get("text", "")))


def write_raw_source(
    handle,
    source: RawTextSource,
    *,
    split_kind: str,
    target_bytes: int,
    max_pages: int,
    chunk_bytes: int,
    min_record_bytes: int,
    seen_hashes: set[str],
) -> dict[str, Any]:
    start_offset = 0 if split_kind == "train" else source.eval_start_offset
    stats = {
        "dataset": source.dataset,
        "config": source.config,
        "split": source.split,
        "start_offset": start_offset,
        "target_bytes": target_bytes,
        "max_pages": max_pages,
        "read_rows": 0,
        "written_records": 0,
        "written_bytes": 0,
        "skipped": Counter(),
    }
    if target_bytes <= 0:
        stats["skipped"]["target_zero"] += 1
        stats["skipped"] = dict(stats["skipped"])
        return stats
    for raw in hf_rows(
        source.dataset,
        source.config,
        source.split,
        start_offset=start_offset,
        max_pages=max_pages,
    ):
        stats["read_rows"] += 1
        source_text = clean_text(raw.get(source.text_field))
        if not source_text:
            stats["skipped"]["empty_text"] += 1
            continue
        for chunk_index, chunk in enumerate(
            byte_chunks(source_text, chunk_bytes=chunk_bytes, min_bytes=min_record_bytes)
        ):
            if not safe_text(chunk):
                stats["skipped"]["unsafe_or_too_short"] += 1
                continue
            text_hash = stable_hash(chunk)
            if text_hash in seen_hashes:
                stats["skipped"]["duplicate_text"] += 1
                continue
            seen_hashes.add(text_hash)
            row = {
                "id": f"{source.key}_{split_kind}_{raw.get('__row_idx')}_{chunk_index}",
                "source": {
                    "dataset": source.dataset,
                    "config": source.config,
                    "split": source.split,
                    "row_idx": raw.get("__row_idx"),
                    "license": source.license,
                },
                "domain": source.domain,
                "stream": source.stream,
                "text": chunk,
            }
            written = write_record(handle, row)
            stats["written_records"] += 1
            stats["written_bytes"] += written
            if stats["written_bytes"] >= target_bytes:
                stats["skipped"] = dict(stats["skipped"])
                return stats
    stats["skipped"] = dict(stats["skipped"])
    return stats


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_local_seed(
    handle,
    path: Path,
    *,
    split_kind: str,
    target_bytes: int,
    seen_hashes: set[str],
    rng: random.Random,
) -> dict[str, Any]:
    stats = {
        "path": str(path),
        "target_bytes": target_bytes,
        "read_rows": 0,
        "written_records": 0,
        "written_bytes": 0,
        "skipped": Counter(),
    }
    if target_bytes <= 0:
        stats["skipped"]["target_zero"] += 1
        stats["skipped"] = dict(stats["skipped"])
        return stats
    if not path.exists():
        stats["skipped"]["missing_path"] += 1
        stats["skipped"] = dict(stats["skipped"])
        return stats
    rows = list(iter_jsonl(path))
    rng.shuffle(rows)
    for raw in rows:
        stats["read_rows"] += 1
        text = clean_text(raw.get("text"))
        if not text or not safe_text(text):
            stats["skipped"]["unsafe_or_empty"] += 1
            continue
        text_hash = stable_hash(text)
        if text_hash in seen_hashes:
            stats["skipped"]["duplicate_text"] += 1
            continue
        seen_hashes.add(text_hash)
        stream = str(raw.get("stream") or "instruction_seed")
        row = {
            "id": f"local_seed_{split_kind}_{raw.get('id') or stats['read_rows']}",
            "source": {
                "dataset": "local_capability_balanced_external_seq512",
                "path": str(path),
                "original_source": raw.get("source"),
            },
            "domain": f"pretrain_seed:{stream}",
            "stream": "pretrain_seed",
            "text": text,
        }
        written = write_record(handle, row)
        stats["written_records"] += 1
        stats["written_bytes"] += written
        if stats["written_bytes"] >= target_bytes:
            break
    stats["skipped"] = dict(stats["skipped"])
    return stats


def build_jsonl_split(
    output_path: Path,
    *,
    split_kind: str,
    args: argparse.Namespace,
    seen_hashes: set[str],
    rng: random.Random,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "path": str(output_path),
        "split": split_kind,
        "sources": {},
        "local_seed": None,
        "records": 0,
        "bytes": 0,
        "streams": Counter(),
        "domains": Counter(),
        "max_record_bytes": 0,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        for source in raw_sources():
            target = int(
                getattr(
                    args,
                    source.train_target_bytes_arg if split_kind == "train" else source.eval_target_bytes_arg,
                )
            )
            max_pages = int(
                getattr(
                    args,
                    source.train_max_pages_arg if split_kind == "train" else source.eval_max_pages_arg,
                )
            )
            source_stats = write_raw_source(
                handle,
                source,
                split_kind=split_kind,
                target_bytes=target,
                max_pages=max_pages,
                chunk_bytes=args.chunk_bytes,
                min_record_bytes=args.min_record_bytes,
                seen_hashes=seen_hashes,
            )
            stats["sources"][source.key] = source_stats
            print(
                f"[{split_kind}] {source.key}: wrote {source_stats['written_records']} "
                f"records / {source_stats['written_bytes']} bytes",
                file=sys.stderr,
                flush=True,
            )
        local_target = args.local_seed_train_bytes if split_kind == "train" else args.local_seed_eval_bytes
        local_path = args.local_seed_train_jsonl if split_kind == "train" else args.local_seed_eval_jsonl
        stats["local_seed"] = write_local_seed(
            handle,
            local_path,
            split_kind=split_kind,
            target_bytes=local_target,
            seen_hashes=seen_hashes,
            rng=rng,
        )
        print(
            f"[{split_kind}] local_seed: wrote {stats['local_seed']['written_records']} "
            f"records / {stats['local_seed']['written_bytes']} bytes",
            file=sys.stderr,
            flush=True,
        )

    for row in iter_jsonl(output_path):
        text = str(row.get("text", ""))
        size = byte_len(text)
        stats["records"] += 1
        stats["bytes"] += size
        stats["max_record_bytes"] = max(stats["max_record_bytes"], size)
        stats["streams"][str(row.get("stream", "unknown"))] += 1
        stats["domains"][str(row.get("domain", "unknown"))] += 1
    stats["streams"] = dict(stats["streams"].most_common())
    stats["domains"] = dict(stats["domains"].most_common())
    return stats


def byte_tokens(text: str) -> list[int]:
    return [byte + 4 for byte in text.encode("utf-8", errors="replace")]


def stream_tokenized_memmap(
    input_path: Path,
    output_dir: Path,
    *,
    vocab_size: int,
    label_field: str = "domain",
    text_field: str = "text",
    dtype: str = "uint16",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = output_dir / f"tokens.{dtype}.bin"
    offsets: list[int] = []
    lengths: list[int] = []
    category_ids: list[int] = []
    category_to_id: dict[str, int] = {}
    records = 0
    token_count = 0
    token_dtype = np.dtype(dtype)
    if vocab_size > np.iinfo(token_dtype).max + 1:
        raise ValueError(f"vocab_size {vocab_size} does not fit dtype {dtype}")
    with input_path.open("r", encoding="utf-8") as src, tokens_path.open("wb") as token_out:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get(text_field, ""))
            if not text:
                continue
            record_tokens = byte_tokens(text) + [EOS_TOKEN_ID]
            if max(record_tokens, default=0) >= vocab_size:
                raise ValueError("vocab_size is too small for TAC byte tokens")
            offsets.append(token_count)
            lengths.append(len(record_tokens))
            np.asarray(record_tokens, dtype=token_dtype).tofile(token_out)
            token_count += len(record_tokens)
            category = str(row.get(label_field, ""))
            category_id = category_to_id.setdefault(category, len(category_to_id)) if category else -1
            category_ids.append(category_id)
            records += 1
    offsets_path = output_dir / "record_offsets.int64.npy"
    lengths_path = output_dir / "record_lengths.int32.npy"
    category_ids_path = output_dir / "category_ids.int16.npy"
    categories_path = output_dir / "categories.json"
    np.save(offsets_path, np.asarray(offsets, dtype=np.int64))
    np.save(lengths_path, np.asarray(lengths, dtype=np.int32))
    np.save(category_ids_path, np.asarray(category_ids, dtype=np.int16))
    categories_path.write_text(
        json.dumps({k: v for k, v in sorted(category_to_id.items(), key=lambda item: item[1])}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "input_path": str(input_path),
        "records": records,
        "tokens": token_count,
        "vocab_size": vocab_size,
        "dtype": dtype,
        "tokenizer": "tac_byte",
        "text_field": text_field,
        "label_field": label_field,
        "eos_token_id": EOS_TOKEN_ID,
        "append_eos": True,
        "tokens_path": str(tokens_path),
        "record_offsets_path": str(offsets_path),
        "record_lengths_path": str(lengths_path),
        "category_ids_path": str(category_ids_path),
        "categories_path": str(categories_path),
        "categories": {k: v for k, v in sorted(category_to_id.items(), key=lambda item: item[1])},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_sampling_weights(path: Path) -> dict[str, float]:
    weights = {
        "*": 1.0,
        "pretrain_english": 1.0,
        "pretrain_textbook": 1.2,
        "pretrain_seed": 2.0,
    }
    path.write_text(json.dumps(weights, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return weights


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a TAC byte-level pretraining corpus sized for a ~20M parameter from-scratch run."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--chunk-bytes", type=positive_int, default=4096)
    parser.add_argument("--min-record-bytes", type=positive_int, default=768)
    parser.add_argument("--dtype", choices=["uint16", "uint32"], default="uint16")
    parser.add_argument("--skip-tokenized", action="store_true")

    parser.add_argument("--fineweb-train-bytes", type=positive_int, default=120_000_000)
    parser.add_argument("--fineweb-eval-bytes", type=positive_int, default=2_000_000)
    parser.add_argument("--cosmopedia-train-bytes", type=positive_int, default=50_000_000)
    parser.add_argument("--cosmopedia-eval-bytes", type=positive_int, default=1_000_000)
    parser.add_argument("--local-seed-train-bytes", type=positive_int, default=20_000_000)
    parser.add_argument("--local-seed-eval-bytes", type=positive_int, default=2_000_000)

    parser.add_argument("--fineweb-train-max-pages", type=positive_int, default=450)
    parser.add_argument("--fineweb-eval-max-pages", type=positive_int, default=20)
    parser.add_argument("--cosmopedia-train-max-pages", type=positive_int, default=180)
    parser.add_argument("--cosmopedia-eval-max-pages", type=positive_int, default=15)
    parser.add_argument(
        "--local-seed-train-jsonl",
        type=Path,
        default=Path("runs/capability_balanced_external_seq512_2026_06_07/train.completions.jsonl"),
    )
    parser.add_argument(
        "--local-seed-eval-jsonl",
        type=Path,
        default=Path("runs/capability_balanced_external_seq512_2026_06_07/eval.completions.jsonl"),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_hashes: set[str] = set()
    train_stats = build_jsonl_split(
        args.output_dir / "train.jsonl",
        split_kind="train",
        args=args,
        seen_hashes=train_hashes,
        rng=rng,
    )
    eval_hashes = set(train_hashes)
    eval_stats = build_jsonl_split(
        args.output_dir / "eval.jsonl",
        split_kind="eval",
        args=args,
        seen_hashes=eval_hashes,
        rng=rng,
    )
    tokenized = None
    if not args.skip_tokenized:
        tokenized = {
            "train_manifest": stream_tokenized_memmap(
                args.output_dir / "train.jsonl",
                args.output_dir / "tokenized" / "train",
                vocab_size=args.vocab_size,
                dtype=args.dtype,
            ),
            "eval_manifest": stream_tokenized_memmap(
                args.output_dir / "eval.jsonl",
                args.output_dir / "tokenized" / "valid",
                vocab_size=args.vocab_size,
                dtype=args.dtype,
            ),
        }
    sampling_weights = write_sampling_weights(args.output_dir / "sampling_weights.json")
    manifest = {
        "schema": "tac_20m_from_scratch_pretrain_dataset.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "model_target": {
            "parameter_count": "approximately 20M",
            "rule_of_thumb": "compute-optimal pretraining is roughly 20 tokens per parameter; this local corpus targets a practical subset plus instruction seed.",
            "ideal_tokens_for_20m": 400_000_000,
        },
        "vocab_size": args.vocab_size,
        "tokenizer": "tac_byte",
        "train": train_stats,
        "eval": eval_stats,
        "tokenized": tokenized,
        "sampling_weights": sampling_weights,
        "sources": [
            {
                "key": source.key,
                "dataset": source.dataset,
                "config": source.config,
                "split": source.split,
                "domain": source.domain,
                "license": source.license,
            }
            for source in raw_sources()
        ],
        "notes": [
            "This is full-LM pretraining data, not answer-only SFT data.",
            "Use --supervision-mode full_lm when training from scratch.",
            "Raw public text is chunked into bounded records for seq_len 512 sampling.",
            "The local seed stream reuses validated assistant/reasoning/tool/ATS text to expose behavior after raw LM pretraining.",
            "A larger 400M-token build is preferable when disk/time budget allows.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
