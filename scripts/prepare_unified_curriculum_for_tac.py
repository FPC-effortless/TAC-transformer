from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STREAM_FILES = {
    "cpt": "unified_cpt.jsonl",
    "sft": "unified_sft_messages.jsonl",
    "reasoning": "unified_reasoning_traces.jsonl",
    "preference": "unified_preference_pairs.jsonl",
}

MEDIA_KEYS = {
    "audio",
    "file",
    "image",
    "image_url",
    "input_audio",
    "input_file",
    "input_image",
    "mime_type",
    "url",
    "video",
}

MEDIA_TYPES = {
    "audio",
    "file",
    "image",
    "image_url",
    "input_audio",
    "input_file",
    "input_image",
    "video",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = str(item.get("type", "")).strip()
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
                elif "image_url" in item:
                    parts.append(f"<image:{json.dumps(item['image_url'], ensure_ascii=False)}>")
                elif "url" in item:
                    parts.append(f"<image:{item['url']}>")
                elif item_type:
                    parts.append(f"<{item_type}:{json.dumps(item, ensure_ascii=False, sort_keys=True)}>")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def value_has_multimodal(value: Any) -> bool:
    if isinstance(value, dict):
        value_type = str(value.get("type", "")).strip().lower()
        if value_type in MEDIA_TYPES:
            return True
        if any(key in value for key in MEDIA_KEYS):
            return True
        return any(value_has_multimodal(item) for item in value.values())
    if isinstance(value, list):
        return any(value_has_multimodal(item) for item in value)
    return False


def row_has_multimodal(kind: str, row: dict[str, Any]) -> bool:
    if kind == "sft":
        return messages_have_multimodal_content(row.get("messages"))
    if kind == "preference":
        return any(
            messages_have_multimodal_content(row.get(field))
            for field in ("prompt", "chosen", "rejected")
        )
    return False


def messages_have_multimodal_content(messages: Any) -> bool:
    if not isinstance(messages, list):
        return value_has_multimodal(messages)
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            return True
        if value_has_multimodal(content):
            return True
    return False


def messages_to_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return content_to_text(messages)
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "message")).strip() or "message"
        text = content_to_text(message.get("content")).strip()
        if text:
            parts.append(f"<|{role}|>\n{text}")
    return "\n<|end|>\n".join(parts).strip()


def assistant_target_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return content_to_text(messages).strip()
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        text = content_to_text(message.get("content")).strip()
        if text:
            parts.append(text)
    if parts:
        return "\n<|end|>\n".join(parts).strip()
    return messages_to_text(messages).strip()


def source_dataset(row: dict[str, Any]) -> str:
    source = row.get("source")
    if isinstance(source, dict):
        return str(source.get("dataset") or source.get("path") or "unknown")
    return "unknown"


def serialize_row(kind: str, row: dict[str, Any]) -> str:
    if kind == "cpt":
        return str(row.get("text", "")).strip()
    if kind == "sft":
        return messages_to_text(row.get("messages")).strip()
    if kind == "reasoning":
        state = str(row.get("state", "")).strip()
        actions = str(row.get("actions_json", "")).strip()
        next_state = str(row.get("next_state", "")).strip()
        reward = row.get("reward")
        return (
            "<reasoning_trace>\n"
            "<state>\n"
            f"{state}\n"
            "<actions>\n"
            f"{actions}\n"
            "<next_state>\n"
            f"{next_state}\n"
            "<reward>\n"
            f"{reward}\n"
            "</reasoning_trace>"
        ).strip()
    if kind == "preference":
        prompt = messages_to_text(row.get("prompt")).strip()
        chosen = messages_to_text(row.get("chosen")).strip()
        # The rejected answer is kept in the raw preference file. The prepared
        # full-LM view uses only the chosen answer to avoid teaching bad replies.
        return f"{prompt}\n<|end|>\n{chosen}".strip()
    raise ValueError(f"unsupported stream kind: {kind}")


def completion_row(kind: str, row: dict[str, Any]) -> dict[str, Any] | None:
    if kind == "sft":
        messages = row.get("messages")
        if not isinstance(messages, list):
            return None
        assistant_index = None
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, dict) and message.get("role") == "assistant":
                assistant_index = index
                break
        if assistant_index is None:
            return None
        prompt_messages = messages[:assistant_index]
        answer = content_to_text(messages[assistant_index].get("content")).strip()
        prompt = messages_to_text(prompt_messages).strip()
    elif kind == "preference":
        prompt = messages_to_text(row.get("prompt")).strip()
        answer = assistant_target_text(row.get("chosen")).strip()
    else:
        return None
    if not prompt or not answer:
        return None
    dataset = source_dataset(row)
    return {
        "id": row.get("id"),
        "source": row.get("source"),
        "domain": f"{kind}:{dataset}",
        "stream": kind,
        "prompt": prompt,
        "answer": answer,
        "text": f"{prompt}\n<|end|>\n{answer}",
    }


def tac_byte_tokens(text: str) -> int:
    return len(text.encode("utf-8", errors="replace")) + 1


def profile_paths(curriculum_dir: Path, profile: str) -> dict[str, list[tuple[str, Path]]]:
    if profile == "balanced":
        return {
            "train": [
                (kind, curriculum_dir / "splits_balanced" / "train" / filename)
                for kind, filename in STREAM_FILES.items()
            ],
            "validation": [
                (kind, curriculum_dir / "splits" / "validation" / filename)
                for kind, filename in STREAM_FILES.items()
            ],
        }
    if profile == "full":
        return {
            "all": [
                (kind, curriculum_dir / filename)
                for kind, filename in STREAM_FILES.items()
            ],
            "train": [
                (kind, curriculum_dir / "splits" / "train" / filename)
                for kind, filename in STREAM_FILES.items()
            ],
            "validation": [
                (kind, curriculum_dir / "splits" / "validation" / filename)
                for kind, filename in STREAM_FILES.items()
            ],
        }
    raise ValueError(f"unsupported profile: {profile}")


def build_counts(
    curriculum_dir: Path,
    profile: str,
    *,
    exclude_multimodal: bool = False,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema": "tac_prepared_token_count.v1",
        "tokenizer": "tac_byte_utf8_plus_eos",
        "profile": profile,
        "exclude_multimodal": exclude_multimodal,
        "splits": {},
    }
    for split, entries in profile_paths(curriculum_dir, profile).items():
        split_counter: dict[str, Any] = {
            "records": 0,
            "tac_byte_tokens": 0,
            "approx_bpe_tokens_len_div_4": 0,
            "streams": {},
            "datasets": {},
        }
        dataset_tokens: Counter[str] = Counter()
        dataset_records: Counter[str] = Counter()
        for kind, path in entries:
            stream_records = 0
            stream_tokens = 0
            stream_approx = 0
            stream_multimodal_skipped = 0
            for row in read_jsonl(path):
                if exclude_multimodal and row_has_multimodal(kind, row):
                    stream_multimodal_skipped += 1
                    continue
                text = serialize_row(kind, row)
                if not text:
                    continue
                tokens = tac_byte_tokens(text)
                approx = max(1, round(len(text) / 4))
                stream_records += 1
                stream_tokens += tokens
                stream_approx += approx
                dataset = source_dataset(row)
                dataset_records[dataset] += 1
                dataset_tokens[dataset] += tokens
            split_counter["streams"][kind] = {
                "path": str(path),
                "records": stream_records,
                "tac_byte_tokens": stream_tokens,
                "approx_bpe_tokens_len_div_4": stream_approx,
                "multimodal_skipped": stream_multimodal_skipped,
            }
            split_counter["records"] += stream_records
            split_counter["tac_byte_tokens"] += stream_tokens
            split_counter["approx_bpe_tokens_len_div_4"] += stream_approx
        split_counter["datasets"] = {
            dataset: {
                "records": dataset_records[dataset],
                "tac_byte_tokens": dataset_tokens[dataset],
            }
            for dataset in sorted(dataset_records)
        }
        summary["splits"][split] = split_counter
    if "all" in summary["splits"]:
        total_basis = summary["splits"]["all"]
        summary["total_basis"] = "all"
        summary["total_records"] = total_basis["records"]
        summary["total_tac_byte_tokens"] = total_basis["tac_byte_tokens"]
        summary["total_approx_bpe_tokens_len_div_4"] = total_basis[
            "approx_bpe_tokens_len_div_4"
        ]
    else:
        summary["total_basis"] = "sum_of_splits"
        summary["total_records"] = sum(
            split_summary["records"] for split_summary in summary["splits"].values()
        )
        summary["total_tac_byte_tokens"] = sum(
            split_summary["tac_byte_tokens"]
            for split_summary in summary["splits"].values()
        )
        summary["total_approx_bpe_tokens_len_div_4"] = sum(
            split_summary["approx_bpe_tokens_len_div_4"]
            for split_summary in summary["splits"].values()
        )
    return summary


def write_prepared(
    curriculum_dir: Path,
    profile: str,
    output_dir: Path,
    *,
    exclude_multimodal: bool = False,
    include_splits: set[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {}
    split_name_map = {"validation": "eval", "all": "all", "train": "train"}
    for split, entries in profile_paths(curriculum_dir, profile).items():
        if include_splits is not None and split not in include_splits:
            continue
        output_name = split_name_map.get(split, split)
        output_path = output_dir / f"{output_name}.prepared.jsonl"
        records = 0
        tokens = 0
        kind_counts: Counter[str] = Counter()
        dataset_counts: Counter[str] = Counter()
        multimodal_skipped = 0
        with output_path.open("w", encoding="utf-8") as out:
            for kind, path in entries:
                for row in read_jsonl(path):
                    if exclude_multimodal and row_has_multimodal(kind, row):
                        multimodal_skipped += 1
                        continue
                    text = serialize_row(kind, row)
                    if not text:
                        continue
                    dataset = source_dataset(row)
                    prepared = {
                        "id": row.get("id"),
                        "source": row.get("source"),
                        "domain": f"{kind}:{dataset}",
                        "stream": kind,
                        "text": text,
                    }
                    out.write(json.dumps(prepared, ensure_ascii=False) + "\n")
                    records += 1
                    tokens += tac_byte_tokens(text)
                    kind_counts[kind] += 1
                    dataset_counts[dataset] += 1
        written[output_name] = {
            "path": str(output_path),
            "records": records,
            "tac_byte_tokens": tokens,
            "streams": dict(sorted(kind_counts.items())),
            "datasets": dict(sorted(dataset_counts.items())),
            "multimodal_skipped": multimodal_skipped,
        }
    manifest = {
        "schema": "tac_prepared_unified_curriculum.v1",
        "profile": profile,
        "tokenizer": "tac_byte_utf8_plus_eos",
        "exclude_multimodal": exclude_multimodal,
        "files": written,
        "notes": [
            "Prepared files expose a text field for JsonlTextBatcher and train_best_tac_agentic.py.",
            "Preference pairs are serialized as prompt plus chosen response only; raw rejected answers remain in the source preference JSONL for a future DPO objective.",
            "Structured multimodal parts are represented as text/image placeholders because the current TAC trainer is byte-text only.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_completions(
    curriculum_dir: Path,
    profile: str,
    output_dir: Path,
    *,
    exclude_multimodal: bool = False,
    include_splits: set[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {}
    split_name_map = {"validation": "eval", "all": "all", "train": "train"}
    for split, entries in profile_paths(curriculum_dir, profile).items():
        if include_splits is not None and split not in include_splits:
            continue
        output_name = split_name_map.get(split, split)
        output_path = output_dir / f"{output_name}.completions.jsonl"
        records = 0
        tokens = 0
        kind_counts: Counter[str] = Counter()
        dataset_counts: Counter[str] = Counter()
        multimodal_skipped = 0
        with output_path.open("w", encoding="utf-8") as out:
            for kind, path in entries:
                if kind not in {"sft", "preference"}:
                    continue
                for row in read_jsonl(path):
                    if exclude_multimodal and row_has_multimodal(kind, row):
                        multimodal_skipped += 1
                        continue
                    prepared = completion_row(kind, row)
                    if prepared is None:
                        continue
                    out.write(json.dumps(prepared, ensure_ascii=False) + "\n")
                    records += 1
                    tokens += tac_byte_tokens(str(prepared["text"]))
                    kind_counts[kind] += 1
                    dataset = source_dataset(row)
                    dataset_counts[dataset] += 1
        written[output_name] = {
            "path": str(output_path),
            "records": records,
            "tac_byte_tokens": tokens,
            "streams": dict(sorted(kind_counts.items())),
            "datasets": dict(sorted(dataset_counts.items())),
            "multimodal_skipped": multimodal_skipped,
        }
    manifest = {
        "schema": "tac_completion_unified_curriculum.v1",
        "profile": profile,
        "tokenizer": "tac_byte_utf8_plus_eos",
        "exclude_multimodal": exclude_multimodal,
        "files": written,
        "notes": [
            "Completion files expose prompt and answer fields for --supervision-mode answer_only.",
            "Answer fields contain only assistant target content; role markers remain in prompts only.",
            "Only SFT and preference-chosen rows are included; CPT and reasoning traces remain full-LM objectives.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# TAC Curriculum Token Count",
        "",
        f"Profile: `{summary['profile']}`",
        "Tokenizer: `tac_byte_utf8_plus_eos`",
        "",
        f"Total records: `{summary['total_records']}`",
        f"Total TAC byte tokens: `{summary['total_tac_byte_tokens']}`",
        f"Approx BPE tokens (len/4): `{summary['total_approx_bpe_tokens_len_div_4']}`",
        "",
        "## Splits",
    ]
    for split, split_summary in summary["splits"].items():
        lines.extend(
            [
                "",
                f"### {split}",
                f"- Records: `{split_summary['records']}`",
                f"- TAC byte tokens: `{split_summary['tac_byte_tokens']}`",
                f"- Approx BPE tokens: `{split_summary['approx_bpe_tokens_len_div_4']}`",
                "",
                "| Stream | Records | TAC byte tokens | Approx BPE tokens |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for stream, stats in split_summary["streams"].items():
            lines.append(
                f"| `{stream}` | {stats['records']} | {stats['tac_byte_tokens']} | {stats['approx_bpe_tokens_len_div_4']} |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count or prepare the unified TAC curriculum for the byte-level TAC trainers."
    )
    parser.add_argument(
        "--curriculum-dir",
        type=Path,
        default=Path("Training data") / "unified_training_curriculum",
    )
    parser.add_argument("--profile", choices=["balanced", "full"], default="balanced")
    parser.add_argument("--count-json", type=Path, default=None)
    parser.add_argument("--count-md", type=Path, default=None)
    parser.add_argument("--write-prepared", action="store_true")
    parser.add_argument(
        "--write-completions",
        action="store_true",
        help="Write prompt/answer JSONL for answer-only training from SFT and preference rows.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["all", "train", "validation"],
        default=None,
        help="Limit written output splits. Useful with --profile full to avoid all/train duplication.",
    )
    parser.add_argument(
        "--exclude-multimodal",
        action="store_true",
        help="Skip structured multimodal SFT/preference rows when counting or writing TAC text JSONL.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curriculum_dir = args.curriculum_dir.resolve()
    summary = build_counts(
        curriculum_dir,
        args.profile,
        exclude_multimodal=args.exclude_multimodal,
    )
    if args.count_json is not None:
        args.count_json.parent.mkdir(parents=True, exist_ok=True)
        args.count_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.count_md is not None:
        args.count_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown_report(summary, args.count_md)
    result: dict[str, Any] = {"token_count": summary}
    include_splits = set(args.splits) if args.splits is not None else None
    if args.write_prepared:
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = curriculum_dir / f"tac_prepared_{args.profile}"
        result["prepared"] = write_prepared(
            curriculum_dir,
            args.profile,
            output_dir,
            exclude_multimodal=args.exclude_multimodal,
            include_splits=include_splits,
        )
    if args.write_completions:
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = curriculum_dir / f"tac_completions_{args.profile}"
        result["completions"] = write_completions(
            curriculum_dir,
            args.profile,
            output_dir,
            exclude_multimodal=args.exclude_multimodal,
            include_splits=include_splits,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
