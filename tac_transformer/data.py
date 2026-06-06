from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


MASTER_500K_TRAIN = "master_500k/master_train.jsonl"
MASTER_500K_EVAL = "master_500k/master_eval.jsonl"


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        yield from _iter_jsonl(path)
        return
    yield from _iter_json(path)


def serialize_record(row: dict[str, Any]) -> str:
    if _looks_like_agent_record(row):
        return serialize_agent_record(row)
    return serialize_generic_record(row)


def serialize_agent_record(row: dict[str, Any]) -> str:
    parts = [
        f"<record id=\"{_clean(row.get('record_id', ''))}\" source=\"{_clean(row.get('source', ''))}\" domain=\"{_clean(row.get('domain', ''))}\" success=\"{row.get('success', '')}\">",
        f"<prompt>\n{_clean(row.get('prompt', ''))}",
        f"<edge_cases>\n{_json(row.get('edge_cases', []))}",
        f"<retrieved_docs>\n{_json(row.get('retrieved_docs', []))}",
        f"<plan>\n{_json(row.get('plan', []))}",
        f"<tool_results>\n{_json(row.get('tool_results', []))}",
        f"<target_plan>\n{_clean(row.get('target_plan', ''))}",
        f"<final_answer>\n{_clean(row.get('final_answer', ''))}",
        f"<coherence_violations>\n{_json(row.get('coherence_violations', []))}",
        f"<energy_spent>\n{row.get('energy_spent', '')}",
        "</record>",
    ]
    return "\n".join(parts)


def serialize_generic_record(row: dict[str, Any]) -> str:
    label = row.get("layer") or row.get("purpose") or row.get("category") or row.get("algorithm") or "generic"
    return "\n".join(
        [
            f"<record type=\"{_clean(str(label))}\">",
            _json(row),
            "</record>",
        ]
    )


def prepare_jsonl_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    duplicate_cap: int = 3,
    skip_missing_final_answer: bool = True,
    sanitize: bool = True,
    max_records: Optional[int] = None,
) -> dict[str, int]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duplicate_counts: Counter[tuple[str, str, str]] = Counter()
    stats = Counter()

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in iter_records(input_path):
            stats["read"] += 1
            if skip_missing_final_answer and _looks_like_agent_record(row) and not row.get("final_answer"):
                stats["missing_final_answer"] += 1
                continue

            text = serialize_record(row)
            if sanitize:
                sanitized_text = sanitize_training_text(text)
                if sanitized_text != text:
                    stats["sanitized_records"] += 1
                text = sanitized_text
            if _looks_like_agent_record(row):
                key = (
                    "agent",
                    str(row.get("prompt", "")),
                    str(row.get("final_answer", "")),
                    str(row.get("target_plan", "")),
                )
            else:
                key = ("generic", text)
            duplicate_counts[key] += 1
            if duplicate_counts[key] > duplicate_cap:
                stats["duplicate_capped"] += 1
                continue

            output.write(
                json.dumps(
                    {
                        "record_id": row.get("record_id", f"record_{stats['read']}"),
                        "source": row.get("source", "unknown"),
                        "domain": row.get("domain") or row.get("purpose") or row.get("category") or row.get("algorithm") or "unknown",
                        "text": text,
                    },
                    ensure_ascii=False,
                )
            )
            output.write("\n")
            stats["written"] += 1
            stats["serialized_chars"] += len(text)
            if max_records is not None and stats["written"] >= max_records:
                break

    stats["approx_tokens_chars_div_4"] = round(stats["serialized_chars"] / 4)
    return dict(stats)


def normalize_template_text(text: str) -> str:
    """Remove superficial ids/numbers while preserving structural task shape."""

    normalized = re.sub(
        r'(<record id=")[^"]+(")',
        r"\1<ID>\2",
        text,
    )
    normalized = re.sub(r"synthetic_curriculum_\d+_traj_\d+", "synthetic_curriculum_ID", normalized)
    normalized = re.sub(r"\bkw_\d+\b", "kw_ID", normalized)
    normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "<NUM>", normalized)
    normalized = re.sub(r"\bteam-\d+\b", "team-<NUM>", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def stable_text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def dedupe_prepared_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    exact_cap: int = 1,
    template_cap: int = 3,
    max_records: Optional[int] = None,
) -> dict[str, int]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exact_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    stats = Counter()

    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            stats["read"] += 1
            row = json.loads(line)
            text = str(row.get("text", ""))
            exact_hash = stable_text_hash(text)
            template_hash = stable_text_hash(normalize_template_text(text))
            exact_counts[exact_hash] += 1
            template_counts[template_hash] += 1
            if exact_counts[exact_hash] > exact_cap:
                stats["exact_deduped"] += 1
                continue
            if template_counts[template_hash] > template_cap:
                stats["template_capped"] += 1
                continue
            destination.write(json.dumps(row, ensure_ascii=False))
            destination.write("\n")
            stats["written"] += 1
            stats["serialized_chars"] += len(text)
            if max_records is not None and stats["written"] >= max_records:
                break

    stats["approx_tokens_chars_div_4"] = round(stats["serialized_chars"] / 4)
    stats["exact_unique_texts"] = len(exact_counts)
    stats["normalized_template_families"] = len(template_counts)
    return dict(stats)


def default_master_500k_paths(root: str | Path) -> tuple[Path, Path]:
    root = Path(root)
    return root / MASTER_500K_TRAIN, root / MASTER_500K_EVAL


def sanitize_training_text(text: str) -> str:
    replacements = [
        (r"sk_live_[A-Za-z0-9_\\-]+", "<API_KEY>"),
        (r"sk-[A-Za-z0-9_\\-]{12,}", "<API_KEY>"),
        (r'(?i)(password"?\s*[:=]\s*"?)[^"\s,;}]+', r"\1<PASSWORD>"),
        (r'(?i)(token"?\s*[:=]\s*"?)[^"<\s,}]+', r"\1<TOKEN>"),
        (r'(?i)(api[_-]?key"?\s*[:=]\s*"?)[^"\s,}]+', r"\1<API_KEY>"),
    ]
    sanitized = text
    for pattern, replacement in replacements:
        sanitized = re.sub(pattern, replacement, sanitized)
    return sanitized


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare TAC training JSONL from agent/procedural datasets.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duplicate-cap", type=int, default=3)
    parser.add_argument("--keep-missing-final-answer", action="store_true")
    parser.add_argument("--no-sanitize", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args(argv)

    stats = prepare_jsonl_dataset(
        args.input,
        args.output,
        duplicate_cap=args.duplicate_cap,
        skip_missing_final_answer=not args.keep_missing_final_answer,
        sanitize=not args.no_sanitize,
        max_records=args.max_records,
    )
    print(json.dumps(stats, indent=2))


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def _iter_json(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        first = _first_non_ws(handle)
    if first == "[":
        yield from _iter_json_array(path)
        return

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        yielded = False
        for key in ("records", "examples", "items", "transactions", "evolution_events_logged"):
            value = data.get(key)
            if isinstance(value, list):
                yielded = True
                for row in value:
                    if isinstance(row, dict):
                        yield row
        if not yielded:
            yield data


def _iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    finished = False

    with path.open("r", encoding="utf-8") as handle:
        while not finished:
            chunk = handle.read(chunk_size)
            if not chunk and not buffer.strip():
                break
            buffer += chunk

            while True:
                buffer = buffer.lstrip()
                if not started:
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise ValueError(f"{path} is not a JSON array")
                    buffer = buffer[1:]
                    started = True
                    continue

                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                if buffer[0] == "]":
                    finished = True
                    buffer = buffer[1:]
                    break

                try:
                    row, index = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break

                buffer = buffer[index:]
                if isinstance(row, dict):
                    yield row


def _first_non_ws(handle) -> str:
    while True:
        char = handle.read(1)
        if not char:
            return ""
        if not char.isspace():
            return char


def _looks_like_agent_record(row: dict[str, Any]) -> bool:
    return "prompt" in row and ("final_answer" in row or "target_plan" in row)


def _clean(value: Any) -> str:
    return str(value).replace("\r\n", "\n").strip()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    main()
