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
from typing import Any, Callable
from urllib.parse import urlencode

import requests

try:
    from scripts.build_private_reasoning_final_answer_dataset import byte_len, reject_reasons
except ModuleNotFoundError:  # pragma: no cover - direct script execution path.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts.build_private_reasoning_final_answer_dataset import byte_len, reject_reasons


DATASET_SERVER = "https://datasets-server.huggingface.co"
PAGE_SIZE = 100


@dataclass(frozen=True)
class HfSource:
    key: str
    dataset: str
    config: str
    train_split: str
    eval_split: str | None
    converter: Callable[[dict[str, Any], dict[str, Any], int], tuple[dict[str, Any] | None, str]]
    train_cap_arg: str
    eval_cap_arg: str
    stream: str
    license: str
    eval_start_offset: int = 0
    train_max_pages_arg: str = "default_train_max_pages"
    eval_max_pages_arg: str = "default_eval_max_pages"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def clean_text(text: Any) -> str:
    value = "" if text is None else str(text)
    value = value.replace("<|endoftext|>", " ").replace("<|end|>", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def trim_to_bytes(text: str, max_bytes: int) -> str:
    if byte_len(text) <= max_bytes:
        return text
    words = text.split()
    kept: list[str] = []
    for word in words:
        candidate = " ".join([*kept, word])
        if byte_len(candidate) > max_bytes:
            break
        kept.append(word)
    if kept:
        return " ".join(kept).strip()
    return text.encode("utf-8", errors="replace")[:max_bytes].decode("utf-8", errors="ignore").strip()


def trim_answer_to_budget(text: str, max_bytes: int) -> str | None:
    text = clean_text(text)
    if byte_len(text) <= max_bytes:
        return text
    trimmed = trim_to_bytes(text, max_bytes).strip()
    if byte_len(trimmed) < 40:
        return None
    best_end = -1
    for match in re.finditer(r"[.!?](?:\s|$)", trimmed):
        if match.end() >= 60:
            best_end = match.end()
    if best_end >= 0:
        trimmed = trimmed[:best_end].strip()
    else:
        trimmed = trimmed.rstrip(" ,;:")
        if trimmed and trimmed[-1] not in ".!?":
            trimmed += "."
    return trimmed if byte_len(trimmed) <= max_bytes else trim_to_bytes(trimmed, max_bytes)


def completion_text(prompt: str, answer: str) -> str:
    return f"{prompt}\n<|end|>\n{answer}"


def row_total_bytes(row: dict[str, Any]) -> int:
    return byte_len(str(row.get("prompt", "")) + str(row.get("answer", "")))


def source_meta(
    source: HfSource,
    split: str,
    row_idx: int | None,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = {
        "dataset": source.dataset,
        "config": source.config,
        "split": split,
        "row_idx": row_idx,
        "license": source.license,
    }
    if extra:
        meta.update(extra)
    return meta


def make_completion(
    *,
    source: HfSource,
    split: str,
    row_idx: int | None,
    domain: str,
    stream: str,
    prompt: str,
    answer: str,
    source_extra: dict[str, Any] | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    prompt = clean_text(prompt)
    answer = clean_text(answer)
    row = {
        "id": f"{source.key}_{split}_{row_idx}",
        "source": source_meta(source, split, row_idx, extra=source_extra),
        "domain": domain,
        "stream": stream,
        "prompt": prompt,
        "answer": answer,
        "text": completion_text(prompt, answer),
    }
    if task_type:
        row["task_type"] = task_type
    return row


def validate_completion(row: dict[str, Any], max_total_bytes: int) -> str | None:
    prompt = str(row.get("prompt", "")).strip()
    answer = str(row.get("answer", "")).strip()
    if not prompt:
        return "empty_prompt"
    if not answer:
        return "empty_answer"
    if row_total_bytes(row) > max_total_bytes:
        return "too_long_for_context"
    reasons = reject_reasons(prompt + "\n" + answer)
    if reasons:
        return "redteam_" + reasons[0]
    return None


def message_text(message: dict[str, Any]) -> str:
    return clean_text(message.get("content") if "content" in message else message.get("value"))


def role_of(message: dict[str, Any]) -> str:
    role = str(message.get("role") or message.get("from") or "").strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant"}:
        return "assistant"
    if role == "system":
        return "system"
    return role


def last_user_assistant(messages: Any) -> tuple[str | None, str | None]:
    if not isinstance(messages, list):
        return None, None
    assistant_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        item = messages[index]
        if isinstance(item, dict) and role_of(item) == "assistant" and message_text(item):
            assistant_index = index
            break
    if assistant_index is None:
        return None, None
    user_text: str | None = None
    for index in range(assistant_index - 1, -1, -1):
        item = messages[index]
        if isinstance(item, dict) and role_of(item) == "user" and message_text(item):
            user_text = message_text(item)
            break
    if not user_text:
        return None, None
    return user_text, message_text(messages[assistant_index])


def convert_messages_chat(
    row: dict[str, Any],
    source: HfSource,
    split: str,
    max_total_bytes: int,
    *,
    messages_key: str,
    domain: str,
) -> tuple[dict[str, Any] | None, str]:
    user, assistant = last_user_assistant(row.get(messages_key))
    if not user or not assistant:
        return None, "missing_user_assistant_turn"
    prompt = f"<|user|>\n{trim_to_bytes(user, min(240, max_total_bytes // 2))}"
    answer_budget = max_total_bytes - byte_len(prompt) - 8
    answer = trim_answer_to_budget(assistant, answer_budget)
    if not answer:
        return None, "assistant_answer_too_short_after_trim"
    completion = make_completion(
        source=source,
        split=split,
        row_idx=row.get("__row_idx"),
        domain=domain,
        stream="assistant_qna",
        prompt=prompt,
        answer=answer,
    )
    reason = validate_completion(completion, max_total_bytes)
    return (None, reason) if reason else (completion, "ok")


def convert_ultrachat(
    row: dict[str, Any], source: HfSource, split: str, max_total_bytes: int
) -> tuple[dict[str, Any] | None, str]:
    return convert_messages_chat(
        row,
        source,
        split,
        max_total_bytes,
        messages_key="messages",
        domain="assistant_qna:ultrachat_200k",
    )


def convert_slimorca(
    row: dict[str, Any], source: HfSource, split: str, max_total_bytes: int
) -> tuple[dict[str, Any] | None, str]:
    return convert_messages_chat(
        row,
        source,
        split,
        max_total_bytes,
        messages_key="conversations",
        domain="assistant_qna:slimorca_dedup",
    )


def continuation_from_text(
    row: dict[str, Any],
    source: HfSource,
    split: str,
    max_total_bytes: int,
    *,
    domain: str,
) -> tuple[dict[str, Any] | None, str]:
    text = clean_text(row.get("text"))
    if byte_len(text) < 260:
        return None, "source_text_too_short"
    prompt_budget = min(220, max_total_bytes // 2 - 20)
    answer_budget = min(240, max_total_bytes - prompt_budget - 40)
    prefix = trim_to_bytes(text, prompt_budget)
    suffix_source = text[len(prefix) :].strip()
    if byte_len(suffix_source) < 80:
        return None, "continuation_too_short"
    answer = trim_to_bytes(suffix_source, answer_budget)
    prompt = f"Continue the passage:\n{prefix}"
    completion = make_completion(
        source=source,
        split=split,
        row_idx=row.get("__row_idx"),
        domain=domain,
        stream="english_lm_continuation",
        prompt=prompt,
        answer=answer,
    )
    reason = validate_completion(completion, max_total_bytes)
    return (None, reason) if reason else (completion, "ok")


def convert_fineweb(
    row: dict[str, Any], source: HfSource, split: str, max_total_bytes: int
) -> tuple[dict[str, Any] | None, str]:
    return continuation_from_text(
        row,
        source,
        split,
        max_total_bytes,
        domain="english_lm_continuation:fineweb_edu_100bt",
    )


def convert_cosmopedia(
    row: dict[str, Any], source: HfSource, split: str, max_total_bytes: int
) -> tuple[dict[str, Any] | None, str]:
    return continuation_from_text(
        row,
        source,
        split,
        max_total_bytes,
        domain="english_lm_continuation:cosmopedia_web_samples",
    )


def extract_gsm8k_final(answer: str) -> str | None:
    match = re.search(r"####\s*(.+?)\s*$", answer, re.DOTALL)
    if not match:
        return None
    final = clean_text(match.group(1)).strip(".")
    return final or None


def convert_gsm8k(
    row: dict[str, Any], source: HfSource, split: str, max_total_bytes: int
) -> tuple[dict[str, Any] | None, str]:
    question = clean_text(row.get("question"))
    final = extract_gsm8k_final(str(row.get("answer", "")))
    if not question or not final:
        return None, "missing_question_or_final"
    prompt = f"Solve the math problem. Give only the final answer.\nProblem: {question}"
    completion = make_completion(
        source=source,
        split=split,
        row_idx=row.get("__row_idx"),
        domain="private_reasoning_final_answer:gsm8k_final_answer",
        stream="private_reasoning_final_answer",
        prompt=prompt,
        answer=final,
        task_type="gsm8k_final_answer",
    )
    reason = validate_completion(completion, max_total_bytes)
    return (None, reason) if reason else (completion, "ok")


def first_tagged_block(text: str, tag: str) -> str | None:
    pattern = rf"{tag}:\s*(.*?)(?=\n\s*(?:USER|ASSISTANT|FUNCTION RESPONSE):|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return clean_text(match.group(1))


def extract_tool_names(system_text: str) -> list[str]:
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', system_text)
    unique: list[str] = []
    for name in names:
        if name not in unique:
            unique.append(name)
    return unique[:4]


def convert_glaive(
    row: dict[str, Any], source: HfSource, split: str, max_total_bytes: int
) -> tuple[dict[str, Any] | None, str]:
    chat = str(row.get("chat", ""))
    user = first_tagged_block(chat, "USER")
    assistant = first_tagged_block(chat, "ASSISTANT")
    if not user or not assistant:
        return None, "missing_user_or_assistant"
    tool_names = extract_tool_names(str(row.get("system", "")))
    tool_line = "Available tools: " + (", ".join(tool_names) if tool_names else "none")
    prompt = f"{tool_line}\nUser: {trim_to_bytes(user, 260)}\nNext assistant action or answer."
    completion = make_completion(
        source=source,
        split=split,
        row_idx=row.get("__row_idx"),
        domain="agentic_next_action:glaive_function_calling_v2",
        stream="agentic_next_action",
        prompt=prompt,
        answer=assistant,
        source_extra={"tool_names": tool_names},
        task_type="function_call_or_refusal",
    )
    reason = validate_completion(completion, max_total_bytes)
    return (None, reason) if reason else (completion, "ok")


def hf_rows(
    dataset: str,
    config: str,
    split: str,
    *,
    start_offset: int,
    max_pages: int,
    page_size: int = PAGE_SIZE,
    timeout: int = 60,
):
    session = requests.Session()
    headers = {"User-Agent": "tac-external-dataset-builder/1.0"}
    request_delay = float(os.environ.get("TAC_HF_REQUEST_DELAY_SECONDS", "0.25"))
    offset = start_offset
    for _ in range(max_pages):
        params = {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": page_size,
        }
        url = f"{DATASET_SERVER}/rows?{urlencode(params)}"
        last_error: Exception | None = None
        response: requests.Response | None = None
        for attempt in range(5):
            try:
                response = session.get(url, headers=headers, timeout=timeout)
                if response.status_code == 429:
                    last_error = RuntimeError(f"HTTP {response.status_code} for {url}")
                    time.sleep(max(15.0, 5.0 * (attempt + 1)))
                    continue
                if response.status_code in {500, 502, 503, 504}:
                    last_error = RuntimeError(f"HTTP {response.status_code} for {url}")
                    time.sleep(2**attempt)
                    continue
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(2**attempt)
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
        if len(rows) < page_size:
            return
        if request_delay > 0:
            time.sleep(request_delay)
        offset += page_size


def write_row(
    handle,
    row: dict[str, Any],
    *,
    seen_prompt_answer: set[str],
    exclude_prompt_hashes: set[str],
    exclude_prompt_answer_hashes: set[str],
) -> str:
    prompt = str(row.get("prompt", ""))
    answer = str(row.get("answer", ""))
    key = prompt + "\0" + answer
    prompt_hash = stable_hash(prompt)
    prompt_answer_hash = stable_hash(key)
    if key in seen_prompt_answer:
        return "duplicate_prompt_answer"
    if prompt_hash in exclude_prompt_hashes or prompt_answer_hash in exclude_prompt_answer_hashes:
        return "train_eval_overlap"
    seen_prompt_answer.add(key)
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return "written"


def write_local_rows(
    handle,
    path: Path,
    *,
    max_total_bytes: int,
    seen_prompt_answer: set[str],
    exclude_prompt_hashes: set[str],
    exclude_prompt_answer_hashes: set[str],
    prompt_hashes: set[str],
    prompt_answer_hashes: set[str],
) -> dict[str, Any]:
    stats = {"path": str(path), "read": 0, "written": 0, "skipped": Counter(), "streams": Counter()}
    if not path.exists():
        stats["skipped"]["missing_local_path"] += 1
        stats["skipped"] = dict(stats["skipped"])
        stats["streams"] = dict(stats["streams"])
        return stats
    for row in read_jsonl(path):
        stats["read"] += 1
        reason = validate_completion(row, max_total_bytes)
        if reason:
            stats["skipped"][reason] += 1
            continue
        result = write_row(
            handle,
            row,
            seen_prompt_answer=seen_prompt_answer,
            exclude_prompt_hashes=exclude_prompt_hashes,
            exclude_prompt_answer_hashes=exclude_prompt_answer_hashes,
        )
        if result == "written":
            prompt = str(row.get("prompt", ""))
            answer = str(row.get("answer", ""))
            prompt_hashes.add(stable_hash(prompt))
            prompt_answer_hashes.add(stable_hash(prompt + "\0" + answer))
            stats["written"] += 1
            stats["streams"][str(row.get("stream", "unknown"))] += 1
        else:
            stats["skipped"][result] += 1
    stats["skipped"] = dict(stats["skipped"])
    stats["streams"] = dict(stats["streams"])
    return stats


def process_hf_source(
    handle,
    source: HfSource,
    *,
    split_kind: str,
    cap: int,
    max_pages: int,
    max_total_bytes: int,
    seen_prompt_answer: set[str],
    exclude_prompt_hashes: set[str],
    exclude_prompt_answer_hashes: set[str],
    prompt_hashes: set[str],
    prompt_answer_hashes: set[str],
) -> dict[str, Any]:
    split = source.train_split if split_kind == "train" else (source.eval_split or source.train_split)
    start_offset = 0 if split_kind == "train" or source.eval_split else source.eval_start_offset
    stats = {
        "dataset": source.dataset,
        "config": source.config,
        "split": split,
        "start_offset": start_offset,
        "cap": cap,
        "max_pages": max_pages,
        "read": 0,
        "converted": 0,
        "written": 0,
        "skipped": Counter(),
    }
    if cap <= 0:
        stats["skipped"]["cap_zero"] += 1
        stats["skipped"] = dict(stats["skipped"])
        return stats

    for raw in hf_rows(
        source.dataset,
        source.config,
        split,
        start_offset=start_offset,
        max_pages=max_pages,
    ):
        stats["read"] += 1
        converted, reason = source.converter(raw, source, split, max_total_bytes)
        if converted is None:
            stats["skipped"][reason] += 1
            continue
        stats["converted"] += 1
        result = write_row(
            handle,
            converted,
            seen_prompt_answer=seen_prompt_answer,
            exclude_prompt_hashes=exclude_prompt_hashes,
            exclude_prompt_answer_hashes=exclude_prompt_answer_hashes,
        )
        if result == "written":
            prompt = str(converted.get("prompt", ""))
            answer = str(converted.get("answer", ""))
            prompt_hashes.add(stable_hash(prompt))
            prompt_answer_hashes.add(stable_hash(prompt + "\0" + answer))
            stats["written"] += 1
            if stats["written"] >= cap:
                break
        else:
            stats["skipped"][result] += 1

    stats["skipped"] = dict(stats["skipped"])
    return stats


def build_sources() -> list[HfSource]:
    return [
        HfSource(
            key="ultrachat_200k",
            dataset="HuggingFaceH4/ultrachat_200k",
            config="default",
            train_split="train_sft",
            eval_split="test_sft",
            converter=convert_ultrachat,
            train_cap_arg="ultrachat_train_cap",
            eval_cap_arg="ultrachat_eval_cap",
            stream="assistant_qna",
            license="mit",
            train_max_pages_arg="chat_train_max_pages",
            eval_max_pages_arg="chat_eval_max_pages",
        ),
        HfSource(
            key="slimorca_dedup",
            dataset="Open-Orca/SlimOrca-Dedup",
            config="default",
            train_split="train",
            eval_split=None,
            converter=convert_slimorca,
            train_cap_arg="slimorca_train_cap",
            eval_cap_arg="slimorca_eval_cap",
            stream="assistant_qna",
            license="mit",
            eval_start_offset=300000,
            train_max_pages_arg="chat_train_max_pages",
            eval_max_pages_arg="chat_eval_max_pages",
        ),
        HfSource(
            key="fineweb_edu_100bt",
            dataset="HuggingFaceFW/fineweb_edu_100BT-shuffled",
            config="default",
            train_split="train",
            eval_split=None,
            converter=convert_fineweb,
            train_cap_arg="fineweb_train_cap",
            eval_cap_arg="fineweb_eval_cap",
            stream="english_lm_continuation",
            license="odc-by",
            eval_start_offset=100000,
            train_max_pages_arg="lm_train_max_pages",
            eval_max_pages_arg="lm_eval_max_pages",
        ),
        HfSource(
            key="cosmopedia_web",
            dataset="HuggingFaceTB/cosmopedia",
            config="web_samples_v1",
            train_split="train",
            eval_split=None,
            converter=convert_cosmopedia,
            train_cap_arg="cosmopedia_train_cap",
            eval_cap_arg="cosmopedia_eval_cap",
            stream="english_lm_continuation",
            license="apache-2.0",
            eval_start_offset=100000,
            train_max_pages_arg="lm_train_max_pages",
            eval_max_pages_arg="lm_eval_max_pages",
        ),
        HfSource(
            key="gsm8k",
            dataset="openai/gsm8k",
            config="main",
            train_split="train",
            eval_split="test",
            converter=convert_gsm8k,
            train_cap_arg="gsm8k_train_cap",
            eval_cap_arg="gsm8k_eval_cap",
            stream="private_reasoning_final_answer",
            license="mit",
            train_max_pages_arg="math_train_max_pages",
            eval_max_pages_arg="math_eval_max_pages",
        ),
        HfSource(
            key="glaive_function_calling_v2",
            dataset="glaiveai/glaive-function-calling-v2",
            config="default",
            train_split="train",
            eval_split=None,
            converter=convert_glaive,
            train_cap_arg="glaive_train_cap",
            eval_cap_arg="glaive_eval_cap",
            stream="agentic_next_action",
            license="apache-2.0",
            eval_start_offset=80000,
            train_max_pages_arg="tool_train_max_pages",
            eval_max_pages_arg="tool_eval_max_pages",
        ),
    ]


def build_split(
    output_path: Path,
    *,
    split_kind: str,
    args: argparse.Namespace,
    train_prompt_hashes: set[str] | None = None,
    train_prompt_answer_hashes: set[str] | None = None,
) -> tuple[dict[str, Any], set[str], set[str]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_prompt_answer: set[str] = set()
    prompt_hashes: set[str] = set()
    prompt_answer_hashes: set[str] = set()
    train_prompt_hashes = train_prompt_hashes or set()
    train_prompt_answer_hashes = train_prompt_answer_hashes or set()
    local_path = args.local_train_jsonl if split_kind == "train" else args.local_eval_jsonl

    stats: dict[str, Any] = {
        "path": str(output_path),
        "split": split_kind,
        "max_total_bytes": args.max_total_bytes,
        "local": None,
        "sources": {},
        "written": 0,
        "streams": Counter(),
        "domains": Counter(),
        "max_observed_total_bytes": 0,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        if not args.no_local_base:
            stats["local"] = write_local_rows(
                handle,
                local_path,
                max_total_bytes=args.max_total_bytes,
                seen_prompt_answer=seen_prompt_answer,
                exclude_prompt_hashes=train_prompt_hashes,
                exclude_prompt_answer_hashes=train_prompt_answer_hashes,
                prompt_hashes=prompt_hashes,
                prompt_answer_hashes=prompt_answer_hashes,
            )
            print(
                f"[{split_kind}] local: wrote {stats['local']['written']} from {local_path}",
                file=sys.stderr,
                flush=True,
            )
        for source in build_sources():
            cap = int(getattr(args, source.train_cap_arg if split_kind == "train" else source.eval_cap_arg))
            max_pages = int(
                getattr(
                    args,
                    source.train_max_pages_arg if split_kind == "train" else source.eval_max_pages_arg,
                )
            )
            source_stats = process_hf_source(
                handle,
                source,
                split_kind=split_kind,
                cap=cap,
                max_pages=max_pages,
                max_total_bytes=args.max_total_bytes,
                seen_prompt_answer=seen_prompt_answer,
                exclude_prompt_hashes=train_prompt_hashes,
                exclude_prompt_answer_hashes=train_prompt_answer_hashes,
                prompt_hashes=prompt_hashes,
                prompt_answer_hashes=prompt_answer_hashes,
            )
            stats["sources"][source.key] = source_stats
            print(
                f"[{split_kind}] {source.key}: wrote {source_stats['written']} "
                f"read {source_stats['read']} skipped {source_stats['skipped']}",
                file=sys.stderr,
                flush=True,
            )

    for row in read_jsonl(output_path):
        stats["written"] += 1
        stats["streams"][str(row.get("stream", "unknown"))] += 1
        stats["domains"][str(row.get("domain", "unknown"))] += 1
        stats["max_observed_total_bytes"] = max(stats["max_observed_total_bytes"], row_total_bytes(row))
    stats["streams"] = dict(stats["streams"].most_common())
    stats["domains"] = dict(stats["domains"].most_common())
    return stats, prompt_hashes, prompt_answer_hashes


def write_sampling_weights(path: Path) -> dict[str, float]:
    weights = {
        "*": 1.0,
        "assistant_qna": 3.5,
        "english_lm_continuation": 1.2,
        "agentic_next_action": 4.0,
        "private_reasoning_final_answer": 3.0,
        "ats_transfer_supervised": 2.0,
    }
    path.write_text(json.dumps(weights, indent=2, sort_keys=True), encoding="utf-8")
    return weights


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a bounded external TAC capability dataset from Hugging Face Dataset Viewer rows."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-total-bytes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument(
        "--local-train-jsonl",
        type=Path,
        default=Path("runs/capability_balanced_clean_seq512_2026_06_07/train.completions.jsonl"),
    )
    parser.add_argument(
        "--local-eval-jsonl",
        type=Path,
        default=Path("runs/capability_balanced_clean_seq512_2026_06_07/eval.completions.jsonl"),
    )
    parser.add_argument("--no-local-base", action="store_true")

    parser.add_argument("--ultrachat-train-cap", type=positive_int, default=25000)
    parser.add_argument("--ultrachat-eval-cap", type=positive_int, default=2000)
    parser.add_argument("--slimorca-train-cap", type=positive_int, default=5000)
    parser.add_argument("--slimorca-eval-cap", type=positive_int, default=500)
    parser.add_argument("--fineweb-train-cap", type=positive_int, default=20000)
    parser.add_argument("--fineweb-eval-cap", type=positive_int, default=1000)
    parser.add_argument("--cosmopedia-train-cap", type=positive_int, default=10000)
    parser.add_argument("--cosmopedia-eval-cap", type=positive_int, default=500)
    parser.add_argument("--gsm8k-train-cap", type=positive_int, default=7500)
    parser.add_argument("--gsm8k-eval-cap", type=positive_int, default=1000)
    parser.add_argument("--glaive-train-cap", type=positive_int, default=25000)
    parser.add_argument("--glaive-eval-cap", type=positive_int, default=2000)

    parser.add_argument("--chat-train-max-pages", type=positive_int, default=500)
    parser.add_argument("--chat-eval-max-pages", type=positive_int, default=80)
    parser.add_argument("--lm-train-max-pages", type=positive_int, default=260)
    parser.add_argument("--lm-eval-max-pages", type=positive_int, default=40)
    parser.add_argument("--math-train-max-pages", type=positive_int, default=80)
    parser.add_argument("--math-eval-max-pages", type=positive_int, default=20)
    parser.add_argument("--tool-train-max-pages", type=positive_int, default=350)
    parser.add_argument("--tool-eval-max-pages", type=positive_int, default=80)
    parser.add_argument("--default-train-max-pages", type=positive_int, default=200)
    parser.add_argument("--default-eval-max-pages", type=positive_int, default=50)
    args = parser.parse_args()

    random.seed(args.seed)
    train_stats, train_prompt_hashes, train_prompt_answer_hashes = build_split(
        args.output_dir / "train.completions.jsonl",
        split_kind="train",
        args=args,
    )
    eval_stats, _, _ = build_split(
        args.output_dir / "eval.completions.jsonl",
        split_kind="eval",
        args=args,
        train_prompt_hashes=train_prompt_hashes,
        train_prompt_answer_hashes=train_prompt_answer_hashes,
    )
    sampling_weights = write_sampling_weights(args.output_dir / "sampling_weights.json")
    manifest = {
        "schema": "tac_external_capability_dataset.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "max_total_bytes": args.max_total_bytes,
        "data_access": "Hugging Face Dataset Viewer /rows API; no full dataset download required.",
        "disk_safety": "Rows are capped per source to keep the artifact small on the local machine.",
        "train": train_stats,
        "eval": eval_stats,
        "sampling_weights": sampling_weights,
        "sources": [
            {
                "key": source.key,
                "dataset": source.dataset,
                "config": source.config,
                "train_split": source.train_split,
                "eval_split": source.eval_split,
                "stream": source.stream,
                "license": source.license,
            }
            for source in build_sources()
        ],
        "notes": [
            "Use with --supervision-mode answer_only, --prompt-field prompt, --completion-field answer.",
            "Local TAC capability rows are included by default to preserve ATS/private-reasoning pressure.",
            "GSM8K is converted to final-answer-only targets; visible chain-of-thought is not trained.",
            "Function-calling rows are converted to compact next-action/tool-call targets.",
            "Every written row passes the strict local red-team reject patterns and byte-length gate.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
