from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


STRICT_REJECT_PATTERNS = {
    "secret_openai_key_strict": re.compile(
        r"\bsk-(?:proj-[A-Za-z0-9_\-]{40,}|[A-Za-z0-9]{48,})\b",
        re.IGNORECASE,
    ),
    "secret_hf_or_github": re.compile(
        r"\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|"
        r"AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{35})\b",
        re.IGNORECASE,
    ),
    "pii_email": re.compile(
        r"(?<![@\w.\-])[A-Za-z0-9._%+\-]{1,64}@"
        r"[A-Za-z0-9.\-]{1,253}\.[A-Za-z]{2,}\b",
        re.IGNORECASE,
    ),
    "prompt_injection": re.compile(
        r"\b(?:ignore|disregard|forget)\s+(?:all\s+)?"
        r"(?:previous|prior|above|system|developer)\b|"
        r"\b(?:override\s+(?:system|developer)|reveal\s+system\s+prompt|"
        r"print\s+system\s+prompt|developer\s+mode|jailbreak|"
        r"do\s+anything\s+now|dan\s+mode|bypass\s+(?:safety|guardrail)|"
        r"hidden\s+instructions|prompt\s+injection)\b",
        re.IGNORECASE,
    ),
    "cyber_abuse": re.compile(
        r"\b(?:malware|ransomware|keylogger|reverse\s+shell|metasploit|"
        r"sql\s+injection|xss|csrf|credential\s+stuffing|phishing|"
        r"exfiltrat(?:e|ion)|exploit|CVE-[0-9]|shellcode|"
        r"bypass\s+authentication|password\s+cracking|hashcat|nmap|botnet)\b",
        re.IGNORECASE,
    ),
    "physical_harm": re.compile(
        r"\b(?:make\s+a\s+bomb|build\s+a\s+bomb|explosive(?:s)?|"
        r"poison(?:ing|ed)?|ricin|napalm|firearm(?:s)?|ghost\s+gun|"
        r"3d\s+printed\s+gun|silencer|detonator(?:s)?)\b",
        re.IGNORECASE,
    ),
    "self_harm": re.compile(
        r"\b(?:suicide|self-harm|self\s+harm|kill\s+myself|end\s+my\s+life)\b",
        re.IGNORECASE,
    ),
    "identity_drift": re.compile(
        r"\b(?:you\s+are\s+now|roleplay|pretend\s+to\s+be|"
        r"as\s+an\s+ai\s+language\s+model)\b",
        re.IGNORECASE,
    ),
    "think_marker": re.compile(r"</?(?:think|thinking|reasoning)>", re.IGNORECASE),
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


def byte_len(text: str) -> int:
    return len(text.encode("utf-8", errors="replace"))


def source_dataset(row: dict[str, Any]) -> str:
    source = row.get("source")
    if isinstance(source, dict):
        return str(source.get("dataset") or source.get("path") or "unknown")
    return "unknown"


def reject_reasons(text: str) -> list[str]:
    return [name for name, pattern in STRICT_REJECT_PATTERNS.items() if pattern.search(text)]


def concise_jackrong_answer(next_state: str) -> str | None:
    text = next_state.strip()
    if not text:
        return None
    first_line = text.splitlines()[0].strip()
    first_line = re.sub(r"\s+", " ", first_line).strip()

    correct = re.search(
        r"(?i)\b(?:the\s+)?correct\s+answer\s+is\s+\*{0,2}([A-D])"
        r"(?:\s*:\s*([^*.\n]+))?",
        first_line,
    )
    if correct:
        letter = correct.group(1).strip()
        value = (correct.group(2) or "").strip()
        return f"{letter}: {value}" if value else letter

    answer = re.search(
        r"(?i)\b(?:answer|final answer)\s*(?:is|:)\s+\*{0,2}(.+?)(?:\*{0,2})?$",
        first_line,
    )
    if answer:
        value = answer.group(1).strip(" .")
        return value[:120] if value else None

    simple = re.match(r"^([-+]?\d+(?:\.\d+)?(?:/\d+)?|[A-D])(?:[.:\s]|$)", first_line)
    if simple:
        return simple.group(1).strip()

    sentence = re.split(r"(?<=[.!?])\s+", first_line)[0].strip(" .")
    if 0 < len(sentence) <= 120:
        return sentence
    return None


def sudoku_move_answer(actions_json: str) -> str | None:
    try:
        action = json.loads(actions_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(action, dict):
        return None
    cell = action.get("cell")
    value = action.get("value")
    if (
        not isinstance(cell, list)
        or len(cell) != 2
        or not all(isinstance(item, int) for item in cell)
        or not isinstance(value, int)
    ):
        return None
    return f"r{cell[0] + 1}c{cell[1] + 1}={value}"


def rationale_summary_answer(next_state: str) -> str | None:
    try:
        state = json.loads(next_state)
    except json.JSONDecodeError:
        return None
    if not isinstance(state, dict):
        return None
    summary = str(state.get("rationale_summary", "")).strip()
    if not summary:
        return None
    return re.sub(r"\s+", " ", summary)


def build_reasoning_completion(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    dataset = source_dataset(row)
    state = str(row.get("state", "")).strip()
    actions_json = str(row.get("actions_json", "")).strip()
    next_state = str(row.get("next_state", "")).strip()
    if not state or not actions_json or not next_state:
        return None

    task_type: str
    prompt: str
    answer: str | None
    if dataset == "enriched_sudoku_dataset":
        answer = sudoku_move_answer(actions_json)
        task_type = "sudoku_next_move"
        prompt = f"Sudoku 0=blank. Return move rNcM=V.\n{state}"
    elif dataset == "Jackrong__Claude-opus-4.6-TraceInversion-9000x":
        answer = concise_jackrong_answer(next_state)
        task_type = "trace_inversion_final_answer"
        prompt = f"Answer the problem. Return only the final answer.\nProblem: {state}"
    else:
        answer = rationale_summary_answer(next_state)
        task_type = "rationale_summary_final_answer"
        prompt = (
            "Read the passage and return one concise rationale summary.\n"
            f"Passage: {state}"
        )

    if not answer:
        return None
    completion = {
        "id": row.get("id"),
        "source": row.get("source"),
        "domain": f"private_reasoning_final_answer:{task_type}:{dataset}",
        "stream": "private_reasoning_final_answer",
        "task_type": task_type,
        "prompt": prompt,
        "answer": answer,
        "text": f"{prompt}\n<|end|>\n{answer}",
    }
    metadata = {
        "id": row.get("id"),
        "domain": completion["domain"],
        "private_reasoning": actions_json,
        "raw_next_state": next_state,
        "reward": row.get("reward"),
    }
    return completion, metadata


def write_split(
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    *,
    max_total_bytes: int,
    redteam_filter: bool,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "input": str(input_path),
        "output": str(output_path),
        "metadata": str(metadata_path),
        "read": 0,
        "written": 0,
        "skipped": Counter(),
        "sources": Counter(),
        "task_types": Counter(),
        "max_total_bytes": 0,
        "max_prompt_bytes": 0,
        "max_answer_bytes": 0,
        "reject_reasons": Counter(),
    }
    with output_path.open("w", encoding="utf-8") as out, metadata_path.open(
        "w",
        encoding="utf-8",
    ) as meta:
        for row in read_jsonl(input_path):
            stats["read"] += 1
            built = build_reasoning_completion(row)
            if built is None:
                stats["skipped"]["no_clean_final_answer"] += 1
                continue
            completion, metadata = built
            prompt = str(completion["prompt"])
            answer = str(completion["answer"])
            total_bytes = byte_len(prompt + answer)
            if total_bytes > max_total_bytes:
                stats["skipped"]["too_long_for_context"] += 1
                continue
            if redteam_filter:
                reasons = reject_reasons(prompt + "\n" + answer)
                if reasons:
                    for reason in reasons:
                        stats["reject_reasons"][reason] += 1
                    stats["skipped"]["redteam_reject"] += 1
                    continue
            out.write(json.dumps(completion, ensure_ascii=False, separators=(",", ":")) + "\n")
            meta.write(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats["written"] += 1
            stats["sources"][source_dataset(row)] += 1
            stats["task_types"][completion["task_type"]] += 1
            stats["max_total_bytes"] = max(stats["max_total_bytes"], total_bytes)
            stats["max_prompt_bytes"] = max(stats["max_prompt_bytes"], byte_len(prompt))
            stats["max_answer_bytes"] = max(stats["max_answer_bytes"], byte_len(answer))
    stats["skipped"] = dict(stats["skipped"])
    stats["sources"] = dict(stats["sources"].most_common())
    stats["task_types"] = dict(stats["task_types"].most_common())
    stats["reject_reasons"] = dict(stats["reject_reasons"].most_common())
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build final-answer-only reasoning supervision with private trace metadata."
    )
    parser.add_argument(
        "--train-reasoning-jsonl",
        type=Path,
        default=Path("Training data/unified_training_curriculum/splits/train/unified_reasoning_traces.jsonl"),
    )
    parser.add_argument(
        "--eval-reasoning-jsonl",
        type=Path,
        default=Path(
            "Training data/unified_training_curriculum/splits/validation/unified_reasoning_traces.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/private_reasoning_final_answer_2026_06_07"),
    )
    parser.add_argument("--max-total-bytes", type=int, default=176)
    parser.add_argument("--no-redteam-filter", action="store_true")
    args = parser.parse_args()

    redteam_filter = not args.no_redteam_filter
    train = write_split(
        args.train_reasoning_jsonl,
        args.output_dir / "train.completions.jsonl",
        args.output_dir / "train.private_reasoning_metadata.jsonl",
        max_total_bytes=args.max_total_bytes,
        redteam_filter=redteam_filter,
    )
    eval_stats = write_split(
        args.eval_reasoning_jsonl,
        args.output_dir / "eval.completions.jsonl",
        args.output_dir / "eval.private_reasoning_metadata.jsonl",
        max_total_bytes=args.max_total_bytes,
        redteam_filter=redteam_filter,
    )
    manifest = {
        "schema": "private_reasoning_final_answer_dataset.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "redteam_filter": redteam_filter,
        "max_total_bytes": args.max_total_bytes,
        "train": train,
        "eval": eval_stats,
        "notes": [
            "Training JSONL contains prompt/answer only for --supervision-mode answer_only.",
            "Private reasoning traces are stored in separate metadata files and are not part of model-visible prompt/answer text.",
            "This is final-answer distillation from reasoning traces, not a trainer-level latent scratchpad objective.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
