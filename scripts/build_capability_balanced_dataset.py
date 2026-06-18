from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.build_private_reasoning_final_answer_dataset import byte_len, reject_reasons
except ModuleNotFoundError:  # pragma: no cover - direct script execution path.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts.build_private_reasoning_final_answer_dataset import byte_len, reject_reasons


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


def row_total_bytes(row: dict[str, Any]) -> int:
    return byte_len(str(row.get("prompt", "")) + str(row.get("answer", "")))


def fits_and_clean(row: dict[str, Any], max_total_bytes: int) -> bool:
    prompt = str(row.get("prompt", ""))
    answer = str(row.get("answer", ""))
    if not prompt.strip() or not answer.strip():
        return False
    if row_total_bytes(row) > max_total_bytes:
        return False
    return not reject_reasons(prompt + "\n" + answer)


def normalize_completion(
    row: dict[str, Any],
    *,
    family: str,
    index: int,
    domain_prefix: str | None = None,
) -> dict[str, Any]:
    prompt = str(row.get("prompt", "")).strip()
    answer = str(row.get("answer", "")).strip()
    original_domain = str(row.get("domain") or row.get("stream") or family)
    domain = f"{domain_prefix}:{original_domain}" if domain_prefix else original_domain
    normalized = {
        "id": f"{family}_{row.get('id') or row.get('record_id') or index}",
        "source": row.get("source"),
        "domain": domain,
        "stream": family,
        "prompt": prompt,
        "answer": answer,
        "text": f"{prompt}\n<|end|>\n{answer}",
    }
    if "task_type" in row:
        normalized["task_type"] = row["task_type"]
    return normalized


def sample_rows(rows: list[dict[str, Any]], cap: int | None, rng: random.Random) -> list[dict[str, Any]]:
    if cap is None or cap < 0 or len(rows) <= cap:
        return rows
    indices = sorted(rng.sample(range(len(rows)), cap))
    return [rows[index] for index in indices]


def source_dataset(row: dict[str, Any]) -> str:
    source = row.get("source")
    if isinstance(source, dict):
        return str(source.get("dataset") or source.get("path") or "unknown")
    if isinstance(source, str):
        return source
    return "unknown"


def parse_agentic_trajectory(text: str) -> tuple[str, list[dict[str, Any]]] | None:
    if not text.startswith('<record type="agentic_trajectory">') or "<payload>" not in text:
        return None
    payload = text.split("<payload>", 1)[1]
    payload = payload.split("</record>", 1)[0]
    if "<training_target>" not in payload:
        return None
    metadata_text, target_text = payload.split("<training_target>", 1)
    metadata_text = metadata_text.strip()
    target_text = target_text.strip()
    if target_text.endswith("</record>"):
        target_text = target_text[: -len("</record>")].strip()
    try:
        metadata = json.loads(metadata_text)
        steps = json.loads(target_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(metadata, dict) or not isinstance(steps, list):
        return None
    task = metadata.get("task")
    if not isinstance(task, dict):
        return None
    goal = str(task.get("user_goal", "")).strip()
    if not goal:
        return None
    return goal, [step for step in steps if isinstance(step, dict)]


def summarize_observation(observation: Any) -> str:
    if not isinstance(observation, dict):
        return "none"
    status = str(observation.get("status", "")).strip()
    parsed = observation.get("parsed")
    if isinstance(parsed, dict):
        failing = parsed.get("failing_tests")
        error_type = parsed.get("error_type")
        if isinstance(failing, list) and failing:
            test_name = str(failing[0]).split("::")[-1]
            return f"{status} {test_name} {error_type or ''}".strip()
    if "content_excerpt" in observation:
        return f"{status} read implementation"
    if "exit_code" in observation:
        return f"{status} exit {observation.get('exit_code')}"
    return status or "observed"


def format_action(action: Any) -> str | None:
    if not isinstance(action, dict):
        return None
    tool = str(action.get("tool", "")).strip()
    arguments = action.get("arguments")
    if not tool or not isinstance(arguments, dict):
        return None
    if tool == "run_shell":
        cmd = str(arguments.get("cmd", "")).strip()
        return f"run_shell: {cmd}" if cmd else None
    if tool == "read_file":
        path = str(arguments.get("path", "")).strip()
        return f"read_file: {path}" if path else None
    if tool == "write_patch":
        files = arguments.get("files_changed")
        if isinstance(files, list) and files:
            return f"write_patch: {files[0]}"
        patch = str(arguments.get("patch", "")).strip()
        changed = re.search(r"\*\*\* Update File: ([^\n]+)", patch)
        if changed:
            return f"write_patch: {changed.group(1).strip()}"
        return "write_patch"
    if tool == "pytest":
        target = str(arguments.get("target") or arguments.get("cmd") or "").strip()
        return f"pytest: {target}" if target else "pytest"
    return f"{tool}: {json.dumps(arguments, ensure_ascii=False, sort_keys=True)[:80]}"


def agentic_rows_from_cpt(path: Path, *, max_total_bytes: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_index, row in enumerate(read_jsonl(path)):
        parsed = parse_agentic_trajectory(str(row.get("text", "")))
        if parsed is None:
            continue
        goal, steps = parsed
        previous_observation = "none"
        for step_index, step in enumerate(steps):
            action = format_action(step.get("action"))
            if not action:
                continue
            if step_index == 0:
                prompt = f"Task: {goal}\nChoose first tool."
            else:
                prompt = f"Task: {goal}\nObs: {previous_observation}\nNext tool."
            completion = {
                "id": f"agentic_{row.get('id')}_{step_index}",
                "source": row.get("source"),
                "domain": f"agentic_next_action:{source_dataset(row)}",
                "stream": "agentic_next_action",
                "prompt": prompt,
                "answer": action,
                "text": f"{prompt}\n<|end|>\n{action}",
            }
            if fits_and_clean(completion, max_total_bytes):
                rows.append(completion)
            previous_observation = summarize_observation(step.get("observation"))
    return rows


def load_completion_rows(
    path: Path,
    *,
    max_total_bytes: int,
    family: str,
    domain_prefix: str,
    stream_filter: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(read_jsonl(path)):
        if stream_filter is not None and row.get("stream") != stream_filter:
            continue
        normalized = normalize_completion(
            row,
            family=family,
            index=index,
            domain_prefix=domain_prefix,
        )
        if fits_and_clean(normalized, max_total_bytes):
            rows.append(normalized)
    return rows


def write_mix(
    output_path: Path,
    rows_by_family: dict[str, list[dict[str, Any]]],
    caps: dict[str, int | None],
    rng: random.Random,
    *,
    exclude_prompt_hashes: set[str] | None = None,
    exclude_prompt_answer_hashes: set[str] | None = None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []
    input_counts = {family: len(rows) for family, rows in rows_by_family.items()}
    selected_counts: dict[str, int] = {}
    excluded_by_train_overlap = 0
    exclude_prompt_hashes = exclude_prompt_hashes or set()
    exclude_prompt_answer_hashes = exclude_prompt_answer_hashes or set()
    for family, rows in rows_by_family.items():
        sampled = sample_rows(rows, caps.get(family), rng)
        selected.extend(sampled)
        selected_counts[family] = len(sampled)
    rng.shuffle(selected)
    seen = set()
    written = 0
    duplicate_prompt_answer = 0
    task_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    max_total = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            prompt = str(row.get("prompt", ""))
            answer = str(row.get("answer", ""))
            key = prompt + "\0" + answer
            prompt_hash = stable_hash(prompt)
            prompt_answer_hash = stable_hash(key)
            if (
                prompt_hash in exclude_prompt_hashes
                or prompt_answer_hash in exclude_prompt_answer_hashes
            ):
                excluded_by_train_overlap += 1
                continue
            if key in seen:
                duplicate_prompt_answer += 1
                continue
            seen.add(key)
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            task_counts[str(row.get("stream", "unknown"))] += 1
            domain_counts[str(row.get("domain", "unknown")).split(":", 1)[0]] += 1
            max_total = max(max_total, row_total_bytes(row))
    return {
        "path": str(output_path),
        "input_counts": input_counts,
        "selected_counts": selected_counts,
        "written": written,
        "duplicate_prompt_answer_skipped": duplicate_prompt_answer,
        "excluded_by_train_overlap": excluded_by_train_overlap,
        "streams": dict(task_counts.most_common()),
        "domain_prefixes": dict(domain_counts.most_common()),
        "max_total_bytes": max_total,
    }


def stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def collect_prompt_hashes(path: Path) -> tuple[set[str], set[str]]:
    prompt_hashes: set[str] = set()
    prompt_answer_hashes: set[str] = set()
    for row in read_jsonl(path):
        prompt = str(row.get("prompt", ""))
        answer = str(row.get("answer", ""))
        prompt_hashes.add(stable_hash(prompt))
        prompt_answer_hashes.add(stable_hash(prompt + "\0" + answer))
    return prompt_hashes, prompt_answer_hashes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a clean capability-balanced TAC prompt/completion mixture."
    )
    parser.add_argument(
        "--assistant-train-jsonl",
        type=Path,
        default=Path("runs/benchmarks/unified_curriculum_redteam_2026_06_07/safe_anchor_candidates.completions.jsonl"),
    )
    parser.add_argument(
        "--assistant-eval-jsonl",
        type=Path,
        default=Path("Training data/unified_training_curriculum/tac_completions_full_text_only_clean/eval.completions.jsonl"),
    )
    parser.add_argument(
        "--reasoning-train-jsonl",
        type=Path,
        default=Path("runs/private_reasoning_final_answer_2026_06_07/train.completions.jsonl"),
    )
    parser.add_argument(
        "--reasoning-eval-jsonl",
        type=Path,
        default=Path("runs/private_reasoning_final_answer_2026_06_07/eval.completions.jsonl"),
    )
    parser.add_argument(
        "--ats-train-jsonl",
        type=Path,
        default=Path("runs/ats_transfer_10k_anchor_mix_redteam_clean_2026_06_07/train.completions.jsonl"),
    )
    parser.add_argument(
        "--ats-eval-jsonl",
        type=Path,
        default=Path("runs/ats_transfer_10k_anchor_mix_redteam_clean_2026_06_07/eval.completions.jsonl"),
    )
    parser.add_argument(
        "--agentic-train-jsonl",
        type=Path,
        default=Path("Training data/unified_training_curriculum/splits/train/unified_cpt.jsonl"),
    )
    parser.add_argument(
        "--agentic-eval-jsonl",
        type=Path,
        default=Path("Training data/unified_training_curriculum/splits/validation/unified_cpt.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-total-bytes", type=int, default=176)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--assistant-cap", type=int, default=-1)
    parser.add_argument("--agentic-cap", type=int, default=12000)
    parser.add_argument("--reasoning-cap", type=int, default=12000)
    parser.add_argument("--ats-cap", type=int, default=5000)
    parser.add_argument("--assistant-eval-cap", type=int, default=-1)
    parser.add_argument("--agentic-eval-cap", type=int, default=1000)
    parser.add_argument("--reasoning-eval-cap", type=int, default=1000)
    parser.add_argument("--ats-eval-cap", type=int, default=2000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cap = lambda value: None if value is None or value < 0 else int(value)

    train_rows = {
        "assistant_qna": load_completion_rows(
            args.assistant_train_jsonl,
            max_total_bytes=args.max_total_bytes,
            family="assistant_qna",
            domain_prefix="assistant_qna",
        ),
        "agentic_next_action": agentic_rows_from_cpt(
            args.agentic_train_jsonl,
            max_total_bytes=args.max_total_bytes,
        ),
        "private_reasoning_final_answer": load_completion_rows(
            args.reasoning_train_jsonl,
            max_total_bytes=args.max_total_bytes,
            family="private_reasoning_final_answer",
            domain_prefix="private_reasoning_final_answer",
        ),
        "ats_transfer": load_completion_rows(
            args.ats_train_jsonl,
            max_total_bytes=args.max_total_bytes,
            family="ats_transfer",
            domain_prefix=None,
            stream_filter="ats_transfer",
        ),
    }
    eval_rows = {
        "assistant_qna": load_completion_rows(
            args.assistant_eval_jsonl,
            max_total_bytes=args.max_total_bytes,
            family="assistant_qna",
            domain_prefix="assistant_qna",
        ),
        "agentic_next_action": agentic_rows_from_cpt(
            args.agentic_eval_jsonl,
            max_total_bytes=args.max_total_bytes,
        ),
        "private_reasoning_final_answer": load_completion_rows(
            args.reasoning_eval_jsonl,
            max_total_bytes=args.max_total_bytes,
            family="private_reasoning_final_answer",
            domain_prefix="private_reasoning_final_answer",
        ),
        "ats_transfer": load_completion_rows(
            args.ats_eval_jsonl,
            max_total_bytes=args.max_total_bytes,
            family="ats_transfer",
            domain_prefix=None,
            stream_filter="ats_transfer",
        ),
    }
    train = write_mix(
        args.output_dir / "train.completions.jsonl",
        train_rows,
        {
            "assistant_qna": cap(args.assistant_cap),
            "agentic_next_action": cap(args.agentic_cap),
            "private_reasoning_final_answer": cap(args.reasoning_cap),
            "ats_transfer": cap(args.ats_cap),
        },
        rng,
    )
    train_prompt_hashes, train_prompt_answer_hashes = collect_prompt_hashes(
        args.output_dir / "train.completions.jsonl"
    )
    eval_stats = write_mix(
        args.output_dir / "eval.completions.jsonl",
        eval_rows,
        {
            "assistant_qna": cap(args.assistant_eval_cap),
            "agentic_next_action": cap(args.agentic_eval_cap),
            "private_reasoning_final_answer": cap(args.reasoning_eval_cap),
            "ats_transfer": cap(args.ats_eval_cap),
        },
        rng,
        exclude_prompt_hashes=train_prompt_hashes,
        exclude_prompt_answer_hashes=train_prompt_answer_hashes,
    )
    weights = {
        "*": 1.0,
        "assistant_qna": 4.0,
        "agentic_next_action": 4.0,
        "private_reasoning_final_answer": 3.0,
        "ats_transfer_supervised": 2.0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sampling_weights.json").write_text(
        json.dumps(weights, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "schema": "tac_capability_balanced_dataset.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "max_total_bytes": args.max_total_bytes,
        "seed": args.seed,
        "train": train,
        "eval": eval_stats,
        "sampling_weights": weights,
        "notes": [
            "This is prompt/answer data for --supervision-mode answer_only.",
            "Assistant rows teach concise English assistant answers.",
            "Agentic rows teach compact state-to-next-tool behavior from long trajectories.",
            "Private reasoning rows teach final answers while keeping traces out of visible text.",
            "ATS rows preserve exact-answer transfer pressure.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
