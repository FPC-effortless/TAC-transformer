from __future__ import annotations

import json
import random
import re
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

import torch

from .agentic_controller import AgenticScratchpadState
from .model import TACConfig, TACTransformerLM, VanillaTransformerLM


PHASE_D_TASK_FAMILIES = {
    "multi_hop_chain_retrieval": "memory_intensive",
    "long_context_retrieval_4096": "memory_intensive",
    "episodic_fact_update": "memory_intensive",
    "tool_selection": "agentic",
    "delayed_goal_binding": "agentic",
}

PHASE_D_TASK_IDS = tuple(PHASE_D_TASK_FAMILIES)
PHASE_D_EOS_TOKEN_ID = 3
PHASE_D_BYTE_TOKEN_OFFSET = 4
PHASE_D_MIN_BYTE_VOCAB_SIZE = PHASE_D_BYTE_TOKEN_OFFSET + 256


def build_phase_d_task_suite(
    *,
    seed: int,
    examples_per_task: int = 8,
    context_length: int = 4096,
) -> dict[str, Any]:
    """Build deterministic Phase D benchmark examples for one seed."""

    rng = random.Random(int(seed))
    examples = []
    for task_id in PHASE_D_TASK_IDS:
        for index in range(int(examples_per_task)):
            examples.append(
                _build_phase_d_example(
                    task_id,
                    seed=int(seed),
                    index=index,
                    rng=rng,
                    context_length=int(context_length),
                )
            )
    return {
        "phase": "D",
        "seed": int(seed),
        "examples_per_task": int(examples_per_task),
        "context_length": int(context_length),
        "task_ids": list(PHASE_D_TASK_IDS),
        "example_count": len(examples),
        "examples": examples,
    }


def score_phase_d_predictions(
    examples: Iterable[dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
    *,
    control_id: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Score prediction rows into `aggregate_phase_d_benchmark_results` rows."""

    example_rows = [row for row in examples if isinstance(row, dict)]
    prediction_by_example = {
        str(row.get("example_id")): row
        for row in predictions
        if isinstance(row, dict)
        and str(row.get("control_id", control_id)) == str(control_id)
    }
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in example_rows:
        by_task[str(example.get("task_id"))].append(example)

    rows = []
    for task_id in sorted(by_task):
        task_examples = by_task[task_id]
        correct = 0
        missing = 0
        scored = []
        tokens_per_second = []
        wall_clock_seconds = []
        for example in task_examples:
            prediction = prediction_by_example.get(str(example.get("id")))
            if prediction is None:
                missing += 1
                scored.append(False)
                continue
            is_correct = _exact_match(
                prediction.get("prediction"),
                example.get("answer"),
            )
            scored.append(is_correct)
            correct += int(is_correct)
            _append_number(tokens_per_second, prediction.get("tokens_per_second"))
            _append_number(wall_clock_seconds, prediction.get("wall_clock_seconds"))

        total = len(scored)
        primary_score = (correct / total) if total else 0.0
        rows.append(
            {
                "task_id": task_id,
                "family": PHASE_D_TASK_FAMILIES.get(task_id, "unknown"),
                "control_id": str(control_id),
                "seed": int(seed),
                "primary_metric": "exact_match",
                "primary_score": primary_score,
                "correct_count": correct,
                "example_count": total,
                "missing_prediction_count": missing,
                "tokens_per_second": (
                    mean(tokens_per_second) if tokens_per_second else None
                ),
                "wall_clock_seconds": (
                    sum(wall_clock_seconds) if wall_clock_seconds else None
                ),
            }
        )
    return rows


def format_agentic_scratchpad_context(state: AgenticScratchpadState) -> str:
    """Render verified scratchpad state as a compact Phase D prompt prefix."""

    verified_items = [item for item in state.items if item.verified]
    if not verified_items:
        return ""
    lines = ["Verified scratchpad:"]
    for item in verified_items:
        payload = str(item.payload).strip()
        if not payload:
            continue
        item_id = str(item.item_id).strip() or "item"
        kind = str(item.kind).strip() or "note"
        lines.append(f"- {kind}:{item_id}: {payload}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def augment_phase_d_prompt_with_scratchpad(
    example: Mapping[str, Any],
    state: AgenticScratchpadState,
) -> str:
    """Attach verified scratchpad context to a Phase D example prompt."""

    context = format_agentic_scratchpad_context(state)
    prompt = str(example.get("prompt", ""))
    if not context:
        return prompt
    return f"{context}\n\nTask prompt:\n{prompt}"


def run_phase_d_scratchpad_state_predictions(
    *,
    examples: Iterable[dict[str, Any]],
    scratchpad_by_example: Mapping[str, AgenticScratchpadState],
    control_id: str,
    seed: int,
    answer_extraction: str = "raw",
) -> dict[str, Any]:
    """Run a verifier-gated scratchpad state through the Phase D row contract."""

    state_by_example = {
        str(example_id): state for example_id, state in scratchpad_by_example.items()
    }
    rows = []
    for example in examples:
        example_id = str(example.get("id"))
        state = state_by_example.get(example_id) or AgenticScratchpadState.empty(budget=0)
        augmented_prompt = augment_phase_d_prompt_with_scratchpad(example, state)
        raw_completion = _phase_d_scratchpad_answer_payload(state)
        prediction = (
            extract_phase_d_answer(raw_completion, mode=answer_extraction)
            if raw_completion
            else ""
        )
        verified_item_ids = [
            str(item.item_id) for item in state.items if item.verified
        ]
        rows.append(
            {
                "example_id": example_id,
                "task_id": str(example.get("task_id")),
                "family": str(example.get("family", "")),
                "control_id": str(control_id),
                "seed": int(seed),
                "prediction": prediction,
                "raw_completion": raw_completion,
                "answer_extraction": answer_extraction,
                "generated_token_count": 0,
                "prompt_token_count": len(
                    phase_d_text_to_token_ids(
                        augmented_prompt,
                        vocab_size=PHASE_D_MIN_BYTE_VOCAB_SIZE,
                    )
                ),
                "truncated_prompt_token_count": 0,
                "context_window": None,
                "tokens_per_second": None,
                "wall_clock_seconds": 0.0,
                "scratchpad_used": bool(raw_completion),
                "scratchpad_item_ids": list(state.item_ids()),
                "verified_scratchpad_item_ids": verified_item_ids,
                "augmented_prompt": augmented_prompt,
            }
        )
    return {
        "phase": "D",
        "schema": "tac_control_v1_phase_d_scratchpad_state_predictions.v1",
        "control_id": str(control_id),
        "seed": int(seed),
        "prediction_count": len(rows),
        "answer_extraction": answer_extraction,
        "rows": rows,
    }


def stage_phase_d_benchmark_suite(
    *,
    output_dir: str | Path,
    seeds: Iterable[int] = (11, 23, 37),
    examples_per_task: int = 8,
    context_length: int = 4096,
) -> dict[str, Any]:
    """Write deterministic Phase D task JSONL files and a manifest."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seed_list = [int(seed) for seed in seeds]
    suites = []
    example_count = 0
    for seed in seed_list:
        suite = build_phase_d_task_suite(
            seed=seed,
            examples_per_task=examples_per_task,
            context_length=context_length,
        )
        seed_dir = root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        tasks_path = seed_dir / "tasks.jsonl"
        _write_jsonl(tasks_path, suite["examples"])
        template_path = seed_dir / "predictions_template.jsonl"
        _write_jsonl(
            template_path,
            [
                {
                    "example_id": example["id"],
                    "task_id": example["task_id"],
                    "control_id": "<control_id>",
                    "prediction": "<model_answer>",
                    "tokens_per_second": None,
                    "wall_clock_seconds": None,
                }
                for example in suite["examples"]
            ],
        )
        suites.append(
            {
                "seed": seed,
                "tasks_jsonl": str(tasks_path),
                "predictions_template_jsonl": str(template_path),
                "example_count": suite["example_count"],
            }
        )
        example_count += int(suite["example_count"])

    manifest = {
        "phase": "D",
        "schema": "tac_control_v1_phase_d_benchmark_suite.v1",
        "seeds": seed_list,
        "task_ids": list(PHASE_D_TASK_IDS),
        "examples_per_task": int(examples_per_task),
        "context_length": int(context_length),
        "example_count": example_count,
        "suites": suites,
        "prediction_schema": {
            "required": ["example_id", "control_id", "prediction"],
            "optional": ["tokens_per_second", "wall_clock_seconds"],
        },
    }
    (root / "phase_d_benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (root / "RESULTS.md").write_text(
        format_phase_d_benchmark_suite_markdown(manifest),
        encoding="utf-8",
    )
    return manifest


def format_phase_d_benchmark_suite_markdown(manifest: dict[str, Any]) -> str:
    """Format a staged Phase D benchmark suite manifest as Markdown."""

    lines = [
        "# TAC-Control-v1 Phase D Benchmark Suite",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in manifest.get('seeds', []))}",
        f"- Tasks: {', '.join(manifest.get('task_ids', []))}",
        f"- Examples per task: {manifest.get('examples_per_task')}",
        f"- Context length: {manifest.get('context_length')}",
        f"- Total examples: {manifest.get('example_count')}",
        "",
        "| Seed | Examples | Tasks JSONL | Prediction Template |",
        "| ---: | ---: | --- | --- |",
    ]
    for suite in manifest.get("suites", []):
        lines.append(
            "| {seed} | {count} | `{tasks}` | `{template}` |".format(
                seed=suite.get("seed"),
                count=suite.get("example_count"),
                tasks=suite.get("tasks_jsonl"),
                template=suite.get("predictions_template_jsonl"),
            )
        )
    lines.extend(
        [
            "",
            "Prediction rows require `example_id`, `control_id`, and `prediction`.",
            "Scored rows are compatible with `experiments/aggregate_phase_d_benchmarks.py`.",
        ]
    )
    return "\n".join(lines)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def phase_d_text_to_token_ids(
    text: str,
    *,
    vocab_size: int,
    append_eos: bool = False,
) -> list[int]:
    """Encode Phase D prompt text with the training byte-level token contract."""

    if int(vocab_size) < PHASE_D_MIN_BYTE_VOCAB_SIZE:
        raise ValueError(
            f"vocab_size must be at least {PHASE_D_MIN_BYTE_VOCAB_SIZE} "
            "for Phase D byte-level prompts"
        )
    token_ids = [
        byte + PHASE_D_BYTE_TOKEN_OFFSET
        for byte in str(text).encode("utf-8", errors="replace")
    ]
    if append_eos:
        token_ids.append(PHASE_D_EOS_TOKEN_ID)
    return token_ids


def phase_d_token_ids_to_text(
    token_ids: Iterable[int],
    *,
    stop_at_eos: bool = True,
) -> str:
    """Decode generated byte-level token IDs into text for answer extraction."""

    bytes_out = bytearray()
    for token_id in token_ids:
        token = int(token_id)
        if token == PHASE_D_EOS_TOKEN_ID and stop_at_eos:
            break
        byte_value = token - PHASE_D_BYTE_TOKEN_OFFSET
        if 0 <= byte_value <= 255:
            bytes_out.append(byte_value)
    return bytes(bytes_out).decode("utf-8", errors="replace")


def extract_phase_d_answer(completion: str, *, mode: str = "first_token") -> str:
    """Extract the exact-match prediction from a raw generated completion."""

    if mode not in {"raw", "first_line", "first_token"}:
        raise ValueError("answer extraction mode must be raw, first_line, or first_token")
    text = str(completion)
    if mode == "raw":
        return text.strip()
    first_line = ""
    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            first_line = candidate
            break
    if mode == "first_line":
        return first_line
    tokens = first_line.split()
    if not tokens:
        return ""
    return tokens[0].strip("`'\" .,:;")


def load_phase_d_checkpoint_model(
    checkpoint_path: str | Path,
    *,
    model_type: str = "auto",
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a TAC or vanilla checkpoint saved by the Kaggle training scripts."""

    if model_type not in {"auto", "tac", "vanilla"}:
        raise ValueError("model_type must be auto, tac, or vanilla")
    resolved_device = torch.device(device)
    checkpoint = _torch_load_checkpoint(Path(checkpoint_path), resolved_device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint must be a dict: {checkpoint_path}")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"checkpoint is missing model_state_dict: {checkpoint_path}")

    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError(f"model_state_dict must be a dict: {checkpoint_path}")
    resolved_type = _resolve_phase_d_model_type(model_type, state_dict)
    config = _coerce_phase_d_config(checkpoint.get("config"))
    model: torch.nn.Module
    if resolved_type == "tac":
        model = TACTransformerLM(config)
    else:
        model = VanillaTransformerLM(config)
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": _optional_int(checkpoint.get("step")),
        "best_eval_loss": _optional_float(checkpoint.get("best_eval_loss")),
        "model_type": resolved_type,
        "config": asdict(config),
        "parameter_counts": checkpoint.get("parameter_counts"),
    }
    return model, metadata


def generate_phase_d_completion(
    model: torch.nn.Module,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    device: str | torch.device = "cpu",
    precision: str = "fp32",
) -> dict[str, Any]:
    """Greedily generate a byte-level completion from a loaded Phase D model."""

    if int(max_new_tokens) < 1:
        raise ValueError("max_new_tokens must be at least 1")
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("model must expose a TACConfig-compatible config")
    resolved_device = torch.device(device)
    prompt_tokens = phase_d_text_to_token_ids(
        prompt,
        vocab_size=int(config.vocab_size),
        append_eos=False,
    )
    if not prompt_tokens:
        prompt_tokens = [PHASE_D_EOS_TOKEN_ID]
    context_window = max(1, int(config.max_seq_len))
    all_tokens = list(prompt_tokens)
    generated_tokens: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(int(max_new_tokens)):
            window = all_tokens[-context_window:]
            input_ids = torch.tensor([window], dtype=torch.long, device=resolved_device)
            with _phase_d_autocast_context(resolved_device, precision):
                output = model(input_ids, collect_auxiliary=False)
            next_token = int(output.logits[0, -1].argmax().detach().cpu())
            if next_token == PHASE_D_EOS_TOKEN_ID:
                break
            generated_tokens.append(next_token)
            all_tokens.append(next_token)
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "completion": phase_d_token_ids_to_text(generated_tokens),
        "generated_token_ids": generated_tokens,
        "generated_token_count": len(generated_tokens),
        "prompt_token_count": len(prompt_tokens),
        "truncated_prompt_token_count": max(0, len(prompt_tokens) - context_window),
        "context_window": context_window,
        "wall_clock_seconds": elapsed,
        "tokens_per_second": len(generated_tokens) / elapsed,
    }


def run_phase_d_checkpoint_predictions(
    *,
    tasks_jsonl: str | Path,
    checkpoint_path: str | Path,
    control_id: str,
    seed: int,
    output_jsonl: str | Path | None = None,
    model_type: str = "auto",
    device: str | torch.device = "cpu",
    precision: str = "fp32",
    max_new_tokens: int = 32,
    answer_extraction: str = "first_token",
) -> dict[str, Any]:
    """Run a loaded checkpoint over a Phase D task JSONL and write predictions."""

    examples = load_jsonl(tasks_jsonl)
    model, checkpoint_metadata = load_phase_d_checkpoint_model(
        checkpoint_path,
        model_type=model_type,
        device=device,
    )
    rows = []
    for example in examples:
        result = generate_phase_d_completion(
            model,
            str(example.get("prompt", "")),
            max_new_tokens=max_new_tokens,
            device=device,
            precision=precision,
        )
        raw_completion = str(result["completion"])
        rows.append(
            {
                "example_id": str(example.get("id")),
                "task_id": str(example.get("task_id")),
                "family": str(example.get("family", "")),
                "control_id": str(control_id),
                "seed": int(seed),
                "prediction": extract_phase_d_answer(
                    raw_completion,
                    mode=answer_extraction,
                ),
                "raw_completion": raw_completion,
                "answer_extraction": answer_extraction,
                "generated_token_count": int(result["generated_token_count"]),
                "prompt_token_count": int(result["prompt_token_count"]),
                "truncated_prompt_token_count": int(
                    result["truncated_prompt_token_count"]
                ),
                "context_window": int(result["context_window"]),
                "tokens_per_second": float(result["tokens_per_second"]),
                "wall_clock_seconds": float(result["wall_clock_seconds"]),
                "checkpoint": checkpoint_metadata["checkpoint"],
                "checkpoint_step": checkpoint_metadata["checkpoint_step"],
                "model_type": checkpoint_metadata["model_type"],
            }
        )
    if output_jsonl is not None:
        _write_jsonl(Path(output_jsonl), rows)
    return {
        "phase": "D",
        "schema": "tac_control_v1_phase_d_checkpoint_predictions.v1",
        "tasks_jsonl": str(tasks_jsonl),
        "checkpoint": checkpoint_metadata["checkpoint"],
        "checkpoint_step": checkpoint_metadata["checkpoint_step"],
        "model_type": checkpoint_metadata["model_type"],
        "control_id": str(control_id),
        "seed": int(seed),
        "prediction_count": len(rows),
        "answer_extraction": answer_extraction,
        "max_new_tokens": int(max_new_tokens),
        "rows": rows,
    }


def _build_phase_d_example(
    task_id: str,
    *,
    seed: int,
    index: int,
    rng: random.Random,
    context_length: int,
) -> dict[str, Any]:
    builders = {
        "multi_hop_chain_retrieval": _multi_hop_example,
        "long_context_retrieval_4096": _long_context_example,
        "episodic_fact_update": _episodic_update_example,
        "tool_selection": _tool_selection_example,
        "delayed_goal_binding": _delayed_goal_binding_example,
    }
    if task_id not in builders:
        raise ValueError(f"unknown Phase D task_id: {task_id}")
    example = builders[task_id](
        seed=seed,
        index=index,
        rng=rng,
        context_length=context_length,
    )
    return {
        "id": f"{task_id}_seed{seed}_{index:04d}",
        "task_id": task_id,
        "family": PHASE_D_TASK_FAMILIES[task_id],
        "primary_metric": "exact_match",
        **example,
    }


def _phase_d_scratchpad_answer_payload(state: AgenticScratchpadState) -> str:
    for item in state.items:
        if not item.verified:
            continue
        item_id = str(item.item_id).lower()
        kind = str(item.kind).lower()
        if item_id in {"answer", "final_answer"} or kind in {"answer", "final_answer"}:
            return str(item.payload).strip()
    return ""


def _multi_hop_example(
    *,
    seed: int,
    index: int,
    rng: random.Random,
    context_length: int,
) -> dict[str, Any]:
    del context_length
    names = _sample_labels(rng, prefix="node", count=4)
    prompt = (
        "Follow the directed chain exactly two hops.\n"
        f"{names[0]} -> {names[1]}\n"
        f"{names[1]} -> {names[2]}\n"
        f"{names[2]} -> {names[3]}\n"
        f"Question: starting at {names[0]}, where are you after two hops?\n"
        "Answer with only the node label."
    )
    return {
        "prompt": prompt,
        "answer": names[2],
        "metadata": {"seed": seed, "index": index, "hops": 2},
    }


def _long_context_example(
    *,
    seed: int,
    index: int,
    rng: random.Random,
    context_length: int,
) -> dict[str, Any]:
    target_key = f"target_{seed}_{index}"
    answer = f"value_{rng.randrange(10_000, 99_999)}"
    distractors = [
        f"key_{index}_{i}=value_{rng.randrange(1000, 9999)}"
        for i in range(max(16, context_length // 32))
    ]
    insertion = rng.randrange(4, max(5, len(distractors) - 4))
    distractors.insert(insertion, f"{target_key}={answer}")
    context = " | ".join(distractors)
    prompt = (
        "Retrieve the exact value for the requested key from the long context.\n"
        f"Context: {context}\n"
        f"Question: what is the value of {target_key}?\n"
        "Answer with only the value."
    )
    if len(prompt) < context_length:
        filler = " | filler=ignore"
        repeats = ((context_length - len(prompt)) // len(filler)) + 1
        prompt = prompt.replace("\nQuestion:", filler * repeats + "\nQuestion:")
    return {
        "prompt": prompt,
        "answer": answer,
        "metadata": {"seed": seed, "index": index, "target_key": target_key},
    }


def _episodic_update_example(
    *,
    seed: int,
    index: int,
    rng: random.Random,
    context_length: int,
) -> dict[str, Any]:
    del context_length
    subject = f"case_{seed}_{index}"
    old_value = f"status_{rng.randrange(100, 999)}"
    new_value = f"status_{rng.randrange(1000, 9999)}"
    prompt = (
        "Use the latest episode, not the stale fact.\n"
        f"Episode 1: {subject} has status {old_value}.\n"
        f"Episode 2: correction, {subject} now has status {new_value}.\n"
        f"Question: what is the current status for {subject}?\n"
        "Answer with only the status."
    )
    return {
        "prompt": prompt,
        "answer": new_value,
        "metadata": {"seed": seed, "index": index, "stale": old_value},
    }


def _tool_selection_example(
    *,
    seed: int,
    index: int,
    rng: random.Random,
    context_length: int,
) -> dict[str, Any]:
    del seed, context_length
    tools = [
        ("search_docs", "look up current documentation"),
        ("run_tests", "execute the test suite"),
        ("edit_file", "modify source code"),
        ("inspect_logs", "read runtime logs"),
    ]
    selected = tools[(index + rng.randrange(len(tools))) % len(tools)]
    distractors = ", ".join(f"{name}: {desc}" for name, desc in tools)
    prompt = (
        "Select the single best tool for the goal.\n"
        f"Available tools: {distractors}.\n"
        f"Goal: {selected[1]}.\n"
        "Answer with only the tool name."
    )
    return {
        "prompt": prompt,
        "answer": selected[0],
        "metadata": {"index": index, "tool_description": selected[1]},
    }


def _delayed_goal_binding_example(
    *,
    seed: int,
    index: int,
    rng: random.Random,
    context_length: int,
) -> dict[str, Any]:
    del context_length
    goal = f"goal_{seed}_{index}"
    answer = f"commit_{rng.randrange(1000, 9999)}"
    prompt = (
        f"Initial goal token: {goal}.\n"
        "Several irrelevant notes follow: alpha, beta, gamma.\n"
        f"If the initial goal token was {goal}, bind the final action to {answer}.\n"
        "Question: what final action is bound to the initial goal?\n"
        "Answer with only the action token."
    )
    return {
        "prompt": prompt,
        "answer": answer,
        "metadata": {"seed": seed, "index": index, "goal": goal},
    }


def _sample_labels(rng: random.Random, *, prefix: str, count: int) -> list[str]:
    return [
        f"{prefix}_{rng.randrange(1000, 9999)}"
        for _ in range(count)
    ]


def _exact_match(prediction: Any, answer: Any) -> bool:
    return _normalize_answer(prediction) == _normalize_answer(answer)


def _normalize_answer(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip("`'\" .,:;")


def _append_number(values: list[float], value: Any) -> None:
    try:
        if value is None:
            return
        number = float(value)
    except (TypeError, ValueError):
        return
    values.append(number)


def _torch_load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch compatibility.
        return torch.load(path, map_location=device)


def _coerce_phase_d_config(config_like: Any) -> TACConfig:
    if isinstance(config_like, TACConfig):
        return config_like
    if is_dataclass(config_like) and not isinstance(config_like, type):
        config_like = asdict(config_like)
    if not isinstance(config_like, dict):
        raise ValueError("checkpoint is missing a TACConfig-compatible config")
    known_fields = {field.name for field in fields(TACConfig)}
    config_dict = {
        str(name): value
        for name, value in config_like.items()
        if str(name) in known_fields
    }
    if "vocab_size" not in config_dict:
        raise ValueError("checkpoint config is missing vocab_size")
    for name in (
        "semantic_route_allowed_programs",
        "semantic_route_suppressed_programs",
    ):
        value = config_dict.get(name)
        if isinstance(value, list):
            config_dict[name] = tuple(int(item) for item in value)
    return TACConfig(**config_dict)


def _resolve_phase_d_model_type(model_type: str, state_dict: dict[str, Any]) -> str:
    if model_type != "auto":
        return model_type
    for name in state_dict:
        if ".identity_field." in str(name) or str(name).startswith("memory_adapter"):
            return "tac"
    return "vanilla"


def _phase_d_autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError("precision must be fp32, fp16, or bf16")


def _optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
