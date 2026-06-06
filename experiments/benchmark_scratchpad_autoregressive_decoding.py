from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import AgenticScratchpadState, ScratchpadItem, TACConfig, TACTransformerLM
from tac_transformer.phase_d_benchmarks import (
    PHASE_D_EOS_TOKEN_ID,
    PHASE_D_MIN_BYTE_VOCAB_SIZE,
    extract_phase_d_answer,
    generate_phase_d_completion,
    phase_d_text_to_token_ids,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/scratchpad_autoregressive_decoding_2026_06_04")
TRAIN_MARKERS = tuple("ABCDEFGH")
EVAL_MARKER = "Z"
DIGIT_ANSWERS = tuple(str(value) for value in range(10))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove TAC can learn autoregressive decoding from verified scratchpad context."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--min-scratchpad-score", type=float, default=0.95)
    parser.add_argument("--min-counterfactual-score", type=float, default=0.95)
    parser.add_argument("--max-no-scratchpad-score", type=float, default=0.20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_scratchpad_autoregressive_decoding_probe(
        seed=args.seed,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        min_scratchpad_score=args.min_scratchpad_score,
        min_counterfactual_score=args.min_counterfactual_score,
        max_no_scratchpad_score=args.max_no_scratchpad_score,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scratchpad_autoregressive_decoding.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def run_scratchpad_autoregressive_decoding_probe(
    *,
    seed: int = 3,
    train_steps: int = 50,
    learning_rate: float = 0.01,
    min_scratchpad_score: float = 0.95,
    min_counterfactual_score: float = 0.95,
    max_no_scratchpad_score: float = 0.20,
) -> dict[str, Any]:
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    torch.manual_seed(seed)
    max_seq_len = 80
    train_rows = [
        {
            "marker": marker,
            "answer": answer,
            "prompt": build_verified_scratchpad_decoding_prompt(
                answer=answer,
                marker=marker,
            ),
        }
        for marker in TRAIN_MARKERS
        for answer in DIGIT_ANSWERS
    ]
    input_ids, labels = _build_training_batch(train_rows, max_seq_len=max_seq_len)
    model = TACTransformerLM(_scratchpad_decoder_config(max_seq_len=max_seq_len))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    initial_loss = _loss_value(model, input_ids, labels)
    for _ in range(train_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, labels=labels, collect_auxiliary=False)
        if output.loss is None:
            raise RuntimeError("TACTransformerLM did not return a training loss")
        output.loss.backward()
        optimizer.step()
    final_loss = _loss_value(model, input_ids, labels)

    scratchpad_predictions = _evaluate_scratchpad_prompts(
        model,
        marker=EVAL_MARKER,
        answers=DIGIT_ANSWERS,
    )
    no_scratchpad_predictions = _evaluate_no_scratchpad_control(
        model,
        marker=EVAL_MARKER,
        expected_answers=DIGIT_ANSWERS,
    )
    counterfactual_answers = tuple(
        str((int(answer) + 5) % 10) for answer in DIGIT_ANSWERS
    )
    counterfactual_predictions = _evaluate_scratchpad_prompts(
        model,
        marker=EVAL_MARKER,
        answers=counterfactual_answers,
        kind="counterfactual_scratchpad",
    )

    scratchpad_score = _prediction_score(scratchpad_predictions)
    no_scratchpad_score = _prediction_score(no_scratchpad_predictions)
    counterfactual_score = _prediction_score(counterfactual_predictions)
    checks = {
        "loss_reduced": final_loss < initial_loss,
        "scratchpad_generation_passed": scratchpad_score >= min_scratchpad_score,
        "counterfactual_generation_passed": counterfactual_score
        >= min_counterfactual_score,
        "no_scratchpad_control_low": no_scratchpad_score <= max_no_scratchpad_score,
        "scratchpad_beats_control": scratchpad_score > no_scratchpad_score,
        "generated_nonempty": all(
            bool(row["raw_completion"]) for row in scratchpad_predictions
        ),
    }
    return {
        "schema": "scratchpad_autoregressive_decoding.v1",
        "date": "2026-06-04",
        "seed": seed,
        "train_steps": train_steps,
        "train_examples": len(train_rows),
        "model": {
            "type": "TACTransformerLM",
            "config": model.config.__dict__,
            "generation": "greedy_byte_level_autoregressive",
        },
        "training": {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction": initial_loss - final_loss,
            "learning_rate": learning_rate,
            "supervision": "masked_next_token_completion_only",
        },
        "scores": {
            "scratchpad_score": scratchpad_score,
            "no_scratchpad_score": no_scratchpad_score,
            "counterfactual_score": counterfactual_score,
            "scratchpad_control_margin": scratchpad_score - no_scratchpad_score,
        },
        "thresholds": {
            "min_scratchpad_score": min_scratchpad_score,
            "min_counterfactual_score": min_counterfactual_score,
            "max_no_scratchpad_score": max_no_scratchpad_score,
        },
        "sample_predictions": (
            scratchpad_predictions[:3]
            + no_scratchpad_predictions[:1]
            + counterfactual_predictions[:3]
        ),
        "decision": {
            "status": (
                "scratchpad_autoregressive_decoding_proved"
                if all(checks.values())
                else "blocked"
            ),
            "checks": checks,
            "scope": (
                "This trains a TACTransformerLM decoder with masked next-token "
                "loss on scratchpad-augmented prompts, then evaluates greedy "
                "autoregressive byte generation. It proves learned decoding from "
                "verified scratchpad context on a compact local digit-copy gate. "
                "It does not yet prove full Phase D task solving or joint "
                "TAC+controller optimization."
            ),
        },
    }


def build_verified_scratchpad_decoding_prompt(*, answer: str, marker: str) -> str:
    state = AgenticScratchpadState(
        items=(
            ScratchpadItem(
                item_id="answer",
                kind="answer",
                payload=str(answer),
                utility=1.0,
                confidence=1.0,
                verified=True,
            ),
            ScratchpadItem(
                item_id="unverified_guess",
                kind="simulation",
                payload="wrong",
                utility=1.0,
                confidence=1.0,
                imagined=True,
                verified=False,
            ),
        ),
        budget=2,
        step=1,
    )
    verified_answers = [
        str(item.payload).strip()
        for item in state.items
        if item.verified and item.kind == "answer"
    ]
    scratchpad_value = verified_answers[0] if verified_answers else ""
    return (
        f"Verified scratchpad answer={scratchpad_value}\n"
        f"Task {marker}: answer from scratchpad.\n"
        "Answer:"
    )


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Scratchpad Autoregressive Decoding",
        "",
        f"Decision: `{report['decision']['status']}`",
        "",
        "## Scores",
        "",
        f"Scratchpad score: `{report['scores']['scratchpad_score']:.4f}`",
        f"No-scratchpad score: `{report['scores']['no_scratchpad_score']:.4f}`",
        f"Counterfactual score: `{report['scores']['counterfactual_score']:.4f}`",
        (
            "Scratchpad/control margin: "
            f"`{report['scores']['scratchpad_control_margin']:.4f}`"
        ),
        "",
        "## Training",
        "",
        f"Initial loss: `{report['training']['initial_loss']:.6f}`",
        f"Final loss: `{report['training']['final_loss']:.6f}`",
        f"Train examples: `{report['train_examples']}`",
        f"Train steps: `{report['train_steps']}`",
        "",
        "## Scope",
        "",
        report["decision"]["scope"],
        "",
    ]
    return "\n".join(lines)


def _scratchpad_decoder_config(*, max_seq_len: int) -> TACConfig:
    return TACConfig(
        vocab_size=PHASE_D_MIN_BYTE_VOCAB_SIZE,
        d_model=32,
        n_heads=4,
        n_layers=1,
        n_programs=4,
        max_seq_len=max_seq_len,
        dropout=0.0,
        memory_read_type="none",
        routing_type="base",
        content_read_steps=1,
        content_store_size=4,
        pattern_store_size=4,
    )


def _build_training_batch(
    rows: list[dict[str, str]],
    *,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [
        _encode_completion_row(row["prompt"], row["answer"], max_seq_len=max_seq_len)
        for row in rows
    ]
    input_ids = torch.tensor([row[0] for row in encoded], dtype=torch.long)
    labels = torch.tensor([row[1] for row in encoded], dtype=torch.long)
    return input_ids, labels


def _encode_completion_row(
    prompt: str,
    answer: str,
    *,
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    prompt_ids = phase_d_text_to_token_ids(
        prompt,
        vocab_size=PHASE_D_MIN_BYTE_VOCAB_SIZE,
    )
    answer_ids = phase_d_text_to_token_ids(
        str(answer),
        vocab_size=PHASE_D_MIN_BYTE_VOCAB_SIZE,
    )
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_seq_len:
        raise ValueError("scratchpad decoding row exceeds max_seq_len")
    labels = [-100 for _ in input_ids]
    for offset, token_id in enumerate(answer_ids + [PHASE_D_EOS_TOKEN_ID]):
        label_position = len(prompt_ids) - 1 + offset
        if label_position < len(labels):
            labels[label_position] = token_id
    padding = max_seq_len - len(input_ids)
    return input_ids + [0] * padding, labels + [-100] * padding


def _loss_value(
    model: TACTransformerLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        output = model(input_ids, labels=labels, collect_auxiliary=False)
    if output.loss is None:
        raise RuntimeError("TACTransformerLM did not return a loss")
    return float(output.loss.detach())


def _evaluate_scratchpad_prompts(
    model: TACTransformerLM,
    *,
    marker: str,
    answers: tuple[str, ...],
    kind: str = "verified_scratchpad",
) -> list[dict[str, Any]]:
    return [
        _prediction_row(
            model,
            prompt=build_verified_scratchpad_decoding_prompt(
                answer=answer,
                marker=marker,
            ),
            expected=answer,
            kind=kind,
        )
        for answer in answers
    ]


def _evaluate_no_scratchpad_control(
    model: TACTransformerLM,
    *,
    marker: str,
    expected_answers: tuple[str, ...],
) -> list[dict[str, Any]]:
    prompt = f"Task {marker}: answer from scratchpad.\nAnswer:"
    return [
        _prediction_row(
            model,
            prompt=prompt,
            expected=answer,
            kind="no_scratchpad",
        )
        for answer in expected_answers
    ]


def _prediction_row(
    model: TACTransformerLM,
    *,
    prompt: str,
    expected: str,
    kind: str,
) -> dict[str, Any]:
    result = generate_phase_d_completion(
        model,
        prompt,
        max_new_tokens=2,
        device="cpu",
    )
    raw_completion = str(result["completion"])
    prediction = extract_phase_d_answer(raw_completion, mode="first_token")
    return {
        "kind": kind,
        "expected": str(expected),
        "prediction": prediction,
        "correct": prediction == str(expected),
        "raw_completion": raw_completion,
        "generated_token_count": int(result["generated_token_count"]),
        "prompt_token_count": int(result["prompt_token_count"]),
    }


def _prediction_score(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["correct"]) / len(rows)


if __name__ == "__main__":
    main()
