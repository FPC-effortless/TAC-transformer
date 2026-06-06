from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    TACConfig,
    TACTransformerLM,
    VanillaTransformerLM,
    build_ats_transfer_suite,
    extract_phase_d_answer,
    generate_phase_d_completion,
    phase_d_text_to_token_ids,
    score_ats_transfer_predictions,
)
from tac_transformer.phase_d_benchmarks import PHASE_D_EOS_TOKEN_ID
from tac_transformer.training import parameter_matched_baseline_config


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/ats_answer_copy_training_2026_06_05")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TAC and vanilla controls with answer-only ATS copy loss."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--examples-per-domain", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--max-seq-len", type=int, default=176)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-programs", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--min-tac-train-score", type=float, default=0.50)
    parser.add_argument("--min-tac-test-score", type=float, default=0.25)
    parser.add_argument("--min-test-margin", type=float, default=0.10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_ats_answer_copy_training_probe(
        seed=args.seed,
        examples_per_domain=args.examples_per_domain,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        max_new_tokens=args.max_new_tokens,
        min_tac_train_score=args.min_tac_train_score,
        min_tac_test_score=args.min_tac_test_score,
        min_test_margin=args.min_test_margin,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ats_answer_copy_training.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def run_ats_answer_copy_training_probe(
    *,
    seed: int = 37,
    examples_per_domain: int = 4,
    train_steps: int = 300,
    learning_rate: float = 0.003,
    max_seq_len: int = 176,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    n_programs: int = 16,
    max_new_tokens: int = 24,
    min_tac_train_score: float = 0.50,
    min_tac_test_score: float = 0.25,
    min_test_margin: float = 0.10,
) -> dict[str, Any]:
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    torch.manual_seed(seed)
    suite = build_ats_transfer_suite(
        seed=seed,
        examples_per_domain=examples_per_domain,
    )
    train_examples = [
        example for example in suite["examples"] if example["split"] == "train"
    ]
    input_ids, labels = _build_answer_only_batch(
        train_examples,
        max_seq_len=max_seq_len,
    )
    tac_config = _ats_tac_config(
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_programs=n_programs,
    )
    controls = {
        "tac_answer_only": TACTransformerLM(tac_config),
        "vanilla_answer_only": VanillaTransformerLM(
            parameter_matched_baseline_config(tac_config)
        ),
    }
    training: dict[str, dict[str, Any]] = {}
    predictions: dict[str, list[dict[str, Any]]] = {}
    scores: dict[str, dict[str, Any]] = {}
    score_rows: dict[str, list[dict[str, Any]]] = {}
    for control_id, model in controls.items():
        training[control_id] = _train_answer_only_model(
            model,
            input_ids,
            labels,
            train_steps=train_steps,
            learning_rate=learning_rate,
        )
        predictions[control_id] = _predict_examples(
            model,
            suite["examples"],
            control_id=control_id,
            seed=seed,
            max_new_tokens=max_new_tokens,
        )
        rows = score_ats_transfer_predictions(suite["examples"], predictions[control_id])
        score_rows[control_id] = rows
        scores[control_id] = _scores_by_split_and_task(rows)

    tac_train = scores["tac_answer_only"]["train"]["mean_score"]
    tac_test = scores["tac_answer_only"]["test"]["mean_score"]
    vanilla_test = scores["vanilla_answer_only"]["test"]["mean_score"]
    checks = {
        "tac_loss_reduced": training["tac_answer_only"]["final_loss"]
        < training["tac_answer_only"]["initial_loss"],
        "vanilla_loss_reduced": training["vanilla_answer_only"]["final_loss"]
        < training["vanilla_answer_only"]["initial_loss"],
        "tac_train_score_passed": tac_train >= min_tac_train_score,
        "tac_test_score_passed": tac_test >= min_tac_test_score,
        "tac_beats_vanilla_test": (tac_test - vanilla_test) >= min_test_margin,
        "no_prompt_truncation": all(
            row["truncated_prompt_token_count"] == 0
            for control_rows in predictions.values()
            for row in control_rows
        ),
    }
    return {
        "schema": "ats_answer_copy_training.v1",
        "date": "2026-06-05",
        "seed": seed,
        "suite": {
            "examples_per_domain": examples_per_domain,
            "example_count": suite["example_count"],
            "train_domains": suite["train_domains"],
            "test_domains": suite["test_domains"],
            "task_ids": suite["task_ids"],
        },
        "controls": list(controls),
        "config": {
            "max_seq_len": max_seq_len,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "n_programs": n_programs,
            "max_new_tokens": max_new_tokens,
        },
        "training": training,
        "scores": scores,
        "score_rows": score_rows,
        "sample_predictions": {
            control_id: rows[:4] for control_id, rows in predictions.items()
        },
        "decision": {
            "status": (
                "ats_answer_copy_training_proved"
                if all(checks.values())
                else "blocked"
            ),
            "checks": checks,
            "thresholds": {
                "min_tac_train_score": min_tac_train_score,
                "min_tac_test_score": min_tac_test_score,
                "min_test_margin": min_test_margin,
            },
            "scope": (
                "This is an answer-only local training probe for the ATS scorer. "
                "It diagnoses whether a masked completion objective can train "
                "TAC and parameter-matched vanilla controls to emit exact ATS "
                "answers. It is not a replacement for the full prepared-JSONL "
                "Kaggle-scale comparison."
            ),
        },
    }


def format_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# ATS Answer-Copy Training",
        "",
        f"Decision: `{report['decision']['status']}`",
        "",
        "| Control | Train | Test |",
        "| --- | ---: | ---: |",
    ]
    for control_id in report["controls"]:
        train = report["scores"][control_id]["train"]["mean_score"]
        test = report["scores"][control_id]["test"]["mean_score"]
        lines.append(f"| {control_id} | {train:.4f} | {test:.4f} |")
    lines.extend(["", "## Scope", "", report["decision"]["scope"], ""])
    return "\n".join(lines)


def _ats_tac_config(
    *,
    max_seq_len: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    n_programs: int,
) -> TACConfig:
    return TACConfig(
        vocab_size=512,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_programs=n_programs,
        max_seq_len=max_seq_len,
        routing_type="base_semantic",
        routing_top_k=min(2, n_programs),
        memory_read_type="content_addressed",
        identity_attention_type="identity_first",
        memory_adapter_type="gated_residual",
        content_read_steps=2,
        content_read_gate_type="synthesis",
    )


def _build_answer_only_batch(
    examples: Sequence[Mapping[str, Any]],
    *,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [
        _encode_completion_row(
            str(example["prompt"]),
            str(example["answer"]),
            max_seq_len=max_seq_len,
        )
        for example in examples
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
    prompt_ids = phase_d_text_to_token_ids(prompt, vocab_size=512, append_eos=False)
    answer_ids = phase_d_text_to_token_ids(str(answer), vocab_size=512, append_eos=False)
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_seq_len:
        raise ValueError(
            f"ATS answer row exceeds max_seq_len: {len(input_ids)} > {max_seq_len}"
        )
    labels = [-100 for _ in input_ids]
    for offset, token_id in enumerate(answer_ids + [PHASE_D_EOS_TOKEN_ID]):
        label_position = len(prompt_ids) - 1 + offset
        if label_position < len(labels):
            labels[label_position] = token_id
    padding = max_seq_len - len(input_ids)
    return input_ids + [0] * padding, labels + [-100] * padding


def _train_answer_only_model(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    train_steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    initial_loss = _loss_value(model, input_ids, labels)
    for _ in range(train_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, labels=labels, collect_auxiliary=False)
        if output.loss is None:
            raise RuntimeError("model did not return answer-only training loss")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    final_loss = _loss_value(model, input_ids, labels)
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": initial_loss - final_loss,
        "train_steps": train_steps,
        "learning_rate": learning_rate,
        "supervision": "masked_next_token_answer_only",
    }


def _loss_value(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        output = model(input_ids, labels=labels, collect_auxiliary=False)
    if output.loss is None:
        raise RuntimeError("model did not return answer-only loss")
    return float(output.loss.detach())


def _predict_examples(
    model: torch.nn.Module,
    examples: Sequence[Mapping[str, Any]],
    *,
    control_id: str,
    seed: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    for example in examples:
        result = generate_phase_d_completion(
            model,
            str(example["prompt"]),
            max_new_tokens=max_new_tokens,
            device="cpu",
            precision="fp32",
        )
        raw_completion = str(result["completion"])
        rows.append(
            {
                "schema": "ats_answer_copy_prediction.v1",
                "example_id": str(example["id"]),
                "task_id": str(example["task_id"]),
                "family": str(example["family"]),
                "split": str(example["split"]),
                "domain": str(example["domain"]),
                "control_id": control_id,
                "seed": seed,
                "prediction": extract_phase_d_answer(
                    raw_completion,
                    mode="first_token",
                ),
                "raw_completion": raw_completion,
                "generated_token_count": int(result["generated_token_count"]),
                "prompt_token_count": int(result["prompt_token_count"]),
                "truncated_prompt_token_count": int(
                    result["truncated_prompt_token_count"]
                ),
            }
        )
    return rows


def _scores_by_split_and_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for split in sorted({str(row["split"]) for row in rows}):
        split_rows = [row for row in rows if str(row["split"]) == split]
        splits[split] = {
            "mean_score": (
                sum(float(row["primary_score"]) for row in split_rows)
                / len(split_rows)
                if split_rows
                else 0.0
            ),
            "tasks": {
                str(row["task_id"]): float(row["primary_score"])
                for row in split_rows
            },
        }
    return splits


if __name__ == "__main__":
    main()
