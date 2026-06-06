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

from experiments.benchmark_scratchpad_autoregressive_decoding import (
    DIGIT_ANSWERS,
    EVAL_MARKER,
    TRAIN_MARKERS,
    build_verified_scratchpad_decoding_prompt,
)
from tac_transformer import (
    AgenticPolicyController,
    SimulationBranch,
    TACConfig,
    TACTransformerLM,
    agentic_controller_supervised_loss,
    build_agentic_policy_features_from_tac_output,
)
from tac_transformer.phase_d_benchmarks import (
    PHASE_D_EOS_TOKEN_ID,
    PHASE_D_MIN_BYTE_VOCAB_SIZE,
    extract_phase_d_answer,
    generate_phase_d_completion,
    phase_d_text_to_token_ids,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/joint_tac_controller_training_2026_06_04")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove joint TAC decoder and AgenticPolicyController training."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--train-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--controller-loss-weight", type=float, default=0.5)
    parser.add_argument("--min-scratchpad-score", type=float, default=0.95)
    parser.add_argument("--max-no-scratchpad-score", type=float, default=0.20)
    parser.add_argument("--min-controller-score", type=float, default=0.95)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_joint_tac_controller_training_probe(
        seed=args.seed,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        controller_loss_weight=args.controller_loss_weight,
        min_scratchpad_score=args.min_scratchpad_score,
        max_no_scratchpad_score=args.max_no_scratchpad_score,
        min_controller_score=args.min_controller_score,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "joint_tac_controller_training.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def run_joint_tac_controller_training_probe(
    *,
    seed: int = 5,
    train_steps: int = 50,
    learning_rate: float = 0.01,
    controller_loss_weight: float = 0.5,
    min_scratchpad_score: float = 0.95,
    max_no_scratchpad_score: float = 0.20,
    min_controller_score: float = 0.95,
) -> dict[str, Any]:
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if controller_loss_weight < 0.0:
        raise ValueError("controller_loss_weight must be non-negative")
    torch.manual_seed(seed)
    max_seq_len = 80
    branches = _training_branches()
    input_ids, labels = _build_training_batch(max_seq_len=max_seq_len)
    targets = _controller_targets(input_ids.shape[0])

    model = TACTransformerLM(_joint_training_config(max_seq_len=max_seq_len))
    controller = AgenticPolicyController()
    initial_tac_params = _parameter_snapshot(model)
    initial_controller_params = _parameter_snapshot(controller)

    initial = _joint_metrics(model, controller, input_ids, labels, targets, branches)
    tac_policy_grad_abs_sum = _tac_policy_gradient_abs_sum(
        model,
        controller,
        input_ids,
        targets,
        branches,
    )

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(controller.parameters()),
        lr=learning_rate,
    )
    for _ in range(train_steps):
        model.train()
        controller.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, labels=labels, collect_auxiliary=True)
        if output.loss is None:
            raise RuntimeError("TACTransformerLM did not return decoder loss")
        controller_losses = _controller_losses_from_output(
            output,
            controller,
            targets,
            branches,
        )
        joint_loss = output.loss + controller_loss_weight * controller_losses["loss"]
        joint_loss.backward()
        optimizer.step()

    final = _joint_metrics(model, controller, input_ids, labels, targets, branches)
    parameter_updates = {
        "tac_max_abs_delta": _max_parameter_delta(model, initial_tac_params),
        "controller_max_abs_delta": _max_parameter_delta(
            controller,
            initial_controller_params,
        ),
    }
    scratchpad_predictions = _evaluate_scratchpad_generation(
        model,
        marker=EVAL_MARKER,
        answers=DIGIT_ANSWERS,
    )
    no_scratchpad_predictions = _evaluate_no_scratchpad_generation(
        model,
        marker=EVAL_MARKER,
        expected_answers=DIGIT_ANSWERS,
    )
    scratchpad_score = _prediction_score(scratchpad_predictions)
    no_scratchpad_score = _prediction_score(no_scratchpad_predictions)
    checks = {
        "decoder_loss_reduced": final["decoder_loss"] < initial["decoder_loss"],
        "controller_loss_reduced": final["controller_loss"] < initial["controller_loss"],
        "scratchpad_generation_passed": scratchpad_score >= min_scratchpad_score,
        "no_scratchpad_control_low": no_scratchpad_score <= max_no_scratchpad_score,
        "controller_scratchpad_passed": final["policy"]["scratchpad_policy_score"]
        >= min_controller_score,
        "controller_simulation_passed": final["policy"]["simulation_policy_score"]
        >= min_controller_score,
        "controller_teaching_passed": final["policy"]["teaching_policy_score"]
        >= min_controller_score,
        "tac_parameters_updated": parameter_updates["tac_max_abs_delta"] > 0.0,
        "controller_parameters_updated": parameter_updates["controller_max_abs_delta"] > 0.0,
        "policy_loss_reaches_tac": tac_policy_grad_abs_sum > 0.0,
    }
    return {
        "schema": "joint_tac_controller_training.v1",
        "date": "2026-06-04",
        "seed": seed,
        "train_steps": train_steps,
        "train_examples": int(input_ids.shape[0]),
        "training": {
            "initial_decoder_loss": initial["decoder_loss"],
            "final_decoder_loss": final["decoder_loss"],
            "decoder_loss_reduction": initial["decoder_loss"] - final["decoder_loss"],
            "initial_controller_loss": initial["controller_loss"],
            "final_controller_loss": final["controller_loss"],
            "controller_loss_reduction": (
                initial["controller_loss"] - final["controller_loss"]
            ),
            "learning_rate": learning_rate,
            "controller_loss_weight": controller_loss_weight,
            "mode": "joint_tac_and_controller_optimizer",
        },
        "scores": {
            "scratchpad_score": scratchpad_score,
            "no_scratchpad_score": no_scratchpad_score,
            "scratchpad_control_margin": scratchpad_score - no_scratchpad_score,
            "controller_scratchpad_score": final["policy"]["scratchpad_policy_score"],
            "controller_simulation_score": final["policy"]["simulation_policy_score"],
            "controller_teaching_score": final["policy"]["teaching_policy_score"],
        },
        "gradient_flow": {
            "tac_policy_grad_abs_sum": tac_policy_grad_abs_sum,
        },
        "parameter_updates": parameter_updates,
        "sample_predictions": scratchpad_predictions[:3] + no_scratchpad_predictions[:1],
        "decision": {
            "status": (
                "joint_tac_controller_training_proved"
                if all(checks.values())
                else "blocked"
            ),
            "checks": checks,
            "thresholds": {
                "min_scratchpad_score": min_scratchpad_score,
                "max_no_scratchpad_score": max_no_scratchpad_score,
                "min_controller_score": min_controller_score,
            },
            "scope": (
                "This gate jointly optimizes TAC decoder parameters and "
                "AgenticPolicyController parameters with a single optimizer over "
                "masked decoder loss plus supervised controller action loss. It "
                "proves local joint optimization, nonzero policy-gradient flow "
                "into TAC, parameter updates on both modules, scratchpad-based "
                "autoregressive decoding, and learned controller actions. It does "
                "not prove large-scale RL rollout training or external Phase B/D "
                "benchmark improvement."
            ),
        },
    }


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Joint TAC Controller Training",
        "",
        f"Decision: `{report['decision']['status']}`",
        "",
        "## Losses",
        "",
        f"Decoder loss: `{report['training']['initial_decoder_loss']:.6f}` -> `{report['training']['final_decoder_loss']:.6f}`",
        f"Controller loss: `{report['training']['initial_controller_loss']:.6f}` -> `{report['training']['final_controller_loss']:.6f}`",
        "",
        "## Scores",
        "",
        f"Scratchpad score: `{report['scores']['scratchpad_score']:.4f}`",
        f"No-scratchpad score: `{report['scores']['no_scratchpad_score']:.4f}`",
        f"Controller scratchpad score: `{report['scores']['controller_scratchpad_score']:.4f}`",
        f"Controller simulation score: `{report['scores']['controller_simulation_score']:.4f}`",
        f"Controller teaching score: `{report['scores']['controller_teaching_score']:.4f}`",
        "",
        "## Updates",
        "",
        f"TAC max parameter delta: `{report['parameter_updates']['tac_max_abs_delta']:.8f}`",
        f"Controller max parameter delta: `{report['parameter_updates']['controller_max_abs_delta']:.8f}`",
        f"TAC policy-gradient abs sum: `{report['gradient_flow']['tac_policy_grad_abs_sum']:.8f}`",
        "",
        "## Scope",
        "",
        report["decision"]["scope"],
        "",
    ]
    return "\n".join(lines)


def _joint_training_config(*, max_seq_len: int) -> TACConfig:
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
        detach_identity_state=False,
    )


def _build_training_batch(*, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [
        (
            build_verified_scratchpad_decoding_prompt(answer=answer, marker=marker),
            answer,
        )
        for marker in TRAIN_MARKERS
        for answer in DIGIT_ANSWERS
    ]
    encoded = [
        _encode_completion_row(prompt, answer, max_seq_len=max_seq_len)
        for prompt, answer in rows
    ]
    return (
        torch.tensor([row[0] for row in encoded], dtype=torch.long),
        torch.tensor([row[1] for row in encoded], dtype=torch.long),
    )


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
        raise ValueError("joint training row exceeds max_seq_len")
    labels = [-100 for _ in input_ids]
    for offset, token_id in enumerate(answer_ids + [PHASE_D_EOS_TOKEN_ID]):
        label_position = len(prompt_ids) - 1 + offset
        if label_position < len(labels):
            labels[label_position] = token_id
    padding = max_seq_len - len(input_ids)
    return input_ids + [0] * padding, labels + [-100] * padding


def _training_branches() -> tuple[SimulationBranch, ...]:
    return (
        SimulationBranch("safe", ("read_scratchpad", "answer"), 0.75, 0.1, risk=0.0),
        SimulationBranch("deep", ("think", "answer"), 0.85, 0.8, risk=0.1),
        SimulationBranch("risky", ("guess",), 0.99, 0.05, risk=0.9),
    )


def _controller_targets(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "scratchpad_targets": torch.tensor(
            [[1.0, 1.0, 0.0] for _ in range(batch_size)],
            dtype=torch.float32,
        ),
        "simulation_targets": torch.zeros(batch_size, dtype=torch.long),
        "process_targets": torch.tensor(
            [[0, 1, 2, 3] for _ in range(batch_size)],
            dtype=torch.long,
        ),
        "verifier_scores": torch.tensor(
            [[1.0, 1.0, 0.75, 1.0] for _ in range(batch_size)],
            dtype=torch.float32,
        ),
    }


def _controller_losses_from_output(
    output: Any,
    controller: AgenticPolicyController,
    targets: dict[str, torch.Tensor],
    branches: tuple[SimulationBranch, ...],
) -> dict[str, torch.Tensor]:
    features = build_agentic_policy_features_from_tac_output(
        output,
        branches=branches,
        scratchpad_slots=3,
    )
    controller_outputs = controller(
        scratchpad_features=features.scratchpad_features,
        simulation_features=features.simulation_features,
        context_features=features.context_features,
    )
    return agentic_controller_supervised_loss(
        controller_outputs,
        scratchpad_targets=targets["scratchpad_targets"],
        simulation_targets=targets["simulation_targets"],
        process_targets=targets["process_targets"],
        verifier_scores=targets["verifier_scores"],
    )


def _joint_metrics(
    model: TACTransformerLM,
    controller: AgenticPolicyController,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    targets: dict[str, torch.Tensor],
    branches: tuple[SimulationBranch, ...],
) -> dict[str, Any]:
    model.eval()
    controller.eval()
    with torch.no_grad():
        output = model(input_ids, labels=labels, collect_auxiliary=True)
        if output.loss is None:
            raise RuntimeError("TACTransformerLM did not return decoder loss")
        controller_losses = _controller_losses_from_output(
            output,
            controller,
            targets,
            branches,
        )
        policy = _score_policy_from_output(output, controller, targets, branches)
    return {
        "decoder_loss": float(output.loss.detach()),
        "controller_loss": float(controller_losses["loss"].detach()),
        "policy": policy,
    }


def _score_policy_from_output(
    output: Any,
    controller: AgenticPolicyController,
    targets: dict[str, torch.Tensor],
    branches: tuple[SimulationBranch, ...],
) -> dict[str, float]:
    features = build_agentic_policy_features_from_tac_output(
        output,
        branches=branches,
        scratchpad_slots=3,
    )
    controller_outputs = controller(
        scratchpad_features=features.scratchpad_features,
        simulation_features=features.simulation_features,
        context_features=features.context_features,
    )
    scratchpad_predictions = (
        controller_outputs["scratchpad_logits"].sigmoid() >= 0.5
    ).to(dtype=targets["scratchpad_targets"].dtype)
    exact_scratchpad = (
        scratchpad_predictions == targets["scratchpad_targets"]
    ).all(dim=-1)
    simulation_predictions = controller_outputs["simulation_logits"].argmax(dim=-1)
    process_predictions = controller_outputs["process_logits"].argmax(dim=-1)
    exact_process = (process_predictions == targets["process_targets"]).all(dim=-1)
    return {
        "scratchpad_policy_score": float(exact_scratchpad.float().mean()),
        "simulation_policy_score": float(
            (simulation_predictions == targets["simulation_targets"]).float().mean()
        ),
        "teaching_policy_score": float(exact_process.float().mean()),
    }


def _tac_policy_gradient_abs_sum(
    model: TACTransformerLM,
    controller: AgenticPolicyController,
    input_ids: torch.Tensor,
    targets: dict[str, torch.Tensor],
    branches: tuple[SimulationBranch, ...],
) -> float:
    model.zero_grad(set_to_none=True)
    controller.zero_grad(set_to_none=True)
    output = model(input_ids, collect_auxiliary=True)
    losses = _controller_losses_from_output(output, controller, targets, branches)
    losses["loss"].backward()
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().abs().sum())
    model.zero_grad(set_to_none=True)
    controller.zero_grad(set_to_none=True)
    return total


def _evaluate_scratchpad_generation(
    model: TACTransformerLM,
    *,
    marker: str,
    answers: tuple[str, ...],
) -> list[dict[str, Any]]:
    return [
        _prediction_row(
            model,
            prompt=build_verified_scratchpad_decoding_prompt(
                answer=answer,
                marker=marker,
            ),
            expected=answer,
            kind="verified_scratchpad",
        )
        for answer in answers
    ]


def _evaluate_no_scratchpad_generation(
    model: TACTransformerLM,
    *,
    marker: str,
    expected_answers: tuple[str, ...],
) -> list[dict[str, Any]]:
    prompt = f"Task {marker}: answer from scratchpad.\nAnswer:"
    return [
        _prediction_row(model, prompt=prompt, expected=answer, kind="no_scratchpad")
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


def _parameter_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


def _max_parameter_delta(
    module: torch.nn.Module,
    before: dict[str, torch.Tensor],
) -> float:
    deltas = []
    for name, parameter in module.named_parameters():
        if name in before:
            deltas.append(float((parameter.detach() - before[name]).abs().max()))
    return max(deltas) if deltas else 0.0


if __name__ == "__main__":
    main()
