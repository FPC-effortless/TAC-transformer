from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    AgenticPolicyController,
    AgenticPolicyControllerConfig,
    AgenticScratchpadState,
    ScratchpadItem,
    TACConfig,
    TACTransformerLM,
    apply_agentic_scratchpad_transition,
)
from tac_transformer.phase_d_benchmarks import (
    PHASE_D_TASK_IDS,
    build_phase_d_task_suite,
    phase_d_text_to_token_ids,
    run_phase_d_scratchpad_state_predictions,
    score_phase_d_predictions,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/live_phase_d_scratchpad_policy_2026_06_05")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Connect live TAC Phase D prompt features to AgenticPolicyController "
            "scratchpad commits and score scratchpad-vs-empty controls."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--examples-per-task", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-scratchpad-score", type=float, default=0.95)
    parser.add_argument("--min-score-margin", type=float, default=0.50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_live_phase_d_scratchpad_policy_probe(
        seed=args.seed,
        examples_per_task=args.examples_per_task,
        context_length=args.context_length,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        min_scratchpad_score=args.min_scratchpad_score,
        min_score_margin=args.min_score_margin,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "live_phase_d_scratchpad_policy.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


def run_live_phase_d_scratchpad_policy_probe(
    *,
    seed: int = 23,
    examples_per_task: int = 2,
    context_length: int = 128,
    train_steps: int = 120,
    learning_rate: float = 0.05,
    min_scratchpad_score: float = 0.95,
    min_score_margin: float = 0.50,
) -> dict[str, Any]:
    if examples_per_task <= 0:
        raise ValueError("examples_per_task must be positive")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")

    torch.manual_seed(seed)
    suite = build_phase_d_task_suite(
        seed=seed,
        examples_per_task=examples_per_task,
        context_length=context_length,
    )
    examples = list(suite["examples"])
    candidates = _build_phase_d_scratchpad_candidates(examples)

    model = TACTransformerLM(
        TACConfig(
            vocab_size=260,
            d_model=16,
            n_heads=4,
            n_layers=1,
            n_programs=6,
            max_seq_len=max(64, min(256, int(context_length) + 64)),
            detach_identity_state=False,
        )
    )
    features, feature_report = _build_live_candidate_features(
        model,
        examples,
        candidates,
    )
    targets = _candidate_targets(candidates)

    controller = AgenticPolicyController(
        AgenticPolicyControllerConfig(
            scratchpad_feature_dim=features.shape[-1],
            simulation_feature_dim=5,
            context_feature_dim=4,
            hidden_dim=32,
        )
    )

    gradient_probe = _measure_live_tac_gradient_flow(
        model,
        controller,
        features,
        targets,
    )
    detached_features = features.detach()
    optimizer = torch.optim.AdamW(controller.parameters(), lr=learning_rate)
    initial_loss = _scratchpad_loss(controller, detached_features, targets)
    for _ in range(int(train_steps)):
        controller.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _scratchpad_loss_tensor(controller, detached_features, targets)
        loss.backward()
        optimizer.step()
        if _selection_score(controller, detached_features, targets) >= 1.0:
            break
    final_loss = _scratchpad_loss(controller, detached_features, targets)

    scratchpad_states, transition_reports = _build_policy_scratchpad_states(
        controller,
        detached_features,
        examples,
        candidates,
    )
    scratchpad_predictions = run_phase_d_scratchpad_state_predictions(
        examples=examples,
        scratchpad_by_example=scratchpad_states,
        control_id="live_phase_d_scratchpad_policy",
        seed=seed,
    )
    empty_states = {
        str(example["id"]): AgenticScratchpadState.empty(budget=1)
        for example in examples
    }
    no_scratchpad_predictions = run_phase_d_scratchpad_state_predictions(
        examples=examples,
        scratchpad_by_example=empty_states,
        control_id="no_scratchpad",
        seed=seed,
    )
    scratchpad_scores = score_phase_d_predictions(
        examples,
        scratchpad_predictions["rows"],
        control_id="live_phase_d_scratchpad_policy",
        seed=seed,
    )
    no_scratchpad_scores = score_phase_d_predictions(
        examples,
        no_scratchpad_predictions["rows"],
        control_id="no_scratchpad",
        seed=seed,
    )
    scratchpad_mean = _mean_primary_score(scratchpad_scores)
    no_scratchpad_mean = _mean_primary_score(no_scratchpad_scores)
    unverified_payloads = {
        row["wrong_payload"]
        for candidate_set in candidates.values()
        for row in candidate_set
        if not row["is_correct"]
    }
    unverified_prompt_leak_count = sum(
        1
        for row in scratchpad_predictions["rows"]
        for payload in unverified_payloads
        if payload and payload in str(row.get("augmented_prompt", ""))
    )
    contamination_rate = mean(
        float(report["hypothesis_contamination_rate"])
        for report in transition_reports
    )
    selection_score = _selection_score(controller, detached_features, targets)
    score_margin = scratchpad_mean - no_scratchpad_mean
    checks = {
        "loss_reduced": final_loss < initial_loss,
        "live_tac_gradient_flow": gradient_probe["token_embedding_grad_abs_sum"] > 0.0,
        "scratchpad_selection_learned": selection_score >= 1.0,
        "scratchpad_score_passed": scratchpad_mean >= min_scratchpad_score,
        "scratchpad_beats_no_scratchpad": score_margin >= min_score_margin,
        "unverified_payloads_excluded": unverified_prompt_leak_count == 0,
        "hypothesis_contamination_blocked": contamination_rate == 0.0,
    }
    return {
        "schema": "live_phase_d_scratchpad_policy.v1",
        "date": "2026-06-05",
        "seed": seed,
        "examples_per_task": examples_per_task,
        "context_length": context_length,
        "task_ids": list(PHASE_D_TASK_IDS),
        "example_count": len(examples),
        "training": {
            "train_steps_requested": int(train_steps),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction": initial_loss - final_loss,
        },
        "features": feature_report,
        "gradient_flow": gradient_probe,
        "policy": {
            "scratchpad_selection_score": selection_score,
            "candidate_count_per_example": len(next(iter(candidates.values()))),
        },
        "scratchpad": {
            "transition_count": len(transition_reports),
            "committed_answer_count": sum(
                1 for state in scratchpad_states.values() if state.items
            ),
            "hypothesis_contamination_rate": contamination_rate,
            "unverified_prompt_leak_count": unverified_prompt_leak_count,
        },
        "scores": {
            "scratchpad_mean_score": scratchpad_mean,
            "no_scratchpad_mean_score": no_scratchpad_mean,
            "score_margin": score_margin,
            "scratchpad_rows": scratchpad_scores,
            "no_scratchpad_rows": no_scratchpad_scores,
        },
        "decision": {
            "status": (
                "live_phase_d_scratchpad_policy_proved"
                if all(checks.values())
                else "blocked"
            ),
            "checks": checks,
            "thresholds": {
                "min_scratchpad_score": min_scratchpad_score,
                "min_score_margin": min_score_margin,
            },
            "scope": (
                "This is a local Phase D wiring gate. It proves live TAC-derived "
                "Phase D candidate features can train AgenticPolicyController "
                "scratchpad logits, pass through verifier-gated scratchpad state, "
                "and beat an empty-scratchpad Phase D control. It does not prove "
                "external-scale rollout RL or held-out OOD capability advantage."
            ),
        },
    }


def _build_phase_d_scratchpad_candidates(
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    for index, example in enumerate(examples):
        example_id = str(example["id"])
        answer = str(example["answer"])
        wrong_payload = f"wrong_{index}_{answer[::-1]}"
        correct = {
            "item_id": f"{example_id}:answer",
            "kind": "answer",
            "payload": answer,
            "wrong_payload": "",
            "is_correct": True,
        }
        wrong = {
            "item_id": f"{example_id}:wrong",
            "kind": "simulation",
            "payload": wrong_payload,
            "wrong_payload": wrong_payload,
            "is_correct": False,
        }
        candidates[example_id] = [correct, wrong] if index % 2 == 0 else [wrong, correct]
    return candidates


def _build_live_candidate_features(
    model: TACTransformerLM,
    examples: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    rows = []
    live_abs_sum = 0.0
    prompt_overlap_count = 0
    for example in examples:
        example_id = str(example["id"])
        prompt = str(example.get("prompt", ""))
        candidate_features = []
        for slot, candidate in enumerate(candidates[example_id]):
            payload = str(candidate["payload"])
            encoded = _encode_phase_d_candidate(model, prompt, payload)
            live = _live_feature_vector(encoded)
            live_abs_sum += float(live.abs().sum().detach())
            payload_in_prompt = 1.0 if payload in prompt else 0.0
            prompt_overlap_count += int(payload_in_prompt > 0.0)
            slot_fraction = slot / max(len(candidates[example_id]) - 1, 1)
            feature = torch.stack(
                [
                    torch.tensor(slot_fraction, dtype=live.dtype),
                    live[0],
                    live[1],
                    live[2],
                    live[3],
                    live[4],
                    torch.tensor(payload_in_prompt, dtype=live.dtype),
                    torch.tensor(min(len(payload) / 32.0, 1.0), dtype=live.dtype),
                ]
            )
            candidate_features.append(feature)
        rows.append(torch.stack(candidate_features, dim=0))
    features = torch.stack(rows, dim=0)
    return features, {
        "scratchpad_feature_shape": list(features.shape),
        "feature_dim": int(features.shape[-1]),
        "live_tac_feature_abs_sum": live_abs_sum,
        "candidate_prompt_overlap_count": prompt_overlap_count,
    }


def _encode_phase_d_candidate(
    model: TACTransformerLM,
    prompt: str,
    payload: str,
) -> Any:
    config = model.config
    text = f"{prompt}\nCandidate scratchpad answer: {payload}\nCommit?"
    token_ids = phase_d_text_to_token_ids(text, vocab_size=config.vocab_size)
    if not token_ids:
        token_ids = [3]
    window = token_ids[-int(config.max_seq_len) :]
    input_ids = torch.tensor([window], dtype=torch.long)
    return model(input_ids, collect_auxiliary=True)


def _live_feature_vector(tac_output: Any) -> torch.Tensor:
    hidden = tac_output.hidden_states
    if hidden is None or hidden.ndim != 3:
        raise ValueError("TAC output must include hidden_states [batch, tokens, d_model]")
    final_hidden = hidden[0, -1, :]
    hidden_scale = max(final_hidden.numel(), 1) ** 0.5
    hidden_norm = final_hidden.norm() / hidden_scale
    hidden_mean = final_hidden.mean()
    aux = tac_output.aux
    route_fraction = (
        aux.token_selected_program_mask[0, -1, :].float().mean()
        if aux.token_selected_program_mask is not None
        else final_hidden.new_tensor(0.0)
    )
    activation_mean = (
        aux.token_program_activations[0, -1, :].mean()
        if aux.token_program_activations is not None
        else final_hidden.new_tensor(0.0)
    )
    identity_state = tac_output.identity_states[-1] if tac_output.identity_states else None
    memory_norm = (
        identity_state.program_memory[0].norm(dim=-1).mean()
        / (identity_state.program_memory.shape[-1] ** 0.5)
        if identity_state is not None
        else final_hidden.new_tensor(0.0)
    )
    return torch.stack(
        [
            hidden_norm,
            hidden_mean,
            route_fraction,
            activation_mean,
            memory_norm,
        ]
    )


def _candidate_targets(
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
) -> torch.Tensor:
    return torch.tensor(
        [
            [1.0 if candidate["is_correct"] else 0.0 for candidate in candidate_set]
            for candidate_set in candidates.values()
        ],
        dtype=torch.float32,
    )


def _measure_live_tac_gradient_flow(
    model: TACTransformerLM,
    controller: AgenticPolicyController,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    controller.zero_grad(set_to_none=True)
    loss = _scratchpad_loss_tensor(controller, features, targets)
    loss.backward(retain_graph=True)
    grad = model.token_embedding.weight.grad
    grad_abs_sum = 0.0 if grad is None else float(grad.abs().sum().detach())
    model.zero_grad(set_to_none=True)
    controller.zero_grad(set_to_none=True)
    return {"token_embedding_grad_abs_sum": grad_abs_sum}


def _scratchpad_loss(
    controller: AgenticPolicyController,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    return float(_scratchpad_loss_tensor(controller, features, targets).detach())


def _scratchpad_loss_tensor(
    controller: AgenticPolicyController,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    batch_size = features.shape[0]
    outputs = controller(
        scratchpad_features=features,
        simulation_features=torch.zeros(batch_size, 3, 5, dtype=features.dtype),
        context_features=features.mean(dim=1)[:, :4],
    )
    return F.binary_cross_entropy_with_logits(outputs["scratchpad_logits"], targets)


def _selection_score(
    controller: AgenticPolicyController,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    controller.eval()
    with torch.no_grad():
        batch_size = features.shape[0]
        outputs = controller(
            scratchpad_features=features,
            simulation_features=torch.zeros(batch_size, 3, 5, dtype=features.dtype),
            context_features=features.mean(dim=1)[:, :4],
        )
        selected = (outputs["scratchpad_logits"].sigmoid() >= 0.5).to(targets.dtype)
    return float((selected == targets).all(dim=-1).float().mean().item())


def _build_policy_scratchpad_states(
    controller: AgenticPolicyController,
    features: torch.Tensor,
    examples: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, AgenticScratchpadState], list[dict[str, Any]]]:
    controller.eval()
    batch_size = features.shape[0]
    with torch.no_grad():
        outputs = controller(
            scratchpad_features=features,
            simulation_features=torch.zeros(batch_size, 3, 5, dtype=features.dtype),
            context_features=features.mean(dim=1)[:, :4],
        )
    states: dict[str, AgenticScratchpadState] = {}
    reports: list[dict[str, Any]] = []
    for row_index, example in enumerate(examples):
        example_id = str(example["id"])
        candidate_rows = list(candidates[example_id])
        items = [
            ScratchpadItem(
                item_id=str(candidate["item_id"]),
                kind=str(candidate["kind"]),
                payload=str(candidate["payload"]),
                utility=1.0 if candidate["is_correct"] else 0.25,
                confidence=1.0,
                imagined=not bool(candidate["is_correct"]),
                verified=False,
            )
            for candidate in candidate_rows
        ]
        supported_ids = {
            str(candidate["item_id"])
            for candidate in candidate_rows
            if candidate["is_correct"]
        }
        state, report = apply_agentic_scratchpad_transition(
            AgenticScratchpadState.empty(budget=1),
            items,
            commit_logits=outputs["scratchpad_logits"][row_index].detach(),
            verifier_supported_ids=supported_ids,
        )
        states[example_id] = state
        reports.append(report)
    return states, reports


def _mean_primary_score(rows: Iterable[Mapping[str, Any]]) -> float:
    values = [float(row["primary_score"]) for row in rows]
    return mean(values) if values else 0.0


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Live Phase D Scratchpad Policy",
        "",
        f"Decision: `{report['decision']['status']}`",
        "",
        "## Scores",
        "",
        f"Scratchpad mean score: `{report['scores']['scratchpad_mean_score']:.4f}`",
        f"No-scratchpad mean score: `{report['scores']['no_scratchpad_mean_score']:.4f}`",
        f"Score margin: `{report['scores']['score_margin']:.4f}`",
        "",
        "## Policy",
        "",
        f"Scratchpad selection score: `{report['policy']['scratchpad_selection_score']:.4f}`",
        (
            "Live TAC token embedding grad abs sum: "
            f"`{report['gradient_flow']['token_embedding_grad_abs_sum']:.6f}`"
        ),
        "",
        "## Scratchpad Safety",
        "",
        (
            "Hypothesis contamination rate: "
            f"`{report['scratchpad']['hypothesis_contamination_rate']:.4f}`"
        ),
        (
            "Unverified prompt leak count: "
            f"`{report['scratchpad']['unverified_prompt_leak_count']}`"
        ),
        "",
        "## Scope",
        "",
        report["decision"]["scope"],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
