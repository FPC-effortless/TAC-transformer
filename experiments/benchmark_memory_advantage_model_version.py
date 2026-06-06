from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    TACTransformerLM,
    VanillaTransformerLM,
    best_tac_config,
    memory_advantage_config,
    memory_advantage_training_kwargs,
)
from tac_transformer.training import count_parameters, parameter_matched_baseline_config


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/memory_advantage_model_version_2026_06_05")

PRIMARY_QUESTION = (
    "Does persistent computational identity create a measurable long-horizon memory "
    "advantage over transformer plus retrieval baselines under equal resource constraints?"
)

TARGET_GRAPHS = [
    "Context Tokens Required vs Task Success",
    "Days Since Instruction vs Accuracy",
]

RESEARCH_ADVANTAGES = [
    {
        "id": "modern_backbone",
        "mechanisms": ["RMSNorm", "SwiGLU", "RoPE", "linear_expert"],
        "reason": "Retains the strongest promoted TAC backbone found in the harder architecture matrix.",
    },
    {
        "id": "learnable_semantic_routing",
        "mechanisms": ["base_semantic", "top_k_2", "load_balance_0p05"],
        "reason": "Keeps the learnable routing path instead of fixed identity-rule dispatch.",
    },
    {
        "id": "persistent_trainable_memory",
        "mechanisms": ["program_conditioned_writes", "creb_allocation", "memory_separation"],
        "reason": "Makes persistent identity memory a trainable subsystem with anti-collapse pressure.",
    },
    {
        "id": "content_addressed_synthesis_reads",
        "mechanisms": ["content_addressed", "two_step_read", "synthesis_gate", "reconsolidation"],
        "reason": "Uses the promoted long-context memory retrieval and rewrite path.",
    },
    {
        "id": "identity_first_memory_path",
        "mechanisms": ["identity_first_attention", "gated_residual_memory_adapter"],
        "reason": "Preserves the memory-to-token path that stayed strongest after cheaper identity ablations.",
    },
    {
        "id": "coalition_memory_context",
        "mechanisms": ["program_memory_graph"],
        "reason": "Adds the opt-in cue-chain candidate needed for multi-session memory consistency probes.",
    },
    {
        "id": "optimizer_health",
        "mechanisms": ["fp32", "nonzero_gradient_gate", "fail_fast"],
        "reason": "Carries the TAC-187 stability repair into this candidate before external validation.",
    },
]

EQUAL_RESOURCE_CONTROLS = [
    {
        "id": "parameter_matched_vanilla_window",
        "model": "TransformerLM",
        "resource_contract": "Match parameter count, training tokens, eval examples, and context-token budget curve.",
        "tests": ["no_retrieval_window", "fixed_context_limit"],
    },
    {
        "id": "parameter_matched_vanilla_retrieval",
        "model": "TransformerLM + retrieval",
        "resource_contract": "Use the same train/eval records and charge retrieval context tokens in the budget.",
        "tests": ["top_k_sparse_reminders", "distractor_retrieval_noise"],
    },
    {
        "id": "parameter_matched_vanilla_memory_db",
        "model": "TransformerLM + memory database",
        "resource_contract": "Same parameter budget plus explicit accounting for memory read/write context.",
        "tests": ["identity_keyed_memory", "cross_session_memory_collision"],
    },
    {
        "id": "current_best_tac",
        "model": "best_tac_config",
        "resource_contract": "Use the current promoted TAC preset as an internal ablation baseline.",
        "tests": ["persistent_state_without_tac188_candidate_extras"],
    },
]

LONG_HORIZON_EVALUATION_AXES = [
    {
        "id": "sparse_reminder_gap",
        "description": "Instruction appears once, later sessions contain sparse reminders or no reminder.",
        "metric": "accuracy_by_session_gap",
    },
    {
        "id": "identity_continuity",
        "description": "Multiple identities require different latent computations over shared surface prompts.",
        "metric": "carried_state_advantage_over_reset",
    },
    {
        "id": "multi_session_reasoning",
        "description": "Later answers require composing facts introduced across prior sessions.",
        "metric": "multi_hop_exact_match",
    },
    {
        "id": "context_efficiency",
        "description": "Score success while progressively reducing context tokens supplied to the model.",
        "metric": "context_tokens_required_for_target_success",
    },
]

ADVERSARIAL_STRESS_AXES = [
    "identity_collision",
    "distractor_memory_noise",
    "shuffled_identity_state",
    "reset_state",
    "long_gap_decay",
    "task_family_shift",
]


def run_memory_advantage_model_version(
    *,
    vocab_size: int = 512,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    n_programs: int | None = None,
    max_seq_len: int = 256,
    content_store_size: int | None = None,
    memory_allocation_k: int | None = None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "max_seq_len": max_seq_len,
    }
    if n_programs is not None:
        overrides["n_programs"] = n_programs
    if content_store_size is not None:
        overrides["content_store_size"] = content_store_size
    if memory_allocation_k is not None:
        overrides["memory_allocation_k"] = memory_allocation_k

    config = memory_advantage_config(vocab_size=vocab_size, **overrides)
    tac_model = TACTransformerLM(config)
    vanilla_config = parameter_matched_baseline_config(config)
    vanilla_model = VanillaTransformerLM(vanilla_config)
    best_config = best_tac_config(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_programs=config.n_programs,
        max_seq_len=max_seq_len,
        content_store_size=config.content_store_size,
    )
    best_model = TACTransformerLM(best_config)
    training = memory_advantage_training_kwargs()

    return {
        "schema": "memory_advantage_model_version.v1",
        "primary_question": PRIMARY_QUESTION,
        "model_version": {
            "preset": "memory_advantage",
            "ticket": "TAC-188",
            "role": "opt_in_candidate",
            "config": asdict(config),
            "training_defaults": training,
        },
        "research_advantages_enabled": RESEARCH_ADVANTAGES,
        "equal_resource_controls": EQUAL_RESOURCE_CONTROLS,
        "long_horizon_evaluation_axes": LONG_HORIZON_EVALUATION_AXES,
        "adversarial_stress_axes": ADVERSARIAL_STRESS_AXES,
        "target_graphs": TARGET_GRAPHS,
        "parameter_counts": {
            "memory_advantage_tac": count_parameters(tac_model),
            "parameter_matched_vanilla": count_parameters(vanilla_model),
            "current_best_tac": count_parameters(best_model),
        },
        "resource_contract": {
            "same_training_tokens": True,
            "same_eval_examples": True,
            "same_or_accounted_context_tokens": True,
            "same_parameter_budget": True,
            "same_random_seed_matrix": True,
            "charge_retrieval_context": True,
            "report_wall_clock_and_tokens_per_second": True,
        },
        "promotion_gate": {
            "minimum_tac_advantage_over_best_control": 0.10,
            "minimum_long_gap_accuracy": 0.75,
            "maximum_reset_state_accuracy_for_identity_tasks": 0.35,
            "require_adversarial_stress_report": True,
            "require_context_efficiency_curve": True,
        },
        "boundary": {
            "claims_trained_checkpoint_advantage": False,
            "claims_real_world_memory_advantage": False,
            "reason": (
                "This artifact creates the model version and benchmark contract. "
                "It does not replace equal-resource trained checkpoint evidence."
            ),
        },
        "decision": {
            "status": "memory_advantage_model_version_ready",
            "next_step": "Train and evaluate the preset against the equal-resource controls.",
        },
    }


def format_memory_advantage_markdown(result: dict[str, Any]) -> str:
    controls = "\n".join(
        f"- {control['id']}: {control['model']}"
        for control in result["equal_resource_controls"]
    )
    axes = "\n".join(
        f"- {axis['id']}: {axis['metric']}"
        for axis in result["long_horizon_evaluation_axes"]
    )
    graphs = "\n".join(f"- {graph}" for graph in result["target_graphs"])
    counts = result["parameter_counts"]
    return f"""# Memory Advantage Model Version

## Primary Question

{result["primary_question"]}

## Preset

- Name: {result["model_version"]["preset"]}
- Decision: {result["decision"]["status"]}
- Role: {result["model_version"]["role"]}

## Equal-Resource Controls

{controls}

## Long-Horizon Axes

{axes}

## Target Graphs

{graphs}

## Parameter Counts

- memory_advantage_tac: {counts["memory_advantage_tac"]["total"]}
- parameter_matched_vanilla: {counts["parameter_matched_vanilla"]["total"]}
- current_best_tac: {counts["current_best_tac"]["total"]}

## Boundary

This is a model-version and benchmark-contract artifact. It does not claim a trained
checkpoint advantage until the equal-resource controls are run.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the TAC-188 memory-advantage model version manifest."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-programs", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--content-store-size", type=int, default=None)
    parser.add_argument("--memory-allocation-k", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_memory_advantage_model_version(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        max_seq_len=args.max_seq_len,
        content_store_size=args.content_store_size,
        memory_allocation_k=args.memory_allocation_k,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / "memory_advantage_model_version.json"
    artifact_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "RESULTS.md").write_text(
        format_memory_advantage_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps({"artifact": str(artifact_path), "decision": result["decision"]}))


if __name__ == "__main__":
    main()
