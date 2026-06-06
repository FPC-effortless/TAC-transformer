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

from experiments import benchmark_local_tac_efficiency_matrix as matrix
from tac_transformer import (
    TACTransformerLM,
    cpu_research_tac_config,
    cpu_research_tac_training_kwargs,
    kaggle_fast_tac_config,
)
from tac_transformer.training import count_parameters, parameter_matched_baseline_config


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/cpu_research_tac_version_2026_06_05")
PRIOR_LOCAL_EFFICIENCY_REFERENCES: list[dict[str, Any]] = [
    {
        "variant": "tac_aux_every_4",
        "label": "TAC aux every 4",
        "source_ticket": "TAC-192",
        "source_artifact": (
            "runs/benchmarks/local_tac_efficiency_matrix_2026_06_05/"
            "local_tac_efficiency_matrix.json"
        ),
        "tokens_per_second": 2276.89,
        "speed_ratio_vs_full_aux_tac": 1.15,
        "eval_loss_delta_vs_full_aux_tac": -0.0006,
        "comparison_scope": (
            "Prior local efficiency matrix result. The 1.15x ratio is against "
            "the TAC eager full-aux baseline from TAC-192, not against the "
            "current CPU research benchmark baseline."
        ),
    }
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the opt-in CPU research TAC version."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=193)
    parser.add_argument("--max-loss-delta", type=float, default=0.35)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_cpu_research_tac_version(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cpu_research_tac_version.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


def run_cpu_research_tac_version(args: argparse.Namespace) -> dict[str, Any]:
    previous_threads = torch.get_num_threads()
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    interop_status: dict[str, Any] = {
        "requested": args.interop_threads,
        "changed": False,
        "error": None,
    }
    try:
        if args.torch_threads > 0:
            torch.set_num_threads(args.torch_threads)
        if args.interop_threads > 0:
            try:
                torch.set_num_interop_threads(args.interop_threads)
                interop_status["changed"] = True
            except RuntimeError as exc:
                interop_status["error"] = str(exc)
        torch.manual_seed(args.seed)
        return _run_cpu_research_tac_version(args, interop_status)
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        if args.torch_threads > 0:
            torch.set_num_threads(previous_threads)


def _run_cpu_research_tac_version(
    args: argparse.Namespace,
    interop_status: dict[str, Any],
) -> dict[str, Any]:
    device = matrix._select_device(args.device)
    fast_config = kaggle_fast_tac_config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
    )
    cpu_config = cpu_research_tac_config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
    )
    cpu_training = cpu_research_tac_training_kwargs()
    vanilla_config = parameter_matched_baseline_config(cpu_config)
    train_batch = matrix._make_batch(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
    )
    eval_batch = matrix._make_batch(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
    )
    fast_initial_state = matrix._cloned_state_dict(TACTransformerLM(fast_config))
    cpu_initial_state = matrix._cloned_state_dict(TACTransformerLM(cpu_config))

    fast_profile = matrix._profile_tac_variant(
        variant="kaggle_fast_tac_reference",
        config=fast_config,
        initial_state=fast_initial_state,
        train_batch=train_batch,
        eval_batch=eval_batch,
        warmup=args.warmup,
        iters=args.iters,
        learning_rate=args.learning_rate,
        device=device,
        collect_metrics=False,
        auxiliary_loss_cadence=1,
        compiled=False,
        max_loss_delta=args.max_loss_delta,
    )
    baseline_tps = float(fast_profile["tokens_per_second"])
    baseline_eval_loss = float(fast_profile["eval_loss"])
    fast_aux4_profile = matrix._profile_tac_variant(
        variant="kaggle_fast_tac_aux_every_4",
        config=fast_config,
        initial_state=fast_initial_state,
        train_batch=train_batch,
        eval_batch=eval_batch,
        warmup=args.warmup,
        iters=args.iters,
        learning_rate=args.learning_rate,
        device=device,
        collect_metrics=False,
        auxiliary_loss_cadence=4,
        compiled=False,
        baseline_tps=baseline_tps,
        baseline_eval_loss=baseline_eval_loss,
        max_loss_delta=args.max_loss_delta,
    )
    cpu_arch_full_aux_profile = matrix._profile_tac_variant(
        variant="cpu_research_arch_full_aux",
        config=cpu_config,
        initial_state=cpu_initial_state,
        train_batch=train_batch,
        eval_batch=eval_batch,
        warmup=args.warmup,
        iters=args.iters,
        learning_rate=args.learning_rate,
        device=device,
        collect_metrics=False,
        auxiliary_loss_cadence=1,
        compiled=False,
        baseline_tps=baseline_tps,
        baseline_eval_loss=baseline_eval_loss,
        max_loss_delta=args.max_loss_delta,
    )
    cpu_profile = matrix._profile_tac_variant(
        variant="cpu_research_tac",
        config=cpu_config,
        initial_state=cpu_initial_state,
        train_batch=train_batch,
        eval_batch=eval_batch,
        warmup=args.warmup,
        iters=args.iters,
        learning_rate=args.learning_rate,
        device=device,
        collect_metrics=False,
        auxiliary_loss_cadence=int(cpu_training["aux_loss_cadence"]),
        compiled=False,
        baseline_tps=baseline_tps,
        baseline_eval_loss=baseline_eval_loss,
        max_loss_delta=args.max_loss_delta,
    )
    vanilla_profile = matrix._profile_vanilla_reference(
        config=vanilla_config,
        train_batch=train_batch,
        eval_batch=eval_batch,
        warmup=args.warmup,
        iters=args.iters,
        learning_rate=args.learning_rate,
        device=device,
        baseline_tps=baseline_tps,
        baseline_eval_loss=baseline_eval_loss,
        max_loss_delta=args.max_loss_delta,
    )
    fast_profile["config"] = _config_summary(fast_config)
    fast_aux4_profile["config"] = _config_summary(fast_config)
    cpu_arch_full_aux_profile["config"] = _config_summary(cpu_config)
    cpu_profile["config"] = _config_summary(cpu_config)
    vanilla_profile["config"] = {
        "d_model": vanilla_config.d_model,
        "n_layers": vanilla_config.n_layers,
        "n_heads": vanilla_config.n_heads,
    }
    profiles = [
        fast_profile,
        fast_aux4_profile,
        cpu_arch_full_aux_profile,
        cpu_profile,
        vanilla_profile,
    ]
    combination_analysis = _combination_analysis(
        fast_profile=fast_profile,
        fast_aux4_profile=fast_aux4_profile,
        cpu_arch_full_aux_profile=cpu_arch_full_aux_profile,
        cpu_profile=cpu_profile,
        vanilla_profile=vanilla_profile,
    )
    cpu_speed_ratio = cpu_profile["speed_ratio_vs_baseline"]
    cpu_loss_delta = cpu_profile["capability_proxy"]["eval_loss_delta_vs_baseline"]
    status = (
        "cpu_research_tac_speed_candidate"
        if cpu_speed_ratio >= 1.0 and cpu_loss_delta <= args.max_loss_delta
        else "cpu_research_tac_version_ready"
    )
    fast_counts = count_parameters(TACTransformerLM(fast_config))
    cpu_counts = count_parameters(TACTransformerLM(cpu_config))
    return {
        "schema": "cpu_research_tac_version.v1",
        "ticket": "TAC-193",
        "date": "2026-06-05",
        "environment": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "interop_thread_set_status": interop_status,
        },
        "model_version": {
            "preset": "cpu_research_tac",
            "base_class": "TACTransformerLM",
            "research_only": True,
        },
        "applied_cpu_tactics": [
            "hard_lower_k_routing",
            "smaller_program_bank",
            "sparse_content_reads",
            "single_step_content_read",
            "local_attention_window",
            "cheaper_residual_memory_adapter",
            "auxiliary_loss_cadence",
            "cpu_thread_pinning",
        ],
        "benchmark_shape": {
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "iters": args.iters,
            "learning_rate": args.learning_rate,
        },
        "parameter_counts": {
            "kaggle_fast_tac_reference": fast_counts,
            "cpu_research_tac": cpu_counts,
            "identity_field_share_delta": (
                cpu_counts["identity_field"] / max(cpu_counts["total"], 1)
                - fast_counts["identity_field"] / max(fast_counts["total"], 1)
            ),
        },
        "training_defaults": cpu_training,
        "profiles": profiles,
        "combination_analysis": combination_analysis,
        "prior_local_efficiency_references": [
            dict(reference) for reference in PRIOR_LOCAL_EFFICIENCY_REFERENCES
        ],
        "decision": {
            "status": status,
            "cpu_research_speed_ratio_vs_kaggle_fast": cpu_speed_ratio,
            "cpu_research_eval_loss_delta_vs_kaggle_fast": cpu_loss_delta,
            "same_run_combined_speed_ratio_vs_fast_full_aux": combination_analysis[
                "combined_speed_ratio_vs_fast_full_aux"
            ],
            "same_run_aux_every_4_speed_ratio_on_cpu_architecture": (
                combination_analysis["aux_every_4_speed_ratio_on_cpu_architecture"]
            ),
            "prior_aux_every_4_reference": dict(PRIOR_LOCAL_EFFICIENCY_REFERENCES[0]),
            "use_as_default": False,
            "next_step": (
                "Use --preset cpu_research_tac for local CPU research runs, then "
                "validate capability with the long-horizon and ATS gates before "
                "promoting any tactic into main TAC."
            ),
        },
        "boundary": {
            "changes_main_tac_architecture": False,
            "claims_capability_preserved": False,
            "claims_gpu_speedup": False,
            "reason": (
                "This is a separate opt-in preset for CPU experiments. It changes "
                "routing/read/adapter capacity and auxiliary cadence, so it is a "
                "research version rather than a drop-in replacement."
            ),
        },
    }


def _config_summary(config) -> dict[str, Any]:
    return {
        "routing_type": config.routing_type,
        "routing_top_k": config.routing_top_k,
        "n_programs": config.n_programs,
        "memory_read_type": config.memory_read_type,
        "content_store_size": config.content_store_size,
        "content_read_steps": config.content_read_steps,
        "content_read_query_top_k": config.content_read_query_top_k,
        "attention_window_size": config.attention_window_size,
        "memory_adapter_type": config.memory_adapter_type,
        "identity_attention_type": config.identity_attention_type,
        "routing_load_balance_weight": config.routing_load_balance_weight,
    }


def _profile_tps(profile: dict[str, Any]) -> float:
    return float(profile["tokens_per_second"])


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _combination_analysis(
    *,
    fast_profile: dict[str, Any],
    fast_aux4_profile: dict[str, Any],
    cpu_arch_full_aux_profile: dict[str, Any],
    cpu_profile: dict[str, Any],
    vanilla_profile: dict[str, Any],
) -> dict[str, float]:
    fast_tps = _profile_tps(fast_profile)
    fast_aux4_tps = _profile_tps(fast_aux4_profile)
    cpu_arch_full_aux_tps = _profile_tps(cpu_arch_full_aux_profile)
    cpu_combined_tps = _profile_tps(cpu_profile)
    vanilla_tps = _profile_tps(vanilla_profile)
    return {
        "aux_every_4_speed_ratio_on_fast_tac": _safe_ratio(
            fast_aux4_tps,
            fast_tps,
        ),
        "cpu_architecture_speed_ratio_with_full_aux": _safe_ratio(
            cpu_arch_full_aux_tps,
            fast_tps,
        ),
        "combined_speed_ratio_vs_fast_full_aux": _safe_ratio(
            cpu_combined_tps,
            fast_tps,
        ),
        "aux_every_4_speed_ratio_on_cpu_architecture": _safe_ratio(
            cpu_combined_tps,
            cpu_arch_full_aux_tps,
        ),
        "cpu_architecture_speed_ratio_after_aux_every_4": _safe_ratio(
            cpu_combined_tps,
            fast_aux4_tps,
        ),
        "vanilla_speed_ratio_vs_combined": _safe_ratio(
            vanilla_tps,
            cpu_combined_tps,
        ),
    }


def _technique_stack(variant: str) -> str:
    return {
        "kaggle_fast_tac_reference": "baseline fast TAC, full aux",
        "kaggle_fast_tac_aux_every_4": "aux every 4 only",
        "cpu_research_arch_full_aux": "CPU research architecture only",
        "cpu_research_tac": "CPU research architecture + aux every 4",
        "vanilla_reference": "vanilla reference",
    }.get(variant, "")


def format_markdown(result: dict[str, Any]) -> str:
    rows = []
    for profile in result["profiles"]:
        rows.append(
            (
                "| {variant} | {stack} | {tps:.2f} | {ratio:.4f} | "
                "{params} | {delta:.4f} |"
            ).format(
                variant=profile["variant"],
                stack=_technique_stack(profile["variant"]),
                tps=profile["tokens_per_second"],
                ratio=profile["speed_ratio_vs_baseline"],
                params=profile["parameter_count"],
                delta=profile["capability_proxy"]["eval_loss_delta_vs_baseline"],
            )
        )
    analysis = result["combination_analysis"]
    analysis_rows = [
        (
            "Aux every 4 on fast TAC",
            analysis["aux_every_4_speed_ratio_on_fast_tac"],
        ),
        (
            "CPU research architecture with full aux",
            analysis["cpu_architecture_speed_ratio_with_full_aux"],
        ),
        (
            "CPU research architecture + aux every 4",
            analysis["combined_speed_ratio_vs_fast_full_aux"],
        ),
        (
            "Aux every 4 gain on CPU research architecture",
            analysis["aux_every_4_speed_ratio_on_cpu_architecture"],
        ),
        (
            "CPU architecture gain after aux every 4",
            analysis["cpu_architecture_speed_ratio_after_aux_every_4"],
        ),
        (
            "Vanilla speed vs combined CPU research TAC",
            analysis["vanilla_speed_ratio_vs_combined"],
        ),
    ]
    analysis_markdown_rows = [
        f"| {label} | {ratio:.4f} |" for label, ratio in analysis_rows
    ]
    reference_rows = []
    for reference in result["prior_local_efficiency_references"]:
        reference_rows.append(
            "| {variant} | {tps:.2f} | {ratio:.2f} | {delta:.4f} | {scope} |".format(
                variant=reference["variant"],
                tps=reference["tokens_per_second"],
                ratio=reference["speed_ratio_vs_full_aux_tac"],
                delta=reference["eval_loss_delta_vs_full_aux_tac"],
                scope=reference["comparison_scope"],
            )
        )
    return "\n".join(
        [
            "# CPU Research TAC Version",
            "",
            f"Decision: `{result['decision']['status']}`",
            "",
            "## Same-Run Ablation",
            "",
            (
                "| Variant | Technique stack | Tokens/s | Speed vs fast full-aux TAC | "
                "Parameters | Eval loss delta |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "## Combination Analysis",
            "",
            "| Comparison | Speed ratio |",
            "| --- | ---: |",
            *analysis_markdown_rows,
            "",
            "## Prior Local Efficiency Reference",
            "",
            "| Variant | Tokens/s | Speed vs full-aux TAC | Eval loss delta | Scope |",
            "| --- | ---: | ---: | ---: | --- |",
            *reference_rows,
            "",
            "## Applied CPU Tactics",
            "",
            *[f"- `{name}`" for name in result["applied_cpu_tactics"]],
            "",
            "## Boundary",
            "",
            result["boundary"]["reason"],
            "",
        ]
    )


if __name__ == "__main__":
    main()
