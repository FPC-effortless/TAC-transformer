from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    TACConfig,
    TACTransformerLM,
    VanillaTransformerLM,
    kaggle_fast_tac_config,
    run5b_capability_config,
)
from tac_transformer.training import (
    count_parameters,
    forward_language_model_window,
    parameter_matched_baseline_config,
)


DEFAULT_OUTPUT_DIR = Path(
    "runs/benchmarks/kaggle_tac_training_speed_profile_2026_06_05"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the opt-in Kaggle fast TAC training profile."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-programs", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=190)
    parser.add_argument(
        "--defer-train-metrics",
        action="store_true",
        help="Skip logging-only TAC metrics during measured training steps.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_kaggle_tac_training_speed_profile(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "kaggle_tac_training_speed_profile.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


def run_kaggle_tac_training_speed_profile(
    args: argparse.Namespace,
) -> dict[str, Any]:
    previous_threads = torch.get_num_threads()
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        if args.torch_threads > 0:
            torch.set_num_threads(args.torch_threads)
        torch.manual_seed(args.seed)
        return _run_kaggle_tac_training_speed_profile(args)
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        if args.torch_threads > 0:
            torch.set_num_threads(previous_threads)


def _run_kaggle_tac_training_speed_profile(
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = _select_device(args.device)
    base_config = run5b_capability_config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        max_seq_len=args.seq_len,
    )
    fast_config = kaggle_fast_tac_config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        max_seq_len=args.seq_len,
    )
    vanilla_config = parameter_matched_baseline_config(fast_config)

    batch = _make_batch(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
    )
    variants = [
        (
            "base_tac_run5b",
            TACTransformerLM(base_config),
            base_config,
        ),
        (
            "kaggle_fast_tac",
            TACTransformerLM(fast_config),
            fast_config,
        ),
        (
            "parameter_matched_vanilla",
            VanillaTransformerLM(vanilla_config),
            vanilla_config,
        ),
    ]
    profiles = [
        profile_training_variant(
            name=name,
            model=model,
            config=config,
            batch=batch,
            warmup=args.warmup,
            iters=args.iters,
            learning_rate=args.learning_rate,
            device=device,
            collect_metrics=not args.defer_train_metrics,
        )
        for name, model, config in variants
    ]
    by_name = {profile["variant"]: profile for profile in profiles}
    fast = by_name["kaggle_fast_tac"]
    base = by_name["base_tac_run5b"]
    vanilla = by_name["parameter_matched_vanilla"]
    fast_fraction = float(fast["content_read_query_fraction"])
    structural_gate_passed = fast_fraction <= 0.25
    decision_status = (
        "kaggle_fast_tac_profile_ready_for_external_validation"
        if structural_gate_passed
        else "kaggle_fast_tac_profile_blocked_by_read_query_gate"
    )

    return {
        "schema": "kaggle_tac_training_speed_profile.v1",
        "ticket": "TAC-190",
        "date": "2026-06-05",
        "benchmark_shape": {
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "n_programs": args.n_programs,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "iters": args.iters,
            "torch_threads": torch.get_num_threads(),
            "device": str(device),
            "collect_metrics": not args.defer_train_metrics,
        },
        "profiles": profiles,
        "structural_gate": {
            "max_allowed_fast_tac_content_read_query_fraction": 0.25,
            "fast_tac_content_read_query_fraction": fast_fraction,
            "passed": structural_gate_passed,
        },
        "interpretation": {
            "fast_tac_tps_ratio_vs_base_tac": _ratio(
                fast["tokens_per_second"],
                base["tokens_per_second"],
            ),
            "fast_tac_tps_ratio_vs_vanilla": _ratio(
                fast["tokens_per_second"],
                vanilla["tokens_per_second"],
            ),
            "vanilla_gap": {
                "vanilla_tokens_per_second": vanilla["tokens_per_second"],
                "fast_tac_tokens_per_second": fast["tokens_per_second"],
                "vanilla_over_fast_tac_ratio": _ratio(
                    vanilla["tokens_per_second"],
                    fast["tokens_per_second"],
                ),
            },
            "read_work_reduction_vs_full": 1.0 - fast_fraction,
            "external_validation_required": True,
        },
        "boundary": {
            "claims_kaggle_t4_wall_clock_speedup": False,
            "claims_capability_preserved": False,
            "reason": (
                "This benchmark verifies the opt-in profile and local training-step "
                "telemetry. Kaggle wall-clock throughput and downstream capability "
                "still require an external run."
            ),
        },
        "decision": {
            "status": decision_status,
            "next_step": (
                "Run kaggle/train_best_tac_agentic.py with --preset kaggle_fast_tac "
                "and compare metrics.jsonl tokens_per_second against the vanilla job."
            ),
        },
    }


def profile_training_variant(
    *,
    name: str,
    model: torch.nn.Module,
    config: TACConfig,
    batch: tuple[torch.Tensor, torch.Tensor],
    warmup: int,
    iters: int,
    learning_rate: float,
    device: torch.device,
    collect_metrics: bool = True,
) -> dict[str, Any]:
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    input_ids, labels = batch

    for _ in range(warmup):
        _training_step(
            model,
            optimizer,
            input_ids,
            labels,
            collect_metrics=collect_metrics,
        )
    _sync(device)
    start = time.perf_counter()
    output = None
    loss = None
    for _ in range(iters):
        output, loss = _training_step(
            model,
            optimizer,
            input_ids,
            labels,
            collect_metrics=collect_metrics,
        )
    _sync(device)
    elapsed = max(time.perf_counter() - start, 1e-9)
    tokens = input_ids.numel() * iters
    metrics = output.aux.metrics if output is not None else {}
    content_read_query_fraction = float(
        metrics.get(
            "content_read_query_fraction",
            input_ids.new_tensor(0.0, dtype=torch.float32),
        ).detach()
    )
    content_read_queries = float(
        metrics.get(
            "content_read_queries",
            input_ids.new_tensor(0.0, dtype=torch.float32),
        ).detach()
    )
    return {
        "variant": name,
        "tokens_per_second": tokens / elapsed,
        "elapsed_seconds": elapsed,
        "loss": float(loss.detach()) if loss is not None else 0.0,
        "parameter_count": count_parameters(model)["total"],
        "config": {
            "routing_type": config.routing_type,
            "routing_top_k": config.routing_top_k,
            "n_programs": config.n_programs,
            "memory_read_type": config.memory_read_type,
            "content_read_steps": config.content_read_steps,
            "content_read_query_top_k": config.content_read_query_top_k,
            "attention_window_size": config.attention_window_size,
            "collect_metrics": collect_metrics,
        },
        "content_read_queries": content_read_queries,
        "content_read_query_fraction": content_read_query_fraction,
        "content_read_skipped_fraction": 1.0 - content_read_query_fraction,
    }


def _training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    collect_metrics: bool = True,
) -> tuple[Any, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    output, next_token_loss, _ = forward_language_model_window(
        model,
        input_ids,
        labels,
        chunked_state_within_batch=True,
        collect_metrics=collect_metrics,
    )
    aux_loss = sum(output.aux.losses.values(), output.logits.new_zeros(()))
    loss = next_token_loss + 0.01 * aux_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return output, loss


def _make_batch(
    *,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len + 1),
        device=device,
    )
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def _select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1e-9)


def format_markdown(result: dict[str, Any]) -> str:
    rows = []
    for profile in result["profiles"]:
        rows.append(
            "| {variant} | {tps:.2f} | {fraction:.4f} | {params} |".format(
                variant=profile["variant"],
                tps=profile["tokens_per_second"],
                fraction=profile["content_read_query_fraction"],
                params=profile["parameter_count"],
            )
        )
    return "\n".join(
        [
            "# Kaggle TAC Training Speed Profile",
            "",
            f"Decision: `{result['decision']['status']}`",
            "",
            "## Profiles",
            "",
            "| Variant | Tokens/s | Read-query fraction | Parameters |",
            "| --- | ---: | ---: | ---: |",
            *rows,
            "",
            "## Interpretation",
            "",
            "- Fast TAC TPS vs base TAC: "
            f"{result['interpretation']['fast_tac_tps_ratio_vs_base_tac']:.4f}",
            "- Fast TAC TPS vs parameter-matched vanilla: "
            f"{result['interpretation']['fast_tac_tps_ratio_vs_vanilla']:.4f}",
            "- Structural read-work reduction vs full content reads: "
            f"{result['interpretation']['read_work_reduction_vs_full']:.4f}",
            "",
            "## Boundary",
            "",
            result["boundary"]["reason"],
            "",
        ]
    )


if __name__ == "__main__":
    main()
