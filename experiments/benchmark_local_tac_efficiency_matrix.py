from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import TACConfig, TACTransformerLM, VanillaTransformerLM, kaggle_fast_tac_config
from tac_transformer.training import (
    count_parameters,
    forward_language_model_window,
    parameter_matched_baseline_config,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/local_tac_efficiency_matrix_2026_06_05")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local TAC efficiency matrix over low-risk speed tactics."
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
    parser.add_argument("--interop-threads", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=192)
    parser.add_argument("--max-loss-delta", type=float, default=0.25)
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Record torch.compile as skipped instead of running the CPU compiler.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_local_tac_efficiency_matrix(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "local_tac_efficiency_matrix.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        format_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


def run_local_tac_efficiency_matrix(args: argparse.Namespace) -> dict[str, Any]:
    previous_threads = torch.get_num_threads()
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    interop_set_status: dict[str, Any] = {
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
                interop_set_status["changed"] = True
            except RuntimeError as exc:
                interop_set_status["error"] = str(exc)
        torch.manual_seed(args.seed)
        return _run_local_tac_efficiency_matrix(args, interop_set_status)
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        if args.torch_threads > 0:
            torch.set_num_threads(previous_threads)


def _run_local_tac_efficiency_matrix(
    args: argparse.Namespace,
    interop_set_status: dict[str, Any],
) -> dict[str, Any]:
    device = _select_device(args.device)
    config = kaggle_fast_tac_config(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        max_seq_len=args.seq_len,
    )
    vanilla_config = parameter_matched_baseline_config(config)
    train_batch = _make_batch(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
    )
    eval_batch = _make_batch(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
    )

    initial_tac_state = _cloned_state_dict(TACTransformerLM(config))
    variants: list[dict[str, Any]] = []
    variants.append(
        _profile_tac_variant(
            variant="eager_full_aux",
            config=config,
            initial_state=initial_tac_state,
            train_batch=train_batch,
            eval_batch=eval_batch,
            warmup=args.warmup,
            iters=args.iters,
            learning_rate=args.learning_rate,
            device=device,
            collect_metrics=True,
            auxiliary_loss_cadence=1,
            compiled=False,
            max_loss_delta=args.max_loss_delta,
        )
    )
    baseline = variants[0]
    baseline_eval_loss = float(baseline["eval_loss"])
    baseline_tps = float(baseline["tokens_per_second"])

    for variant, collect_metrics, cadence in [
        ("eager_metrics_deferred", False, 1),
        ("eager_aux_every_2", False, 2),
        ("eager_aux_every_4", False, 4),
    ]:
        variants.append(
            _profile_tac_variant(
                variant=variant,
                config=config,
                initial_state=initial_tac_state,
                train_batch=train_batch,
                eval_batch=eval_batch,
                warmup=args.warmup,
                iters=args.iters,
                learning_rate=args.learning_rate,
                device=device,
                collect_metrics=collect_metrics,
                auxiliary_loss_cadence=cadence,
                compiled=False,
                baseline_tps=baseline_tps,
                baseline_eval_loss=baseline_eval_loss,
                max_loss_delta=args.max_loss_delta,
            )
        )

    variants.append(
        _compile_variant(
            args=args,
            config=config,
            initial_state=initial_tac_state,
            train_batch=train_batch,
            eval_batch=eval_batch,
            device=device,
            baseline_tps=baseline_tps,
            baseline_eval_loss=baseline_eval_loss,
        )
    )
    variants.append(
        _profile_vanilla_reference(
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
    )
    variants.extend(_deferred_or_not_applicable_rows(device))

    completed_tac = [
        row
        for row in variants
        if row.get("status") == "completed" and row.get("model_family") == "tac"
    ]
    eligible = [
        row
        for row in completed_tac
        if row.get("capability_proxy", {}).get("within_loss_delta_tolerance", False)
    ]
    best = max(
        eligible,
        key=lambda row: float(row.get("tokens_per_second", 0.0)),
        default=baseline,
    )
    return {
        "schema": "local_tac_efficiency_matrix.v1",
        "ticket": "TAC-192",
        "date": "2026-06-05",
        "environment": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "interop_thread_set_status": interop_set_status,
            "torch_compile_available": hasattr(torch, "compile"),
        },
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
            "learning_rate": args.learning_rate,
        },
        "baseline_variant": "eager_full_aux",
        "variants": variants,
        "decision": {
            "recommended_local_profile": best["variant"],
            "recommended_speed_ratio_vs_baseline": best.get(
                "speed_ratio_vs_baseline",
                1.0,
            ),
            "recommended_is_default_safe": best["variant"] in {
                "eager_full_aux",
                "eager_metrics_deferred",
                "torch_compile_reduce_overhead",
            },
            "accepted_loss_delta_threshold": args.max_loss_delta,
            "status": _decision_status(best, baseline),
        },
        "boundary": {
            "claims_no_capability_loss": False,
            "claims_gpu_speedup": False,
            "claims_torch_compile_fullgraph_success": any(
                row.get("variant") == "torch_compile_reduce_overhead"
                and row.get("status") == "completed"
                for row in variants
            ),
            "reason": (
                "This is a local CPU microbenchmark with a held-out next-token loss "
                "proxy. Aux-loss cadence changes the training objective cadence and "
                "must remain opt-in until longer capability benchmarks pass."
            ),
        },
    }


def _profile_tac_variant(
    *,
    variant: str,
    config: TACConfig,
    initial_state: dict[str, torch.Tensor],
    train_batch: tuple[torch.Tensor, torch.Tensor],
    eval_batch: tuple[torch.Tensor, torch.Tensor],
    warmup: int,
    iters: int,
    learning_rate: float,
    device: torch.device,
    collect_metrics: bool,
    auxiliary_loss_cadence: int,
    compiled: bool,
    baseline_tps: float | None = None,
    baseline_eval_loss: float | None = None,
    max_loss_delta: float = 0.25,
) -> dict[str, Any]:
    model = TACTransformerLM(config)
    model.load_state_dict(initial_state)
    model.to(device)
    if compiled:
        model = torch.compile(model, backend="inductor", mode="reduce-overhead", fullgraph=False)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    input_ids, labels = train_batch

    step_index = 0
    for _ in range(warmup):
        step_index += 1
        _training_step_tac(
            model,
            optimizer,
            input_ids,
            labels,
            collect_metrics=collect_metrics,
            auxiliary_loss_cadence=auxiliary_loss_cadence,
            step_index=step_index,
        )
    _sync(device)
    start = time.perf_counter()
    output = None
    loss = None
    for _ in range(iters):
        step_index += 1
        output, loss = _training_step_tac(
            model,
            optimizer,
            input_ids,
            labels,
            collect_metrics=collect_metrics,
            auxiliary_loss_cadence=auxiliary_loss_cadence,
            step_index=step_index,
        )
    _sync(device)
    elapsed = max(time.perf_counter() - start, 1e-9)
    tokens = input_ids.numel() * iters
    eval_loss = _eval_tac_loss(model, eval_batch)
    train_loss = float(loss.detach()) if loss is not None else 0.0
    output_metrics = output.aux.metrics if output is not None else {}
    aux_collected_last_step = _should_collect_auxiliary(
        auxiliary_loss_cadence,
        step_index,
    )
    return _completed_row(
        variant=variant,
        model_family="tac",
        optimization_category="auxiliary_and_dispatch",
        tokens_per_second=tokens / elapsed,
        elapsed_seconds=elapsed,
        train_loss=train_loss,
        eval_loss=eval_loss,
        parameter_count=count_parameters(model)["total"],
        baseline_tps=baseline_tps,
        baseline_eval_loss=baseline_eval_loss,
        max_loss_delta=max_loss_delta,
        collect_metrics=collect_metrics,
        auxiliary_loss_cadence=auxiliary_loss_cadence,
        compiled=compiled,
        notes=(
            "Lossless metric collection change."
            if auxiliary_loss_cadence == 1 and not collect_metrics
            else (
                "Auxiliary losses are computed every step."
                if auxiliary_loss_cadence == 1
                else "Objective-changing auxiliary loss cadence; opt-in only."
            )
        ),
        extra={
            "auxiliary_collected_on_last_step": aux_collected_last_step,
            "content_read_query_fraction": _metric_float(
                output_metrics,
                "content_read_query_fraction",
            ),
        },
    )


def _compile_variant(
    *,
    args: argparse.Namespace,
    config: TACConfig,
    initial_state: dict[str, torch.Tensor],
    train_batch: tuple[torch.Tensor, torch.Tensor],
    eval_batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    baseline_tps: float,
    baseline_eval_loss: float,
) -> dict[str, Any]:
    if args.skip_compile:
        return {
            "variant": "torch_compile_reduce_overhead",
            "status": "skipped",
            "model_family": "tac",
            "optimization_category": "fusion",
            "reason": "--skip-compile was set for this run",
            "compiled": False,
            "auxiliary_loss_cadence": 1,
        }
    if not hasattr(torch, "compile"):
        return {
            "variant": "torch_compile_reduce_overhead",
            "status": "skipped",
            "model_family": "tac",
            "optimization_category": "fusion",
            "reason": "torch.compile is not available in this PyTorch build",
            "compiled": False,
            "auxiliary_loss_cadence": 1,
        }
    try:
        return _profile_tac_variant(
            variant="torch_compile_reduce_overhead",
            config=config,
            initial_state=initial_state,
            train_batch=train_batch,
            eval_batch=eval_batch,
            warmup=args.warmup,
            iters=args.iters,
            learning_rate=args.learning_rate,
            device=device,
            collect_metrics=False,
            auxiliary_loss_cadence=1,
            compiled=True,
            baseline_tps=baseline_tps,
            baseline_eval_loss=baseline_eval_loss,
            max_loss_delta=args.max_loss_delta,
        )
    except Exception as exc:  # pragma: no cover - compiler failures are platform-specific.
        return {
            "variant": "torch_compile_reduce_overhead",
            "status": "skipped",
            "model_family": "tac",
            "optimization_category": "fusion",
            "reason": f"torch.compile failed locally: {type(exc).__name__}: {exc}",
            "compiled": False,
            "auxiliary_loss_cadence": 1,
        }


def _profile_vanilla_reference(
    *,
    config: TACConfig,
    train_batch: tuple[torch.Tensor, torch.Tensor],
    eval_batch: tuple[torch.Tensor, torch.Tensor],
    warmup: int,
    iters: int,
    learning_rate: float,
    device: torch.device,
    baseline_tps: float,
    baseline_eval_loss: float,
    max_loss_delta: float,
) -> dict[str, Any]:
    model = VanillaTransformerLM(config)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    input_ids, labels = train_batch
    for _ in range(warmup):
        _training_step_vanilla(model, optimizer, input_ids, labels)
    _sync(device)
    start = time.perf_counter()
    loss = None
    for _ in range(iters):
        loss = _training_step_vanilla(model, optimizer, input_ids, labels)
    _sync(device)
    elapsed = max(time.perf_counter() - start, 1e-9)
    tokens = input_ids.numel() * iters
    eval_loss = _eval_vanilla_loss(model, eval_batch)
    return _completed_row(
        variant="vanilla_reference",
        model_family="vanilla",
        optimization_category="reference",
        tokens_per_second=tokens / elapsed,
        elapsed_seconds=elapsed,
        train_loss=float(loss.detach()) if loss is not None else 0.0,
        eval_loss=eval_loss,
        parameter_count=count_parameters(model)["total"],
        baseline_tps=baseline_tps,
        baseline_eval_loss=baseline_eval_loss,
        max_loss_delta=max_loss_delta,
        collect_metrics=False,
        auxiliary_loss_cadence=0,
        compiled=False,
        notes="Parameter-matched vanilla reference; loss is not a capability comparison.",
    )


def _training_step_tac(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    collect_metrics: bool,
    auxiliary_loss_cadence: int,
    step_index: int,
) -> tuple[Any, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    collect_auxiliary = _should_collect_auxiliary(auxiliary_loss_cadence, step_index)
    output, next_token_loss, _ = forward_language_model_window(
        model,
        input_ids,
        labels,
        chunked_state_within_batch=True,
        collect_auxiliary=collect_auxiliary,
        collect_metrics=collect_metrics,
    )
    aux_loss = sum(output.aux.losses.values(), output.logits.new_zeros(()))
    loss = next_token_loss + 0.01 * aux_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return output, loss


def _training_step_vanilla(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    output = model(input_ids, labels=labels, collect_metrics=False)
    loss = output.loss
    if loss is None:
        loss = F.cross_entropy(
            output.logits.reshape(-1, model.config.vocab_size),
            labels.reshape(-1),
        )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss


def _eval_tac_loss(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
) -> float:
    was_training = model.training
    model.eval()
    input_ids, labels = batch
    with torch.no_grad():
        _, loss, _ = forward_language_model_window(
            model,
            input_ids,
            labels,
            chunked_state_within_batch=True,
            collect_auxiliary=False,
            collect_metrics=False,
        )
    if was_training:
        model.train()
    return float(loss.detach())


def _eval_vanilla_loss(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
) -> float:
    was_training = model.training
    model.eval()
    input_ids, labels = batch
    with torch.no_grad():
        output = model(input_ids, labels=labels, collect_metrics=False)
        loss = output.loss
        if loss is None:
            loss = F.cross_entropy(
                output.logits.reshape(-1, model.config.vocab_size),
                labels.reshape(-1),
            )
    if was_training:
        model.train()
    return float(loss.detach())


def _completed_row(
    *,
    variant: str,
    model_family: str,
    optimization_category: str,
    tokens_per_second: float,
    elapsed_seconds: float,
    train_loss: float,
    eval_loss: float,
    parameter_count: int,
    baseline_tps: float | None,
    baseline_eval_loss: float | None,
    max_loss_delta: float,
    collect_metrics: bool,
    auxiliary_loss_cadence: int,
    compiled: bool,
    notes: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    speed_ratio = 1.0 if baseline_tps is None else tokens_per_second / max(baseline_tps, 1e-9)
    eval_delta = 0.0 if baseline_eval_loss is None else eval_loss - baseline_eval_loss
    row: dict[str, Any] = {
        "variant": variant,
        "status": "completed",
        "model_family": model_family,
        "optimization_category": optimization_category,
        "tokens_per_second": tokens_per_second,
        "elapsed_seconds": elapsed_seconds,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "speed_ratio_vs_baseline": speed_ratio,
        "parameter_count": parameter_count,
        "collect_metrics": collect_metrics,
        "auxiliary_loss_cadence": auxiliary_loss_cadence,
        "compiled": compiled,
        "capability_proxy": {
            "eval_loss_delta_vs_baseline": eval_delta,
            "accepted_loss_delta_threshold": max_loss_delta,
            "within_loss_delta_tolerance": eval_delta <= max_loss_delta,
        },
        "notes": notes,
    }
    if extra:
        row.update(extra)
    return row


def _deferred_or_not_applicable_rows(device: torch.device) -> list[dict[str, Any]]:
    triton_status = "deferred" if device.type == "cuda" else "not_applicable"
    triton_reason = (
        "CUDA is available, but no custom Triton identity/routing kernel is implemented yet."
        if device.type == "cuda"
        else "Local run is CPU-only; Triton CUDA kernels cannot execute here."
    )
    return [
        {
            "variant": "triton_identity_kernel",
            "status": triton_status,
            "model_family": "tac",
            "optimization_category": "fusion",
            "reason": triton_reason,
        },
        {
            "variant": "foreach_identity_ops",
            "status": "deferred",
            "model_family": "tac",
            "optimization_category": "fusion",
            "reason": (
                "Current identity state tensors are already stacked; foreach helps "
                "only after refactoring fragmented per-program/per-head tensor lists."
            ),
        },
        {
            "variant": "routing_cache_or_hard_routing",
            "status": "deferred",
            "model_family": "tac",
            "optimization_category": "routing_memory",
            "reason": (
                "Routing cache and hard straight-through routing change route dynamics; "
                "they need a longer capability benchmark before promotion."
            ),
        },
        {
            "variant": "two_pass_amortized_state",
            "status": "deferred",
            "model_family": "tac",
            "optimization_category": "two_pass_training",
            "reason": (
                "Single-pass cached-state training changes the batch objective and "
                "requires sequential-corpus validation."
            ),
        },
        {
            "variant": "parameter_reallocation",
            "status": "deferred",
            "model_family": "tac",
            "optimization_category": "architecture",
            "reason": (
                "Shrinking routing/adapters and widening dense layers changes model "
                "capacity allocation; it needs an equal-parameter retraining study."
            ),
        },
    ]


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


def _cloned_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _should_collect_auxiliary(cadence: int, step_index: int) -> bool:
    if cadence <= 1:
        return True
    return step_index % cadence == 0


def _metric_float(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    if value is None:
        return 0.0
    if isinstance(value, torch.Tensor):
        return float(value.detach())
    return float(value)


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


def _decision_status(best: dict[str, Any], baseline: dict[str, Any]) -> str:
    if best["variant"] == baseline["variant"]:
        return "no_local_efficiency_promotion"
    if best.get("auxiliary_loss_cadence", 1) > 1:
        return "opt_in_aux_cadence_candidate"
    return "local_efficiency_profile_promote"


def format_markdown(result: dict[str, Any]) -> str:
    rows = []
    for row in result["variants"]:
        if row["status"] == "completed":
            rows.append(
                "| {variant} | {status} | {tps:.2f} | {ratio:.4f} | {delta:.4f} | {cadence} |".format(
                    variant=row["variant"],
                    status=row["status"],
                    tps=row["tokens_per_second"],
                    ratio=row["speed_ratio_vs_baseline"],
                    delta=row["capability_proxy"]["eval_loss_delta_vs_baseline"],
                    cadence=row.get("auxiliary_loss_cadence", ""),
                )
            )
        else:
            rows.append(
                "| {variant} | {status} | - | - | - | {cadence} |".format(
                    variant=row["variant"],
                    status=row["status"],
                    cadence=row.get("auxiliary_loss_cadence", ""),
                )
            )
    return "\n".join(
        [
            "# Local TAC Efficiency Matrix",
            "",
            f"Decision: `{result['decision']['status']}`",
            f"Recommended profile: `{result['decision']['recommended_local_profile']}`",
            "",
            "## Variants",
            "",
            "| Variant | Status | Tokens/s | Speed vs baseline | Eval loss delta | Aux cadence |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "## Boundary",
            "",
            result["boundary"]["reason"],
            "",
        ]
    )


if __name__ == "__main__":
    main()
