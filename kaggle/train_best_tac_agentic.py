from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle.analyze_program_specialization import (
    analyze_program_specialization,
    write_attribution_csv,
)
from tac_transformer import (
    TACTransformerLM,
    best_tac_config,
    cpu_research_tac_config,
    cpu_research_tac_training_kwargs,
    kaggle_fast_tac_config,
    kaggle_fast_tac_training_kwargs,
    memory_advantage_config,
    memory_advantage_training_kwargs,
    run5b_best_capability_fast_config,
    run5b_best_capability_fast_training_kwargs,
    run5_capability_config,
    run5_capability_training_kwargs,
    run5b_capability_config,
    run5b_capability_training_kwargs,
)
from tac_transformer.optimization import TACOptimizerConfig, build_tac_optimizer
from tac_transformer.training import (
    JsonlCompletionBatcher,
    JsonlLabeledCompletionBatcher,
    JsonlLabeledTextBatcher,
    JsonlTextBatcher,
    JsonlWeightedCompletionBatcher,
    JsonlWeightedTextBatcher,
    TokenizedMemmapBatcher,
    category_program_mi_loss,
    category_route_loss,
    selected_program_mi_loss,
    count_parameters,
    evaluate_language_model,
    forward_language_model_window,
)


MODEL_SCALES: dict[str, dict[str, int]] = {
    "smoke": {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "n_programs": 16,
        "seq_len": 64,
        "batch_size": 8,
        "grad_accum_steps": 1,
    },
    "small": {
        "d_model": 192,
        "n_heads": 6,
        "n_layers": 6,
        "n_programs": 24,
        "seq_len": 256,
        "batch_size": 8,
        "grad_accum_steps": 4,
    },
    "base": {
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 8,
        "n_programs": 32,
        "seq_len": 256,
        "batch_size": 6,
        "grad_accum_steps": 6,
    },
    "large": {
        "d_model": 384,
        "n_heads": 8,
        "n_layers": 10,
        "n_programs": 48,
        "seq_len": 384,
        "batch_size": 2,
        "grad_accum_steps": 16,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the best TAC architecture on agentic traces within a Kaggle time budget."
    )
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--eval-jsonl", type=Path, default=None)
    parser.add_argument(
        "--train-tokenized-manifest",
        type=Path,
        default=None,
        help="Train from a tokenized memmap manifest instead of JSONL text.",
    )
    parser.add_argument(
        "--eval-tokenized-manifest",
        type=Path,
        default=None,
        help="Evaluate from a tokenized memmap manifest instead of JSONL text.",
    )
    parser.add_argument(
        "--supervision-mode",
        choices=["full_lm", "answer_only"],
        default="full_lm",
        help=(
            "Training label contract for prepared JSONL rows. full_lm keeps "
            "standard next-token loss over the text field. answer_only expects "
            "separate prompt and answer fields and masks prompt tokens."
        ),
    )
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--completion-field", default="answer")
    parser.add_argument(
        "--sampling-weights-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON object mapping domain labels, dataset names, stream names, "
            "or '*' to sampling weights for JSONL training."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scale", choices=sorted(MODEL_SCALES), default="base")
    parser.add_argument(
        "--preset",
        choices=[
            "best",
            "run5_capability",
            "run5b_capability",
            "memory_advantage",
            "run5b_best_capability_fast",
            "kaggle_fast_tac",
            "cpu_research_tac",
        ],
        default="best",
        help=(
            "Architecture preset to train. run5_capability applies the post-Run-4 "
            "low-weight semantic candidate; run5b_capability keeps that architecture "
            "with safer optimizer-health defaults; memory_advantage composes the "
            "TAC-188 long-horizon memory candidate; run5b_best_capability_fast "
            "adds TAC-169 cue-chain reads and speed defaults for a Run 5B "
            "capability launch; kaggle_fast_tac keeps the "
            "semantic TAC path while reducing content-read and attention work; "
            "cpu_research_tac is a CPU-only research profile that trims routing, "
            "memory-read, and adapter overhead for local experiments."
        ),
    )
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--n-programs", type=int, default=None)
    parser.add_argument(
        "--mlp-ratio",
        type=int,
        default=None,
        help="Override transformer MLP expansion ratio without changing identity programs.",
    )
    parser.add_argument(
        "--program-compute-type",
        choices=[
            "embedding",
            "linear_expert",
            "sparse_linear_expert",
            "low_rank_linear_expert",
        ],
        default=None,
        help="Override the per-program compute module used by the identity field.",
    )
    parser.add_argument(
        "--program-expert-rank",
        type=int,
        default=None,
        help="Rank for --program-compute-type low_rank_linear_expert.",
    )
    parser.add_argument("--energy-budget", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument(
        "--routing-type",
        choices=[
            "energy",
            "expert_choice",
            "base",
            "hash",
            "sparse_ensemble",
            "base_semantic",
            "base_semantic_soft",
            "authority_gated",
        ],
        default=None,
        help="Override the preset router. Leave unset to use the promoted default.",
    )
    parser.add_argument("--routing-top-k", type=int, default=None)
    parser.add_argument("--routing-load-balance-weight", type=float, default=None)
    parser.add_argument(
        "--semantic-route-allowed-programs",
        type=int,
        nargs="+",
        default=None,
        help="Restrict base_semantic extra program selection to these program IDs.",
    )
    parser.add_argument(
        "--semantic-route-suppressed-programs",
        type=int,
        nargs="+",
        default=None,
        help="Prevent base_semantic extra program selection from using these program IDs.",
    )
    parser.add_argument(
        "--category-route-weight",
        type=float,
        default=None,
        help="Add a domain-label program-routing objective for labeled JSONL corpora.",
    )
    parser.add_argument(
        "--category-route-start-step",
        type=int,
        default=0,
        help="Keep category-route loss weight at zero before this optimizer step.",
    )
    parser.add_argument(
        "--category-route-warmup-steps",
        type=int,
        default=0,
        help="Linearly warm category-route weight from zero after category-route-start-step.",
    )
    parser.add_argument(
        "--category-route-objective",
        choices=["fixed", "mi", "selected_mi"],
        default=None,
        help="Use fixed targets, token-level MI, or selected record-level MI.",
    )
    parser.add_argument(
        "--semantic-routing-start-step",
        type=int,
        default=0,
        help="Temporarily run BASE top-1 routing before this step when the target router is semantic.",
    )
    parser.add_argument(
        "--program-memory-update-type",
        choices=["shared", "program_conditioned"],
        default=None,
        help="Override how candidate program-memory writes are generated.",
    )
    parser.add_argument(
        "--memory-adapter-type",
        choices=["none", "residual", "gated_residual"],
        default=None,
        help="Override whether identity memory is adapted back into token hidden states.",
    )
    parser.add_argument(
        "--identity-attention-type",
        choices=[
            "none",
            "compressed_memory",
            "coherence_sparse",
            "coherence_sparse_compressed",
            "identity_first",
        ],
        default=None,
        help="Override how identity memory interacts with attention.",
    )
    parser.add_argument(
        "--memory-read-type",
        choices=["none", "program_memory", "pattern_completion", "content_addressed"],
        default=None,
        help="Override how identity memory is read back into token states.",
    )
    parser.add_argument(
        "--content-read-steps",
        type=int,
        default=None,
        help="Override the number of iterative content-addressed read steps.",
    )
    parser.add_argument(
        "--content-read-gate-type",
        choices=["learned", "confidence", "confidence_margin", "cue_match", "synthesis"],
        default=None,
        help="Override how multi-step content-addressed reads are blended.",
    )
    parser.add_argument(
        "--content-read-confidence-margin",
        type=float,
        default=None,
        help="Confidence gap below which confidence-margin reads continue to the second hop.",
    )
    parser.add_argument(
        "--content-read-cue-match-threshold",
        type=float,
        default=None,
        help="Cue-match score above which cue-match reads continue to the second hop.",
    )
    parser.add_argument(
        "--content-read-query-top-k",
        type=int,
        default=None,
        help="Limit content-addressed memory reads to the top-k token positions per batch.",
    )
    parser.add_argument(
        "--attention-window-size",
        type=int,
        default=None,
        help="Use local causal token attention with this window size.",
    )
    parser.add_argument(
        "--coalition-context-type",
        choices=[
            "none",
            "program_memory",
            "program_memory_graph",
            "program_memory_task_graph",
        ],
        default=None,
        help="Inject an optional coalition context signal into selected program execution.",
    )
    parser.add_argument(
        "--coalition-context-scale",
        type=float,
        default=None,
        help="Scale applied to the optional coalition context projection.",
    )
    parser.add_argument(
        "--program-residual-scale",
        type=float,
        default=None,
        help="Scale the routed program-context residual added inside each TAC block.",
    )
    parser.add_argument(
        "--coherence-attention-scale",
        type=float,
        default=None,
        help="Scale identity coherence bias added to attention logits.",
    )
    parser.add_argument(
        "--memory-allocation-type",
        choices=["stability", "creb"],
        default=None,
        help="Override the program-memory write allocation policy.",
    )
    parser.add_argument("--memory-allocation-k", type=int, default=None)
    parser.add_argument("--memory-separation-weight", type=float, default=None)
    parser.add_argument(
        "--aux-loss-scale",
        type=float,
        default=1.0,
        help="Multiply all TAC auxiliary losses; use values below 1.0 to prioritize next-token learning.",
    )
    parser.add_argument(
        "--aux-loss-warmup-steps",
        type=int,
        default=0,
        help="Linearly warm auxiliary loss pressure from 0 to aux-loss-scale over this many steps.",
    )
    parser.add_argument(
        "--aux-loss-cadence",
        type=int,
        default=1,
        help="Compute TAC auxiliary losses every N optimizer steps; values above 1 are opt-in speed experiments.",
    )
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=None,
        help="Derive warmup steps as a fraction of total steps; useful for short diagnostics.",
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--max-seconds", type=int, default=8 * 60 * 60 + 30 * 60)
    parser.add_argument("--stop-buffer-seconds", type=int, default=20 * 60)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="Set torch intra-op threads before training; 0 leaves the current setting unchanged.",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=0,
        help="Set torch inter-op threads before training; 0 leaves the current setting unchanged.",
    )
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument(
        "--min-healthy-gradient-norm",
        type=float,
        default=None,
        help="Flag optimizer health as failed when the clipped pre-step gradient norm is below this value.",
    )
    parser.add_argument(
        "--fail-on-unhealthy-optimization",
        action="store_true",
        help="Abort training when optimizer-health telemetry fails the configured gate.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help=(
            "Resume from output-dir/last.pt when present, otherwise search attached "
            "Kaggle input datasets for last.pt or best.pt."
        ),
    )
    parser.add_argument(
        "--carry-state-across-batches",
        action="store_true",
        help="Opt in only for contiguous trajectory batches; random JSONL windows reset state by default.",
    )
    parser.add_argument(
        "--no-chunked-state-within-batch",
        action="store_true",
        help="Disable context/query splitting inside each JSONL window.",
    )
    parser.add_argument(
        "--analyze-specialization-at-end",
        action="store_true",
        help=(
            "After training, run program attribution, program-category MI, and "
            "program knockout analysis on best.pt."
        ),
    )
    parser.add_argument(
        "--skip-end-specialization-on-time-stop",
        action="store_true",
        help=(
            "When training stops early because of the wall-clock guard, skip the "
            "expensive end specialization pass so a resume run can spend the next "
            "session on training. Completed target-step runs still analyze best.pt."
        ),
    )
    parser.add_argument(
        "--specialization-jsonl",
        type=Path,
        default=None,
        help=(
            "Labeled hard-agentic JSONL for specialization analysis. If omitted, "
            "the trainer searches for hard_agentic_eval.generated.jsonl, then eval.prepared.jsonl."
        ),
    )
    parser.add_argument(
        "--specialization-output-dir",
        type=Path,
        default=None,
        help="Directory for specialization artifacts. Defaults to <output-dir>/specialization.",
    )
    parser.add_argument("--specialization-max-records-per-category", type=int, default=64)
    parser.add_argument("--specialization-top-k", type=int, default=3)
    parser.add_argument(
        "--specialization-knockout-programs",
        nargs="+",
        default=None,
        help="Program IDs to ablate; omit for the full one-program knockout matrix.",
    )
    parser.add_argument(
        "--specialization-device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
        help="Device for the post-training specialization pass. CPU avoids GPU OOM after training.",
    )
    parser.add_argument(
        "--specialization-checkpoints",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Training steps at which to save a model-only snapshot and run lightweight "
            "specialization analysis."
        ),
    )
    parser.add_argument(
        "--specialization-checkpoint-max-records-per-category",
        type=int,
        default=16,
        help="Records per category for periodic specialization checkpoint analysis.",
    )
    parser.add_argument(
        "--specialization-checkpoint-run-knockouts",
        action="store_true",
        help="Run program knockouts during periodic specialization checkpoints.",
    )
    parser.add_argument(
        "--specialization-checkpoint-knockout-programs",
        nargs="+",
        default=None,
        help="Program IDs for periodic checkpoint knockouts; omit for all when checkpoint knockouts are enabled.",
    )
    parser.add_argument("--num-workers-note", action="store_true")
    return apply_training_preset_defaults(parser.parse_args(argv))


def apply_training_preset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.warmup_ratio is not None:
        ratio = float(args.warmup_ratio)
        if ratio < 0.0:
            raise ValueError("warmup_ratio must be non-negative")
        derived_warmup = int(round(max(args.steps, 0) * ratio))
        if ratio > 0.0:
            derived_warmup = max(1, derived_warmup)
        args.warmup_steps = derived_warmup
    if args.preset in {
        "run5_capability",
        "run5b_capability",
        "memory_advantage",
        "run5b_best_capability_fast",
        "kaggle_fast_tac",
        "cpu_research_tac",
    }:
        if args.preset == "memory_advantage":
            defaults = memory_advantage_training_kwargs()
        elif args.preset == "run5b_best_capability_fast":
            defaults = run5b_best_capability_fast_training_kwargs()
        elif args.preset == "cpu_research_tac":
            defaults = cpu_research_tac_training_kwargs()
        elif args.preset == "kaggle_fast_tac":
            defaults = kaggle_fast_tac_training_kwargs()
        elif args.preset == "run5b_capability":
            defaults = run5b_capability_training_kwargs()
        else:
            defaults = run5_capability_training_kwargs()
        if args.category_route_weight is None:
            args.category_route_weight = float(defaults["category_route_weight"])
        if args.category_route_objective is None:
            args.category_route_objective = str(defaults["category_route_objective"])
        if args.warmup_steps is None:
            args.warmup_steps = int(defaults["warmup_steps"])
        if "aux_loss_cadence" in defaults and args.aux_loss_cadence == 1:
            args.aux_loss_cadence = int(defaults["aux_loss_cadence"])
        if "torch_threads" in defaults and args.torch_threads == 0:
            args.torch_threads = int(defaults["torch_threads"])
        if "torch_interop_threads" in defaults and args.torch_interop_threads == 0:
            args.torch_interop_threads = int(defaults["torch_interop_threads"])
        if "precision" in defaults:
            if args.precision == "auto":
                args.precision = str(defaults["precision"])
            if args.min_healthy_gradient_norm is None:
                args.min_healthy_gradient_norm = float(defaults["min_healthy_gradient_norm"])
            if defaults.get("fail_on_unhealthy_optimization", 0):
                args.fail_on_unhealthy_optimization = True
    else:
        if args.category_route_weight is None:
            args.category_route_weight = 0.0
        if args.category_route_objective is None:
            args.category_route_objective = "fixed"
        if args.warmup_steps is None:
            args.warmup_steps = 500
    if args.min_healthy_gradient_norm is None:
        args.min_healthy_gradient_norm = 0.0
    return args


def apply_torch_thread_settings(args: argparse.Namespace) -> dict[str, Any]:
    status: dict[str, Any] = {
        "requested_torch_threads": int(getattr(args, "torch_threads", 0)),
        "requested_torch_interop_threads": int(
            getattr(args, "torch_interop_threads", 0)
        ),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "interop_error": None,
    }
    if status["requested_torch_threads"] > 0:
        torch.set_num_threads(status["requested_torch_threads"])
        status["torch_threads"] = torch.get_num_threads()
    if status["requested_torch_interop_threads"] > 0:
        try:
            torch.set_num_interop_threads(status["requested_torch_interop_threads"])
        except RuntimeError as exc:
            status["interop_error"] = str(exc)
        status["torch_interop_threads"] = torch.get_num_interop_threads()
    return status


def batcher_record_count(batcher: Any) -> int:
    if hasattr(batcher, "offsets"):
        return len(batcher.offsets)
    if hasattr(batcher, "record_offsets"):
        return int(batcher.record_offsets.shape[0])
    return 0


def load_sampling_weights(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--sampling-weights-json must contain a JSON object")
    weights: dict[str, float] = {}
    for key, value in payload.items():
        weight = float(value)
        if weight < 0.0:
            raise ValueError("sampling weights must be non-negative")
        weights[str(key)] = weight
    if not weights:
        raise ValueError("--sampling-weights-json must not be empty")
    if sum(weights.values()) <= 0.0:
        raise ValueError("at least one sampling weight must be positive")
    return weights


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    thread_settings = apply_torch_thread_settings(args)
    torch.manual_seed(args.seed)
    distributed = init_distributed_if_needed()
    rank = distributed_rank()
    world_size = distributed_world_size()
    device = select_device(args.device)
    precision = select_precision(args.precision, device)
    output_dir = args.output_dir or default_kaggle_output_dir()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    tokenized_training = (
        args.train_tokenized_manifest is not None
        or args.eval_tokenized_manifest is not None
    )
    sampling_weights = load_sampling_weights(args.sampling_weights_json)
    if tokenized_training:
        if args.train_tokenized_manifest is None or args.eval_tokenized_manifest is None:
            raise ValueError(
                "--train-tokenized-manifest and --eval-tokenized-manifest must be provided together"
            )
        if args.train_jsonl is not None or args.eval_jsonl is not None:
            raise ValueError("Use either JSONL paths or tokenized manifests, not both")
        if args.supervision_mode != "full_lm":
            raise ValueError("tokenized manifests currently support full_lm supervision only")
        if args.category_route_weight > 0.0:
            raise ValueError("category-route training is not supported with tokenized manifests yet")
        if sampling_weights is not None:
            raise ValueError("sampling weights are only supported with JSONL training")
        train_path = None
        eval_path = None
    else:
        train_path = args.train_jsonl or discover_prepared_jsonl("train.prepared.jsonl")
        eval_path = args.eval_jsonl or discover_prepared_jsonl("eval.prepared.jsonl")
    scale = resolved_scale(args)
    config = build_training_config(args, scale)
    base_model = TACTransformerLM(config).to(device)
    model = wrap_distributed_model(base_model, device, distributed)
    optimizer = build_tac_optimizer(
        model,
        TACOptimizerConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
    )
    scaler = make_grad_scaler(device, precision)
    start_step = 0
    best_eval_loss = math.inf
    resume_checkpoint = resolve_resume_checkpoint(
        args.resume,
        output_dir,
        auto_resume=args.auto_resume,
    )

    if resume_checkpoint is not None:
        start_step, best_eval_loss = load_checkpoint(
            resume_checkpoint,
            model=unwrap_model(model),
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

    if tokenized_training:
        train_batcher = TokenizedMemmapBatcher.from_manifest(
            args.train_tokenized_manifest,
            seq_len=config.max_seq_len,
            seed=args.seed + rank,
        )
        eval_batcher = TokenizedMemmapBatcher.from_manifest(
            args.eval_tokenized_manifest,
            seq_len=config.max_seq_len,
            seed=args.seed + 1000,
        )
        if train_batcher.vocab_size != config.vocab_size:
            raise ValueError(
                "train tokenized manifest vocab_size must match --vocab-size "
                f"({train_batcher.vocab_size} != {config.vocab_size})"
            )
        if eval_batcher.vocab_size != config.vocab_size:
            raise ValueError(
                "eval tokenized manifest vocab_size must match --vocab-size "
                f"({eval_batcher.vocab_size} != {config.vocab_size})"
            )
    elif args.supervision_mode == "answer_only":
        batcher_cls = (
            JsonlWeightedCompletionBatcher
            if sampling_weights is not None
            else JsonlCompletionBatcher
        )
        train_kwargs: dict[str, Any] = {}
        if sampling_weights is not None:
            train_kwargs["category_weights"] = sampling_weights
        train_batcher = batcher_cls(
            train_path,
            seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            seed=args.seed + rank,
            prompt_field=args.prompt_field,
            completion_field=args.completion_field,
            **train_kwargs,
        )
    else:
        batcher_cls = (
            JsonlWeightedTextBatcher
            if sampling_weights is not None
            else JsonlTextBatcher
        )
        train_kwargs: dict[str, Any] = {}
        if sampling_weights is not None:
            train_kwargs["category_weights"] = sampling_weights
        train_batcher = batcher_cls(
            train_path,
            seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            seed=args.seed + rank,
            **train_kwargs,
        )
    category_batcher = None
    if args.category_route_weight > 0.0:
        if args.supervision_mode == "answer_only":
            category_batcher = JsonlLabeledCompletionBatcher(
                train_path,
                seq_len=config.max_seq_len,
                vocab_size=config.vocab_size,
                seed=args.seed + rank + 2000,
                prompt_field=args.prompt_field,
                completion_field=args.completion_field,
                category_weights=sampling_weights,
            )
        else:
            category_batcher = JsonlLabeledTextBatcher(
                train_path,
                seq_len=config.max_seq_len,
                vocab_size=config.vocab_size,
                seed=args.seed + rank + 2000,
                category_weights=sampling_weights,
            )
    if tokenized_training:
        pass
    elif args.supervision_mode == "answer_only":
        eval_batcher = JsonlCompletionBatcher(
            eval_path,
            seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            seed=args.seed + 1000,
            prompt_field=args.prompt_field,
            completion_field=args.completion_field,
        )
    else:
        eval_batcher = JsonlTextBatcher(
            eval_path,
            seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            seed=args.seed + 1000,
        )
    run_manifest = {
        "device": str(device),
        "precision": precision,
        "supervision_mode": args.supervision_mode,
        "prompt_field": args.prompt_field,
        "completion_field": args.completion_field,
        "sampling_weights_json": None
        if args.sampling_weights_json is None
        else str(args.sampling_weights_json),
        "sampling_weights": sampling_weights,
        "torch_thread_settings": thread_settings,
        "distributed": distributed,
        "rank": rank,
        "world_size": world_size,
        "train_jsonl": None if train_path is None else str(train_path),
        "eval_jsonl": None if eval_path is None else str(eval_path),
        "train_tokenized_manifest": None
        if args.train_tokenized_manifest is None
        else str(args.train_tokenized_manifest),
        "eval_tokenized_manifest": None
        if args.eval_tokenized_manifest is None
        else str(args.eval_tokenized_manifest),
        "output_dir": str(output_dir),
        "start_step": start_step,
        "resume_checkpoint": None
        if resume_checkpoint is None
        else str(resume_checkpoint),
        "auto_resume": bool(args.auto_resume),
        "target_steps": args.steps,
        "max_seconds": args.max_seconds,
        "stop_buffer_seconds": args.stop_buffer_seconds,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "warmup_ratio": args.warmup_ratio,
        "effective_warmup_fraction": args.warmup_steps / max(args.steps, 1),
        "min_lr_ratio": args.min_lr_ratio,
        "scale": args.scale,
        "preset": args.preset,
        "parameter_counts": count_parameters(unwrap_model(model)),
        "config": asdict(config),
        "train_records": batcher_record_count(train_batcher),
        "eval_records": batcher_record_count(eval_batcher),
        "specialization_analysis_enabled": bool(args.analyze_specialization_at_end),
        "skip_end_specialization_on_time_stop": bool(
            args.skip_end_specialization_on_time_stop
        ),
        "specialization_jsonl": None
        if args.specialization_jsonl is None
        else str(args.specialization_jsonl),
        "specialization_max_records_per_category": args.specialization_max_records_per_category,
        "specialization_top_k": args.specialization_top_k,
        "specialization_knockout_programs": args.specialization_knockout_programs,
        "specialization_device": args.specialization_device,
        "specialization_checkpoints": args.specialization_checkpoints,
        "specialization_checkpoint_max_records_per_category": (
            args.specialization_checkpoint_max_records_per_category
        ),
        "specialization_checkpoint_run_knockouts": (
            args.specialization_checkpoint_run_knockouts
        ),
        "specialization_checkpoint_knockout_programs": (
            args.specialization_checkpoint_knockout_programs
        ),
        "category_route_weight": args.category_route_weight,
        "category_route_start_step": args.category_route_start_step,
        "category_route_warmup_steps": args.category_route_warmup_steps,
        "category_route_objective": args.category_route_objective,
        "semantic_routing_start_step": args.semantic_routing_start_step,
        "aux_loss_scale": args.aux_loss_scale,
        "aux_loss_warmup_steps": args.aux_loss_warmup_steps,
        "aux_loss_cadence": args.aux_loss_cadence,
        "optimization_health": {
            "min_gradient_norm": args.min_healthy_gradient_norm,
            "fail_on_unhealthy_optimization": bool(args.fail_on_unhealthy_optimization),
        },
        "semantic_route_allowed_programs": args.semantic_route_allowed_programs,
        "semantic_route_suppressed_programs": args.semantic_route_suppressed_programs,
        "category_route_categories": []
        if category_batcher is None
        else getattr(category_batcher, "categories", []),
        "per_device_batch_size": scale["batch_size"],
        "grad_accum_steps": scale["grad_accum_steps"],
        "effective_batch_sequences": (
            scale["batch_size"] * scale["grad_accum_steps"] * world_size
        ),
        "tokens_per_optimizer_step": (
            scale["batch_size"]
            * scale["grad_accum_steps"]
            * (train_batcher.seq_len - 1)
            * world_size
        ),
        "estimated_total_train_tokens": (
            args.steps
            * scale["batch_size"]
            * scale["grad_accum_steps"]
            * (train_batcher.seq_len - 1)
            * world_size
        ),
        "estimated_dataset_record_passes": (
            args.steps
            * scale["batch_size"]
            * scale["grad_accum_steps"]
            * world_size
            / max(batcher_record_count(train_batcher), 1)
        ),
    }
    if rank == 0:
        write_json(output_dir / "run_manifest.json", run_manifest)
        print(json.dumps(run_manifest, indent=2), flush=True)

    metrics = train_until_done(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        train_batcher=train_batcher,
        category_batcher=category_batcher,
        eval_batcher=eval_batcher,
        args=args,
        scale=scale,
        device=device,
        precision=precision,
        output_dir=output_dir,
        start_step=start_step,
        best_eval_loss=best_eval_loss,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
    )
    if rank == 0:
        if should_run_end_specialization(args, metrics):
            specialization_analysis = run_end_specialization_analysis(args, output_dir)
            if specialization_analysis is not None:
                metrics["specialization_analysis"] = specialization_analysis
        elif args.analyze_specialization_at_end:
            metrics["specialization_analysis"] = {
                "enabled": False,
                "skipped_reason": "stopped_for_time",
                "checkpoint": str(output_dir / "best.pt"),
            }
        write_json(output_dir / "final_summary.json", metrics)
        print(json.dumps({"final_summary": metrics}, indent=2), flush=True)
    if distributed:
        dist.destroy_process_group()


def resolved_scale(args: argparse.Namespace) -> dict[str, int]:
    scale = dict(MODEL_SCALES[args.scale])
    for arg_name, scale_name in [
        ("seq_len", "seq_len"),
        ("d_model", "d_model"),
        ("n_heads", "n_heads"),
        ("n_layers", "n_layers"),
        ("n_programs", "n_programs"),
        ("batch_size", "batch_size"),
        ("grad_accum_steps", "grad_accum_steps"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            scale[scale_name] = int(value)
    return scale


def build_training_config(args: argparse.Namespace, scale: dict[str, int]):
    overrides: dict[str, Any] = {}
    if args.mlp_ratio is not None:
        overrides["mlp_ratio"] = args.mlp_ratio
    if args.program_compute_type is not None:
        overrides["program_compute_type"] = args.program_compute_type
    if args.program_expert_rank is not None:
        overrides["program_expert_rank"] = args.program_expert_rank
    if args.routing_type is not None:
        overrides["routing_type"] = args.routing_type
    if args.routing_top_k is not None:
        overrides["routing_top_k"] = args.routing_top_k
    if args.routing_load_balance_weight is not None:
        overrides["routing_load_balance_weight"] = args.routing_load_balance_weight
    if args.semantic_route_allowed_programs is not None:
        overrides["semantic_route_allowed_programs"] = tuple(
            args.semantic_route_allowed_programs
        )
    if args.semantic_route_suppressed_programs is not None:
        overrides["semantic_route_suppressed_programs"] = tuple(
            args.semantic_route_suppressed_programs
        )
    if args.program_memory_update_type is not None:
        overrides["program_memory_update_type"] = args.program_memory_update_type
    if args.memory_adapter_type is not None:
        overrides["memory_adapter_type"] = args.memory_adapter_type
    if args.identity_attention_type is not None:
        overrides["identity_attention_type"] = args.identity_attention_type
    if args.memory_read_type is not None:
        overrides["memory_read_type"] = args.memory_read_type
    if args.content_read_steps is not None:
        overrides["content_read_steps"] = args.content_read_steps
    if args.content_read_gate_type is not None:
        overrides["content_read_gate_type"] = args.content_read_gate_type
    if args.content_read_confidence_margin is not None:
        overrides["content_read_confidence_margin"] = args.content_read_confidence_margin
    if args.content_read_cue_match_threshold is not None:
        overrides["content_read_cue_match_threshold"] = args.content_read_cue_match_threshold
    if args.content_read_query_top_k is not None:
        overrides["content_read_query_top_k"] = args.content_read_query_top_k
    if args.attention_window_size is not None:
        overrides["attention_window_size"] = args.attention_window_size
    if args.coalition_context_type is not None:
        overrides["coalition_context_type"] = args.coalition_context_type
    if args.coalition_context_scale is not None:
        overrides["coalition_context_scale"] = args.coalition_context_scale
    if args.program_residual_scale is not None:
        overrides["program_residual_scale"] = args.program_residual_scale
    if args.coherence_attention_scale is not None:
        overrides["coherence_attention_scale"] = args.coherence_attention_scale
    if args.memory_allocation_type is not None:
        overrides["memory_allocation_type"] = args.memory_allocation_type
    if args.memory_allocation_k is not None:
        overrides["memory_allocation_k"] = args.memory_allocation_k
    if args.memory_separation_weight is not None:
        overrides["memory_separation_weight"] = args.memory_separation_weight
    if args.preset == "memory_advantage":
        preset_builder = memory_advantage_config
    elif args.preset == "run5b_best_capability_fast":
        preset_builder = run5b_best_capability_fast_config
    elif args.preset == "cpu_research_tac":
        preset_builder = cpu_research_tac_config
    elif args.preset == "kaggle_fast_tac":
        preset_builder = kaggle_fast_tac_config
    elif args.preset == "run5b_capability":
        preset_builder = run5b_capability_config
    elif args.preset == "run5_capability":
        preset_builder = run5_capability_config
    else:
        preset_builder = best_tac_config
    config_kwargs: dict[str, Any] = {
        "vocab_size": args.vocab_size,
        "d_model": scale["d_model"],
        "n_heads": scale["n_heads"],
        "n_layers": scale["n_layers"],
        "max_seq_len": scale["seq_len"],
        "beta": args.beta,
        "energy_budget": args.energy_budget,
        **overrides,
    }
    if (
        args.preset
        not in {
            "run5_capability",
            "run5b_capability",
            "memory_advantage",
            "run5b_best_capability_fast",
            "kaggle_fast_tac",
            "cpu_research_tac",
        }
        or args.n_programs is not None
    ):
        config_kwargs["n_programs"] = scale["n_programs"]
    return preset_builder(**config_kwargs)


def should_collect_train_metrics(
    args: argparse.Namespace,
    *,
    step: int,
    specialization_checkpoints: set[int],
) -> bool:
    if step >= int(args.steps):
        return True
    if args.log_every and step % int(args.log_every) == 0:
        return True
    if args.eval_every and step % int(args.eval_every) == 0:
        return True
    if args.checkpoint_every and step % int(args.checkpoint_every) == 0:
        return True
    return step in specialization_checkpoints


def should_collect_auxiliary_losses(args: argparse.Namespace, *, step: int) -> bool:
    cadence = int(getattr(args, "aux_loss_cadence", 1))
    if cadence < 1:
        raise ValueError("aux_loss_cadence must be >= 1")
    return step % cadence == 0


def train_until_done(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    train_batcher: JsonlTextBatcher | JsonlCompletionBatcher,
    category_batcher: JsonlLabeledTextBatcher | JsonlLabeledCompletionBatcher | None,
    eval_batcher: JsonlTextBatcher | JsonlCompletionBatcher,
    args: argparse.Namespace,
    scale: dict[str, int],
    device: torch.device,
    precision: str,
    output_dir: Path,
    start_step: int,
    best_eval_loss: float,
    rank: int = 0,
    world_size: int = 1,
    distributed: bool = False,
) -> dict[str, Any]:
    model.train()
    started = time.perf_counter()
    metrics_path = output_dir / "metrics.jsonl"
    identity_states = None
    latest_metrics: dict[str, Any] = {}
    specialization_checkpoints = set(args.specialization_checkpoints or [])
    specialization_checkpoint_summaries = load_specialization_checkpoint_summaries(
        output_dir / "specialization_checkpoints" / "summaries.jsonl"
    )
    step = start_step
    optimizer.zero_grad(set_to_none=True)
    final_routing_type = unwrap_model(model).config.routing_type
    final_routing_top_k = unwrap_model(model).config.routing_top_k

    while step < args.steps:
        if should_stop_for_time(started, args.max_seconds, args.stop_buffer_seconds):
            break

        step += 1
        active_routing_type, active_routing_top_k = apply_semantic_routing_schedule(
            model,
            args,
            step=step,
            final_routing_type=final_routing_type,
            final_routing_top_k=final_routing_top_k,
        )
        active_category_route_weight = category_route_multiplier(args, step=step)
        total_loss = None
        total_next_token_loss = 0.0
        total_aux_loss = 0.0
        total_content_addressed_hit = 0.0
        total_content_synthesis_gate = 0.0
        total_content_gate_entropy = 0.0
        total_content_cue_cosine = 0.0
        total_content_reconsolidation_gate = 0.0
        total_program_memory_cosine = 0.0
        total_category_route_loss = 0.0
        aux_loss_components: dict[str, float] = {}
        weighted_aux_loss_components: dict[str, float] = {}
        aux_metrics: dict[str, float] = {}
        collect_train_metrics = should_collect_train_metrics(
            args,
            step=step,
            specialization_checkpoints=specialization_checkpoints,
        )
        collect_train_auxiliary = should_collect_auxiliary_losses(args, step=step)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(scale["grad_accum_steps"]):
            category_ids = None
            if category_batcher is None:
                input_ids, labels = train_batcher.next_batch(
                    scale["batch_size"],
                    device=device,
                )
            else:
                input_ids, labels, category_ids = category_batcher.next_batch(
                    scale["batch_size"],
                    device=device,
                )
            with autocast_context(device, precision):
                carried_state = identity_states if args.carry_state_across_batches else None
                output, next_token_loss, _ = forward_language_model_window(
                    model,
                    input_ids,
                    labels,
                    identity_states=carried_state,
                    chunked_state_within_batch=not args.no_chunked_state_within_batch,
                    collect_auxiliary=collect_train_auxiliary,
                    collect_metrics=collect_train_metrics,
                )
                aux_multiplier = aux_loss_multiplier(args, step=step)
                aux_loss = aux_multiplier * sum(
                    default_aux_weight(name, model) * loss
                    for name, loss in output.aux.losses.items()
                )
                route_loss = output.logits.new_zeros(())
                if category_ids is not None:
                    if args.category_route_objective == "selected_mi":
                        route_loss = selected_program_mi_loss(
                            output.aux.program_activations,
                            output.aux.selected_program_mask,
                            category_ids,
                            n_categories=len(category_batcher.categories),
                        )
                    elif args.category_route_objective == "mi":
                        route_loss = category_program_mi_loss(
                            output.aux.token_program_activations,
                            category_ids,
                            n_categories=len(category_batcher.categories),
                        )
                    else:
                        route_loss = category_route_loss(
                            output.aux.token_program_activations,
                            category_ids,
                        )
                weighted_route_loss = active_category_route_weight * route_loss
                loss = (
                    next_token_loss
                    + aux_loss
                    + weighted_route_loss
                ) / scale["grad_accum_steps"]
            backward(loss, scaler=scaler)
            identity_states = (
                detach_identity_states(output.identity_states)
                if args.carry_state_across_batches
                else None
            )
            total_loss = loss.detach() if total_loss is None else total_loss + loss.detach()
            total_next_token_loss += float(next_token_loss.detach())
            total_aux_loss += float(aux_loss.detach())
            total_category_route_loss += float(route_loss.detach())
            if collect_train_metrics:
                for name, loss_component in output.aux.losses.items():
                    scalar = scalar_float(loss_component)
                    if scalar is None:
                        continue
                    aux_loss_components[f"aux_loss_{name}"] = (
                        aux_loss_components.get(f"aux_loss_{name}", 0.0) + scalar
                    )
                    weighted_aux_loss_components[f"weighted_aux_loss_{name}"] = (
                        weighted_aux_loss_components.get(
                            f"weighted_aux_loss_{name}",
                            0.0,
                        )
                        + aux_multiplier * default_aux_weight(name, model) * scalar
                    )
                accumulate_scalar_metrics(
                    aux_metrics,
                    output.aux.metrics,
                    prefix="metric_",
                )
                total_content_addressed_hit += float(
                    output.aux.metrics.get(
                        "content_addressed_hit",
                        output.logits.new_zeros(()),
                    ).detach()
                )
                total_content_synthesis_gate += float(
                    output.aux.metrics.get(
                        "content_synthesis_gate",
                        output.logits.new_zeros(()),
                    ).detach()
                )
                total_content_gate_entropy += float(
                    output.aux.metrics.get(
                        "content_gate_entropy",
                        output.logits.new_zeros(()),
                    ).detach()
                )
                total_content_cue_cosine += float(
                    output.aux.metrics.get(
                        "content_cue_cosine",
                        output.logits.new_zeros(()),
                    ).detach()
                )
                total_content_reconsolidation_gate += float(
                    output.aux.metrics.get(
                        "content_reconsolidation_gate",
                        output.logits.new_zeros(()),
                    ).detach()
                )
                total_program_memory_cosine += float(
                    output.aux.metrics.get(
                        "program_memory_cosine",
                        output.logits.new_zeros(()),
                    ).detach()
                )

        if scaler is not None:
            scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        set_learning_rate(optimizer, step, args)
        grad_scaler_scale_before = scaler.get_scale() if scaler is not None else 1.0
        optimizer_step(optimizer, scaler=scaler)
        grad_scaler_scale_after = scaler.get_scale() if scaler is not None else 1.0

        elapsed = max(time.perf_counter() - started, 1e-9)
        sequences_seen = (
            (step - start_step)
            * scale["grad_accum_steps"]
            * scale["batch_size"]
            * world_size
        )
        tokens = (
            sequences_seen
            * (train_batcher.seq_len - 1)
        )
        latest_metrics = {
            "step": step,
            "loss": float(total_loss) if total_loss is not None else 0.0,
            "next_token_loss": total_next_token_loss / scale["grad_accum_steps"],
            "aux_loss": total_aux_loss / scale["grad_accum_steps"],
            "content_addressed_hit": total_content_addressed_hit
            / scale["grad_accum_steps"],
            "content_synthesis_gate": total_content_synthesis_gate
            / scale["grad_accum_steps"],
            "content_gate_entropy": total_content_gate_entropy
            / scale["grad_accum_steps"],
            "content_cue_cosine": total_content_cue_cosine
            / scale["grad_accum_steps"],
            "content_reconsolidation_gate": total_content_reconsolidation_gate
            / scale["grad_accum_steps"],
            "program_memory_cosine": total_program_memory_cosine
            / scale["grad_accum_steps"],
            "category_route_loss": total_category_route_loss
            / scale["grad_accum_steps"],
            "weighted_category_route_loss": (
                active_category_route_weight
                * total_category_route_loss
                / scale["grad_accum_steps"]
            ),
            "category_route_weight_active": active_category_route_weight,
            "active_routing_type": active_routing_type,
            "active_routing_top_k": active_routing_top_k,
            "aux_loss_multiplier": aux_loss_multiplier(args, step=step),
            "aux_loss_cadence": args.aux_loss_cadence,
            "auxiliary_loss_collected": collect_train_auxiliary,
            "gradient_norm": scalar_float(gradient_norm) or 0.0,
            "grad_clip_max_norm": 1.0,
            "grad_scaler_scale_before": float(grad_scaler_scale_before),
            "grad_scaler_scale_after": float(grad_scaler_scale_after),
            "grad_scaler_scale": float(grad_scaler_scale_after),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "tokens_seen": tokens,
            "sequences_seen": sequences_seen,
            "epoch_equivalent": sequences_seen / max(len(train_batcher.offsets), 1),
            "tokens_per_second": tokens / elapsed,
            "elapsed_seconds": elapsed,
        }
        latest_metrics.update(
            average_scalar_metrics(aux_loss_components, scale["grad_accum_steps"])
        )
        latest_metrics.update(
            average_scalar_metrics(
                weighted_aux_loss_components,
                scale["grad_accum_steps"],
            )
        )
        latest_metrics.update(
            average_scalar_metrics(aux_metrics, scale["grad_accum_steps"])
        )
        latest_metrics.update(cuda_memory_metrics(device))
        latest_metrics["optimization_health"] = optimization_health_status(
            gradient_norm=latest_metrics["gradient_norm"],
            precision=precision,
            grad_scaler_scale=latest_metrics["grad_scaler_scale"],
            min_gradient_norm=args.min_healthy_gradient_norm,
        )
        if (
            args.fail_on_unhealthy_optimization
            and latest_metrics["optimization_health"]["status"] != "passed"
        ):
            raise RuntimeError(
                "Optimizer health gate failed: "
                + json.dumps(latest_metrics["optimization_health"], sort_keys=True)
            )
        if rank == 0 and args.log_every and step % args.log_every == 0:
            print(json.dumps({"train": latest_metrics}), flush=True)
        if rank == 0 and args.checkpoint_every and step % args.checkpoint_every == 0:
            save_checkpoint(
                output_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                best_eval_loss=best_eval_loss,
                metrics=latest_metrics,
            )
        if args.eval_every and step % args.eval_every == 0:
            if rank == 0:
                eval_metrics = evaluate_language_model(
                    unwrap_model(model),
                    eval_batcher,
                    batches=args.eval_batches,
                    batch_size=args.eval_batch_size or scale["batch_size"],
                    device=device,
                    carry_state_across_batches=args.carry_state_across_batches,
                    chunked_state_within_batch=not args.no_chunked_state_within_batch,
                )
                latest_metrics = {**latest_metrics, "eval": eval_metrics}
                append_jsonl(metrics_path, latest_metrics)
                print(json.dumps({"eval": {"step": step, **eval_metrics}}), flush=True)
                model.train()
                if eval_metrics["loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["loss"]
                    save_checkpoint(
                        output_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=step,
                        best_eval_loss=best_eval_loss,
                        metrics=latest_metrics,
                    )
            if distributed:
                dist.barrier()
        if rank == 0 and step in specialization_checkpoints:
            summary = run_specialization_checkpoint_analysis(
                args,
                output_dir,
                step=step,
                model=unwrap_model(model),
                best_eval_loss=best_eval_loss,
                latest_metrics=latest_metrics,
            )
            specialization_checkpoint_summaries.append(summary)
            append_jsonl(
                output_dir / "specialization_checkpoints" / "summaries.jsonl",
                summary,
            )
        if distributed and step in specialization_checkpoints:
            dist.barrier()

    if rank == 0:
        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=step,
            best_eval_loss=best_eval_loss,
            metrics=latest_metrics,
        )
    if distributed:
        dist.barrier()
    return {
        "completed_steps": step,
        "target_steps": args.steps,
        "stopped_for_time": step < args.steps,
        "best_eval_loss": best_eval_loss,
        "latest_metrics": latest_metrics,
        "last_checkpoint": str(output_dir / "last.pt"),
        "best_checkpoint": str(output_dir / "best.pt"),
        "specialization_checkpoints": specialization_checkpoint_summaries,
    }


def run_end_specialization_analysis(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any] | None:
    if not args.analyze_specialization_at_end:
        return None

    checkpoint_path = output_dir / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = output_dir / "last.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Specialization analysis was requested, but neither best.pt nor last.pt exists."
        )

    artifact_dir = args.specialization_output_dir or (output_dir / "specialization")
    return run_specialization_analysis_for_checkpoint(
        args,
        checkpoint_path=checkpoint_path,
        artifact_dir=artifact_dir,
        max_records_per_category=args.specialization_max_records_per_category,
        knockout_programs=parse_knockout_programs(args.specialization_knockout_programs),
        run_knockouts=True,
        label="end",
    )


def should_run_end_specialization(
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> bool:
    if not args.analyze_specialization_at_end:
        return False
    if args.skip_end_specialization_on_time_stop and metrics.get("stopped_for_time"):
        return False
    return True


def load_specialization_checkpoint_summaries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    summaries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                summaries.append(json.loads(line))
    return summaries


def run_specialization_checkpoint_analysis(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    step: int,
    model: TACTransformerLM,
    best_eval_loss: float,
    latest_metrics: dict[str, Any],
) -> dict[str, Any]:
    artifact_dir = output_dir / "specialization_checkpoints" / f"step_{step:06d}"
    checkpoint_path = artifact_dir / "checkpoint.pt"
    save_model_snapshot(
        checkpoint_path,
        model=model,
        step=step,
        best_eval_loss=best_eval_loss,
        metrics=latest_metrics,
    )
    return run_specialization_analysis_for_checkpoint(
        args,
        checkpoint_path=checkpoint_path,
        artifact_dir=artifact_dir,
        max_records_per_category=args.specialization_checkpoint_max_records_per_category,
        knockout_programs=parse_knockout_programs(
            args.specialization_checkpoint_knockout_programs
        ),
        run_knockouts=args.specialization_checkpoint_run_knockouts,
        label=f"step_{step}",
    )


def run_specialization_analysis_for_checkpoint(
    args: argparse.Namespace,
    *,
    checkpoint_path: Path,
    artifact_dir: Path,
    max_records_per_category: int,
    knockout_programs: list[int] | None,
    run_knockouts: bool,
    label: str,
) -> dict[str, Any]:
    jsonl_path = resolve_specialization_jsonl(args.specialization_jsonl)
    report_path = artifact_dir / "program_specialization.json"
    csv_path = artifact_dir / "program_attribution.csv"

    started = time.perf_counter()
    print(
        json.dumps(
            {
                "specialization_analysis": "started",
                "label": label,
                "checkpoint": str(checkpoint_path),
                "jsonl": str(jsonl_path),
                "max_records_per_category": max_records_per_category,
                "run_knockouts": run_knockouts,
                "knockout_programs": (
                    "all" if run_knockouts and knockout_programs is None else knockout_programs
                ),
                "device": args.specialization_device,
            }
        ),
        flush=True,
    )
    report = analyze_program_specialization(
        checkpoint_path,
        jsonl_path,
        max_records_per_category=max_records_per_category,
        top_k=args.specialization_top_k,
        knockout_programs=knockout_programs,
        run_knockouts=run_knockouts,
        device=args.specialization_device,
    )
    write_json(report_path, report)
    write_attribution_csv(report, csv_path)
    summary = summarize_specialization_report(
        report,
        jsonl_path=jsonl_path,
        report_path=report_path,
        csv_path=csv_path,
        elapsed_seconds=time.perf_counter() - started,
        run_knockouts=run_knockouts,
        label=label,
    )
    print(json.dumps({"specialization_analysis": summary}, indent=2), flush=True)
    return summary


def resolve_specialization_jsonl(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Specialization JSONL not found: {explicit_path}")
        return explicit_path
    try:
        return discover_prepared_jsonl("hard_agentic_eval.generated.jsonl")
    except FileNotFoundError:
        return discover_prepared_jsonl("eval.prepared.jsonl")


def parse_knockout_programs(values: list[str] | None) -> list[int] | None:
    if values is None:
        return None
    lowered = [str(value).lower() for value in values]
    if lowered == ["all"]:
        return None
    return [int(value) for value in values]


def summarize_specialization_report(
    report: dict[str, Any],
    *,
    jsonl_path: Path,
    report_path: Path,
    csv_path: Path,
    elapsed_seconds: float,
    run_knockouts: bool = True,
    label: str = "end",
) -> dict[str, Any]:
    mutual_information = report.get("mutual_information", {})
    histogram = report.get("activation_histogram", {})
    by_category = histogram.get("by_category", {})
    dominant_program_by_category = {}
    for category, values in by_category.items():
        counts = list(values.get("top_program_counts", []))
        if not counts:
            continue
        program = max(range(len(counts)), key=lambda index: counts[index])
        dominant_program_by_category[category] = {
            "program": program,
            "count": counts[program],
            "records": values.get("records", 0),
        }

    ablations = sorted(
        report.get("ablations", []),
        key=lambda row: abs(float(row.get("loss_delta", 0.0))),
        reverse=True,
    )
    top_ablation_loss_deltas = [
        {
            "program": int(row.get("program", -1)),
            "loss_delta": float(row.get("loss_delta", 0.0)),
        }
        for row in ablations[:10]
    ]
    return {
        "enabled": True,
        "label": label,
        "checkpoint": report.get("checkpoint"),
        "checkpoint_step": int(report.get("checkpoint_step", 0)),
        "jsonl": str(jsonl_path),
        "report": str(report_path),
        "attribution_csv": str(csv_path),
        "elapsed_seconds": elapsed_seconds,
        "run_knockouts": run_knockouts,
        "records": len(report.get("records", [])),
        "categories": report.get("categories", []),
        "mi_bits": float(mutual_information.get("mi_bits", 0.0)),
        "normalized_mi": float(mutual_information.get("normalized_mi", 0.0)),
        "category_entropy_bits": float(
            mutual_information.get("category_entropy_bits", 0.0)
        ),
        "program_entropy_bits": float(
            mutual_information.get("program_entropy_bits", 0.0)
        ),
        "dominant_program_by_category": dominant_program_by_category,
        "top_ablation_loss_deltas": top_ablation_loss_deltas,
    }


def scalar_float(value: Any) -> float | None:
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        result = float(value.detach())
    elif isinstance(value, (int, float)):
        result = float(value)
    else:
        return None
    return result if math.isfinite(result) else None


def accumulate_scalar_metrics(
    totals: dict[str, float],
    values: dict[str, Any],
    *,
    prefix: str,
) -> None:
    for name, value in values.items():
        scalar = scalar_float(value)
        if scalar is None:
            continue
        key = f"{prefix}{name}"
        totals[key] = totals.get(key, 0.0) + scalar


def average_scalar_metrics(
    totals: dict[str, float],
    count: int,
) -> dict[str, float]:
    denominator = max(count, 1)
    return {name: value / denominator for name, value in sorted(totals.items())}


def cuda_memory_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    mib = 1024.0 * 1024.0
    return {
        "cuda_memory_allocated_mib": torch.cuda.memory_allocated(index) / mib,
        "cuda_memory_reserved_mib": torch.cuda.memory_reserved(index) / mib,
        "cuda_max_memory_allocated_mib": torch.cuda.max_memory_allocated(index) / mib,
        "cuda_max_memory_reserved_mib": torch.cuda.max_memory_reserved(index) / mib,
    }


def default_aux_weight(name: str, model: torch.nn.Module) -> float:
    config = unwrap_model(model).config
    weights = {
        "coherence": 0.05,
        "program_reuse": 0.05,
        "energy": 0.01,
        "multi_token": getattr(config, "multi_token_loss_weight", 0.0),
        "separation": getattr(config, "memory_separation_weight", 0.0),
        "content_cue_separation": getattr(
            config,
            "content_cue_separation_weight",
            0.0,
        ),
        "content_gate_entropy": getattr(
            config,
            "content_gate_entropy_weight",
            0.0,
        ),
        "routing_load_balance": getattr(
            config,
            "routing_load_balance_weight",
            0.0,
        ),
    }
    return weights.get(name, 0.0)


def aux_loss_multiplier(args: argparse.Namespace, *, step: int) -> float:
    scale = float(args.aux_loss_scale)
    if scale < 0.0:
        raise ValueError("aux_loss_scale must be non-negative")
    warmup_steps = int(args.aux_loss_warmup_steps)
    if warmup_steps < 0:
        raise ValueError("aux_loss_warmup_steps must be non-negative")
    if warmup_steps == 0:
        return scale
    return scale * min(max(step, 0), warmup_steps) / warmup_steps


def category_route_multiplier(args: argparse.Namespace, *, step: int) -> float:
    weight = float(args.category_route_weight)
    if weight < 0.0:
        raise ValueError("category_route_weight must be non-negative")
    start_step = int(args.category_route_start_step)
    warmup_steps = int(args.category_route_warmup_steps)
    if start_step < 0:
        raise ValueError("category_route_start_step must be non-negative")
    if warmup_steps < 0:
        raise ValueError("category_route_warmup_steps must be non-negative")
    if step < start_step:
        return 0.0
    if warmup_steps == 0:
        return weight
    return weight * min(max(step - start_step, 0), warmup_steps) / warmup_steps


def effective_routing_mode(args: argparse.Namespace, *, step: int) -> tuple[str, int]:
    target_type, target_top_k = target_routing_mode(args)
    if (
        step < int(args.semantic_routing_start_step)
        and target_type in {"base_semantic", "base_semantic_soft"}
    ):
        return "base", 1
    return target_type, target_top_k


def target_routing_mode(args: argparse.Namespace) -> tuple[str, int]:
    if args.preset == "kaggle_fast_tac":
        config = kaggle_fast_tac_config(vocab_size=args.vocab_size)
    elif args.preset == "cpu_research_tac":
        config = cpu_research_tac_config(vocab_size=args.vocab_size)
    elif args.preset == "run5b_best_capability_fast":
        config = run5b_best_capability_fast_config(vocab_size=args.vocab_size)
    elif args.preset == "memory_advantage":
        config = memory_advantage_config(vocab_size=args.vocab_size)
    elif args.preset == "run5b_capability":
        config = run5b_capability_config(vocab_size=args.vocab_size)
    elif args.preset == "run5_capability":
        config = run5_capability_config(vocab_size=args.vocab_size)
    else:
        config = best_tac_config(vocab_size=args.vocab_size)
    target_type = args.routing_type or config.routing_type
    target_top_k = args.routing_top_k or config.routing_top_k
    return target_type, int(target_top_k)


def apply_semantic_routing_schedule(
    model: torch.nn.Module,
    args: argparse.Namespace,
    *,
    step: int,
    final_routing_type: str,
    final_routing_top_k: int,
) -> tuple[str, int]:
    if (
        step < int(args.semantic_routing_start_step)
        and final_routing_type in {"base_semantic", "base_semantic_soft"}
    ):
        active_type, active_top_k = "base", 1
    else:
        active_type, active_top_k = final_routing_type, int(final_routing_top_k)
    base_model = unwrap_model(model)
    for module in base_model.modules():
        config = getattr(module, "config", None)
        if config is None:
            continue
        if hasattr(config, "routing_type"):
            object.__setattr__(config, "routing_type", active_type)
        if hasattr(config, "routing_top_k"):
            object.__setattr__(config, "routing_top_k", active_top_k)
    return active_type, active_top_k


def detach_identity_states(identity_states):
    if not identity_states:
        return None
    detached = []
    for state in identity_states:
        values = {}
        for field in fields(state):
            value = getattr(state, field.name)
            values[field.name] = value.detach() if torch.is_tensor(value) else value
        detached.append(type(state)(**values))
    return detached


def set_learning_rate(
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> None:
    if step <= args.warmup_steps:
        scale = step / max(args.warmup_steps, 1)
    else:
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        scale = args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine
    for group in optimizer.param_groups:
        group["lr"] = (
            args.learning_rate
            * float(group.get("tac_lr_mult", 1.0))
            * scale
        )


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    step: int,
    best_eval_loss: float,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "best_eval_loss": best_eval_loss,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": None if scaler is None else scaler.state_dict(),
            "config": asdict(unwrap_model(model).config),
            "metrics": metrics,
            "parameter_counts": count_parameters(unwrap_model(model)),
        },
        path,
    )


def save_model_snapshot(
    path: Path,
    *,
    model: torch.nn.Module,
    step: int,
    best_eval_loss: float,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "best_eval_loss": best_eval_loss,
            "model_state_dict": model.state_dict(),
            "config": asdict(model.config),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    model: TACTransformerLM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return int(checkpoint.get("step", 0)), float(checkpoint.get("best_eval_loss", math.inf))


def resolve_resume_checkpoint(
    resume: Path | None,
    output_dir: Path,
    *,
    auto_resume: bool,
    input_roots: Sequence[Path] | None = None,
) -> Path | None:
    if resume is not None:
        if not resume.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        return resume
    if not auto_resume:
        return None

    candidates = [output_dir / "last.pt", output_dir / "best.pt"]
    roots = list(input_roots) if input_roots is not None else [Path("/kaggle/input")]
    for root in roots:
        if not root.exists():
            continue
        candidates.extend(sorted(root.glob("**/last.pt")))
        candidates.extend(sorted(root.glob("**/best.pt")))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def select_device(requested: str) -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    if requested == "auto" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def init_distributed_if_needed() -> bool:
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return True


def distributed_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def distributed_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def wrap_distributed_model(
    model: TACTransformerLM,
    device: torch.device,
    distributed: bool,
) -> torch.nn.Module:
    if not distributed:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        output_device=device.index if device.type == "cuda" else None,
        find_unused_parameters=True,
    )


def unwrap_model(model: torch.nn.Module) -> TACTransformerLM:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model  # type: ignore[return-value]


def select_precision(requested: str, device: torch.device) -> str:
    if requested != "auto":
        return requested
    if device.type != "cuda":
        return "fp32"
    if torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp16"


def make_grad_scaler(
    device: torch.device,
    precision: str,
) -> torch.amp.GradScaler | None:
    if device.type == "cuda" and precision == "fp16":
        return torch.amp.GradScaler("cuda")
    return None


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision in {"fp16", "bf16"}:
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.amp.autocast("cuda", dtype=dtype)
    return torch.amp.autocast("cpu", enabled=False)


def backward(loss: torch.Tensor, *, scaler: torch.amp.GradScaler | None) -> None:
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()


def optimizer_step(
    optimizer: torch.optim.Optimizer,
    *,
    scaler: torch.amp.GradScaler | None,
) -> None:
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()


def optimization_health_status(
    *,
    gradient_norm: float,
    precision: str,
    grad_scaler_scale: float,
    min_gradient_norm: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    finite_gradient = math.isfinite(gradient_norm)
    if not finite_gradient:
        reasons.append("gradient_norm_non_finite")
    elif gradient_norm < min_gradient_norm:
        reasons.append("gradient_norm_below_min")
    if precision == "fp16" and (
        not math.isfinite(grad_scaler_scale) or grad_scaler_scale <= 0.0
    ):
        reasons.append("grad_scaler_scale_collapsed")
    return {
        "status": "failed" if reasons else "passed",
        "gradient_norm": float(gradient_norm),
        "min_gradient_norm": float(min_gradient_norm),
        "precision": precision,
        "grad_scaler_scale": float(grad_scaler_scale),
        "reasons": reasons,
    }


def should_stop_for_time(
    started: float,
    max_seconds: int,
    stop_buffer_seconds: int,
) -> bool:
    return time.perf_counter() - started >= max(max_seconds - stop_buffer_seconds, 0)


def discover_prepared_jsonl(filename: str) -> Path:
    candidates = [
        Path("/kaggle/input") / "tac-hard-agentic-corpus" / filename,
        Path("/kaggle/input")
        / "datasets"
        / "jeffwilliamsr"
        / "tac-hard-agentic-corpus"
        / filename,
        Path("/kaggle/input") / "tac-agentic-corpus" / filename,
        Path("/kaggle/input") / "tac-1b-agentic-corpus" / filename,
        Path("runs") / "prepared_corpus_agentic_hard" / filename,
        Path("runs") / "prepared_corpus_1b" / filename,
        Path("runs") / "prepared_corpus" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = sorted(kaggle_input.glob(f"**/{filename}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find {filename}. Pass --train-jsonl/--eval-jsonl explicitly."
    )


def default_kaggle_output_dir() -> Path:
    if bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/working").exists():
        return Path("/kaggle/working") / "best_tac_agentic"
    return Path("runs") / "best_tac_agentic"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row) + "\n")


def write_json(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
