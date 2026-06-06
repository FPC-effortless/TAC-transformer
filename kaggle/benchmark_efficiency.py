from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import TACConfig
from tac_transformer.training import benchmark_data_energy_efficiency


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TAC data/energy efficiency budget curve benchmark."
    )
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-programs", type=int, default=16)
    parser.add_argument("--n-sink-programs", type=int, default=0)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--energy-budget", type=float, default=4.0)
    parser.add_argument("--norm-type", choices=["layernorm", "rmsnorm"], default="layernorm")
    parser.add_argument("--mlp-type", choices=["gelu", "swiglu"], default="gelu")
    parser.add_argument("--position-type", choices=["learned", "rope"], default="learned")
    parser.add_argument(
        "--program-compute-type",
        choices=["embedding", "linear_expert", "sparse_linear_expert"],
        default="embedding",
    )
    parser.add_argument(
        "--routing-type",
        choices=["energy", "expert_choice", "base", "hash", "sparse_ensemble", "base_semantic", "base_semantic_soft", "authority_gated"],
        default="energy",
    )
    parser.add_argument("--routing-top-k", type=int, default=1)
    parser.add_argument("--state-update-type", choices=["fixed", "gated"], default="fixed")
    parser.add_argument("--memory-write-type", choices=["standard", "novelty_gated"], default="standard")
    parser.add_argument("--memory-tier-type", choices=["flat", "hierarchical"], default="flat")
    parser.add_argument(
        "--memory-read-type",
        choices=["none", "program_memory", "pattern_completion", "content_addressed"],
        default="none",
    )
    parser.add_argument("--pattern-store-size", type=int, default=4)
    parser.add_argument("--content-store-size", type=int, default=8)
    parser.add_argument("--content-read-steps", type=int, default=1)
    parser.add_argument(
        "--content-read-gate-type",
        choices=["learned", "confidence", "synthesis"],
        default="learned",
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
        default="none",
    )
    parser.add_argument("--attention-window-size", type=int, default=None)
    parser.add_argument(
        "--memory-adapter-type",
        choices=["none", "residual", "gated_residual"],
        default="none",
    )
    parser.add_argument("--memory-lookup-type", choices=["none", "product_key"], default="none")
    parser.add_argument("--memory-lookup-slots", type=int, default=64)
    parser.add_argument("--residual-stream-type", choices=["single", "dual_stream"], default="single")
    parser.add_argument("--n-prediction-heads", type=int, default=1)
    parser.add_argument("--multi-token-loss-weight", type=float, default=0.0)
    parser.add_argument("--memory-separation-weight", type=float, default=0.0)
    parser.add_argument("--memory-allocation-type", choices=["stability", "creb"], default="stability")
    parser.add_argument("--memory-allocation-k", type=int, default=1)
    parser.add_argument("--creb-alpha", type=float, default=1.0)
    parser.add_argument("--creb-beta", type=float, default=1.0)
    parser.add_argument("--creb-gamma", type=float, default=0.25)
    parser.add_argument("--creb-delta", type=float, default=0.0)
    parser.add_argument("--creb-frequency-decay", type=float, default=0.9)
    parser.add_argument("--memory-reconsolidate", action="store_true")
    parser.add_argument(
        "--reconsolidate-gate-type",
        choices=["linear", "mlp"],
        default="linear",
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 60, 120])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.0)
    parser.add_argument("--memory-read-loss-weight", type=float, default=0.0)
    parser.add_argument("--memory-injection-weight", type=float, default=0.0)
    parser.add_argument("--memory-adapter-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--match-baseline-parameters", action="store_true")
    parser.add_argument("--min-value-accuracy-delta", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = TACConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        n_sink_programs=args.n_sink_programs,
        max_seq_len=args.seq_len,
        beta=args.beta,
        energy_budget=args.energy_budget,
        norm_type=args.norm_type,
        mlp_type=args.mlp_type,
        position_type=args.position_type,
        program_compute_type=args.program_compute_type,
        routing_type=args.routing_type,
        routing_top_k=args.routing_top_k,
        state_update_type=args.state_update_type,
        memory_write_type=args.memory_write_type,
        memory_tier_type=args.memory_tier_type,
        memory_lookup_type=args.memory_lookup_type,
        memory_lookup_slots=args.memory_lookup_slots,
        memory_read_type=args.memory_read_type,
        pattern_store_size=args.pattern_store_size,
        content_store_size=args.content_store_size,
        content_read_steps=args.content_read_steps,
        content_read_gate_type=args.content_read_gate_type,
        identity_attention_type=args.identity_attention_type,
        attention_window_size=args.attention_window_size,
        memory_adapter_type=args.memory_adapter_type,
        residual_stream_type=args.residual_stream_type,
        n_prediction_heads=args.n_prediction_heads,
        multi_token_loss_weight=args.multi_token_loss_weight,
        memory_separation_weight=args.memory_separation_weight,
        memory_allocation_type=args.memory_allocation_type,
        memory_allocation_k=args.memory_allocation_k,
        creb_alpha=args.creb_alpha,
        creb_beta=args.creb_beta,
        creb_gamma=args.creb_gamma,
        creb_delta=args.creb_delta,
        creb_frequency_decay=args.creb_frequency_decay,
        memory_reconsolidate=args.memory_reconsolidate,
        reconsolidate_gate_type=args.reconsolidate_gate_type,
        detach_identity_state=(
            args.memory_read_loss_weight == 0.0
            and args.memory_adapter_weight == 0.0
            and args.identity_attention_type == "none"
        ),
    )
    result = benchmark_data_energy_efficiency(
        config,
        budgets=args.budgets,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        value_loss_weight=args.value_loss_weight,
        memory_read_loss_weight=args.memory_read_loss_weight,
        memory_injection_weight=args.memory_injection_weight,
        memory_adapter_weight=args.memory_adapter_weight,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        device=select_device(args.device),
        match_baseline_parameters=args.match_baseline_parameters,
        min_value_accuracy_delta=args.min_value_accuracy_delta,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
