from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import best_chunked_memory_training_kwargs, best_tac_config
from tac_transformer.training import benchmark_chunked_memory


VARIANTS: dict[str, dict[str, object]] = {
    "hash_attention_best": {
        "sequence_mixer_type": "attention",
        "program_compute_type": "linear_expert",
    },
    "jamba_alternating": {
        "sequence_mixer_type": "alternating",
        "program_compute_type": "linear_expert",
    },
    "jamba_hybrid": {
        "sequence_mixer_type": "hybrid",
        "program_compute_type": "linear_expert",
    },
    "state_only": {
        "sequence_mixer_type": "state",
        "program_compute_type": "linear_expert",
    },
    "blackmamba_sparse_state": {
        "sequence_mixer_type": "state",
        "program_compute_type": "sparse_linear_expert",
    },
    "blackmamba_sparse_hybrid": {
        "sequence_mixer_type": "hybrid",
        "program_compute_type": "sparse_linear_expert",
    },
    "mamba_selective_state": {
        "sequence_mixer_type": "selective_state",
        "program_compute_type": "linear_expert",
    },
    "rwkv_time_mix": {
        "sequence_mixer_type": "rwkv",
        "program_compute_type": "linear_expert",
    },
    "xlstm_gated": {
        "sequence_mixer_type": "xlstm",
        "program_compute_type": "linear_expert",
    },
    "rwkv_sparse_expert": {
        "sequence_mixer_type": "rwkv",
        "program_compute_type": "sparse_linear_expert",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Jamba/BlackMamba-inspired TAC sequence-mixer ablations."
    )
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/benchmarks/hybrid_mixer_matrix_2026_05_28"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_kwargs = best_chunked_memory_training_kwargs()
    per_seed = []

    selected_variants = {
        name: overrides
        for name, overrides in VARIANTS.items()
        if args.variants is None or name in args.variants
    }

    for variant_name, overrides in selected_variants.items():
        for seed in args.seeds:
            config = best_tac_config(
                vocab_size=64,
                d_model=64,
                n_heads=4,
                n_layers=2,
                n_programs=16,
                max_seq_len=16,
                beta=1.5,
                energy_budget=4.0,
                **overrides,
            )
            result = benchmark_chunked_memory(
                config,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                eval_batches=args.eval_batches,
                eval_batch_size=args.eval_batch_size,
                seed=seed,
                device=device,
                match_baseline_parameters=True,
                min_value_accuracy_delta=0.0,
                **training_kwargs,
            )
            result["variant"] = variant_name
            result["seed"] = seed
            per_seed.append(result)
            path = args.output_dir / f"{variant_name}_seed{seed}.json"
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(_one_line_result(variant_name, seed, result), flush=True)

    aggregate = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_batches": args.eval_batches,
        "seeds": args.seeds,
        "variants": {
            name: _aggregate_variant(
                [run for run in per_seed if run["variant"] == name]
            )
            for name in selected_variants
        },
    }
    (args.output_dir / "per_seed_hybrid_mixer_matrix.json").write_text(
        json.dumps(per_seed, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "aggregate_hybrid_mixer_matrix.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "RESULTS.md").write_text(
        _format_results_markdown(aggregate),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2), flush=True)


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _one_line_result(variant_name: str, seed: int, result: dict[str, object]) -> str:
    tac_probe = result["tac"]["chunked_probe"]
    baseline_probe = result["baseline"]["chunked_probe"]
    return (
        f"{variant_name} seed={seed} "
        f"carry={tac_probe['carry']['value_accuracy']:.4f} "
        f"reset={tac_probe['reset']['value_accuracy']:.4f} "
        f"shuffled={tac_probe['shuffled']['value_accuracy']:.4f} "
        f"baseline={baseline_probe['carry']['value_accuracy']:.4f}"
    )


def _aggregate_variant(runs: list[dict[str, object]]) -> dict[str, float | int]:
    def mean(path: tuple[str, ...]) -> float:
        values = []
        for run in runs:
            value: object = run
            for part in path:
                value = value[part]  # type: ignore[index]
            values.append(float(value))
        return sum(values) / max(len(values), 1)

    carry = mean(("tac", "chunked_probe", "carry", "value_accuracy"))
    reset = mean(("tac", "chunked_probe", "reset", "value_accuracy"))
    shuffled = mean(("tac", "chunked_probe", "shuffled", "value_accuracy"))
    baseline = mean(("baseline", "chunked_probe", "carry", "value_accuracy"))
    train_tps = mean(("tac", "train", "tokens_per_second"))
    baseline_tps = mean(("baseline", "train", "tokens_per_second"))
    effective = 0
    for run in runs:
        tac_probe = run["tac"]["chunked_probe"]
        baseline_probe = run["baseline"]["chunked_probe"]
        if (
            tac_probe["carry"]["value_accuracy"]
            > tac_probe["reset"]["value_accuracy"]
            and tac_probe["carry"]["value_accuracy"]
            > tac_probe["shuffled"]["value_accuracy"]
            and tac_probe["carry"]["value_accuracy"]
            > baseline_probe["carry"]["value_accuracy"]
        ):
            effective += 1
    return {
        "carry_accuracy": carry,
        "reset_accuracy": reset,
        "shuffled_accuracy": shuffled,
        "baseline_accuracy": baseline,
        "tac_baseline_gap": carry - baseline,
        "carry_reset_delta": carry - reset,
        "shuffled_penalty": carry - shuffled,
        "train_tps_ratio": train_tps / max(baseline_tps, 1e-9),
        "effective": effective,
        "runs": len(runs),
    }


def _format_results_markdown(aggregate: dict[str, object]) -> str:
    lines = [
        "# Hybrid Mixer Matrix",
        "",
        "Jamba/BlackMamba-inspired TAC sequence-mixer ablations.",
        "",
        "| Variant | Effective | Carry | Reset | Shuffled | Baseline | Gap | TPS Ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    variants = aggregate["variants"]
    for name, metrics in variants.items():
        lines.append(
            "| {name} | {effective}/{runs} | {carry:.4f} | {reset:.4f} | "
            "{shuffled:.4f} | {baseline:.4f} | {gap:.4f} | {tps:.4f} |".format(
                name=name,
                effective=metrics["effective"],
                runs=metrics["runs"],
                carry=metrics["carry_accuracy"],
                reset=metrics["reset_accuracy"],
                shuffled=metrics["shuffled_accuracy"],
                baseline=metrics["baseline_accuracy"],
                gap=metrics["tac_baseline_gap"],
                tps=metrics["train_tps_ratio"],
            )
        )
    lines.append("")
    lines.append(
        "Effective means TAC carry accuracy beat reset, shuffled-state, and the parameter-matched baseline on that seed."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
