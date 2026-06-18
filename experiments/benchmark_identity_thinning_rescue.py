from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import torch
from torch import Tensor
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_energy_balanced_tac import (
    VARIANT_WEIGHTS,
    _next_token_loss,
)
from experiments.benchmark_energy_based_model_probe import (
    corrupt_sequences,
    generate_structured_sequences,
    pair_accuracy,
)
from experiments.benchmark_energy_compression_tac import compression_metrics
from experiments.benchmark_identity_compression_phase_boundary import (
    TACStateEnergyModel,
    _identity_context,
    _identity_query,
    _make_generator,
    _mean_metric_value,
    _select_device,
    evaluate_phase_boundary_model,
    representational_thinning_metrics,
)
from tac_transformer import TACConfig
from tac_transformer.training import count_parameters


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/identity_thinning_rescue_tac209_2026_06_07")
DEFAULT_VARIANTS = [
    "compressed_control",
    "norm_floor_rescue",
    "marker_info_rescue",
    "combined_rescue",
]


@dataclass(frozen=True)
class RescueVariant:
    name: str
    norm_floor_weight: float = 0.0
    marker_info_weight: float = 0.0
    target_selected_identity_norm: float = 0.13


VARIANTS = {
    "compressed_control": RescueVariant("compressed_control"),
    "norm_floor_rescue": RescueVariant(
        "norm_floor_rescue",
        norm_floor_weight=1.0,
        target_selected_identity_norm=0.13,
    ),
    "marker_info_rescue": RescueVariant(
        "marker_info_rescue",
        marker_info_weight=0.10,
    ),
    "combined_rescue": RescueVariant(
        "combined_rescue",
        norm_floor_weight=1.0,
        marker_info_weight=0.10,
        target_selected_identity_norm=0.13,
    ),
}


class TACIdentityRescueModel(TACStateEnergyModel):
    def __init__(self, config: TACConfig):
        super().__init__(config)
        self.identity_probe_head = torch.nn.Linear(
            config.n_layers * config.n_programs * config.d_model,
            4,
        )

    def marker_logits_from_output(self, output: Any) -> Tensor:
        return self.identity_probe_head(_flatten_identity_memory_batch(output))


def _flatten_identity_memory_batch(output: Any) -> Tensor:
    memories = [
        state.program_memory.float().flatten(1)
        for state in getattr(output, "identity_states", [])
        if state.program_memory is not None and state.program_memory.numel() > 0
    ]
    if not memories:
        batch_size = output.logits.shape[0]
        return output.logits.new_zeros((batch_size, 1))
    return torch.cat(memories, dim=1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether explicit identity-state mass or marker-information "
            "auxiliary losses can rescue identity retention in the L1-failing regime."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--identity-trials", type=int, default=12)
    parser.add_argument("--distractor-counts", type=int, nargs="+", default=[0, 5, 10, 20, 50])
    parser.add_argument("--rerank-candidates", type=int, default=4)
    parser.add_argument("--compression-strength", type=float, default=0.125)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-programs", type=int, default=8)
    parser.add_argument("--energy-budget", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--corruption-rate", type=float, default=0.30)
    parser.add_argument("--energy-l2-weight", type=float, default=1e-4)
    parser.add_argument("--aux-batch-size", type=int, default=8)
    parser.add_argument("--aux-cadence", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--torch-threads", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = _select_device(args.device)
    result = run_identity_thinning_rescue(
        output_dir=args.output_dir,
        variant_names=args.variants,
        seeds=args.seeds,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        identity_trials=args.identity_trials,
        distractor_counts=args.distractor_counts,
        rerank_candidates=args.rerank_candidates,
        compression_strength=args.compression_strength,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_programs=args.n_programs,
        energy_budget=args.energy_budget,
        learning_rate=args.learning_rate,
        margin=args.margin,
        corruption_rate=args.corruption_rate,
        energy_l2_weight=args.energy_l2_weight,
        aux_batch_size=args.aux_batch_size,
        aux_cadence=args.aux_cadence,
        device=device,
    )
    print(
        json.dumps(
            {
                "artifact": str(args.output_dir / "identity_thinning_rescue.json"),
                "decision": result["decision"]["status"],
                "winner": result["decision"].get("winner"),
                "control_retention": result["decision"].get("control_retention"),
                "winner_retention": result["decision"].get("winner_retention"),
            },
            indent=2,
        ),
        flush=True,
    )


def run_identity_thinning_rescue(
    *,
    output_dir: str | Path,
    variant_names: Iterable[str],
    seeds: Iterable[int],
    steps: int,
    batch_size: int,
    eval_batches: int,
    eval_batch_size: int,
    identity_trials: int,
    distractor_counts: Iterable[int],
    rerank_candidates: int,
    compression_strength: float,
    seq_len: int,
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    n_programs: int,
    energy_budget: float,
    learning_rate: float,
    margin: float,
    corruption_rate: float,
    energy_l2_weight: float,
    aux_batch_size: int,
    aux_cadence: int,
    device: str | torch.device,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_variants = [VARIANTS[name] for name in variant_names]
    selected_seeds = [int(seed) for seed in seeds]
    selected_distractors = [int(value) for value in distractor_counts]
    rows = [
        run_rescue_variant(
            variant=variant,
            seed=seed,
            steps=steps,
            batch_size=batch_size,
            eval_batches=eval_batches,
            eval_batch_size=eval_batch_size,
            identity_trials=identity_trials,
            distractor_counts=selected_distractors,
            rerank_candidates=rerank_candidates,
            compression_strength=compression_strength,
            seq_len=seq_len,
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_programs=n_programs,
            energy_budget=energy_budget,
            learning_rate=learning_rate,
            margin=margin,
            corruption_rate=corruption_rate,
            energy_l2_weight=energy_l2_weight,
            aux_batch_size=aux_batch_size,
            aux_cadence=aux_cadence,
            device=device,
        )
        for variant in selected_variants
        for seed in selected_seeds
    ]
    result = aggregate_rescue_results(rows)
    result["settings"] = {
        "variants": [variant.name for variant in selected_variants],
        "seeds": selected_seeds,
        "steps": steps,
        "batch_size": batch_size,
        "eval_batches": eval_batches,
        "eval_batch_size": eval_batch_size,
        "identity_trials": identity_trials,
        "distractor_counts": selected_distractors,
        "rerank_candidates": rerank_candidates,
        "compression_strength": compression_strength,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "n_programs": n_programs,
        "energy_budget": energy_budget,
        "learning_rate": learning_rate,
        "margin": margin,
        "corruption_rate": corruption_rate,
        "energy_l2_weight": energy_l2_weight,
        "aux_batch_size": aux_batch_size,
        "aux_cadence": aux_cadence,
        "device": str(device),
    }
    result["per_seed"] = rows
    (output / "identity_thinning_rescue.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "RESULTS.md").write_text(format_markdown(result), encoding="utf-8")
    return result


def run_rescue_variant(
    *,
    variant: RescueVariant,
    seed: int,
    steps: int,
    batch_size: int,
    eval_batches: int,
    eval_batch_size: int,
    identity_trials: int,
    distractor_counts: list[int],
    rerank_candidates: int,
    compression_strength: float,
    seq_len: int,
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    n_programs: int,
    energy_budget: float,
    learning_rate: float,
    margin: float,
    corruption_rate: float,
    energy_l2_weight: float,
    aux_batch_size: int,
    aux_cadence: int,
    device: str | torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    config = TACConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_programs=n_programs,
        max_seq_len=seq_len,
        energy_budget=energy_budget,
        state_update_type="gated",
        program_compute_type="linear_expert",
        memory_write_type="novelty_gated",
        detach_identity_state=False,
    )
    model = TACIdentityRescueModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    weights = VARIANT_WEIGHTS["hybrid_energy_strong"]
    train_generator = _make_generator(seed + 70_000, device)
    started = time.perf_counter()
    latest: dict[str, float] = {}
    model.train()
    for step_index in range(steps):
        positives = generate_structured_sequences(
            batch_size,
            seq_len,
            vocab_size,
            generator=train_generator,
            device=device,
        )
        negatives = corrupt_sequences(
            positives,
            vocab_size,
            corruption_rate=corruption_rate,
            generator=train_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        positive_energy, positive_output = model(positives)
        negative_energy, _ = model(negatives)
        lm_loss = _next_token_loss(positive_output.logits, positives)
        contrastive_loss = F.softplus(positive_energy - negative_energy + margin).mean()
        energy_l2 = (positive_energy.pow(2).mean() + negative_energy.pow(2).mean()) * 0.5
        compression = compression_metrics(positive_output)
        compression_loss = compression_strength * compression["activation_l1"]
        aux_losses = _rescue_auxiliary_losses(
            model,
            variant=variant,
            batch_size=aux_batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            generator=train_generator,
            device=device,
            enabled=step_index % max(aux_cadence, 1) == 0,
        )
        loss = (
            weights["lm_weight"] * lm_loss
            + weights["energy_weight"] * contrastive_loss
            + energy_l2_weight * energy_l2
            + compression_loss
            + aux_losses["norm_floor_loss"]
            + aux_losses["marker_info_loss"]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        latest = {
            "loss": float(loss.detach()),
            "lm_loss": float(lm_loss.detach()),
            "contrastive_loss": float(contrastive_loss.detach()),
            "energy_l2": float(energy_l2.detach()),
            "compression_loss": float(compression_loss.detach()),
            "norm_floor_loss": float(aux_losses["norm_floor_loss"].detach()),
            "marker_info_loss": float(aux_losses["marker_info_loss"].detach()),
            "marker_accuracy": float(aux_losses["marker_accuracy"].detach()),
            "aux_selected_identity_norm": float(
                aux_losses["selected_identity_norm"].detach()
            ),
            "energy_pair_accuracy": float(pair_accuracy(positive_energy, negative_energy).detach()),
            "activation_density": float(compression["activation_density"].detach()),
        }

    final_eval = evaluate_phase_boundary_model(
        model,
        batches=eval_batches,
        batch_size=eval_batch_size,
        identity_trials=identity_trials,
        distractor_counts=distractor_counts,
        rerank_candidates=rerank_candidates,
        seq_len=seq_len,
        vocab_size=vocab_size,
        corruption_rate=corruption_rate,
        seed=seed + 80_000,
        device=device,
    )
    final_eval.update(
        evaluate_identity_transfer_generalization(
            model,
            trials=identity_trials,
            distractor_counts=distractor_counts,
            seq_len=seq_len,
            vocab_size=vocab_size,
            seed=seed + 110_000,
            device=device,
        )
    )
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "variant": variant.name,
        "seed": seed,
        "compression_strength": compression_strength,
        "rescue": {
            "norm_floor_weight": variant.norm_floor_weight,
            "marker_info_weight": variant.marker_info_weight,
            "target_selected_identity_norm": variant.target_selected_identity_norm,
        },
        "config": asdict(config),
        "parameter_counts": count_parameters(model),
        "train": {
            **latest,
            "steps": steps,
            "examples_per_second": steps * batch_size * 2 / elapsed,
        },
        "final_eval": final_eval,
    }


@torch.no_grad()
def evaluate_identity_transfer_generalization(
    model: TACIdentityRescueModel,
    *,
    trials: int,
    distractor_counts: list[int],
    seq_len: int,
    vocab_size: int,
    seed: int,
    device: str | torch.device,
) -> dict[str, float]:
    model.eval()
    generator = _make_generator(seed, device)
    transfer_totals = {
        f"identity_transfer_n{count}": [] for count in distractor_counts
    }
    long_totals = {
        f"identity_transfer_long_n{count}": [] for count in distractor_counts
    }
    step_values = [1, 2, 3, 5]
    span = vocab_size - 8
    for _ in range(trials):
        marker = int(torch.randint(0, 4, (1,), generator=generator, device=device).item())
        start = int(torch.randint(0, span, (1,), generator=generator, device=device).item())
        step = step_values[marker]
        context = _identity_context(
            marker=marker,
            start=start,
            step=step,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device,
        )
        _, context_output = model(context)
        for distractor_count in distractor_counts:
            states = context_output.identity_states
            for index in range(distractor_count):
                distractor_marker = (marker + 1 + index) % 4
                distractor_start = (start + 11 + index * 5) % span
                distractor = _identity_context(
                    marker=distractor_marker,
                    start=distractor_start,
                    step=step_values[distractor_marker],
                    seq_len=seq_len,
                    vocab_size=vocab_size,
                    device=device,
                )
                _, distractor_output = model(distractor, identity_states=states)
                states = distractor_output.identity_states

            novel_start = (start + 17) % span
            wrong_step = step_values[(marker + 1) % 4]
            transfer_totals[f"identity_transfer_n{distractor_count}"].append(
                _identity_rule_pair_accuracy(
                    model,
                    states=states,
                    start=novel_start,
                    positive_step=step,
                    negative_step=wrong_step,
                    offset=seq_len,
                    seq_len=seq_len,
                    vocab_size=vocab_size,
                    device=device,
                )
            )
            long_totals[f"identity_transfer_long_n{distractor_count}"].append(
                _identity_rule_pair_accuracy(
                    model,
                    states=states,
                    start=novel_start,
                    positive_step=step,
                    negative_step=wrong_step,
                    offset=seq_len * 3,
                    seq_len=seq_len,
                    vocab_size=vocab_size,
                    device=device,
                )
            )

    metrics = {
        name: mean(values)
        for totals in (transfer_totals, long_totals)
        for name, values in totals.items()
    }
    metrics["identity_transfer_mean"] = mean(
        metrics[f"identity_transfer_n{count}"] for count in distractor_counts
    )
    metrics["identity_transfer_long_mean"] = mean(
        metrics[f"identity_transfer_long_n{count}"] for count in distractor_counts
    )
    metrics["identity_transfer_combined_mean"] = mean(
        [metrics["identity_transfer_mean"], metrics["identity_transfer_long_mean"]]
    )
    return metrics


def _identity_rule_pair_accuracy(
    model: TACIdentityRescueModel,
    *,
    states: Any,
    start: int,
    positive_step: int,
    negative_step: int,
    offset: int,
    seq_len: int,
    vocab_size: int,
    device: str | torch.device,
) -> float:
    positive = _identity_query(
        start=start,
        step=positive_step,
        seq_len=seq_len,
        vocab_size=vocab_size,
        offset=offset,
        device=device,
    )
    negative = _identity_query(
        start=start,
        step=negative_step,
        seq_len=seq_len,
        vocab_size=vocab_size,
        offset=offset,
        device=device,
    )
    positive_energy, _ = model(positive, identity_states=states)
    negative_energy, _ = model(negative, identity_states=states)
    return float(pair_accuracy(positive_energy, negative_energy).detach())


def _rescue_auxiliary_losses(
    model: TACIdentityRescueModel,
    *,
    variant: RescueVariant,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    generator: torch.Generator,
    device: str | torch.device,
    enabled: bool,
) -> dict[str, Tensor]:
    zero = next(model.parameters()).new_zeros(())
    if not enabled or (variant.norm_floor_weight <= 0.0 and variant.marker_info_weight <= 0.0):
        return {
            "norm_floor_loss": zero,
            "marker_info_loss": zero,
            "marker_accuracy": zero,
            "selected_identity_norm": zero,
        }
    contexts, labels = _identity_auxiliary_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        generator=generator,
        device=device,
    )
    _, output = model(contexts)
    representation = representational_thinning_metrics(
        output,
        energy_features=model.energy_features_from_output(output),
    )
    selected_identity_norm = representation["identity_state_selected_norm"]
    norm_floor_loss = (
        variant.norm_floor_weight
        * F.relu(variant.target_selected_identity_norm - selected_identity_norm).pow(2)
    )
    logits = model.marker_logits_from_output(output)
    raw_marker_loss = F.cross_entropy(logits, labels)
    marker_info_loss = variant.marker_info_weight * raw_marker_loss
    marker_accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    return {
        "norm_floor_loss": norm_floor_loss,
        "marker_info_loss": marker_info_loss,
        "marker_accuracy": marker_accuracy,
        "selected_identity_norm": selected_identity_norm,
    }


def _identity_auxiliary_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    generator: torch.Generator,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    step_values = [1, 2, 3, 5]
    contexts = []
    labels = []
    for _ in range(batch_size):
        marker = int(torch.randint(0, 4, (1,), generator=generator, device=device).item())
        start = int(
            torch.randint(0, vocab_size - 8, (1,), generator=generator, device=device).item()
        )
        contexts.append(
            _identity_context(
                marker=marker,
                start=start,
                step=step_values[marker],
                seq_len=seq_len,
                vocab_size=vocab_size,
                device=device,
            ).squeeze(0)
        )
        labels.append(marker)
    return torch.stack(contexts, dim=0), torch.tensor(labels, device=device, dtype=torch.long)


def aggregate_rescue_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    summaries = []
    for variant in sorted({row["variant"] for row in rows}):
        variant_rows = [row for row in rows if row["variant"] == variant]
        final = _mean_final_metrics(variant_rows)
        summaries.append(
            {
                "variant": variant,
                "seeds": [row["seed"] for row in variant_rows],
                **final,
                "examples_per_second": mean(
                    row.get("train", {}).get("examples_per_second", 0.0)
                    for row in variant_rows
                ),
            }
        )
    decision = make_rescue_decision(summaries)
    return {
        "schema": "identity_thinning_rescue.v1",
        "hypothesis": (
            "If representational thinning causes identity failure, an auxiliary "
            "loss that preserves identity-state mass or marker information should "
            "recover identity retention in the high-compression regime while "
            "leaving compression and energy behavior largely intact."
        ),
        "decision": decision,
        "variant_summaries": summaries,
    }


def _mean_final_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = rows[0]["final_eval"].keys()
    return {
        name: _mean_metric_value([row["final_eval"][name] for row in rows])
        for name in metric_names
    }


def make_rescue_decision(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    control = next(
        (row for row in summaries if row["variant"] == "compressed_control"),
        summaries[0],
    )
    candidates = [row for row in summaries if row is not control]
    winner = max(
        candidates or [control],
        key=lambda row: (
            float(row["identity_retention_mean"]),
            float(row["compression_score"]),
            float(row["energy_pair_accuracy"]),
        ),
    )
    retention_gain = float(winner["identity_retention_mean"]) - float(
        control["identity_retention_mean"]
    )
    compression_delta = float(winner["compression_score"]) - float(control["compression_score"])
    energy_delta = float(winner["energy_pair_accuracy"]) - float(
        control["energy_pair_accuracy"]
    )
    rerank_delta = float(winner["rerank_accuracy"]) - float(control["rerank_accuracy"])
    success = (
        retention_gain >= 0.08
        and compression_delta >= -0.04
        and energy_delta >= -0.03
        and rerank_delta >= -0.05
    )
    return {
        "status": "identity_rescue_supported" if success else "identity_rescue_not_supported",
        "winner": winner["variant"],
        "control_retention": control["identity_retention_mean"],
        "winner_retention": winner["identity_retention_mean"],
        "retention_gain": retention_gain,
        "compression_delta": compression_delta,
        "energy_delta": energy_delta,
        "rerank_delta": rerank_delta,
        "claim": (
            "identity retention recovered under explicit identity-state rescue"
            if success
            else "auxiliary identity-state rescue did not recover retention under current settings"
        ),
    }


def format_markdown(result: dict[str, Any]) -> str:
    decision = result["decision"]
    settings = result["settings"]
    lines = [
        "# Identity Thinning Rescue",
        "",
        f"Schema: `{result['schema']}`",
        "",
        "## Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Winner: `{decision.get('winner')}`",
        f"- Retention gain over control: `{decision.get('retention_gain')}`",
        f"- Compression delta: `{decision.get('compression_delta')}`",
        f"- Energy delta: `{decision.get('energy_delta')}`",
        f"- Rerank delta: `{decision.get('rerank_delta')}`",
        "",
        "## Settings",
        "",
        f"- Variants: {settings['variants']}",
        f"- Seeds: {settings['seeds']}",
        f"- Steps per cell: {settings['steps']}",
        f"- Compression strength: {settings['compression_strength']}",
        "",
        "## Variant Summary",
        "",
        "| Variant | Retention | Transfer | Long Transfer | Compression | Density | Sel Act | Sel Id Norm | Id Probe MI | Energy MI | Energy | Rerank | LM |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["variant_summaries"]:
        lines.append(
            "| {variant} | {retention:.3f} | {transfer:.3f} | {long_transfer:.3f} | {compression:.3f} | {density:.3f} | {sel_act:.3f} | {id_sel:.3f} | {id_mi:.3f} | {energy_mi:.3f} | {energy:.3f} | {rerank:.3f} | {lm:.3f} |".format(
                variant=row["variant"],
                retention=row["identity_retention_mean"],
                transfer=row.get("identity_transfer_mean", 0.0),
                long_transfer=row.get("identity_transfer_long_mean", 0.0),
                compression=row["compression_score"],
                density=row["activation_density"],
                sel_act=row["selected_activation_mean"],
                id_sel=row["identity_state_selected_norm"],
                id_mi=row["identity_state_probe_mi_fraction"],
                energy_mi=row["energy_feature_probe_mi_fraction"],
                energy=row["energy_pair_accuracy"],
                rerank=row["rerank_accuracy"],
                lm=row["lm_accuracy"],
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
