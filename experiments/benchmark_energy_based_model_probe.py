from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import TACConfig, TACTransformerLM
from tac_transformer.training import count_parameters


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/energy_based_model_probe_2026_06_07")


class TACSequenceEnergyModel(nn.Module):
    """Small EBM adapter over TAC hidden states and identity-field diagnostics."""

    def __init__(self, config: TACConfig):
        super().__init__()
        self.config = config
        self.backbone = TACTransformerLM(config)
        feature_dim = config.d_model + 4
        self.energy_head = nn.Sequential(
            nn.Linear(feature_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Any]:
        output = self.backbone(
            input_ids,
            collect_auxiliary=True,
            collect_metrics=False,
            update_content_memory=False,
        )
        pooled_hidden = output.hidden_states.mean(dim=1)
        aux_features = torch.stack(
            [
                output.aux.used_energy / max(self.config.energy_budget, 1e-6),
                output.aux.selected_program_mask.float().mean(dim=-1),
                output.aux.coherence.mean(dim=(1, 2)),
                output.aux.program_activations.mean(dim=-1),
            ],
            dim=-1,
        )
        energy = self.energy_head(torch.cat([pooled_hidden, aux_features], dim=-1))
        return energy.squeeze(-1), output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether TAC can behave as an energy-based model by training "
            "a scalar sequence energy with positive/negative contrastive pairs."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19])
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=6)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=24)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-programs", type=int, default=6)
    parser.add_argument("--energy-budget", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--corruption-rate", type=float, default=0.30)
    parser.add_argument("--energy-l2-weight", type=float, default=1e-4)
    parser.add_argument("--routing-energy-weight", type=float, default=0.0)
    parser.add_argument("--min-pair-accuracy", type=float, default=0.70)
    parser.add_argument("--min-energy-gap", type=float, default=0.20)
    parser.add_argument("--min-accuracy-gain", type=float, default=0.10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--torch-threads", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = _select_device(args.device)
    result = run_energy_based_model_probe(
        output_dir=args.output_dir,
        seeds=args.seeds,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
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
        routing_energy_weight=args.routing_energy_weight,
        min_pair_accuracy=args.min_pair_accuracy,
        min_energy_gap=args.min_energy_gap,
        min_accuracy_gain=args.min_accuracy_gain,
        device=device,
    )
    print(
        json.dumps(
            {
                "artifact": str(args.output_dir / "energy_based_model_probe.json"),
                "verdict": result["verdict"],
                "final_pair_accuracy": result["summary"]["final_pair_accuracy"],
                "final_energy_gap": result["summary"]["final_energy_gap"],
                "routing_energy_best_pair_accuracy": result["summary"][
                    "routing_energy_best_pair_accuracy"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


def run_energy_based_model_probe(
    *,
    output_dir: str | Path,
    seeds: Iterable[int] = (7, 19),
    steps: int = 240,
    batch_size: int = 8,
    eval_batches: int = 6,
    eval_batch_size: int = 8,
    seq_len: int = 16,
    vocab_size: int = 64,
    d_model: int = 24,
    n_heads: int = 4,
    n_layers: int = 1,
    n_programs: int = 6,
    energy_budget: float = 3.0,
    learning_rate: float = 3e-3,
    margin: float = 1.0,
    corruption_rate: float = 0.30,
    energy_l2_weight: float = 1e-4,
    routing_energy_weight: float = 0.0,
    min_pair_accuracy: float = 0.70,
    min_energy_gap: float = 0.20,
    min_accuracy_gain: float = 0.10,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_seeds = [int(seed) for seed in seeds]
    rows = [
        run_energy_probe_seed(
            seed=seed,
            steps=steps,
            batch_size=batch_size,
            eval_batches=eval_batches,
            eval_batch_size=eval_batch_size,
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
            routing_energy_weight=routing_energy_weight,
            device=device,
        )
        for seed in selected_seeds
    ]
    result = aggregate_energy_probe_results(
        rows,
        min_pair_accuracy=min_pair_accuracy,
        min_energy_gap=min_energy_gap,
        min_accuracy_gain=min_accuracy_gain,
    )
    result["schema"] = "energy_based_model_probe.v1"
    result["question"] = (
        "Can the TAC Transformer support energy-based modeling when trained "
        "with a scalar sequence energy over positive and corrupted sequences?"
    )
    result["per_seed"] = rows
    result["settings"] = {
        "seeds": selected_seeds,
        "steps": steps,
        "batch_size": batch_size,
        "eval_batches": eval_batches,
        "eval_batch_size": eval_batch_size,
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
        "routing_energy_weight": routing_energy_weight,
        "device": str(device),
    }
    artifact = output / "energy_based_model_probe.json"
    artifact.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "RESULTS.md").write_text(format_markdown(result), encoding="utf-8")
    return result


def run_energy_probe_seed(
    *,
    seed: int,
    steps: int,
    batch_size: int,
    eval_batches: int,
    eval_batch_size: int,
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
    routing_energy_weight: float,
    device: str | torch.device,
) -> dict[str, Any]:
    if vocab_size < 16:
        raise ValueError("vocab_size must be at least 16")
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
    )
    model = TACSequenceEnergyModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_generator = _make_generator(seed + 10_000, device)

    initial_eval = evaluate_energy_model(
        model,
        batches=eval_batches,
        batch_size=eval_batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        corruption_rate=corruption_rate,
        seed=seed + 20_000,
        device=device,
    )

    started = time.perf_counter()
    latest = {"loss": 0.0, "pair_accuracy": 0.0, "energy_gap": 0.0}
    model.train()
    for _ in range(steps):
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
        negative_energy, negative_output = model(negatives)
        contrastive_loss = F.softplus(positive_energy - negative_energy + margin).mean()
        energy_l2 = (positive_energy.pow(2).mean() + negative_energy.pow(2).mean()) * 0.5
        routing_energy = (
            positive_output.aux.used_energy.mean()
            + negative_output.aux.used_energy.mean()
        ) * 0.5 / max(energy_budget, 1e-6)
        loss = (
            contrastive_loss
            + energy_l2_weight * energy_l2
            + routing_energy_weight * routing_energy
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            latest = {
                "loss": float(loss.detach()),
                "contrastive_loss": float(contrastive_loss.detach()),
                "energy_l2": float(energy_l2.detach()),
                "routing_energy": float(routing_energy.detach()),
                "pair_accuracy": float(
                    pair_accuracy(positive_energy, negative_energy).detach()
                ),
                "energy_gap": float((negative_energy - positive_energy).mean().detach()),
            }

    final_eval = evaluate_energy_model(
        model,
        batches=eval_batches,
        batch_size=eval_batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        corruption_rate=corruption_rate,
        seed=seed + 20_000,
        device=device,
    )
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "seed": seed,
        "config": asdict(config),
        "parameter_counts": count_parameters(model),
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "train": {
            **latest,
            "steps": steps,
            "examples_per_second": steps * batch_size * 2 / elapsed,
        },
    }


@torch.no_grad()
def evaluate_energy_model(
    model: TACSequenceEnergyModel,
    *,
    batches: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    corruption_rate: float,
    seed: int,
    device: str | torch.device,
) -> dict[str, float]:
    model.eval()
    generator = _make_generator(seed, device)
    totals: dict[str, list[float]] = {
        "positive_energy": [],
        "negative_energy": [],
        "energy_gap": [],
        "pair_accuracy": [],
        "positive_routing_energy": [],
        "negative_routing_energy": [],
        "routing_energy_gap": [],
        "routing_lower_positive_accuracy": [],
        "routing_higher_positive_accuracy": [],
    }
    for _ in range(batches):
        positives = generate_structured_sequences(
            batch_size,
            seq_len,
            vocab_size,
            generator=generator,
            device=device,
        )
        negatives = corrupt_sequences(
            positives,
            vocab_size,
            corruption_rate=corruption_rate,
            generator=generator,
        )
        positive_energy, positive_output = model(positives)
        negative_energy, negative_output = model(negatives)
        positive_route = positive_output.aux.used_energy
        negative_route = negative_output.aux.used_energy
        totals["positive_energy"].append(float(positive_energy.mean().detach()))
        totals["negative_energy"].append(float(negative_energy.mean().detach()))
        totals["energy_gap"].append(float((negative_energy - positive_energy).mean().detach()))
        totals["pair_accuracy"].append(
            float(pair_accuracy(positive_energy, negative_energy).detach())
        )
        totals["positive_routing_energy"].append(float(positive_route.mean().detach()))
        totals["negative_routing_energy"].append(float(negative_route.mean().detach()))
        totals["routing_energy_gap"].append(
            float((negative_route - positive_route).mean().detach())
        )
        totals["routing_lower_positive_accuracy"].append(
            float(pair_accuracy(positive_route, negative_route).detach())
        )
        totals["routing_higher_positive_accuracy"].append(
            float(pair_accuracy(-positive_route, -negative_route).detach())
        )
    metrics = {name: mean(values) for name, values in totals.items()}
    metrics["routing_energy_best_pair_accuracy"] = max(
        metrics["routing_lower_positive_accuracy"],
        metrics["routing_higher_positive_accuracy"],
    )
    return metrics


def generate_structured_sequences(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    *,
    generator: torch.Generator,
    device: str | torch.device,
) -> Tensor:
    if seq_len < 4:
        raise ValueError("seq_len must be at least 4")
    data_floor = 8
    span = vocab_size - data_floor
    markers = torch.randint(0, 4, (batch_size,), generator=generator, device=device)
    starts = torch.randint(0, span, (batch_size,), generator=generator, device=device)
    step_values = torch.tensor([1, 2, 3, 5], device=device, dtype=torch.long)
    steps = step_values[markers]
    positions = torch.arange(seq_len - 1, device=device, dtype=torch.long)
    values = data_floor + (starts[:, None] + steps[:, None] * positions[None, :]) % span
    tokens = torch.empty(batch_size, seq_len, device=device, dtype=torch.long)
    tokens[:, 0] = markers + 1
    tokens[:, 1:] = values
    return tokens


def corrupt_sequences(
    input_ids: Tensor,
    vocab_size: int,
    *,
    corruption_rate: float,
    generator: torch.Generator,
) -> Tensor:
    if not 0.0 < corruption_rate <= 1.0:
        raise ValueError("corruption_rate must be in (0, 1]")
    data_floor = 8
    span = vocab_size - data_floor
    random_values = torch.randint(
        data_floor,
        vocab_size,
        input_ids.shape,
        generator=generator,
        device=input_ids.device,
        dtype=torch.long,
    )
    random_values = torch.where(
        random_values == input_ids,
        data_floor + ((random_values - data_floor + 1) % span),
        random_values,
    )
    mask = torch.rand(
        input_ids.shape,
        generator=generator,
        device=input_ids.device,
    ) < corruption_rate
    mask[:, 0] = False
    return torch.where(mask, random_values, input_ids)


def pair_accuracy(positive_energy: Tensor, negative_energy: Tensor) -> Tensor:
    lower = positive_energy < negative_energy
    tied = positive_energy == negative_energy
    return lower.float().mean() + 0.5 * tied.float().mean()


def aggregate_energy_probe_results(
    rows: list[dict[str, Any]],
    *,
    min_pair_accuracy: float = 0.70,
    min_energy_gap: float = 0.20,
    min_accuracy_gain: float = 0.10,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    initial_pair_accuracy = mean(row["initial_eval"]["pair_accuracy"] for row in rows)
    final_pair_accuracy = mean(row["final_eval"]["pair_accuracy"] for row in rows)
    final_energy_gap = mean(row["final_eval"]["energy_gap"] for row in rows)
    routing_best = mean(
        row["final_eval"]["routing_energy_best_pair_accuracy"] for row in rows
    )
    accuracy_gain = final_pair_accuracy - initial_pair_accuracy
    learned_energy_passed = (
        final_pair_accuracy >= min_pair_accuracy
        and final_energy_gap >= min_energy_gap
        and accuracy_gain >= min_accuracy_gain
    )
    routing_energy_passed = routing_best >= min_pair_accuracy
    if learned_energy_passed and routing_energy_passed:
        verdict = "yes_existing_routing_energy_and_learned_energy_head"
    elif learned_energy_passed:
        verdict = "yes_with_scalar_energy_head_not_routing_energy_alone"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "summary": {
            "initial_pair_accuracy": initial_pair_accuracy,
            "final_pair_accuracy": final_pair_accuracy,
            "accuracy_gain": accuracy_gain,
            "initial_energy_gap": mean(row["initial_eval"]["energy_gap"] for row in rows),
            "final_energy_gap": final_energy_gap,
            "routing_energy_best_pair_accuracy": routing_best,
            "routing_energy_gap": mean(
                row["final_eval"]["routing_energy_gap"] for row in rows
            ),
            "learned_energy_passed": learned_energy_passed,
            "routing_energy_passed": routing_energy_passed,
            "thresholds": {
                "min_pair_accuracy": min_pair_accuracy,
                "min_energy_gap": min_energy_gap,
                "min_accuracy_gain": min_accuracy_gain,
            },
        },
    }


def format_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    settings = result["settings"]
    lines = [
        "# Energy-Based Model Probe",
        "",
        f"Schema: `{result['schema']}`",
        "",
        "## Question",
        "",
        result["question"],
        "",
        "## Verdict",
        "",
        f"- Verdict: `{result['verdict']}`",
        f"- Initial learned-energy pair accuracy: {summary['initial_pair_accuracy']:.3f}",
        f"- Final learned-energy pair accuracy: {summary['final_pair_accuracy']:.3f}",
        f"- Learned-energy accuracy gain: {summary['accuracy_gain']:.3f}",
        f"- Final learned-energy gap, negative minus positive: {summary['final_energy_gap']:.3f}",
        f"- Routing-energy best-direction pair accuracy: {summary['routing_energy_best_pair_accuracy']:.3f}",
        "",
        "Interpretation: the current TAC routing budget is an energy regularizer, "
        "not a complete data energy. A TAC sequence EBM needs a scalar energy "
        "readout and a contrastive or score-matching objective that explicitly "
        "pushes observed sequences below corrupted negatives.",
        "",
        "## Settings",
        "",
        f"- Seeds: {settings['seeds']}",
        f"- Steps per seed: {settings['steps']}",
        f"- Batch size: {settings['batch_size']}",
        f"- Eval batches: {settings['eval_batches']}",
        f"- Sequence length: {settings['seq_len']}",
        f"- Config: d_model={settings['d_model']}, layers={settings['n_layers']}, heads={settings['n_heads']}, programs={settings['n_programs']}",
        "",
        "## Per-Seed Results",
        "",
        "| Seed | Initial Acc | Final Acc | Final Gap | Routing Best Acc | Train Loss |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["per_seed"]:
        lines.append(
            "| {seed} | {initial:.3f} | {final:.3f} | {gap:.3f} | {route:.3f} | {loss:.3f} |".format(
                seed=row["seed"],
                initial=row["initial_eval"]["pair_accuracy"],
                final=row["final_eval"]["pair_accuracy"],
                gap=row["final_eval"]["energy_gap"],
                route=row["final_eval"]["routing_energy_best_pair_accuracy"],
                loss=row["train"]["loss"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _make_generator(seed: int, device: str | torch.device) -> torch.Generator:
    torch_device = torch.device(device)
    generator_device = "cuda" if torch_device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


if __name__ == "__main__":
    main()
