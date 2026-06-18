from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_forced_identity_objective import (
    BOS,
    QUERY,
    VALUE_COUNT,
    VALUE_START,
    forced_identity_config,
    knockout_program,
)
from experiments.benchmark_identity_readout_bridge import (
    class_accuracy,
    format_value,
    should_trace,
    value_token_ids,
)
from tac_transformer import TACTransformerLM
from tac_transformer.training import count_parameters


DEFAULT_OUTPUT_DIR = (
    ROOT
    / "runs"
    / "benchmarks"
    / "program_specific_supervision_tac216_2026_06_07"
)
VARIANTS = ("forced_state_baseline", "semantic_baseline", "program_supervised")
N_PROGRAMS = 8
KEY_START = 8
IGNORE_PROGRAM = -100


@dataclass
class ProgramHeads:
    global_bridge: nn.Linear
    slot_probe: nn.Linear
    slot_bridge: nn.Linear

    def parameters(self):
        yield from self.global_bridge.parameters()
        yield from self.slot_probe.parameters()
        yield from self.slot_bridge.parameters()

    def train(self) -> None:
        self.global_bridge.train()
        self.slot_probe.train()
        self.slot_bridge.train()

    def eval(self) -> None:
        self.global_bridge.eval()
        self.slot_probe.eval()
        self.slot_bridge.eval()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TAC-216: test whether explicit program-specific responsibility "
            "supervision can make identity slots more usable and causally "
            "localized."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-steps", type=int, default=240)
    parser.add_argument("--head-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--knockout-batches", type=int, default=3)
    parser.add_argument("--n-pairs", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--route-weight", type=float, default=1.0)
    parser.add_argument("--slot-weight", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_program_specific_supervision(
        output_dir=args.output_dir,
        base_steps=args.base_steps,
        head_steps=args.head_steps,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        knockout_batches=args.knockout_batches,
        n_pairs=args.n_pairs,
        seeds=args.seeds,
        variants=args.variants,
        route_weight=args.route_weight,
        slot_weight=args.slot_weight,
        device=args.device,
        torch_threads=args.torch_threads,
    )
    print(json.dumps(report["decision"], indent=2), flush=True)


def run_program_specific_supervision(
    *,
    output_dir: Path,
    base_steps: int = 240,
    head_steps: int = 200,
    batch_size: int = 32,
    eval_batches: int = 8,
    knockout_batches: int = 3,
    n_pairs: int = 3,
    seeds: Sequence[int] = (7, 19, 31),
    variants: Sequence[str] = VARIANTS,
    route_weight: float = 1.0,
    slot_weight: float = 2.0,
    device: str | torch.device = "cpu",
    torch_threads: int = 4,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_threads = torch.get_num_threads()
    if torch_threads > 0:
        torch.set_num_threads(int(torch_threads))
    try:
        started = time.perf_counter()
        rows = []
        for variant in variants:
            if variant not in VARIANTS:
                raise ValueError(f"unknown variant: {variant}")
            for seed in seeds:
                rows.append(
                    run_variant_seed(
                        variant=variant,
                        seed=int(seed),
                        base_steps=int(base_steps),
                        head_steps=int(head_steps),
                        batch_size=int(batch_size),
                        eval_batches=int(eval_batches),
                        knockout_batches=int(knockout_batches),
                        n_pairs=int(n_pairs),
                        route_weight=float(route_weight),
                        slot_weight=float(slot_weight),
                        device=torch.device(device),
                    )
                )
        aggregate = aggregate_rows(rows)
        report = {
            "schema": "program_specific_supervision.v1",
            "created_at": "2026-06-07",
            "question": (
                "Can explicit program responsibility supervision align identity "
                "state with action space and make target program knockouts matter?"
            ),
            "protocol": {
                "base_steps": int(base_steps),
                "head_steps": int(head_steps),
                "batch_size": int(batch_size),
                "eval_batches": int(eval_batches),
                "knockout_batches": int(knockout_batches),
                "n_pairs": int(n_pairs),
                "n_programs": N_PROGRAMS,
                "seeds": [int(seed) for seed in seeds],
                "variants": list(variants),
                "route_weight": float(route_weight),
                "slot_weight": float(slot_weight),
                "assignment": (
                    "Each key token KEY_START+p is assigned to program p. "
                    "semantic_baseline and program_supervised use activation-aware "
                    "base_semantic routing. Program-supervised training applies "
                    "route supervision to support key/value tokens and value "
                    "supervision to the target program slot."
                ),
            },
            "rows": rows,
            "aggregate": aggregate,
            "decision": decide(aggregate),
            "elapsed_seconds": time.perf_counter() - started,
        }
        (output_dir / "program_specific_supervision.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "RESULTS.md").write_text(
            format_markdown(report),
            encoding="utf-8",
        )
        return report
    finally:
        if torch_threads > 0:
            torch.set_num_threads(prior_threads)


def run_variant_seed(
    *,
    variant: str,
    seed: int,
    base_steps: int,
    head_steps: int,
    batch_size: int,
    eval_batches: int,
    knockout_batches: int,
    n_pairs: int,
    route_weight: float,
    slot_weight: float,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    config = config_for_variant(variant)
    if config.n_programs != N_PROGRAMS:
        raise ValueError("forced_identity_config must keep n_programs=8 for TAC-216")
    model = TACTransformerLM(config).to(device)
    supervisor = nn.Linear(config.d_model, VALUE_COUNT).to(device)
    base_trace = train_base(
        model,
        supervisor,
        variant=variant,
        seed=seed,
        steps=base_steps,
        batch_size=batch_size,
        n_pairs=n_pairs,
        route_weight=route_weight,
        slot_weight=slot_weight,
        device=device,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    heads = ProgramHeads(
        global_bridge=nn.Linear(config.d_model, VALUE_COUNT).to(device),
        slot_probe=nn.Linear(config.d_model, VALUE_COUNT).to(device),
        slot_bridge=nn.Linear(config.d_model, VALUE_COUNT).to(device),
    )
    head_trace = train_heads(
        model,
        heads,
        seed=seed * 3001 + 101,
        steps=head_steps,
        batch_size=batch_size,
        n_pairs=n_pairs,
        device=device,
    )
    evaluation = evaluate_variant(
        model,
        heads,
        seed=seed * 4001 + 131,
        batch_size=batch_size,
        eval_batches=eval_batches,
        n_pairs=n_pairs,
        device=device,
    )
    knockout = targeted_knockout_summary(
        model,
        heads,
        seed=seed * 5003 + 151,
        batch_size=batch_size,
        batches_per_program=knockout_batches,
        n_pairs=n_pairs,
        device=device,
    )
    return {
        "variant": variant,
        "seed": int(seed),
        "base_training_trace": base_trace,
        "head_training_trace": head_trace,
        "parameter_counts": count_parameters(model),
        "evaluation": evaluation,
        "targeted_knockout": knockout,
    }


def make_program_batch(
    rng: random.Random,
    *,
    batch_size: int,
    n_pairs: int,
    device: torch.device,
    target_program: int | None = None,
) -> dict[str, torch.Tensor]:
    support_rows = []
    query_rows = []
    targets = []
    target_programs = []
    support_program_targets = []
    for _ in range(batch_size):
        if target_program is None:
            programs = rng.sample(range(N_PROGRAMS), n_pairs)
            target_index = rng.randrange(n_pairs)
        else:
            target = int(target_program)
            if target < 0 or target >= N_PROGRAMS:
                raise ValueError("target_program out of range")
            others = [program for program in range(N_PROGRAMS) if program != target]
            programs = [target] + rng.sample(others, n_pairs - 1)
            rng.shuffle(programs)
            target_index = programs.index(target)
        values = rng.sample(range(VALUE_START, VALUE_START + VALUE_COUNT), n_pairs)
        support = [BOS]
        program_targets = [IGNORE_PROGRAM]
        for program, value in zip(programs, values):
            support.extend([KEY_START + program, value])
            program_targets.extend([program, program])
        target_program_value = programs[target_index]
        query_rows.append([QUERY, KEY_START + target_program_value])
        support_rows.append(support)
        support_program_targets.append(program_targets)
        targets.append(values[target_index])
        target_programs.append(target_program_value)
    return {
        "support": torch.tensor(support_rows, dtype=torch.long, device=device),
        "query": torch.tensor(query_rows, dtype=torch.long, device=device),
        "target": torch.tensor(targets, dtype=torch.long, device=device),
        "target_program": torch.tensor(
            target_programs,
            dtype=torch.long,
            device=device,
        ),
        "support_program_targets": torch.tensor(
            support_program_targets,
            dtype=torch.long,
            device=device,
        ),
    }


def config_for_variant(variant: str):
    config = forced_identity_config()
    if variant in {"semantic_baseline", "program_supervised"}:
        return replace(config, routing_type="base_semantic", routing_top_k=2)
    if variant == "forced_state_baseline":
        return config
    raise ValueError(f"unknown variant: {variant}")


def train_base(
    model: TACTransformerLM,
    supervisor: nn.Linear,
    *,
    variant: str,
    seed: int,
    steps: int,
    batch_size: int,
    n_pairs: int,
    route_weight: float,
    slot_weight: float,
    device: torch.device,
) -> dict[str, list[float]]:
    parameters = list(model.parameters())
    if variant == "program_supervised":
        parameters += list(supervisor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=3e-3, weight_decay=0.01)
    rng = random.Random(seed * 1009 + 17)
    trace = {"total": [], "lm": [], "route": [], "slot": []}
    model.train()
    supervisor.train()
    for step in range(int(steps)):
        batch = make_program_batch(
            rng,
            batch_size=batch_size,
            n_pairs=n_pairs,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        losses = training_losses(
            model,
            supervisor,
            batch,
            variant=variant,
            route_weight=route_weight,
            slot_weight=slot_weight,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if should_trace(step, steps):
            for name, value in losses.items():
                trace[name].append(float(value.detach().cpu()))
    return trace


def training_losses(
    model: TACTransformerLM,
    supervisor: nn.Linear,
    batch: dict[str, torch.Tensor],
    *,
    variant: str,
    route_weight: float,
    slot_weight: float,
) -> dict[str, torch.Tensor]:
    support = model(
        batch["support"],
        collect_auxiliary=True,
        collect_metrics=False,
    )
    query = model(
        batch["query"],
        identity_states=support.identity_states,
        collect_auxiliary=True,
        collect_metrics=False,
    )
    lm_loss = F.cross_entropy(query.logits[:, -1, :], batch["target"])
    zero = lm_loss.new_zeros(())
    route = zero
    slot = zero
    if variant == "program_supervised":
        route = program_route_loss(
            support.aux.token_program_activations,
            batch["support_program_targets"],
        )
        slot_vectors = target_program_slots(
            support.identity_states[-1].program_memory,
            batch["target_program"],
        )
        slot = F.cross_entropy(
            supervisor(slot_vectors),
            batch["target"] - VALUE_START,
        )
    elif variant not in {"forced_state_baseline", "semantic_baseline"}:
        raise ValueError(f"unknown variant: {variant}")
    return {
        "total": lm_loss + route_weight * route + slot_weight * slot,
        "lm": lm_loss,
        "route": route,
        "slot": slot,
    }


def train_heads(
    model: TACTransformerLM,
    heads: ProgramHeads,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    n_pairs: int,
    device: torch.device,
) -> dict[str, list[float]]:
    optimizer = torch.optim.AdamW(list(heads.parameters()), lr=5e-3, weight_decay=0.0)
    rng = random.Random(seed)
    trace = {"total": [], "global_bridge": [], "slot_probe": [], "slot_bridge": []}
    heads.train()
    model.eval()
    for step in range(int(steps)):
        batch = make_program_batch(
            rng,
            batch_size=batch_size,
            n_pairs=n_pairs,
            device=device,
        )
        with torch.no_grad():
            features = extract_program_features(model, batch)
        optimizer.zero_grad(set_to_none=True)
        global_logits = features["base_value_logits"] + heads.global_bridge(
            features["read_vector"]
        )
        slot_probe_logits = heads.slot_probe(features["target_slot"])
        slot_bridge_logits = features["base_value_logits"] + heads.slot_bridge(
            features["target_slot"]
        )
        global_loss = F.cross_entropy(global_logits, features["target_class"])
        probe_loss = F.cross_entropy(slot_probe_logits, features["target_class"])
        bridge_loss = F.cross_entropy(slot_bridge_logits, features["target_class"])
        total = global_loss + probe_loss + bridge_loss
        total.backward()
        optimizer.step()
        if should_trace(step, steps):
            trace["total"].append(float(total.detach().cpu()))
            trace["global_bridge"].append(float(global_loss.detach().cpu()))
            trace["slot_probe"].append(float(probe_loss.detach().cpu()))
            trace["slot_bridge"].append(float(bridge_loss.detach().cpu()))
    return trace


def evaluate_variant(
    model: TACTransformerLM,
    heads: ProgramHeads,
    *,
    seed: int,
    batch_size: int,
    eval_batches: int,
    n_pairs: int,
    device: torch.device,
) -> dict[str, float]:
    rng = random.Random(seed)
    model.eval()
    heads.eval()
    values: dict[str, list[float]] = {
        "base_carry_accuracy": [],
        "base_reset_accuracy": [],
        "direct_memory_read_accuracy": [],
        "global_bridge_accuracy": [],
        "target_slot_probe_accuracy": [],
        "target_slot_bridge_accuracy": [],
        "wrong_slot_bridge_accuracy": [],
        "shuffled_slot_bridge_accuracy": [],
        "zero_slot_bridge_accuracy": [],
        "route_target_argmax_rate": [],
        "route_target_selected_rate": [],
        "route_target_activation": [],
    }
    with torch.inference_mode():
        for _ in range(eval_batches):
            batch = make_program_batch(
                rng,
                batch_size=batch_size,
                n_pairs=n_pairs,
                device=device,
            )
            features = extract_program_features(model, batch)
            target = features["target_class"]
            values["base_carry_accuracy"].append(
                class_accuracy(features["base_value_logits"], target)
            )
            values["base_reset_accuracy"].append(
                class_accuracy(features["reset_value_logits"], target)
            )
            values["direct_memory_read_accuracy"].append(
                class_accuracy(features["memory_read_value_logits"], target)
            )
            values["global_bridge_accuracy"].append(
                class_accuracy(
                    features["base_value_logits"]
                    + heads.global_bridge(features["read_vector"]),
                    target,
                )
            )
            values["target_slot_probe_accuracy"].append(
                class_accuracy(heads.slot_probe(features["target_slot"]), target)
            )
            values["target_slot_bridge_accuracy"].append(
                class_accuracy(
                    features["base_value_logits"]
                    + heads.slot_bridge(features["target_slot"]),
                    target,
                )
            )
            values["wrong_slot_bridge_accuracy"].append(
                class_accuracy(
                    features["base_value_logits"]
                    + heads.slot_bridge(features["wrong_slot"]),
                    target,
                )
            )
            values["shuffled_slot_bridge_accuracy"].append(
                class_accuracy(
                    features["base_value_logits"]
                    + heads.slot_bridge(features["shuffled_slot"]),
                    target,
                )
            )
            values["zero_slot_bridge_accuracy"].append(
                class_accuracy(
                    features["base_value_logits"] + heads.slot_bridge(features["zero_slot"]),
                    target,
                )
            )
            for metric in (
                "route_target_argmax_rate",
                "route_target_selected_rate",
                "route_target_activation",
            ):
                values[metric].append(float(features[metric]))
    result = {key: mean(metric_values) for key, metric_values in values.items()}
    result["base_carry_minus_reset"] = (
        result["base_carry_accuracy"] - result["base_reset_accuracy"]
    )
    result["target_slot_bridge_minus_global_bridge"] = (
        result["target_slot_bridge_accuracy"] - result["global_bridge_accuracy"]
    )
    result["target_slot_bridge_minus_wrong_slot"] = (
        result["target_slot_bridge_accuracy"] - result["wrong_slot_bridge_accuracy"]
    )
    result["target_slot_bridge_minus_shuffled_slot"] = (
        result["target_slot_bridge_accuracy"] - result["shuffled_slot_bridge_accuracy"]
    )
    return result


def targeted_knockout_summary(
    model: TACTransformerLM,
    heads: ProgramHeads,
    *,
    seed: int,
    batch_size: int,
    batches_per_program: int,
    n_pairs: int,
    device: torch.device,
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows = []
    with torch.inference_mode():
        for program in range(N_PROGRAMS):
            batches = [
                make_program_batch(
                    rng,
                    batch_size=batch_size,
                    n_pairs=n_pairs,
                    device=device,
                    target_program=program,
                )
                for _ in range(batches_per_program)
            ]
            baseline = score_slot_bridge_batches(model, heads, batches)
            with knockout_program(model, program):
                target_knockout = score_slot_bridge_batches(model, heads, batches)
            nontarget_program = (program + 1) % N_PROGRAMS
            with knockout_program(model, nontarget_program):
                nontarget_knockout = score_slot_bridge_batches(model, heads, batches)
            rows.append(
                {
                    "program": int(program),
                    "nontarget_program": int(nontarget_program),
                    "baseline_accuracy": baseline,
                    "target_knockout_accuracy": target_knockout,
                    "nontarget_knockout_accuracy": nontarget_knockout,
                    "targeted_knockout_drop": baseline - target_knockout,
                    "nontarget_knockout_drop": baseline - nontarget_knockout,
                    "localized_drop_gap": (
                        baseline - target_knockout
                    )
                    - (baseline - nontarget_knockout),
                }
            )
    return {
        "programs": rows,
        "targeted_knockout_drop": mean(
            [row["targeted_knockout_drop"] for row in rows]
        ),
        "nontarget_knockout_drop": mean(
            [row["nontarget_knockout_drop"] for row in rows]
        ),
        "localized_drop_gap": mean([row["localized_drop_gap"] for row in rows]),
        "max_targeted_knockout_drop": max(
            [max(0.0, row["targeted_knockout_drop"]) for row in rows],
            default=0.0,
        ),
        "programs_with_target_drop_ge_5pct": sum(
            1 for row in rows if row["targeted_knockout_drop"] >= 0.05
        ),
    }


def score_slot_bridge_batches(
    model: TACTransformerLM,
    heads: ProgramHeads,
    batches: Sequence[dict[str, torch.Tensor]],
) -> float:
    scores = []
    for batch in batches:
        features = extract_program_features(model, batch)
        scores.append(
            class_accuracy(
                features["base_value_logits"] + heads.slot_bridge(features["target_slot"]),
                features["target_class"],
            )
        )
    return mean(scores)


def extract_program_features(
    model: TACTransformerLM,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor | float]:
    value_ids = value_token_ids(batch["target"].device)
    support = model(
        batch["support"],
        collect_auxiliary=True,
        collect_metrics=False,
    )
    query = model(
        batch["query"],
        identity_states=support.identity_states,
        collect_auxiliary=False,
        collect_metrics=False,
    )
    reset = model(
        batch["query"],
        collect_auxiliary=False,
        collect_metrics=False,
    )
    key_ids = batch["query"][:, -1]
    read_vector = model.memory_read_vector(key_ids, support.identity_states)
    memory_read_logits = model.memory_read_logits(key_ids, support.identity_states)
    memory = support.identity_states[-1].program_memory
    target_slot = target_program_slots(memory, batch["target_program"])
    wrong_program = (batch["target_program"] + 1) % N_PROGRAMS
    wrong_slot = target_program_slots(memory, wrong_program)
    shuffled_slot = target_slot[torch.randperm(target_slot.shape[0], device=target_slot.device)]
    route = route_alignment(
        support.aux.token_program_activations,
        support.aux.token_selected_program_mask,
        batch["support_program_targets"],
    )
    return {
        "read_vector": read_vector.detach(),
        "target_slot": target_slot.detach(),
        "wrong_slot": wrong_slot.detach(),
        "shuffled_slot": shuffled_slot.detach(),
        "zero_slot": torch.zeros_like(target_slot).detach(),
        "base_value_logits": query.logits[:, -1, value_ids].detach(),
        "reset_value_logits": reset.logits[:, -1, value_ids].detach(),
        "memory_read_value_logits": memory_read_logits[:, value_ids].detach(),
        "target_class": (batch["target"] - VALUE_START).detach(),
        **route,
    }


def target_program_slots(memory: torch.Tensor, programs: torch.Tensor) -> torch.Tensor:
    indices = programs.to(memory.device).view(-1, 1, 1).expand(-1, 1, memory.shape[-1])
    return memory.gather(dim=1, index=indices).squeeze(1)


def program_route_loss(
    token_activations: torch.Tensor | None,
    targets: torch.Tensor,
) -> torch.Tensor:
    if token_activations is None:
        raise ValueError("token_program_activations are required")
    mask = targets >= 0
    selected = token_activations[mask].clamp(1e-4, 1.0 - 1e-4)
    labels = targets[mask]
    one_hot = F.one_hot(labels, num_classes=N_PROGRAMS).to(dtype=selected.dtype)
    bce = F.binary_cross_entropy(selected, one_hot)
    ce = F.cross_entropy(selected, labels)
    return bce + 0.25 * ce


def route_alignment(
    token_activations: torch.Tensor | None,
    token_selected_mask: torch.Tensor | None,
    targets: torch.Tensor,
) -> dict[str, float]:
    if token_activations is None or token_selected_mask is None:
        return {
            "route_target_argmax_rate": 0.0,
            "route_target_selected_rate": 0.0,
            "route_target_activation": 0.0,
        }
    mask = targets >= 0
    labels = targets[mask]
    activations = token_activations[mask]
    selected = token_selected_mask[mask]
    target_activation = activations.gather(1, labels[:, None]).squeeze(1)
    target_selected = selected.gather(1, labels[:, None]).squeeze(1)
    return {
        "route_target_argmax_rate": float(
            (activations.argmax(dim=-1) == labels).float().mean().detach().cpu()
        ),
        "route_target_selected_rate": float(
            (target_selected > 0.0).float().mean().detach().cpu()
        ),
        "route_target_activation": float(target_activation.mean().detach().cpu()),
    }


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    aggregate: dict[str, Any] = {}
    for variant, variant_rows in sorted(by_variant.items()):
        aggregate[variant] = {
            "seeds": [row["seed"] for row in variant_rows],
            "base_final_total_loss": mean_path(
                variant_rows,
                ["base_training_trace", "total", -1],
            ),
            "head_final_total_loss": mean_path(
                variant_rows,
                ["head_training_trace", "total", -1],
            ),
            "base_carry_accuracy": mean_path(
                variant_rows,
                ["evaluation", "base_carry_accuracy"],
            ),
            "base_reset_accuracy": mean_path(
                variant_rows,
                ["evaluation", "base_reset_accuracy"],
            ),
            "base_carry_minus_reset": mean_path(
                variant_rows,
                ["evaluation", "base_carry_minus_reset"],
            ),
            "direct_memory_read_accuracy": mean_path(
                variant_rows,
                ["evaluation", "direct_memory_read_accuracy"],
            ),
            "global_bridge_accuracy": mean_path(
                variant_rows,
                ["evaluation", "global_bridge_accuracy"],
            ),
            "target_slot_probe_accuracy": mean_path(
                variant_rows,
                ["evaluation", "target_slot_probe_accuracy"],
            ),
            "target_slot_bridge_accuracy": mean_path(
                variant_rows,
                ["evaluation", "target_slot_bridge_accuracy"],
            ),
            "wrong_slot_bridge_accuracy": mean_path(
                variant_rows,
                ["evaluation", "wrong_slot_bridge_accuracy"],
            ),
            "shuffled_slot_bridge_accuracy": mean_path(
                variant_rows,
                ["evaluation", "shuffled_slot_bridge_accuracy"],
            ),
            "zero_slot_bridge_accuracy": mean_path(
                variant_rows,
                ["evaluation", "zero_slot_bridge_accuracy"],
            ),
            "target_slot_bridge_minus_global_bridge": mean_path(
                variant_rows,
                ["evaluation", "target_slot_bridge_minus_global_bridge"],
            ),
            "target_slot_bridge_minus_wrong_slot": mean_path(
                variant_rows,
                ["evaluation", "target_slot_bridge_minus_wrong_slot"],
            ),
            "target_slot_bridge_minus_shuffled_slot": mean_path(
                variant_rows,
                ["evaluation", "target_slot_bridge_minus_shuffled_slot"],
            ),
            "route_target_argmax_rate": mean_path(
                variant_rows,
                ["evaluation", "route_target_argmax_rate"],
            ),
            "route_target_selected_rate": mean_path(
                variant_rows,
                ["evaluation", "route_target_selected_rate"],
            ),
            "route_target_activation": mean_path(
                variant_rows,
                ["evaluation", "route_target_activation"],
            ),
            "targeted_knockout_drop": mean_path(
                variant_rows,
                ["targeted_knockout", "targeted_knockout_drop"],
            ),
            "nontarget_knockout_drop": mean_path(
                variant_rows,
                ["targeted_knockout", "nontarget_knockout_drop"],
            ),
            "localized_drop_gap": mean_path(
                variant_rows,
                ["targeted_knockout", "localized_drop_gap"],
            ),
            "max_targeted_knockout_drop": mean_path(
                variant_rows,
                ["targeted_knockout", "max_targeted_knockout_drop"],
            ),
            "programs_with_target_drop_ge_5pct": mean_path(
                variant_rows,
                ["targeted_knockout", "programs_with_target_drop_ge_5pct"],
            ),
        }
    if "forced_state_baseline" in aggregate and "program_supervised" in aggregate:
        base = aggregate["forced_state_baseline"]
        supervised = aggregate["program_supervised"]
        aggregate["program_supervised_minus_forced_state_baseline"] = {
            key: supervised[key] - base[key]
            for key in (
                "global_bridge_accuracy",
                "target_slot_probe_accuracy",
                "target_slot_bridge_accuracy",
                "target_slot_bridge_minus_wrong_slot",
                "route_target_argmax_rate",
                "route_target_selected_rate",
                "targeted_knockout_drop",
                "localized_drop_gap",
            )
        }
    if "semantic_baseline" in aggregate and "program_supervised" in aggregate:
        base = aggregate["semantic_baseline"]
        supervised = aggregate["program_supervised"]
        aggregate["program_supervised_minus_semantic_baseline"] = {
            key: supervised[key] - base[key]
            for key in (
                "global_bridge_accuracy",
                "target_slot_probe_accuracy",
                "target_slot_bridge_accuracy",
                "target_slot_bridge_minus_wrong_slot",
                "route_target_argmax_rate",
                "route_target_selected_rate",
                "targeted_knockout_drop",
                "localized_drop_gap",
            )
        }
    return aggregate


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    supervised = aggregate.get("program_supervised")
    baseline = aggregate.get("semantic_baseline") or aggregate.get("forced_state_baseline")
    if not supervised:
        return {
            "status": "program_supervised_variant_not_run",
            "reason": "TAC-216 requires the program_supervised variant.",
        }
    delta = aggregate.get(
        "program_supervised_minus_semantic_baseline",
        aggregate.get("program_supervised_minus_forced_state_baseline", {}),
    )
    slot_bridge = supervised["target_slot_bridge_accuracy"]
    wrong_gap = supervised["target_slot_bridge_minus_wrong_slot"]
    target_drop = supervised["targeted_knockout_drop"]
    localized_gap = supervised["localized_drop_gap"]
    route_selected = supervised["route_target_selected_rate"]
    if (
        slot_bridge >= 0.55
        and wrong_gap >= 0.20
        and target_drop >= 0.05
        and localized_gap >= 0.03
    ):
        status = "program_supervision_creates_localized_causal_slots"
    elif slot_bridge >= 0.30 and wrong_gap >= 0.10 and target_drop < 0.05:
        status = "program_supervision_improves_slot_alignment_without_causality"
    elif route_selected >= 0.80 and target_drop < 0.05:
        status = "program_routes_align_but_slots_remain_noncausal"
    else:
        status = "program_supervision_not_sufficient"
    return {
        "status": status,
        "target_slot_bridge_accuracy": slot_bridge,
        "target_slot_bridge_minus_wrong_slot": wrong_gap,
        "targeted_knockout_drop": target_drop,
        "localized_drop_gap": localized_gap,
        "route_target_selected_rate": route_selected,
        "supervised_minus_baseline": delta,
        "baseline_target_slot_bridge_accuracy": None
        if baseline is None
        else baseline["target_slot_bridge_accuracy"],
        "interpretation": (
            "TAC-216 tests whether explicit assignment of keys to programs "
            "turns distributed identity signal into program-local, controllable "
            "action state."
        ),
    }


def format_markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# TAC-216 Program-Specific Supervision",
        "",
        f"Decision: `{report['decision']['status']}`.",
        "",
        "| Metric | forced-state baseline | semantic baseline | program-supervised | delta vs semantic |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    baseline = aggregate.get("forced_state_baseline", {})
    semantic = aggregate.get("semantic_baseline", {})
    supervised = aggregate.get("program_supervised", {})
    delta = aggregate.get("program_supervised_minus_semantic_baseline", {})
    rows = [
        ("Base carry accuracy", "base_carry_accuracy"),
        ("Direct memory-read accuracy", "direct_memory_read_accuracy"),
        ("Global readout bridge accuracy", "global_bridge_accuracy"),
        ("Target-slot probe accuracy", "target_slot_probe_accuracy"),
        ("Target-slot bridge accuracy", "target_slot_bridge_accuracy"),
        ("Wrong-slot bridge accuracy", "wrong_slot_bridge_accuracy"),
        ("Target-slot bridge - wrong slot", "target_slot_bridge_minus_wrong_slot"),
        ("Route target argmax rate", "route_target_argmax_rate"),
        ("Route target selected rate", "route_target_selected_rate"),
        ("Targeted knockout drop", "targeted_knockout_drop"),
        ("Nontarget knockout drop", "nontarget_knockout_drop"),
        ("Localized drop gap", "localized_drop_gap"),
        ("Max targeted knockout drop", "max_targeted_knockout_drop"),
    ]
    for label, key in rows:
        lines.append(
            "| "
            f"{label} | "
            f"{format_value(baseline.get(key))} | "
            f"{format_value(semantic.get(key))} | "
            f"{format_value(supervised.get(key))} | "
            f"{format_value(delta.get(key))} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `semantic_baseline` enables activation-aware routing without assignment supervision.",
            "- `program_supervised` assigns key `KEY_START+p` and its value token to program `p` during support processing.",
            "- The target-slot probe/bridge use the assigned program memory slot, not the content-addressed global readout.",
            "- Targeted knockouts remove the assigned program for batches whose query key maps to that program; nontarget knockouts remove the next program as a control.",
            "- A positive modular result requires higher target-slot accuracy and target knockouts that hurt more than nontarget knockouts.",
            "",
        ]
    )
    return "\n".join(lines)


def mean(values: Sequence[float]) -> float:
    vals = [float(value) for value in values]
    return statistics.fmean(vals) if vals else 0.0


def mean_path(rows: Sequence[dict[str, Any]], path: Sequence[str | int]) -> float:
    values = []
    for row in rows:
        current: Any = row
        for key in path:
            current = current[key]
        if current is not None:
            values.append(float(current))
    return mean(values)


if __name__ == "__main__":
    main()
