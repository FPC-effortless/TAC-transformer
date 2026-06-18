from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from dataclasses import replace
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_forced_identity_objective import (
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
from experiments.benchmark_program_specific_supervision import (
    N_PROGRAMS,
    make_program_batch,
    program_route_loss,
    route_alignment,
    target_program_slots,
)
from tac_transformer import TACTransformerLM
from tac_transformer.training import count_parameters


DEFAULT_OUTPUT_DIR = (
    ROOT
    / "runs"
    / "benchmarks"
    / "representation_binding_scrubbing_tac217_2026_06_07"
)
VARIANTS = ("semantic_baseline", "subspace_bound")


@dataclass
class BindingHeads:
    global_bridge: nn.Linear
    raw_slot_bridge: nn.Linear
    bound_slot_probe: nn.Linear
    bound_slot_bridge: nn.Linear

    def parameters(self):
        yield from self.global_bridge.parameters()
        yield from self.raw_slot_bridge.parameters()
        yield from self.bound_slot_probe.parameters()
        yield from self.bound_slot_bridge.parameters()

    def train(self) -> None:
        self.global_bridge.train()
        self.raw_slot_bridge.train()
        self.bound_slot_probe.train()
        self.bound_slot_bridge.train()

    def eval(self) -> None:
        self.global_bridge.eval()
        self.raw_slot_bridge.eval()
        self.bound_slot_probe.eval()
        self.bound_slot_bridge.eval()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TAC-217: test representational binding with orthogonal program "
            "subspaces and causal scrubbing of learned identity/action subspaces."
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
    parser.add_argument("--subspace-weight", type=float, default=0.2)
    parser.add_argument("--contrastive-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_representation_binding_scrubbing(
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
        subspace_weight=args.subspace_weight,
        contrastive_weight=args.contrastive_weight,
        device=args.device,
        torch_threads=args.torch_threads,
    )
    print(json.dumps(report["decision"], indent=2), flush=True)


def run_representation_binding_scrubbing(
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
    subspace_weight: float = 0.2,
    contrastive_weight: float = 0.5,
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
                        subspace_weight=float(subspace_weight),
                        contrastive_weight=float(contrastive_weight),
                        device=torch.device(device),
                    )
                )
        aggregate = aggregate_rows(rows)
        report = {
            "schema": "representation_binding_scrubbing.v1",
            "created_at": "2026-06-07",
            "question": (
                "A: can orthogonal representational binding make identity locally "
                "persistent in program slots? B: does causal scrubbing of learned "
                "representation subspaces remove identity signal more than program "
                "knockout or attention ablation?"
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
                "subspace_weight": float(subspace_weight),
                "contrastive_weight": float(contrastive_weight),
                "binding": (
                    "subspace_bound uses base_semantic routing plus fixed "
                    "orthogonal program subspace masks over d_model dimensions."
                ),
                "scrubbing": (
                    "Projection scrub removes the learned bridge rowspace from "
                    "bound slot vectors; gradient scrub removes the top gradient "
                    "dimensions; attention ablation zeros attention output during "
                    "feature extraction."
                ),
            },
            "rows": rows,
            "aggregate": aggregate,
            "decision": decide(aggregate),
            "elapsed_seconds": time.perf_counter() - started,
        }
        (output_dir / "representation_binding_scrubbing.json").write_text(
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
    subspace_weight: float,
    contrastive_weight: float,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    config = config_for_variant()
    model = TACTransformerLM(config).to(device)
    masks = program_subspace_masks(
        d_model=config.d_model,
        n_programs=config.n_programs,
        device=device,
    )
    supervisor = nn.Linear(config.d_model, VALUE_COUNT).to(device)
    base_trace = train_base(
        model,
        supervisor,
        masks,
        variant=variant,
        seed=seed,
        steps=base_steps,
        batch_size=batch_size,
        n_pairs=n_pairs,
        route_weight=route_weight,
        slot_weight=slot_weight,
        subspace_weight=subspace_weight,
        contrastive_weight=contrastive_weight,
        device=device,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    heads = BindingHeads(
        global_bridge=nn.Linear(config.d_model, VALUE_COUNT).to(device),
        raw_slot_bridge=nn.Linear(config.d_model, VALUE_COUNT).to(device),
        bound_slot_probe=nn.Linear(config.d_model, VALUE_COUNT).to(device),
        bound_slot_bridge=nn.Linear(config.d_model, VALUE_COUNT).to(device),
    )
    head_trace = train_heads(
        model,
        heads,
        masks,
        seed=seed * 3001 + 211,
        steps=head_steps,
        batch_size=batch_size,
        n_pairs=n_pairs,
        device=device,
    )
    binding = evaluate_binding(
        model,
        heads,
        masks,
        seed=seed * 4001 + 223,
        batch_size=batch_size,
        eval_batches=eval_batches,
        n_pairs=n_pairs,
        device=device,
    )
    knockout = targeted_knockout_summary(
        model,
        heads,
        masks,
        seed=seed * 5003 + 227,
        batch_size=batch_size,
        batches_per_program=knockout_batches,
        n_pairs=n_pairs,
        device=device,
    )
    scrubbing = evaluate_scrubbing(
        model,
        heads,
        masks,
        seed=seed * 6007 + 229,
        batch_size=batch_size,
        eval_batches=eval_batches,
        n_pairs=n_pairs,
        device=device,
    )
    return {
        "variant": variant,
        "seed": int(seed),
        "base_training_trace": base_trace,
        "head_training_trace": head_trace,
        "parameter_counts": count_parameters(model),
        "binding": binding,
        "targeted_knockout": knockout,
        "scrubbing": scrubbing,
    }


def config_for_variant():
    return replace(
        forced_identity_config(),
        routing_type="base_semantic",
        routing_top_k=2,
    )


def program_subspace_masks(
    *,
    d_model: int,
    n_programs: int,
    device: torch.device,
) -> torch.Tensor:
    masks = torch.zeros(n_programs, d_model, device=device)
    for dim in range(d_model):
        masks[dim % n_programs, dim] = 1.0
    return masks


def train_base(
    model: TACTransformerLM,
    supervisor: nn.Linear,
    masks: torch.Tensor,
    *,
    variant: str,
    seed: int,
    steps: int,
    batch_size: int,
    n_pairs: int,
    route_weight: float,
    slot_weight: float,
    subspace_weight: float,
    contrastive_weight: float,
    device: torch.device,
) -> dict[str, list[float]]:
    parameters = list(model.parameters())
    if variant == "subspace_bound":
        parameters += list(supervisor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=3e-3, weight_decay=0.01)
    rng = random.Random(seed * 1009 + 17)
    trace = {"total": [], "lm": [], "route": [], "slot": [], "subspace": [], "contrast": []}
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
            masks,
            batch,
            variant=variant,
            route_weight=route_weight,
            slot_weight=slot_weight,
            subspace_weight=subspace_weight,
            contrastive_weight=contrastive_weight,
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
    masks: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    variant: str,
    route_weight: float,
    slot_weight: float,
    subspace_weight: float,
    contrastive_weight: float,
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
    lm = F.cross_entropy(query.logits[:, -1, :], batch["target"])
    zero = lm.new_zeros(())
    route = zero
    slot = zero
    subspace = zero
    contrast = zero
    if variant == "subspace_bound":
        route = program_route_loss(
            support.aux.token_program_activations,
            batch["support_program_targets"],
        )
        memory = support.identity_states[-1].program_memory
        target_slot = target_program_slots(memory, batch["target_program"])
        wrong_slot = target_program_slots(
            memory,
            (batch["target_program"] + 1) % N_PROGRAMS,
        )
        bound_target = apply_program_subspace(
            target_slot,
            batch["target_program"],
            masks,
        )
        wrong_in_target_space = apply_program_subspace(
            wrong_slot,
            batch["target_program"],
            masks,
        )
        target_class = batch["target"] - VALUE_START
        target_logits = supervisor(bound_target)
        wrong_logits = supervisor(wrong_in_target_space)
        slot = F.cross_entropy(target_logits, target_class)
        target_class_logit = target_logits.gather(1, target_class[:, None]).squeeze(1)
        wrong_class_logit = wrong_logits.gather(1, target_class[:, None]).squeeze(1)
        contrast = F.relu(0.5 - target_class_logit + wrong_class_logit).mean()
        subspace = outside_subspace_penalty(memory, masks)
    elif variant != "semantic_baseline":
        raise ValueError(f"unknown variant: {variant}")
    return {
        "total": (
            lm
            + route_weight * route
            + slot_weight * slot
            + subspace_weight * subspace
            + contrastive_weight * contrast
        ),
        "lm": lm,
        "route": route,
        "slot": slot,
        "subspace": subspace,
        "contrast": contrast,
    }


def train_heads(
    model: TACTransformerLM,
    heads: BindingHeads,
    masks: torch.Tensor,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    n_pairs: int,
    device: torch.device,
) -> dict[str, list[float]]:
    optimizer = torch.optim.AdamW(list(heads.parameters()), lr=5e-3, weight_decay=0.0)
    rng = random.Random(seed)
    trace = {
        "total": [],
        "global_bridge": [],
        "raw_slot_bridge": [],
        "bound_slot_probe": [],
        "bound_slot_bridge": [],
    }
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
            features = extract_features(model, masks, batch)
        optimizer.zero_grad(set_to_none=True)
        global_logits = features["base_value_logits"] + heads.global_bridge(
            features["read_vector"]
        )
        raw_logits = features["base_value_logits"] + heads.raw_slot_bridge(
            features["target_slot"]
        )
        bound_probe_logits = heads.bound_slot_probe(features["bound_target_slot"])
        bound_logits = features["base_value_logits"] + heads.bound_slot_bridge(
            features["bound_target_slot"]
        )
        target = features["target_class"]
        global_loss = F.cross_entropy(global_logits, target)
        raw_loss = F.cross_entropy(raw_logits, target)
        probe_loss = F.cross_entropy(bound_probe_logits, target)
        bound_loss = F.cross_entropy(bound_logits, target)
        total = global_loss + raw_loss + probe_loss + bound_loss
        total.backward()
        optimizer.step()
        if should_trace(step, steps):
            trace["total"].append(float(total.detach().cpu()))
            trace["global_bridge"].append(float(global_loss.detach().cpu()))
            trace["raw_slot_bridge"].append(float(raw_loss.detach().cpu()))
            trace["bound_slot_probe"].append(float(probe_loss.detach().cpu()))
            trace["bound_slot_bridge"].append(float(bound_loss.detach().cpu()))
    return trace


def evaluate_binding(
    model: TACTransformerLM,
    heads: BindingHeads,
    masks: torch.Tensor,
    *,
    seed: int,
    batch_size: int,
    eval_batches: int,
    n_pairs: int,
    device: torch.device,
) -> dict[str, float]:
    rng = random.Random(seed)
    values: dict[str, list[float]] = {
        "base_carry_accuracy": [],
        "direct_memory_read_accuracy": [],
        "global_bridge_accuracy": [],
        "raw_slot_bridge_accuracy": [],
        "bound_slot_probe_accuracy": [],
        "bound_slot_bridge_accuracy": [],
        "wrong_bound_slot_bridge_accuracy": [],
        "shuffled_bound_slot_bridge_accuracy": [],
        "zero_bound_slot_bridge_accuracy": [],
        "route_target_argmax_rate": [],
        "route_target_selected_rate": [],
        "outside_subspace_ratio": [],
    }
    model.eval()
    heads.eval()
    with torch.inference_mode():
        for _ in range(eval_batches):
            batch = make_program_batch(
                rng,
                batch_size=batch_size,
                n_pairs=n_pairs,
                device=device,
            )
            features = extract_features(model, masks, batch)
            target = features["target_class"]
            values["base_carry_accuracy"].append(
                class_accuracy(features["base_value_logits"], target)
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
            values["raw_slot_bridge_accuracy"].append(
                class_accuracy(
                    features["base_value_logits"]
                    + heads.raw_slot_bridge(features["target_slot"]),
                    target,
                )
            )
            values["bound_slot_probe_accuracy"].append(
                class_accuracy(heads.bound_slot_probe(features["bound_target_slot"]), target)
            )
            values["bound_slot_bridge_accuracy"].append(
                class_accuracy(
                    features["base_value_logits"]
                    + heads.bound_slot_bridge(features["bound_target_slot"]),
                    target,
                )
            )
            values["wrong_bound_slot_bridge_accuracy"].append(
                class_accuracy(
                    features["base_value_logits"]
                    + heads.bound_slot_bridge(features["wrong_bound_slot"]),
                    target,
                )
            )
            values["shuffled_bound_slot_bridge_accuracy"].append(
                class_accuracy(
                    features["base_value_logits"]
                    + heads.bound_slot_bridge(features["shuffled_bound_slot"]),
                    target,
                )
            )
            values["zero_bound_slot_bridge_accuracy"].append(
                class_accuracy(
                    features["base_value_logits"]
                    + heads.bound_slot_bridge(features["zero_bound_slot"]),
                    target,
                )
            )
            for metric in (
                "route_target_argmax_rate",
                "route_target_selected_rate",
                "outside_subspace_ratio",
            ):
                values[metric].append(float(features[metric]))
    result = {key: mean(metric_values) for key, metric_values in values.items()}
    result["bound_slot_bridge_minus_wrong"] = (
        result["bound_slot_bridge_accuracy"]
        - result["wrong_bound_slot_bridge_accuracy"]
    )
    result["bound_slot_bridge_minus_shuffled"] = (
        result["bound_slot_bridge_accuracy"]
        - result["shuffled_bound_slot_bridge_accuracy"]
    )
    result["bound_slot_bridge_minus_global"] = (
        result["bound_slot_bridge_accuracy"]
        - result["global_bridge_accuracy"]
    )
    return result


def evaluate_scrubbing(
    model: TACTransformerLM,
    heads: BindingHeads,
    masks: torch.Tensor,
    *,
    seed: int,
    batch_size: int,
    eval_batches: int,
    n_pairs: int,
    device: torch.device,
) -> dict[str, float]:
    rng = random.Random(seed)
    batches = [
        make_program_batch(
            rng,
            batch_size=batch_size,
            n_pairs=n_pairs,
            device=device,
        )
        for _ in range(eval_batches)
    ]
    baseline = score_bound_bridge_batches(model, heads, masks, batches)
    basis = bridge_rowspace_basis(heads.bound_slot_bridge.weight.detach())
    random_basis = random_orthonormal_basis(
        d_model=basis.shape[0],
        rank=basis.shape[1],
        seed=seed + 17,
        device=device,
    )
    slot_projected = score_bound_bridge_batches(
        model,
        heads,
        masks,
        batches,
        transform=lambda slot: project_out(slot, basis),
    )
    random_projected = score_bound_bridge_batches(
        model,
        heads,
        masks,
        batches,
        transform=lambda slot: project_out(slot, random_basis),
    )
    gradient_dims = top_gradient_dimensions(
        model,
        heads,
        masks,
        batches[: max(1, min(2, len(batches)))],
        rank=min(max(1, basis.shape[1]), masks.shape[-1]),
    )
    gradient_scrubbed = score_bound_bridge_batches(
        model,
        heads,
        masks,
        batches,
        transform=lambda slot: zero_dimensions(slot, gradient_dims),
    )
    random_dims = random_dimensions(
        d_model=masks.shape[-1],
        count=len(gradient_dims),
        seed=seed + 29,
        device=device,
    )
    random_dim_scrubbed = score_bound_bridge_batches(
        model,
        heads,
        masks,
        batches,
        transform=lambda slot: zero_dimensions(slot, random_dims),
    )
    with attention_output_ablation(model):
        attention_ablated = score_bound_bridge_batches(model, heads, masks, batches)
    return {
        "baseline_bound_slot_bridge_accuracy": baseline,
        "slot_subspace_projected_accuracy": slot_projected,
        "random_subspace_projected_accuracy": random_projected,
        "gradient_scrubbed_accuracy": gradient_scrubbed,
        "random_dim_scrubbed_accuracy": random_dim_scrubbed,
        "attention_ablated_accuracy": attention_ablated,
        "slot_subspace_projection_drop": baseline - slot_projected,
        "random_subspace_projection_drop": baseline - random_projected,
        "gradient_scrub_drop": baseline - gradient_scrubbed,
        "random_dim_scrub_drop": baseline - random_dim_scrubbed,
        "attention_ablation_drop": baseline - attention_ablated,
    }


def targeted_knockout_summary(
    model: TACTransformerLM,
    heads: BindingHeads,
    masks: torch.Tensor,
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
            baseline = score_bound_bridge_batches(model, heads, masks, batches)
            with knockout_program(model, program):
                target_knockout = score_bound_bridge_batches(model, heads, masks, batches)
            other_program = (program + 1) % N_PROGRAMS
            with knockout_program(model, other_program):
                other_knockout = score_bound_bridge_batches(model, heads, masks, batches)
            rows.append(
                {
                    "program": int(program),
                    "nontarget_program": int(other_program),
                    "baseline_accuracy": baseline,
                    "target_knockout_accuracy": target_knockout,
                    "nontarget_knockout_accuracy": other_knockout,
                    "targeted_knockout_drop": baseline - target_knockout,
                    "nontarget_knockout_drop": baseline - other_knockout,
                    "localized_drop_gap": (
                        baseline - target_knockout
                    )
                    - (baseline - other_knockout),
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
    }


def score_bound_bridge_batches(
    model: TACTransformerLM,
    heads: BindingHeads,
    masks: torch.Tensor,
    batches: Sequence[dict[str, torch.Tensor]],
    *,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> float:
    scores = []
    with torch.inference_mode():
        for batch in batches:
            features = extract_features(model, masks, batch)
            slot = features["bound_target_slot"]
            if transform is not None:
                slot = transform(slot)
            scores.append(
                class_accuracy(
                    features["base_value_logits"] + heads.bound_slot_bridge(slot),
                    features["target_class"],
                )
            )
    return mean(scores)


def extract_features(
    model: TACTransformerLM,
    masks: torch.Tensor,
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
    key_ids = batch["query"][:, -1]
    memory = support.identity_states[-1].program_memory
    target_slot = target_program_slots(memory, batch["target_program"])
    wrong_slot = target_program_slots(memory, (batch["target_program"] + 1) % N_PROGRAMS)
    bound_target = apply_program_subspace(target_slot, batch["target_program"], masks)
    wrong_bound = apply_program_subspace(wrong_slot, batch["target_program"], masks)
    shuffled = bound_target[torch.randperm(bound_target.shape[0], device=bound_target.device)]
    route = route_alignment(
        support.aux.token_program_activations,
        support.aux.token_selected_program_mask,
        batch["support_program_targets"],
    )
    return {
        "read_vector": model.memory_read_vector(key_ids, support.identity_states).detach(),
        "target_slot": target_slot.detach(),
        "bound_target_slot": bound_target.detach(),
        "wrong_bound_slot": wrong_bound.detach(),
        "shuffled_bound_slot": shuffled.detach(),
        "zero_bound_slot": torch.zeros_like(bound_target).detach(),
        "base_value_logits": query.logits[:, -1, value_ids].detach(),
        "memory_read_value_logits": model.memory_read_logits(
            key_ids,
            support.identity_states,
        )[:, value_ids].detach(),
        "target_class": (batch["target"] - VALUE_START).detach(),
        "outside_subspace_ratio": outside_subspace_ratio(memory, masks),
        **route,
    }


def apply_program_subspace(
    vectors: torch.Tensor,
    programs: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    selected_masks = masks[programs.to(masks.device)]
    return vectors * selected_masks


def outside_subspace_penalty(memory: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    slot_masks = masks[None, :, :].to(memory.device, dtype=memory.dtype)
    outside = memory * (1.0 - slot_masks)
    return outside.pow(2).mean()


def outside_subspace_ratio(memory: torch.Tensor, masks: torch.Tensor) -> float:
    slot_masks = masks[None, :, :].to(memory.device, dtype=memory.dtype)
    outside = (memory * (1.0 - slot_masks)).pow(2).sum()
    total = memory.pow(2).sum().clamp_min(1e-8)
    return float((outside / total).detach().cpu())


def bridge_rowspace_basis(weight: torch.Tensor) -> torch.Tensor:
    _, singular_values, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    if singular_values.numel() == 0:
        return torch.empty(weight.shape[1], 0, device=weight.device)
    threshold = singular_values.max() * 1e-5
    rank = int((singular_values > threshold).sum().item())
    rank = max(1, min(rank, vh.shape[0]))
    return vh[:rank].T.to(device=weight.device, dtype=weight.dtype)


def random_orthonormal_basis(
    *,
    d_model: int,
    rank: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    matrix = torch.randn(d_model, rank, generator=generator, device=device)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q[:, :rank]


def project_out(vectors: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0:
        return vectors
    basis = basis.to(device=vectors.device, dtype=vectors.dtype)
    return vectors - torch.matmul(torch.matmul(vectors, basis), basis.T)


def top_gradient_dimensions(
    model: TACTransformerLM,
    heads: BindingHeads,
    masks: torch.Tensor,
    batches: Sequence[dict[str, torch.Tensor]],
    *,
    rank: int,
) -> torch.Tensor:
    grads = []
    for batch in batches:
        features = extract_features(model, masks, batch)
        slot = features["bound_target_slot"].detach().requires_grad_(True)
        logits = features["base_value_logits"] + heads.bound_slot_bridge(slot)
        loss = F.cross_entropy(logits, features["target_class"])
        heads.bound_slot_bridge.zero_grad(set_to_none=True)
        loss.backward()
        grads.append(slot.grad.detach().abs().mean(dim=0))
    scores = torch.stack(grads).mean(dim=0)
    return scores.topk(k=min(rank, scores.numel())).indices


def random_dimensions(
    *,
    d_model: int,
    count: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randperm(d_model, generator=generator, device=device)[:count]


def zero_dimensions(vectors: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
    scrubbed = vectors.clone()
    scrubbed[:, dims.to(vectors.device)] = 0.0
    return scrubbed


@contextmanager
def attention_output_ablation(model: TACTransformerLM) -> Iterator[None]:
    originals = []
    for block in model.blocks:
        attention = getattr(block, "attention", None)
        if attention is None:
            continue
        original = attention.forward
        originals.append((attention, original))

        def patched(self, *args, _original=original, **kwargs):
            output = _original(*args, **kwargs)
            return dataclass_replace(output, hidden=torch.zeros_like(output.hidden))

        attention.forward = MethodType(patched, attention)
    try:
        yield
    finally:
        for attention, original in originals:
            attention.forward = original


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    aggregate = {}
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
            "binding": {
                key: mean_path(variant_rows, ["binding", key])
                for key in (
                    "base_carry_accuracy",
                    "direct_memory_read_accuracy",
                    "global_bridge_accuracy",
                    "raw_slot_bridge_accuracy",
                    "bound_slot_probe_accuracy",
                    "bound_slot_bridge_accuracy",
                    "wrong_bound_slot_bridge_accuracy",
                    "shuffled_bound_slot_bridge_accuracy",
                    "zero_bound_slot_bridge_accuracy",
                    "route_target_argmax_rate",
                    "route_target_selected_rate",
                    "outside_subspace_ratio",
                    "bound_slot_bridge_minus_wrong",
                    "bound_slot_bridge_minus_shuffled",
                    "bound_slot_bridge_minus_global",
                )
            },
            "knockout": {
                key: mean_path(variant_rows, ["targeted_knockout", key])
                for key in (
                    "targeted_knockout_drop",
                    "nontarget_knockout_drop",
                    "localized_drop_gap",
                    "max_targeted_knockout_drop",
                )
            },
            "scrubbing": {
                key: mean_path(variant_rows, ["scrubbing", key])
                for key in (
                    "baseline_bound_slot_bridge_accuracy",
                    "slot_subspace_projected_accuracy",
                    "random_subspace_projected_accuracy",
                    "gradient_scrubbed_accuracy",
                    "random_dim_scrubbed_accuracy",
                    "attention_ablated_accuracy",
                    "slot_subspace_projection_drop",
                    "random_subspace_projection_drop",
                    "gradient_scrub_drop",
                    "random_dim_scrub_drop",
                    "attention_ablation_drop",
                )
            },
        }
    if "semantic_baseline" in aggregate and "subspace_bound" in aggregate:
        aggregate["subspace_bound_minus_semantic_baseline"] = {
            "binding": subtract_nested(
                aggregate["subspace_bound"]["binding"],
                aggregate["semantic_baseline"]["binding"],
            ),
            "knockout": subtract_nested(
                aggregate["subspace_bound"]["knockout"],
                aggregate["semantic_baseline"]["knockout"],
            ),
            "scrubbing": subtract_nested(
                aggregate["subspace_bound"]["scrubbing"],
                aggregate["semantic_baseline"]["scrubbing"],
            ),
        }
    return aggregate


def subtract_nested(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: left[key] - right[key] for key in left.keys() & right.keys()}


def decide(aggregate: dict[str, Any]) -> dict[str, Any]:
    bound = aggregate.get("subspace_bound")
    if not bound:
        return {
            "status": "subspace_bound_variant_not_run",
            "reason": "TAC-217 requires subspace_bound for the A/B decision.",
        }
    binding = bound["binding"]
    knockout = bound["knockout"]
    scrub = bound["scrubbing"]
    binding_supported = (
        binding["bound_slot_bridge_minus_wrong"] >= 0.10
        and knockout["targeted_knockout_drop"] >= 0.05
    )
    scrubbing_supported = (
        scrub["slot_subspace_projection_drop"]
        > max(scrub["random_subspace_projection_drop"], knockout["targeted_knockout_drop"]) + 0.05
    )
    if binding_supported and scrubbing_supported:
        status = "local_binding_and_representation_scrubbing_supported"
    elif binding_supported:
        status = "local_binding_supported_without_scrubbing_advantage"
    elif scrubbing_supported:
        status = "representation_scrubbing_supported_without_local_binding"
    else:
        status = "binding_and_scrubbing_not_sufficient"
    return {
        "status": status,
        "direction_a_binding_supported": binding_supported,
        "direction_b_scrubbing_supported": scrubbing_supported,
        "bound_slot_bridge_accuracy": binding["bound_slot_bridge_accuracy"],
        "bound_slot_bridge_minus_wrong": binding["bound_slot_bridge_minus_wrong"],
        "targeted_knockout_drop": knockout["targeted_knockout_drop"],
        "slot_subspace_projection_drop": scrub["slot_subspace_projection_drop"],
        "random_subspace_projection_drop": scrub["random_subspace_projection_drop"],
        "gradient_scrub_drop": scrub["gradient_scrub_drop"],
        "attention_ablation_drop": scrub["attention_ablation_drop"],
        "interpretation": (
            "Direction A asks whether orthogonal subspace binding makes program "
            "slots locally persistent and causally necessary. Direction B asks "
            "whether identity/action signal is more vulnerable to representation "
            "subspace scrubbing than to program-level interventions."
        ),
    }


def format_markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    semantic = aggregate.get("semantic_baseline", {})
    bound = aggregate.get("subspace_bound", {})
    delta = aggregate.get("subspace_bound_minus_semantic_baseline", {})
    lines = [
        "# TAC-217 Representation Binding And Scrubbing",
        "",
        f"Decision: `{report['decision']['status']}`.",
        "",
        "## Direction A: Binding",
        "",
        "| Metric | semantic baseline | subspace-bound | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key in [
        ("Route target selected rate", "route_target_selected_rate"),
        ("Outside subspace ratio", "outside_subspace_ratio"),
        ("Global bridge accuracy", "global_bridge_accuracy"),
        ("Raw slot bridge accuracy", "raw_slot_bridge_accuracy"),
        ("Bound slot probe accuracy", "bound_slot_probe_accuracy"),
        ("Bound slot bridge accuracy", "bound_slot_bridge_accuracy"),
        ("Wrong bound slot bridge accuracy", "wrong_bound_slot_bridge_accuracy"),
        ("Bound slot bridge - wrong", "bound_slot_bridge_minus_wrong"),
        ("Targeted knockout drop", "targeted_knockout_drop"),
        ("Localized knockout gap", "localized_drop_gap"),
    ]:
        source = "knockout" if "knockout" in label.lower() else "binding"
        lines.append(
            "| "
            f"{label} | "
            f"{format_value(semantic.get(source, {}).get(key))} | "
            f"{format_value(bound.get(source, {}).get(key))} | "
            f"{format_value(delta.get(source, {}).get(key))} |"
        )
    lines.extend(
        [
            "",
            "## Direction B: Scrubbing",
            "",
            "| Metric | semantic baseline | subspace-bound | delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for label, key in [
        ("Baseline bound slot bridge accuracy", "baseline_bound_slot_bridge_accuracy"),
        ("Slot subspace projection drop", "slot_subspace_projection_drop"),
        ("Random subspace projection drop", "random_subspace_projection_drop"),
        ("Gradient scrub drop", "gradient_scrub_drop"),
        ("Random dimension scrub drop", "random_dim_scrub_drop"),
        ("Attention ablation drop", "attention_ablation_drop"),
    ]:
        lines.append(
            "| "
            f"{label} | "
            f"{format_value(semantic.get('scrubbing', {}).get(key))} | "
            f"{format_value(bound.get('scrubbing', {}).get(key))} | "
            f"{format_value(delta.get('scrubbing', {}).get(key))} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Direction A is positive only if the bound target slot beats wrong slots and target program knockouts selectively hurt.",
            "- Direction B is positive only if learned representation-subspace scrubbing drops accuracy more than random scrubs and program knockouts.",
            "- Attention ablation here zeros attention output during feature extraction; it is a broad attention-path control, not per-head attribution.",
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
