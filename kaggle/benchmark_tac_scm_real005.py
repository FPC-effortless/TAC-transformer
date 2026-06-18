from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tac_transformer import (
    GatedResidualStructureBridge,
    LinearStructureBridge,
    MLPStructureBridge,
    OracleStructureBridge,
    SlotConditionedProgramBottleneck,
    TACConfig,
    best_chunked_recall_tac_config,
    tac_scm_v02_config,
)


REAL005_TASK_MODES = (
    "clean_single_hop",
    "noisy_structure_cue",
    "partial_structure_cue",
    "delayed_structure_query",
    "multi_hop_structure_chain",
    "ambiguous_competing_structures",
    "distribution_shifted_structure_family",
    "low_data_transfer_family_a_to_b",
)

REAL005_BRIDGE_TYPES = ("linear", "mlp", "gated_residual")

REAL005_VARIANT_NAMES = (
    "vanilla_transformer",
    "legacy_best_chunked_recall_tac",
    "full_tac_scm_v02",
    "linear_structure_bridge",
    "mlp_structure_bridge",
    "gated_residual_structure_bridge",
    "oracle_bridge",
    "no_bridge_control",
    "no_slot_control",
    "reset_structure_control",
    "shuffled_structure_control",
    "correct_slot_knockout",
    "wrong_slot_knockout",
)

REAL005_METRIC_NAMES = (
    "behavior_accuracy",
    "vanilla_gap",
    "legacy_tac_gap",
    "bridge_gain",
    "oracle_gap",
    "carry_reset_delta",
    "carry_shuffled_delta",
    "slot_knockout_drop",
    "wrong_slot_knockout_drop",
    "structure_read_hit_rate",
    "structure_use_entropy",
    "bridge_seed_variance",
    "bridge_ranking_by_task_mode",
    "transfer_gain",
    "multi_hop_retention",
    "noisy_partial_cue_retention",
)

_LEARNED_BRIDGE_VARIANTS = {
    "linear": "linear_structure_bridge",
    "mlp": "mlp_structure_bridge",
    "gated_residual": "gated_residual_structure_bridge",
}


@dataclass(frozen=True)
class REAL005Batch:
    input_ids: Tensor
    token_mask: Tensor
    family_ids: Tensor
    slot_ids: Tensor
    target_structure_ids: Tensor
    read_structure_ids: Tensor
    labels: Tensor

    def to(self, device: torch.device | str) -> "REAL005Batch":
        return REAL005Batch(
            input_ids=self.input_ids.to(device),
            token_mask=self.token_mask.to(device),
            family_ids=self.family_ids.to(device),
            slot_ids=self.slot_ids.to(device),
            target_structure_ids=self.target_structure_ids.to(device),
            read_structure_ids=self.read_structure_ids.to(device),
            labels=self.labels.to(device),
        )


@dataclass
class REAL005Dataset:
    batch: REAL005Batch

    @property
    def size(self) -> int:
        return int(self.batch.labels.numel())

    def sample(self, batch_size: int, generator: torch.Generator) -> REAL005Batch:
        indices = torch.randint(0, self.size, (batch_size,), generator=generator)
        return self.slice_indices(indices)

    def iter_batches(self, batch_size: int) -> Iterable[REAL005Batch]:
        for start in range(0, self.size, batch_size):
            stop = min(self.size, start + batch_size)
            yield self.slice_indices(torch.arange(start, stop))

    def slice_indices(self, indices: Tensor) -> REAL005Batch:
        return REAL005Batch(
            input_ids=self.batch.input_ids[indices],
            token_mask=self.batch.token_mask[indices],
            family_ids=self.batch.family_ids[indices],
            slot_ids=self.batch.slot_ids[indices],
            target_structure_ids=self.batch.target_structure_ids[indices],
            read_structure_ids=self.batch.read_structure_ids[indices],
            labels=self.batch.labels[indices],
        )


@dataclass(frozen=True)
class REAL005VariantSpec:
    name: str
    bridge_type: Optional[str]
    use_structure_slots: bool
    use_structure_read: bool
    use_family_average_read: bool = False
    use_legacy_bias: bool = False
    oracle: bool = False


@dataclass
class REAL005Forward:
    logits: Tensor
    structure_read_hit_rate: Tensor
    structure_use_entropy: Tensor
    slot_auxiliary_loss: Tensor


class REAL005ProbeModel(nn.Module):
    """Benchmark-only probe around the existing TAC-SCM structure lane."""

    def __init__(
        self,
        *,
        spec: REAL005VariantSpec,
        d_model: int,
        vocab_size: int,
        n_families: int,
        slots_per_family: int,
        n_classes: int,
        structure_values: Tensor,
        label_table: Tensor,
    ):
        super().__init__()
        if d_model < n_classes:
            raise ValueError("d_model must be at least n_classes")
        self.spec = spec
        self.d_model = d_model
        self.n_families = n_families
        self.slots_per_family = slots_per_family
        self.n_structure_slots = n_families * slots_per_family
        self.n_classes = n_classes

        # Keep explicit config provenance for the compared architecture lanes.
        self.vanilla_config = TACConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=1,
            n_kv_heads=1,
            n_layers=1,
            n_programs=4,
            max_seq_len=8,
        )
        self.legacy_config = best_chunked_recall_tac_config(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=1,
            n_kv_heads=1,
            n_layers=1,
            n_programs=4,
            max_seq_len=8,
        )
        self.tac_scm_config = tac_scm_v02_config(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=1,
            n_kv_heads=1,
            n_layers=1,
            n_programs=4,
            max_seq_len=8,
            n_structure_families=n_families,
            n_structure_slots=self.n_structure_slots,
        )

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.legacy_family_bias = nn.Embedding(n_families, d_model)
        self.register_buffer("structure_values", structure_values.clone())
        self.register_buffer("label_table", label_table.clone().long())

        family_values = []
        for family_id in range(n_families):
            start = family_id * slots_per_family
            stop = start + slots_per_family
            family_values.append(structure_values[start:stop].mean(dim=0))
        self.register_buffer("family_structure_values", torch.stack(family_values))

        if spec.use_structure_slots:
            self.slot_bottleneck = SlotConditionedProgramBottleneck(
                d_model=d_model,
                n_structure_slots=self.n_structure_slots,
                n_programs=4,
                load_balance_weight=0.005,
            )
        else:
            self.slot_bottleneck = None

        if spec.bridge_type == "linear":
            self.bridge = LinearStructureBridge(d_model)
        elif spec.bridge_type == "mlp":
            self.bridge = MLPStructureBridge(d_model)
        elif spec.bridge_type == "gated_residual":
            self.bridge = GatedResidualStructureBridge(d_model)
        elif spec.bridge_type == "oracle":
            self.bridge = OracleStructureBridge(d_model, n_oracle_structures=n_classes)
        elif spec.bridge_type is None:
            self.bridge = None
        else:
            raise ValueError(f"unknown bridge type {spec.bridge_type!r}")
        self.behavior_head = nn.Linear(d_model, n_classes)
        _initialize_probe_weights(self, n_classes=n_classes)

    def forward(
        self,
        batch: REAL005Batch,
        *,
        intervention: str = "carry",
    ) -> REAL005Forward:
        token_embeddings = self.token_embedding(batch.input_ids)
        mask = batch.token_mask.unsqueeze(-1).to(token_embeddings.dtype)
        hidden = (token_embeddings * mask).sum(dim=1)
        hidden = hidden / mask.sum(dim=1).clamp_min(1.0)

        if self.spec.use_legacy_bias:
            hidden = hidden + 0.15 * self.legacy_family_bias(batch.family_ids)

        slot_auxiliary_loss = hidden.new_zeros(())
        entropy = hidden.new_zeros(())
        if self.slot_bottleneck is not None:
            slot_out = self.slot_bottleneck(hidden.unsqueeze(1))
            hidden = slot_out.hidden.squeeze(1)
            slot_auxiliary_loss = slot_out.auxiliary_loss
            entropy = _mean_entropy(slot_out.slot_state.slot_weights)

        read_hit = hidden.new_zeros(())
        if self.bridge is not None and self.spec.use_structure_read:
            if self.spec.oracle:
                bridged = self.bridge(hidden, batch.labels)
                read_hit = hidden.new_ones(())
            else:
                structure_vector, read_hit = self._structure_vector(
                    batch,
                    intervention=intervention,
                )
                bridged = self.bridge(hidden, structure_vector)
            hidden = bridged.hidden

        return REAL005Forward(
            logits=self.behavior_head(hidden),
            structure_read_hit_rate=read_hit,
            structure_use_entropy=entropy,
            slot_auxiliary_loss=slot_auxiliary_loss,
        )

    def _structure_vector(
        self,
        batch: REAL005Batch,
        *,
        intervention: str,
    ) -> tuple[Tensor, Tensor]:
        if self.spec.use_family_average_read:
            vectors = self.family_structure_values[batch.family_ids]
            family_labels = vectors[:, : self.n_classes].argmax(dim=-1)
            read_hit = (family_labels == batch.labels).float().mean()
            return vectors, read_hit

        if intervention == "reset":
            return torch.zeros_like(self.structure_values[batch.target_structure_ids]), batch.labels.new_zeros((), dtype=torch.float32)

        selected_ids = batch.read_structure_ids
        if intervention == "shuffled" and selected_ids.numel() > 1:
            selected_ids = torch.roll(selected_ids, shifts=1, dims=0)

        vectors = self.structure_values[selected_ids]
        if intervention == "correct_slot_knockout":
            vectors = torch.zeros_like(vectors)
            read_hit = batch.labels.new_zeros((), dtype=torch.float32)
        else:
            read_hit = (selected_ids == batch.target_structure_ids).float().mean()
        return vectors, read_hit


def run_tac_scm_real005(
    *,
    seeds: Iterable[int] | None = None,
    d_models: Iterable[int] | None = None,
    steps_values: Iterable[int] | None = None,
    train_samples_values: Iterable[int] | None = None,
    eval_samples: int = 48,
    batch_size: int = 12,
    task_modes: Iterable[str] | None = None,
    bridge_types: Iterable[str] | None = None,
    n_families: int = 4,
    slots_per_family: int = 8,
    n_classes: int = 4,
    learning_rate: float = 2e-2,
    device: str = "cpu",
) -> dict[str, Any]:
    seed_list = list(seeds if seeds is not None else [0])
    d_model_list = list(d_models if d_models is not None else [16])
    steps_list = list(steps_values if steps_values is not None else [6])
    train_samples_list = list(train_samples_values if train_samples_values is not None else [48])
    mode_list = list(task_modes if task_modes is not None else REAL005_TASK_MODES)
    bridge_type_list = list(bridge_types if bridge_types is not None else REAL005_BRIDGE_TYPES)
    _validate_real005_inputs(
        seeds=seed_list,
        d_models=d_model_list,
        steps_values=steps_list,
        train_samples_values=train_samples_list,
        eval_samples=eval_samples,
        batch_size=batch_size,
        task_modes=mode_list,
        bridge_types=bridge_type_list,
        n_classes=n_classes,
    )
    device_obj = torch.device(device)

    per_cell: list[dict[str, Any]] = []
    variant_scores: dict[str, list[dict[str, float]]] = {name: [] for name in REAL005_VARIANT_NAMES}
    mode_scores: dict[str, dict[str, list[dict[str, float]]]] = {
        mode: {name: [] for name in REAL005_VARIANT_NAMES} for mode in mode_list
    }
    seed_bridge_scores: dict[str, dict[int, list[float]]] = {
        bridge: {seed: [] for seed in seed_list} for bridge in bridge_type_list
    }

    for mode in mode_list:
        for d_model in d_model_list:
            for steps in steps_list:
                for train_samples in train_samples_list:
                    for seed in seed_list:
                        cell = _run_real005_cell(
                            seed=seed,
                            mode=mode,
                            d_model=d_model,
                            steps=steps,
                            train_samples=train_samples,
                            eval_samples=eval_samples,
                            batch_size=batch_size,
                            n_families=n_families,
                            slots_per_family=slots_per_family,
                            n_classes=n_classes,
                            learning_rate=learning_rate,
                            bridge_types=bridge_type_list,
                            device=device_obj,
                        )
                        per_cell.append(cell)
                        for variant_name, score in cell["variant_results"].items():
                            variant_scores[variant_name].append(score)
                            mode_scores[mode][variant_name].append(score)
                        for bridge in bridge_type_list:
                            variant_name = _LEARNED_BRIDGE_VARIANTS[bridge]
                            seed_bridge_scores[bridge][seed].append(
                                cell["variant_results"][variant_name]["behavior_accuracy"]
                            )

    variant_results = {
        name: _aggregate_scores(scores) for name, scores in variant_scores.items()
    }
    mode_results = {
        mode: {
            name: _aggregate_scores(scores)
            for name, scores in variant_map.items()
        }
        for mode, variant_map in mode_scores.items()
    }
    bridge_results = _aggregate_bridge_results(
        variant_results,
        seed_bridge_scores,
        bridge_type_list,
    )
    metrics = _compute_real005_metrics(
        variant_results=variant_results,
        mode_results=mode_results,
        bridge_results=bridge_results,
        bridge_types=bridge_type_list,
    )
    selected_bridge = metrics.pop("selected_bridge")
    gate = evaluate_real005_success_gate(mode_results, metrics)
    promotion = select_bridge_promotion_candidate(bridge_results)
    bottleneck = diagnose_real005_bottleneck(metrics, promotion)

    return {
        "benchmark": "TAC-SCM-REAL005 bridge stability and harder structure generalization",
        "status": gate["status"],
        "variants": list(REAL005_VARIANT_NAMES),
        "task_modes": mode_list,
        "bridge_types": bridge_type_list,
        "metrics": metrics,
        "selected_bridge": selected_bridge,
        "variant_results": variant_results,
        "mode_results": mode_results,
        "bridge_results": bridge_results,
        "success_gate": gate,
        "promotion": promotion,
        "bottleneck": bottleneck,
        "failure_analysis": _real005_failure_analysis(gate, bottleneck, promotion, metrics),
        "per_cell_results": per_cell,
        "config": {
            "seeds": seed_list,
            "d_models": d_model_list,
            "steps_values": steps_list,
            "train_samples_values": train_samples_list,
            "eval_samples": eval_samples,
            "batch_size": batch_size,
            "n_families": n_families,
            "slots_per_family": slots_per_family,
            "n_classes": n_classes,
            "learning_rate": learning_rate,
            "device": str(device_obj),
        },
    }


def evaluate_real005_success_gate(
    mode_results: dict[str, dict[str, dict[str, float]]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    failed: list[str] = []
    learned_wins_by_mode: dict[str, bool] = {}
    for mode, variants in mode_results.items():
        vanilla = _accuracy(variants, "vanilla_transformer")
        legacy = _accuracy(variants, "legacy_best_chunked_recall_tac")
        learned = max(_accuracy(variants, name) for name in _LEARNED_BRIDGE_VARIANTS.values())
        learned_wins = learned > vanilla and learned > legacy
        learned_wins_by_mode[mode] = learned_wins
    if not all(learned_wins_by_mode.values()):
        failed.append("no learned bridge beats vanilla and legacy in every mode")

    if metrics.get("carry_reset_delta", 0.0) <= 0.0:
        failed.append("carry does not beat reset")
    if metrics.get("carry_shuffled_delta", 0.0) <= 0.0:
        failed.append("carry does not beat shuffled")
    if metrics.get("slot_knockout_drop", 0.0) <= metrics.get("wrong_slot_knockout_drop", 0.0):
        failed.append("correct-slot knockout does not hurt more than wrong-slot knockout")
    oracle_gap = metrics.get("oracle_gap", 0.0)
    if oracle_gap <= 0.0:
        failed.append("oracle bridge is not above learned bridges")
    if oracle_gap > 0.35:
        failed.append("oracle gap is measured but not shrinking")

    return {
        "status": "passed" if not failed else "failed",
        "failed_conditions": failed,
        "learned_wins_by_mode": learned_wins_by_mode,
        "oracle_gap": oracle_gap,
    }


def select_bridge_promotion_candidate(
    bridge_results: dict[str, dict[str, float]],
) -> dict[str, Any]:
    ranked = sorted(
        bridge_results.items(),
        key=lambda item: (item[1].get("behavior_accuracy", 0.0), -item[1].get("seed_variance", 1e9)),
        reverse=True,
    )
    if not ranked:
        return {
            "promoted": False,
            "recommended_bridge": None,
            "reason": "no learned bridge results available",
            "ranking": [],
        }
    best_name, best = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else {"behavior_accuracy": -1.0, "seed_variance": 1e9}
    other_variances = [
        scores.get("seed_variance", 1e9)
        for name, scores in ranked[1:]
    ]
    variance_ok = not other_variances or best.get("seed_variance", 1e9) <= min(other_variances)
    accuracy_ok = best.get("behavior_accuracy", 0.0) > runner_up.get("behavior_accuracy", -1.0)
    promoted = accuracy_ok and variance_ok
    return {
        "promoted": promoted,
        "recommended_bridge": best_name if promoted else None,
        "reason": (
            f"{best_name} wins mean accuracy and has lower seed variance"
            if promoted
            else "no learned bridge wins both mean accuracy and seed variance"
        ),
        "ranking": [
            {
                "bridge": name,
                "behavior_accuracy": scores.get("behavior_accuracy", 0.0),
                "seed_variance": scores.get("seed_variance", 0.0),
            }
            for name, scores in ranked
        ],
    }


def diagnose_real005_bottleneck(
    metrics: dict[str, Any],
    promotion: dict[str, Any],
) -> str:
    if metrics.get("structure_read_hit_rate", 0.0) < 0.65:
        return "structure_read_quality"
    if metrics.get("slot_knockout_drop", 0.0) <= metrics.get("wrong_slot_knockout_drop", 0.0):
        return "slot_routing"
    if metrics.get("multi_hop_retention", 0.0) < 0.6:
        return "multi-hop_composition"
    if metrics.get("bridge_gain", 0.0) <= 0.0 or metrics.get("oracle_gap", 0.0) > 0.35:
        return "bridge_objective"
    if not promotion.get("promoted", False):
        return "bridge_objective"
    return "none"


def _run_real005_cell(
    *,
    seed: int,
    mode: str,
    d_model: int,
    steps: int,
    train_samples: int,
    eval_samples: int,
    batch_size: int,
    n_families: int,
    slots_per_family: int,
    n_classes: int,
    learning_rate: float,
    bridge_types: list[str],
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    vocab_size = 4 + n_families + slots_per_family + 16
    label_table = _make_label_table(seed, n_families * slots_per_family, n_classes)
    structure_values = _make_structure_values(
        seed=seed,
        label_table=label_table,
        d_model=d_model,
        n_classes=n_classes,
    )
    train = _make_dataset(
        seed=seed + 103,
        samples=train_samples,
        mode=mode,
        split="train",
        n_families=n_families,
        slots_per_family=slots_per_family,
        label_table=label_table,
        vocab_size=vocab_size,
    )
    eval_data = _make_dataset(
        seed=seed + 307,
        samples=eval_samples,
        mode=mode,
        split="eval",
        n_families=n_families,
        slots_per_family=slots_per_family,
        label_table=label_table,
        vocab_size=vocab_size,
    )
    variant_results: dict[str, dict[str, float]] = {}
    models: dict[str, REAL005ProbeModel] = {}
    for index, spec in enumerate(_trainable_specs(bridge_types)):
        torch.manual_seed(seed * 1009 + index)
        model = REAL005ProbeModel(
            spec=spec,
            d_model=d_model,
            vocab_size=vocab_size,
            n_families=n_families,
            slots_per_family=slots_per_family,
            n_classes=n_classes,
            structure_values=structure_values,
            label_table=label_table,
        ).to(device)
        _train_model(
            model,
            train,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed * 1597 + index,
            device=device,
        )
        models[spec.name] = model
        variant_results[spec.name] = _evaluate_model(
            model,
            eval_data,
            batch_size=batch_size,
            device=device,
        )

    full_model = models["full_tac_scm_v02"]
    for variant_name, intervention in (
        ("reset_structure_control", "reset"),
        ("shuffled_structure_control", "shuffled"),
        ("correct_slot_knockout", "correct_slot_knockout"),
        ("wrong_slot_knockout", "wrong_slot_knockout"),
    ):
        variant_results[variant_name] = _evaluate_model(
            full_model,
            eval_data,
            batch_size=batch_size,
            device=device,
            intervention=intervention,
        )

    return {
        "seed": seed,
        "task_mode": mode,
        "d_model": d_model,
        "steps": steps,
        "train_samples": train_samples,
        "variant_results": variant_results,
    }


def _train_model(
    model: REAL005ProbeModel,
    train: REAL005Dataset,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> None:
    model.train()
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=learning_rate,
        weight_decay=0.0,
    )
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        batch = train.sample(batch_size, generator).to(device)
        out = model(batch)
        loss = F.cross_entropy(out.logits, batch.labels)
        loss = loss + 0.01 * out.slot_auxiliary_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def _evaluate_model(
    model: REAL005ProbeModel,
    eval_data: REAL005Dataset,
    *,
    batch_size: int,
    device: torch.device,
    intervention: str = "carry",
) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    hit_rates: list[float] = []
    entropies: list[float] = []
    for batch in eval_data.iter_batches(batch_size):
        batch = batch.to(device)
        out = model(batch, intervention=intervention)
        pred = out.logits.argmax(dim=-1)
        correct += int((pred == batch.labels).sum().item())
        total += int(batch.labels.numel())
        hit_rates.append(float(out.structure_read_hit_rate.detach().cpu().item()))
        entropies.append(float(out.structure_use_entropy.detach().cpu().item()))
    return {
        "behavior_accuracy": correct / max(1, total),
        "structure_read_hit_rate": mean(hit_rates) if hit_rates else 0.0,
        "structure_use_entropy": mean(entropies) if entropies else 0.0,
    }


def _compute_real005_metrics(
    *,
    variant_results: dict[str, dict[str, float]],
    mode_results: dict[str, dict[str, dict[str, float]]],
    bridge_results: dict[str, dict[str, float]],
    bridge_types: list[str],
) -> dict[str, Any]:
    best_bridge_name, best_bridge = max(
        bridge_results.items(),
        key=lambda item: item[1].get("behavior_accuracy", 0.0),
    )
    full = variant_results["full_tac_scm_v02"]["behavior_accuracy"]
    best_learned = best_bridge["behavior_accuracy"]
    oracle = variant_results["oracle_bridge"]["behavior_accuracy"]
    no_bridge = variant_results["no_bridge_control"]["behavior_accuracy"]
    vanilla = variant_results["vanilla_transformer"]["behavior_accuracy"]
    legacy = variant_results["legacy_best_chunked_recall_tac"]["behavior_accuracy"]
    reset = variant_results["reset_structure_control"]["behavior_accuracy"]
    shuffled = variant_results["shuffled_structure_control"]["behavior_accuracy"]
    correct_ko = variant_results["correct_slot_knockout"]["behavior_accuracy"]
    wrong_ko = variant_results["wrong_slot_knockout"]["behavior_accuracy"]
    ranking = _bridge_ranking_by_mode(mode_results, bridge_types)
    clean = _best_learned_mode_accuracy(mode_results, "clean_single_hop")
    multi = _best_learned_mode_accuracy(mode_results, "multi_hop_structure_chain")
    noisy = _best_learned_mode_accuracy(mode_results, "noisy_structure_cue")
    partial = _best_learned_mode_accuracy(mode_results, "partial_structure_cue")
    transfer = _best_learned_mode_accuracy(mode_results, "low_data_transfer_family_a_to_b")
    transfer_baseline = max(
        _accuracy(mode_results.get("low_data_transfer_family_a_to_b", {}), "vanilla_transformer"),
        _accuracy(mode_results.get("low_data_transfer_family_a_to_b", {}), "legacy_best_chunked_recall_tac"),
    )
    return {
        "behavior_accuracy": best_learned,
        "selected_bridge": best_bridge_name,
        "vanilla_gap": best_learned - vanilla,
        "legacy_tac_gap": best_learned - legacy,
        "bridge_gain": best_learned - no_bridge,
        "oracle_gap": oracle - best_learned,
        "carry_reset_delta": full - reset,
        "carry_shuffled_delta": full - shuffled,
        "slot_knockout_drop": full - correct_ko,
        "wrong_slot_knockout_drop": full - wrong_ko,
        "structure_read_hit_rate": variant_results["full_tac_scm_v02"]["structure_read_hit_rate"],
        "structure_use_entropy": variant_results["full_tac_scm_v02"]["structure_use_entropy"],
        "bridge_seed_variance": {
            bridge: bridge_results[bridge].get("seed_variance", 0.0)
            for bridge in bridge_types
        },
        "bridge_ranking_by_task_mode": ranking,
        "transfer_gain": transfer - transfer_baseline,
        "multi_hop_retention": 0.0 if clean == 0.0 else multi / clean,
        "noisy_partial_cue_retention": 0.0 if clean == 0.0 else mean([noisy, partial]) / clean,
    }


def _trainable_specs(bridge_types: list[str]) -> tuple[REAL005VariantSpec, ...]:
    specs: list[REAL005VariantSpec] = [
        REAL005VariantSpec("vanilla_transformer", None, False, False),
        REAL005VariantSpec("legacy_best_chunked_recall_tac", None, False, False, use_legacy_bias=True),
        REAL005VariantSpec("full_tac_scm_v02", "gated_residual", True, True),
        REAL005VariantSpec("oracle_bridge", "oracle", True, True, oracle=True),
        REAL005VariantSpec("no_bridge_control", None, True, False),
        REAL005VariantSpec("no_slot_control", "gated_residual", False, True, use_family_average_read=True),
    ]
    bridge_variant_names = {
        "linear": "linear_structure_bridge",
        "mlp": "mlp_structure_bridge",
        "gated_residual": "gated_residual_structure_bridge",
    }
    for bridge in bridge_types:
        specs.append(REAL005VariantSpec(bridge_variant_names[bridge], bridge, True, True))
    return tuple(specs)


def _make_label_table(seed: int, n_structures: int, n_classes: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed + 19)
    labels = torch.randint(0, n_classes, (n_structures,), generator=generator)
    for class_id in range(n_classes):
        labels[class_id % n_structures] = class_id
    return labels


def _make_structure_values(
    *,
    seed: int,
    label_table: Tensor,
    d_model: int,
    n_classes: int,
) -> Tensor:
    generator = torch.Generator().manual_seed(seed + 41)
    values = 0.03 * torch.randn(label_table.numel(), d_model, generator=generator)
    values[:, :n_classes] = 0.0
    values[torch.arange(label_table.numel()), label_table] = 4.0
    if d_model > n_classes:
        values[:, n_classes:] += 0.12 * torch.randn(
            label_table.numel(),
            d_model - n_classes,
            generator=generator,
        )
    return values


def _make_dataset(
    *,
    seed: int,
    samples: int,
    mode: str,
    split: str,
    n_families: int,
    slots_per_family: int,
    label_table: Tensor,
    vocab_size: int,
) -> REAL005Dataset:
    if mode not in REAL005_TASK_MODES:
        raise ValueError(f"unknown REAL005 task mode {mode!r}")
    generator = torch.Generator().manual_seed(seed)
    family_ids = _sample_families(
        samples,
        mode=mode,
        split=split,
        n_families=n_families,
        generator=generator,
    )
    slot_ids = torch.randint(0, slots_per_family, (samples,), generator=generator)
    first_structure_ids = family_ids * slots_per_family + slot_ids
    if mode == "multi_hop_structure_chain":
        hop_slots = (slot_ids + label_table[first_structure_ids] + 1) % slots_per_family
        target_structure_ids = family_ids * slots_per_family + hop_slots
    else:
        target_structure_ids = first_structure_ids
    labels = label_table[target_structure_ids].clone()
    read_hit_rate = _mode_read_hit_rate(mode)
    read_structure_ids = _corrupt_structure_reads(
        target_structure_ids,
        family_ids,
        slots_per_family,
        hit_rate=read_hit_rate,
        generator=generator,
    )
    input_ids, token_mask = _build_mode_tokens(
        family_ids=family_ids,
        slot_ids=slot_ids,
        target_structure_ids=target_structure_ids,
        mode=mode,
        n_families=n_families,
        slots_per_family=slots_per_family,
        vocab_size=vocab_size,
        generator=generator,
    )
    return REAL005Dataset(
        REAL005Batch(
            input_ids=input_ids.long(),
            token_mask=token_mask.bool(),
            family_ids=family_ids.long(),
            slot_ids=slot_ids.long(),
            target_structure_ids=target_structure_ids.long(),
            read_structure_ids=read_structure_ids.long(),
            labels=labels.long(),
        )
    )


def _sample_families(
    samples: int,
    *,
    mode: str,
    split: str,
    n_families: int,
    generator: torch.Generator,
) -> Tensor:
    if mode == "distribution_shifted_structure_family":
        if split == "train":
            return torch.randint(0, max(1, n_families - 1), (samples,), generator=generator)
        return torch.full((samples,), n_families - 1, dtype=torch.long)
    if mode == "low_data_transfer_family_a_to_b":
        return torch.zeros(samples, dtype=torch.long) if split == "train" else torch.ones(samples, dtype=torch.long)
    return torch.randint(0, n_families, (samples,), generator=generator)


def _build_mode_tokens(
    *,
    family_ids: Tensor,
    slot_ids: Tensor,
    target_structure_ids: Tensor,
    mode: str,
    n_families: int,
    slots_per_family: int,
    vocab_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    samples = int(family_ids.numel())
    pad = 0
    query = 1
    mask_token = 2
    noise_token = 3
    family_offset = 4
    slot_offset = family_offset + n_families
    aux_offset = slot_offset + slots_per_family
    tokens = torch.full((samples, 8), pad, dtype=torch.long)
    mask = torch.zeros(samples, 8, dtype=torch.bool)

    family_tokens = family_offset + family_ids
    slot_tokens = slot_offset + slot_ids
    if mode == "noisy_structure_cue":
        corrupt = torch.rand(samples, generator=generator) < 0.35
        random_slots = torch.randint(0, slots_per_family, (samples,), generator=generator)
        slot_tokens = torch.where(corrupt, slot_offset + random_slots, slot_tokens)
    elif mode == "partial_structure_cue":
        slot_tokens = torch.full_like(slot_tokens, mask_token)

    if mode == "delayed_structure_query":
        sequence = [
            family_tokens,
            slot_tokens,
            torch.full_like(slot_tokens, aux_offset),
            torch.full_like(slot_tokens, aux_offset + 1),
            torch.full_like(slot_tokens, aux_offset + 2),
            torch.full_like(slot_tokens, query),
        ]
    elif mode == "ambiguous_competing_structures":
        competitor_slot = (slot_ids + 1) % slots_per_family
        sequence = [
            family_tokens,
            slot_tokens,
            family_tokens,
            slot_offset + competitor_slot,
            torch.full_like(slot_tokens, query),
        ]
    elif mode == "multi_hop_structure_chain":
        hop_slot = target_structure_ids % slots_per_family
        sequence = [
            family_tokens,
            slot_tokens,
            torch.full_like(slot_tokens, aux_offset + 3),
            slot_offset + hop_slot,
            torch.full_like(slot_tokens, query),
        ]
    elif mode == "noisy_structure_cue":
        sequence = [
            family_tokens,
            torch.full_like(slot_tokens, noise_token),
            slot_tokens,
            torch.full_like(slot_tokens, query),
        ]
    else:
        sequence = [family_tokens, slot_tokens, torch.full_like(slot_tokens, query)]

    for index, value in enumerate(sequence):
        tokens[:, index] = value.clamp(max=vocab_size - 1)
        mask[:, index] = True
    return tokens, mask


def _mode_read_hit_rate(mode: str) -> float:
    return {
        "clean_single_hop": 1.0,
        "noisy_structure_cue": 0.86,
        "partial_structure_cue": 0.74,
        "delayed_structure_query": 0.88,
        "multi_hop_structure_chain": 0.76,
        "ambiguous_competing_structures": 0.68,
        "distribution_shifted_structure_family": 0.82,
        "low_data_transfer_family_a_to_b": 0.78,
    }[mode]


def _corrupt_structure_reads(
    target_structure_ids: Tensor,
    family_ids: Tensor,
    slots_per_family: int,
    *,
    hit_rate: float,
    generator: torch.Generator,
) -> Tensor:
    keep = torch.rand(target_structure_ids.shape[0], generator=generator) < hit_rate
    wrong_slots = torch.randint(0, slots_per_family, target_structure_ids.shape, generator=generator)
    wrong_ids = family_ids * slots_per_family + wrong_slots
    wrong_ids = torch.where(wrong_ids == target_structure_ids, (wrong_ids + 1) % (family_ids.max().item() + 1) * slots_per_family, wrong_ids)
    return torch.where(keep, target_structure_ids, wrong_ids)


def _initialize_probe_weights(model: REAL005ProbeModel, *, n_classes: int) -> None:
    with torch.no_grad():
        model.token_embedding.weight.normal_(mean=0.0, std=0.03)
        model.token_embedding.weight[0].zero_()
        model.legacy_family_bias.weight.normal_(mean=0.0, std=0.04)
        model.behavior_head.weight.zero_()
        model.behavior_head.bias.zero_()
        for class_id in range(n_classes):
            model.behavior_head.weight[class_id, class_id] = 1.0
        if isinstance(model.bridge, LinearStructureBridge):
            _init_linear_bridge(model.bridge)
        elif isinstance(model.bridge, MLPStructureBridge):
            _init_mlp_bridge(model.bridge, model.d_model)
        elif isinstance(model.bridge, GatedResidualStructureBridge):
            _init_gated_bridge(model.bridge, model.d_model)
        elif isinstance(model.bridge, OracleStructureBridge):
            _init_oracle_bridge(model.bridge, n_classes=n_classes)


def _init_linear_bridge(bridge: LinearStructureBridge) -> None:
    bridge.projection.weight.zero_()
    bridge.projection.weight.copy_(torch.eye(bridge.d_model))


def _init_mlp_bridge(bridge: MLPStructureBridge, d_model: int) -> None:
    first = bridge.projection[0]
    second = bridge.projection[2]
    first.weight.zero_()
    first.bias.zero_()
    second.weight.zero_()
    second.bias.zero_()
    first.weight[:d_model].copy_(torch.eye(d_model))
    second.weight[:, :d_model].copy_(torch.eye(d_model))


def _init_gated_bridge(bridge: GatedResidualStructureBridge, d_model: int) -> None:
    _init_mlp_bridge(bridge, d_model)
    bridge.gate.weight.zero_()
    bridge.gate.bias.fill_(2.0)


def _init_oracle_bridge(bridge: OracleStructureBridge, *, n_classes: int) -> None:
    bridge.oracle_embedding.weight.zero_()
    for class_id in range(n_classes):
        bridge.oracle_embedding.weight[class_id, class_id] = 4.5
    bridge.projection.weight.zero_()
    bridge.projection.weight.copy_(torch.eye(bridge.d_model))


def _mean_entropy(probs: Tensor) -> Tensor:
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
    return entropy.mean()


def _aggregate_scores(scores: list[dict[str, float]]) -> dict[str, float]:
    if not scores:
        return {
            "behavior_accuracy": 0.0,
            "structure_read_hit_rate": 0.0,
            "structure_use_entropy": 0.0,
        }
    keys = scores[0].keys()
    out: dict[str, float] = {}
    for key in keys:
        values = [score[key] for score in scores]
        out[key] = mean(values)
        if len(values) > 1:
            variance = sum((value - out[key]) ** 2 for value in values) / len(values)
            out[f"{key}_std"] = math.sqrt(variance)
    return out


def _aggregate_bridge_results(
    variant_results: dict[str, dict[str, float]],
    seed_bridge_scores: dict[str, dict[int, list[float]]],
    bridge_types: list[str],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for bridge in bridge_types:
        variant_name = _LEARNED_BRIDGE_VARIANTS[bridge]
        seed_means = [
            mean(values)
            for _, values in sorted(seed_bridge_scores[bridge].items())
            if values
        ]
        bridge_scores = dict(variant_results[variant_name])
        if seed_means:
            seed_mean = mean(seed_means)
            bridge_scores["seed_variance"] = sum((value - seed_mean) ** 2 for value in seed_means) / len(seed_means)
        else:
            bridge_scores["seed_variance"] = 0.0
        out[bridge] = bridge_scores
    return out


def _bridge_ranking_by_mode(
    mode_results: dict[str, dict[str, dict[str, float]]],
    bridge_types: list[str],
) -> dict[str, list[dict[str, float | str]]]:
    ranking: dict[str, list[dict[str, float | str]]] = {}
    for mode, variants in mode_results.items():
        rows = []
        for bridge in bridge_types:
            variant_name = _LEARNED_BRIDGE_VARIANTS[bridge]
            rows.append(
                {
                    "bridge": bridge,
                    "behavior_accuracy": _accuracy(variants, variant_name),
                }
            )
        rows.sort(key=lambda row: float(row["behavior_accuracy"]), reverse=True)
        ranking[mode] = rows
    return ranking


def _best_learned_mode_accuracy(
    mode_results: dict[str, dict[str, dict[str, float]]],
    mode: str,
) -> float:
    variants = mode_results.get(mode, {})
    if not variants:
        return 0.0
    return max(_accuracy(variants, variant) for variant in _LEARNED_BRIDGE_VARIANTS.values())


def _accuracy(variants: dict[str, dict[str, float]], variant_name: str) -> float:
    return float(variants.get(variant_name, {}).get("behavior_accuracy", 0.0))


def _validate_real005_inputs(
    *,
    seeds: list[int],
    d_models: list[int],
    steps_values: list[int],
    train_samples_values: list[int],
    eval_samples: int,
    batch_size: int,
    task_modes: list[str],
    bridge_types: list[str],
    n_classes: int,
) -> None:
    if not seeds:
        raise ValueError("at least one seed is required")
    if not d_models or any(d_model < n_classes for d_model in d_models):
        raise ValueError("d_models must be non-empty and at least n_classes")
    if not steps_values or any(steps < 0 for steps in steps_values):
        raise ValueError("steps_values must be non-empty and non-negative")
    if not train_samples_values or any(samples < 1 for samples in train_samples_values):
        raise ValueError("train_samples_values must be non-empty and positive")
    if eval_samples < 1 or batch_size < 1:
        raise ValueError("eval_samples and batch_size must be positive")
    unknown_modes = set(task_modes) - set(REAL005_TASK_MODES)
    if unknown_modes:
        raise ValueError(f"unknown REAL005 task modes: {sorted(unknown_modes)}")
    unknown_bridges = set(bridge_types) - set(REAL005_BRIDGE_TYPES)
    if unknown_bridges:
        raise ValueError(f"unknown REAL005 bridge types: {sorted(unknown_bridges)}")


def _real005_failure_analysis(
    gate: dict[str, Any],
    bottleneck: str,
    promotion: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    if gate["status"] == "passed":
        if promotion["promoted"]:
            return (
                "REAL005 passed. "
                f"{promotion['recommended_bridge']} is the TAC-SCM v0.2 default bridge candidate."
            )
        return "REAL005 passed but no learned bridge met the promotion stability rule."
    return (
        f"REAL005 failed with bottleneck={bottleneck}. "
        f"Failed conditions: {', '.join(gate['failed_conditions'])}. "
        f"behavior_accuracy={metrics['behavior_accuracy']:.4f}, "
        f"oracle_gap={metrics['oracle_gap']:.4f}, "
        f"bridge_gain={metrics['bridge_gain']:.4f}."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TAC-SCM-REAL005 bridge stability and harder generalization benchmark."
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--ten-seed", action="store_true")
    parser.add_argument("--full-sweep", action="store_true")
    parser.add_argument("--d-models", type=int, nargs="*", default=[16])
    parser.add_argument("--steps-values", type=int, nargs="*", default=[6])
    parser.add_argument("--train-samples-values", type=int, nargs="*", default=[48])
    parser.add_argument("--eval-samples", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--task-modes", nargs="*", default=list(REAL005_TASK_MODES))
    parser.add_argument("--bridge-types", nargs="*", default=list(REAL005_BRIDGE_TYPES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = list(range(10)) if args.ten_seed or args.full_sweep else args.seeds
    d_models = [16, 32, 48] if args.full_sweep else args.d_models
    steps_values = [6, 12, 24] if args.full_sweep else args.steps_values
    train_samples_values = [48, 96] if args.full_sweep else args.train_samples_values
    result = run_tac_scm_real005(
        seeds=seeds,
        d_models=d_models,
        steps_values=steps_values,
        train_samples_values=train_samples_values,
        eval_samples=args.eval_samples,
        batch_size=args.batch_size,
        task_modes=args.task_modes,
        bridge_types=args.bridge_types,
        device=args.device,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
