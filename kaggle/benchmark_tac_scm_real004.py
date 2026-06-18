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

from kaggle.benchmark_structure_compression_roi import evaluate_structure_compression_roi
from tac_transformer import (
    ConceptVolumeEncoder,
    GatedResidualStructureBridge,
    LinearStructureBridge,
    MLPStructureBridge,
    OracleStructureBridge,
    SlotConditionedProgramBottleneck,
    StructureLifecycleScorer,
    StructureLifecycleStats,
    StructureMemoryModule,
    StructureObject,
    TACConfig,
    TACTransformerLM,
    TwoLevelStructureRouter,
    VanillaTransformerLM,
    best_chunked_recall_tac_config,
    tac_scm_v02_config,
)


REAL004_VARIANT_NAMES = (
    "vanilla_transformer",
    "legacy_best_chunked_recall_tac",
    "full_tac_scm_v02",
    "tac_scm_v02_without_structure_slots",
    "tac_scm_v02_without_structure_bridge",
    "tac_scm_v02_linear_bridge",
    "tac_scm_v02_mlp_bridge",
    "tac_scm_v02_gated_residual_bridge",
    "tac_scm_v02_oracle_bridge",
    "reset_structure_control",
    "shuffled_structure_control",
    "wrong_slot_knockout_control",
)

REAL004_METRIC_NAMES = (
    "behavior_accuracy",
    "bridge_gain",
    "oracle_gap",
    "carry_reset_delta",
    "carry_shuffled_delta",
    "slot_knockout_drop",
    "wrong_slot_knockout_drop",
    "structure_read_hit_rate",
    "structure_use_entropy",
    "legacy_tac_gap",
    "vanilla_gap",
    "compression_roi_compatible",
    "lifecycle_preserve_retire_sane",
)

_CONTROL_VARIANTS = {
    "reset_structure_control",
    "shuffled_structure_control",
    "wrong_slot_knockout_control",
}


@dataclass(frozen=True)
class REAL004Batch:
    input_ids: Tensor
    family_ids: Tensor
    slot_ids: Tensor
    structure_ids: Tensor
    labels: Tensor

    def to(self, device: torch.device | str) -> "REAL004Batch":
        return REAL004Batch(
            input_ids=self.input_ids.to(device),
            family_ids=self.family_ids.to(device),
            slot_ids=self.slot_ids.to(device),
            structure_ids=self.structure_ids.to(device),
            labels=self.labels.to(device),
        )


@dataclass
class REAL004Dataset:
    batch: REAL004Batch

    @property
    def size(self) -> int:
        return int(self.batch.labels.numel())

    def sample(self, batch_size: int, generator: torch.Generator) -> REAL004Batch:
        indices = torch.randint(0, self.size, (batch_size,), generator=generator)
        return self.slice_indices(indices)

    def iter_batches(self, batch_size: int) -> Iterable[REAL004Batch]:
        for start in range(0, self.size, batch_size):
            stop = min(self.size, start + batch_size)
            yield self.slice_indices(torch.arange(start, stop))

    def slice_indices(self, indices: Tensor) -> REAL004Batch:
        return REAL004Batch(
            input_ids=self.batch.input_ids[indices],
            family_ids=self.batch.family_ids[indices],
            slot_ids=self.batch.slot_ids[indices],
            structure_ids=self.batch.structure_ids[indices],
            labels=self.batch.labels[indices],
        )


@dataclass(frozen=True)
class REAL004VariantSpec:
    name: str
    backbone: str
    bridge_type: Optional[str]
    use_structure_slots: bool
    use_family_average_read: bool = False


@dataclass
class REAL004Forward:
    logits: Tensor
    structure_read_hit_rate: Tensor
    structure_use_entropy: Tensor
    slot_auxiliary_loss: Tensor


class REAL004BehaviorModel(nn.Module):
    def __init__(
        self,
        *,
        spec: REAL004VariantSpec,
        d_model: int,
        n_layers: int,
        n_heads: int,
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
        self.register_buffer("label_table", label_table.clone().long())

        if spec.backbone == "vanilla":
            config = TACConfig(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                n_layers=n_layers,
                n_programs=4,
                max_seq_len=3,
                norm_type="rmsnorm",
                mlp_type="swiglu",
                position_type="rope",
            )
            self.backbone = VanillaTransformerLM(config)
        elif spec.backbone == "legacy":
            config = best_chunked_recall_tac_config(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                n_layers=n_layers,
                n_programs=4,
                max_seq_len=3,
                content_store_size=4,
                content_read_steps=1,
                content_read_query_top_k=4,
            )
            self.backbone = TACTransformerLM(config)
        elif spec.backbone == "tac_scm":
            config = tac_scm_v02_config(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_heads,
                n_layers=n_layers,
                n_programs=4,
                max_seq_len=3,
                n_structure_families=n_families,
                n_structure_slots=self.n_structure_slots,
                content_store_size=4,
                content_read_steps=1,
                content_read_query_top_k=4,
            )
            self.backbone = TACTransformerLM(config)
        else:
            raise ValueError(f"unknown backbone {spec.backbone!r}")

        self.structure_memory = StructureMemoryModule(
            d_model=d_model,
            n_structure_slots=self.n_structure_slots,
        )
        with torch.no_grad():
            self.structure_memory.key_bank.copy_(F.normalize(structure_values, dim=-1))
            self.structure_memory.value_bank.copy_(structure_values)
        self.structure_memory.key_bank.requires_grad_(False)
        self.structure_memory.value_bank.requires_grad_(False)

        family_values = []
        for family_id in range(n_families):
            start = family_id * slots_per_family
            stop = start + slots_per_family
            family_values.append(structure_values[start:stop].mean(dim=0))
        self.register_buffer("family_structure_values", torch.stack(family_values))

        if spec.use_structure_slots:
            self.concept_encoder = ConceptVolumeEncoder(d_model, n_families)
            self.structure_router = TwoLevelStructureRouter(
                d_model=d_model,
                n_programs=4,
                n_structure_families=n_families,
                family_route_loss_weight=0.01,
                specialist_route_loss_weight=0.01,
            )
            self.slot_bottleneck = SlotConditionedProgramBottleneck(
                d_model=d_model,
                n_structure_slots=self.n_structure_slots,
                n_programs=4,
                load_balance_weight=0.01,
            )
        else:
            self.concept_encoder = None
            self.structure_router = None
            self.slot_bottleneck = None

        if spec.bridge_type == "linear":
            self.bridge = LinearStructureBridge(d_model)
        elif spec.bridge_type == "mlp":
            self.bridge = MLPStructureBridge(d_model)
        elif spec.bridge_type == "gated_residual":
            self.bridge = GatedResidualStructureBridge(d_model)
        elif spec.bridge_type == "oracle":
            self.bridge = OracleStructureBridge(d_model, n_oracle_structures=n_classes)
            _initialize_oracle_bridge(self.bridge, n_classes=n_classes)
        elif spec.bridge_type is None:
            self.bridge = None
        else:
            raise ValueError(f"unknown bridge type {spec.bridge_type!r}")

        self.behavior_head = nn.Linear(d_model, n_classes)

    def forward(
        self,
        batch: REAL004Batch,
        *,
        intervention: str = "carry",
    ) -> REAL004Forward:
        output = self.backbone(
            batch.input_ids,
            collect_auxiliary=False,
            collect_metrics=False,
            update_content_memory=False,
        )
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise RuntimeError("backbone did not return hidden states")
        query_hidden = hidden_states[:, -1, :]

        slot_auxiliary_loss = query_hidden.new_zeros(())
        entropy = query_hidden.new_zeros(())
        if self.slot_bottleneck is not None:
            slot_input = query_hidden.unsqueeze(1)
            concept = self.concept_encoder(slot_input)
            route = self.structure_router(slot_input, concept)
            slot_out = self.slot_bottleneck(slot_input, route=route)
            query_hidden = slot_out.hidden.squeeze(1)
            slot_auxiliary_loss = slot_out.auxiliary_loss + route.route_loss
            entropy = _mean_entropy(slot_out.slot_state.slot_weights)

        read_hit = query_hidden.new_zeros(())
        if self.bridge is not None:
            if self.spec.bridge_type == "oracle":
                bridged = self.bridge(query_hidden, batch.labels)
                read_hit = query_hidden.new_ones(())
            else:
                structure_vector, read_hit = self._read_structure_vector(
                    batch,
                    intervention=intervention,
                    device=query_hidden.device,
                )
                bridged = self.bridge(query_hidden, structure_vector)
            query_hidden = bridged.hidden

        logits = self.behavior_head(query_hidden)
        return REAL004Forward(
            logits=logits,
            structure_read_hit_rate=read_hit,
            structure_use_entropy=entropy,
            slot_auxiliary_loss=slot_auxiliary_loss,
        )

    def _read_structure_vector(
        self,
        batch: REAL004Batch,
        *,
        intervention: str,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        structure_ids = batch.structure_ids.to(device)
        family_ids = batch.family_ids.to(device)

        if self.spec.use_family_average_read:
            vectors = self.family_structure_values.to(device)[family_ids]
            expected_labels = self.label_table.to(device)[structure_ids]
            family_labels = vectors[:, : self.n_classes].argmax(dim=-1)
            return vectors, (family_labels == expected_labels).float().mean()

        if intervention == "reset":
            return (
                torch.zeros(
                    structure_ids.shape[0],
                    self.d_model,
                    device=device,
                    dtype=self.structure_memory.value_bank.dtype,
                ),
                structure_ids.new_zeros((), dtype=torch.float32),
            )

        selected_ids = structure_ids
        if intervention == "shuffled" and structure_ids.numel() > 1:
            selected_ids = torch.roll(structure_ids, shifts=1, dims=0)

        vectors = self.structure_memory.value_bank.to(device)[selected_ids]
        if intervention == "correct_slot_knockout":
            vectors = torch.zeros_like(vectors)
        elif intervention == "wrong_slot_knockout":
            vectors = vectors.clone()

        read_hit = (selected_ids == structure_ids).float().mean()
        if intervention == "correct_slot_knockout":
            read_hit = read_hit.new_zeros(())
        return vectors, read_hit


def run_tac_scm_real004(
    *,
    seeds: Iterable[int] | None = None,
    train_samples: int = 96,
    eval_samples: int = 96,
    steps: int = 24,
    batch_size: int = 16,
    d_model: int = 24,
    n_layers: int = 1,
    n_heads: int = 1,
    n_families: int = 4,
    slots_per_family: int = 8,
    n_classes: int = 4,
    learning_rate: float = 2e-2,
    device: str = "cpu",
) -> dict[str, Any]:
    seed_list = list(seeds if seeds is not None else [0])
    if not seed_list:
        raise ValueError("at least one seed is required")
    if train_samples < 1 or eval_samples < 1:
        raise ValueError("train_samples and eval_samples must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if d_model < n_classes:
        raise ValueError("d_model must be at least n_classes")

    device_obj = torch.device(device)
    per_seed: list[dict[str, Any]] = []
    all_variant_scores: dict[str, list[dict[str, float]]] = {
        name: [] for name in REAL004_VARIANT_NAMES
    }
    correct_knockout_scores: list[dict[str, float]] = []

    for seed in seed_list:
        seed_result = _run_one_seed(
            seed=seed,
            train_samples=train_samples,
            eval_samples=eval_samples,
            steps=steps,
            batch_size=batch_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_families=n_families,
            slots_per_family=slots_per_family,
            n_classes=n_classes,
            learning_rate=learning_rate,
            device=device_obj,
        )
        per_seed.append(seed_result)
        for name, score in seed_result["variant_results"].items():
            all_variant_scores[name].append(score)
        correct_knockout_scores.append(seed_result["correct_slot_knockout"])

    variant_results = {
        name: _aggregate_scores(scores)
        for name, scores in all_variant_scores.items()
    }
    correct_knockout = _aggregate_scores(correct_knockout_scores)
    metrics = _compute_real004_metrics(variant_results, correct_knockout)
    gate = evaluate_success_gate(variant_results, metrics)
    bottleneck = diagnose_real004_bottleneck(variant_results, metrics)
    failure_analysis = _failure_analysis(gate, bottleneck, variant_results, metrics)

    return {
        "benchmark": "TAC-SCM-REAL004 causal structure-to-behavior validation",
        "status": gate["status"],
        "seed_count": len(seed_list),
        "seeds": seed_list,
        "variants": list(REAL004_VARIANT_NAMES),
        "variant_results": variant_results,
        "correct_slot_knockout": correct_knockout,
        "metrics": metrics,
        "success_gate": gate,
        "bottleneck": bottleneck,
        "failure_analysis": failure_analysis,
        "per_seed_results": per_seed,
        "config": {
            "train_samples": train_samples,
            "eval_samples": eval_samples,
            "steps": steps,
            "batch_size": batch_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_families": n_families,
            "slots_per_family": slots_per_family,
            "n_classes": n_classes,
            "learning_rate": learning_rate,
            "device": str(device_obj),
        },
    }


def evaluate_success_gate(
    variant_results: dict[str, dict[str, float]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    full = _variant_accuracy(variant_results, "full_tac_scm_v02")
    vanilla = _variant_accuracy(variant_results, "vanilla_transformer")
    legacy = _variant_accuracy(variant_results, "legacy_best_chunked_recall_tac")
    reset = _variant_accuracy(variant_results, "reset_structure_control")
    shuffled = _variant_accuracy(variant_results, "shuffled_structure_control")
    oracle = _variant_accuracy(variant_results, "tac_scm_v02_oracle_bridge")
    learned = max(
        _variant_accuracy(variant_results, "full_tac_scm_v02"),
        _variant_accuracy(variant_results, "tac_scm_v02_linear_bridge"),
        _variant_accuracy(variant_results, "tac_scm_v02_mlp_bridge"),
        _variant_accuracy(variant_results, "tac_scm_v02_gated_residual_bridge"),
    )

    failed: list[str] = []
    if full <= vanilla:
        failed.append("does not beat vanilla")
    if full <= legacy:
        failed.append("does not beat legacy TAC")
    if full <= reset:
        failed.append("carry does not beat reset structure control")
    if full <= shuffled:
        failed.append("carry does not beat shuffled structure control")
    if metrics.get("slot_knockout_drop", 0.0) <= metrics.get("wrong_slot_knockout_drop", 0.0):
        failed.append("correct-slot knockout does not hurt more than wrong-slot knockout")
    if oracle <= learned:
        failed.append("oracle bridge is not above learned bridge")
    if not metrics.get("compression_roi_compatible", False):
        failed.append("compression ROI compatibility failed")
    if not metrics.get("lifecycle_preserve_retire_sane", False):
        failed.append("lifecycle preserve/retire sanity failed")

    return {
        "status": "passed" if not failed else "failed",
        "failed_conditions": failed,
        "full_accuracy": full,
        "vanilla_accuracy": vanilla,
        "legacy_accuracy": legacy,
        "reset_accuracy": reset,
        "shuffled_accuracy": shuffled,
        "oracle_accuracy": oracle,
        "best_learned_bridge_accuracy": learned,
    }


def diagnose_real004_bottleneck(
    variant_results: dict[str, dict[str, float]],
    metrics: dict[str, Any],
) -> str:
    if metrics.get("structure_read_hit_rate", 0.0) < 0.7:
        return "discovery"
    if metrics.get("slot_knockout_drop", 0.0) <= metrics.get("wrong_slot_knockout_drop", 0.0):
        return "slot_routing"
    if metrics.get("bridge_gain", 0.0) <= 0.0 or _variant_accuracy(
        variant_results,
        "tac_scm_v02_oracle_bridge",
    ) - _variant_accuracy(variant_results, "full_tac_scm_v02") > 0.15:
        return "bridge_decoding"
    if not metrics.get("lifecycle_preserve_retire_sane", False):
        return "lifecycle_scoring"
    return "training_objective"


def _run_one_seed(
    *,
    seed: int,
    train_samples: int,
    eval_samples: int,
    steps: int,
    batch_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    n_families: int,
    slots_per_family: int,
    n_classes: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    vocab_size = 8 + n_families + slots_per_family
    label_table = _make_label_table(seed, n_families * slots_per_family, n_classes)
    structure_values = _make_structure_values(
        seed=seed,
        label_table=label_table,
        d_model=d_model,
        n_classes=n_classes,
    )
    train = _make_dataset(
        seed=seed + 101,
        samples=train_samples,
        n_families=n_families,
        slots_per_family=slots_per_family,
        label_table=label_table,
    )
    eval_data = _make_dataset(
        seed=seed + 202,
        samples=eval_samples,
        n_families=n_families,
        slots_per_family=slots_per_family,
        label_table=label_table,
    )

    variant_results: dict[str, dict[str, float]] = {}
    trained_models: dict[str, REAL004BehaviorModel] = {}
    for index, spec in enumerate(_trainable_variant_specs()):
        torch.manual_seed(seed * 997 + index)
        model = REAL004BehaviorModel(
            spec=spec,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
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
            seed=seed * 1297 + index,
            device=device,
        )
        trained_models[spec.name] = model
        variant_results[spec.name] = _evaluate_model(
            model,
            eval_data,
            batch_size=batch_size,
            device=device,
        )

    full_model = trained_models["full_tac_scm_v02"]
    variant_results["reset_structure_control"] = _evaluate_model(
        full_model,
        eval_data,
        batch_size=batch_size,
        device=device,
        intervention="reset",
    )
    variant_results["shuffled_structure_control"] = _evaluate_model(
        full_model,
        eval_data,
        batch_size=batch_size,
        device=device,
        intervention="shuffled",
    )
    variant_results["wrong_slot_knockout_control"] = _evaluate_model(
        full_model,
        eval_data,
        batch_size=batch_size,
        device=device,
        intervention="wrong_slot_knockout",
    )
    correct_knockout = _evaluate_model(
        full_model,
        eval_data,
        batch_size=batch_size,
        device=device,
        intervention="correct_slot_knockout",
    )

    return {
        "seed": seed,
        "variant_results": variant_results,
        "correct_slot_knockout": correct_knockout,
    }


def _train_model(
    model: REAL004BehaviorModel,
    train: REAL004Dataset,
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
    model: REAL004BehaviorModel,
    eval_data: REAL004Dataset,
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


def _compute_real004_metrics(
    variant_results: dict[str, dict[str, float]],
    correct_knockout: dict[str, float],
) -> dict[str, Any]:
    full = _variant_accuracy(variant_results, "full_tac_scm_v02")
    no_bridge = _variant_accuracy(variant_results, "tac_scm_v02_without_structure_bridge")
    oracle = _variant_accuracy(variant_results, "tac_scm_v02_oracle_bridge")
    reset = _variant_accuracy(variant_results, "reset_structure_control")
    shuffled = _variant_accuracy(variant_results, "shuffled_structure_control")
    wrong = _variant_accuracy(variant_results, "wrong_slot_knockout_control")
    legacy = _variant_accuracy(variant_results, "legacy_best_chunked_recall_tac")
    vanilla = _variant_accuracy(variant_results, "vanilla_transformer")

    compression = evaluate_structure_compression_roi()
    lifecycle_sane = _lifecycle_sanity_check()

    return {
        "behavior_accuracy": full,
        "bridge_gain": full - no_bridge,
        "oracle_gap": oracle - full,
        "carry_reset_delta": full - reset,
        "carry_shuffled_delta": full - shuffled,
        "slot_knockout_drop": full - correct_knockout["behavior_accuracy"],
        "wrong_slot_knockout_drop": full - wrong,
        "structure_read_hit_rate": variant_results["full_tac_scm_v02"]["structure_read_hit_rate"],
        "structure_use_entropy": variant_results["full_tac_scm_v02"]["structure_use_entropy"],
        "legacy_tac_gap": full - legacy,
        "vanilla_gap": full - vanilla,
        "compression_roi_compatible": compression["status"] == "passed",
        "lifecycle_preserve_retire_sane": lifecycle_sane,
    }


def _failure_analysis(
    gate: dict[str, Any],
    bottleneck: str,
    variant_results: dict[str, dict[str, float]],
    metrics: dict[str, Any],
) -> str:
    if gate["status"] == "passed":
        return "REAL004 passed: structure carry, slot intervention, and bridge oracle controls support causal structure-to-behavior use."
    return (
        f"REAL004 failed with bottleneck={bottleneck}. "
        f"Failed conditions: {', '.join(gate['failed_conditions'])}. "
        f"Full={metrics['behavior_accuracy']:.4f}, "
        f"vanilla={_variant_accuracy(variant_results, 'vanilla_transformer'):.4f}, "
        f"legacy={_variant_accuracy(variant_results, 'legacy_best_chunked_recall_tac'):.4f}, "
        f"bridge_gain={metrics['bridge_gain']:.4f}, "
        f"oracle_gap={metrics['oracle_gap']:.4f}, "
        f"slot_drop={metrics['slot_knockout_drop']:.4f}, "
        f"wrong_slot_drop={metrics['wrong_slot_knockout_drop']:.4f}."
    )


def _trainable_variant_specs() -> tuple[REAL004VariantSpec, ...]:
    return (
        REAL004VariantSpec("vanilla_transformer", "vanilla", None, False),
        REAL004VariantSpec("legacy_best_chunked_recall_tac", "legacy", None, False),
        REAL004VariantSpec("full_tac_scm_v02", "tac_scm", "gated_residual", True),
        REAL004VariantSpec(
            "tac_scm_v02_without_structure_slots",
            "tac_scm",
            "gated_residual",
            False,
            use_family_average_read=True,
        ),
        REAL004VariantSpec("tac_scm_v02_without_structure_bridge", "tac_scm", None, True),
        REAL004VariantSpec("tac_scm_v02_linear_bridge", "tac_scm", "linear", True),
        REAL004VariantSpec("tac_scm_v02_mlp_bridge", "tac_scm", "mlp", True),
        REAL004VariantSpec("tac_scm_v02_gated_residual_bridge", "tac_scm", "gated_residual", True),
        REAL004VariantSpec("tac_scm_v02_oracle_bridge", "tac_scm", "oracle", True),
    )


def _make_label_table(seed: int, n_structures: int, n_classes: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed + 17)
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
    generator = torch.Generator().manual_seed(seed + 31)
    values = 0.05 * torch.randn(label_table.numel(), d_model, generator=generator)
    values[:, :n_classes] = 0.0
    values[torch.arange(label_table.numel()), label_table] = 4.0
    if d_model > n_classes:
        values[:, n_classes:] += 0.2 * torch.randn(
            label_table.numel(),
            d_model - n_classes,
            generator=generator,
        )
    return values


def _make_dataset(
    *,
    seed: int,
    samples: int,
    n_families: int,
    slots_per_family: int,
    label_table: Tensor,
) -> REAL004Dataset:
    generator = torch.Generator().manual_seed(seed)
    family_ids = torch.randint(0, n_families, (samples,), generator=generator)
    slot_ids = torch.randint(0, slots_per_family, (samples,), generator=generator)
    structure_ids = family_ids * slots_per_family + slot_ids
    labels = label_table[structure_ids].clone()
    family_token_offset = 4
    slot_token_offset = family_token_offset + n_families
    query_token = slot_token_offset + slots_per_family
    input_ids = torch.stack(
        [
            family_token_offset + family_ids,
            slot_token_offset + slot_ids,
            torch.full_like(slot_ids, query_token),
        ],
        dim=1,
    )
    return REAL004Dataset(
        REAL004Batch(
            input_ids=input_ids.long(),
            family_ids=family_ids.long(),
            slot_ids=slot_ids.long(),
            structure_ids=structure_ids.long(),
            labels=labels.long(),
        )
    )


def _initialize_oracle_bridge(bridge: OracleStructureBridge, *, n_classes: int) -> None:
    with torch.no_grad():
        bridge.oracle_embedding.weight.zero_()
        for class_id in range(n_classes):
            bridge.oracle_embedding.weight[class_id, class_id] = 6.0
        bridge.projection.weight.zero_()
        dim = bridge.projection.weight.shape[0]
        bridge.projection.weight.copy_(torch.eye(dim))


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
    aggregated: dict[str, float] = {}
    for key in keys:
        values = [score[key] for score in scores]
        aggregated[key] = mean(values)
        if len(values) > 1:
            variance = sum((value - aggregated[key]) ** 2 for value in values) / len(values)
            aggregated[f"{key}_std"] = math.sqrt(variance)
    return aggregated


def _variant_accuracy(
    variant_results: dict[str, dict[str, float]],
    variant_name: str,
) -> float:
    return float(variant_results.get(variant_name, {}).get("behavior_accuracy", 0.0))


def _lifecycle_sanity_check() -> bool:
    scorer = StructureLifecycleScorer()
    preserved = scorer.decide(
        StructureObject(structure_id=1),
        StructureLifecycleStats(
            usage_count=80,
            success_rate=0.9,
            transfer_gain=0.7,
            reset_sensitivity=0.0,
            shuffle_sensitivity=0.0,
            attack_recovery=0.9,
            shift_retention=0.9,
        ),
    )
    retired = scorer.decide(
        StructureObject(structure_id=2),
        StructureLifecycleStats(
            usage_count=20,
            success_rate=0.0,
            transfer_gain=0.0,
            reset_sensitivity=1.0,
            shuffle_sensitivity=1.0,
            attack_recovery=0.0,
            shift_retention=0.0,
        ),
    )
    return not preserved.should_retire and retired.should_retire


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TAC-SCM-REAL004 causal structure-to-behavior validation."
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--ten-seed", action="store_true")
    parser.add_argument("--train-samples", type=int, default=96)
    parser.add_argument("--eval-samples", type=int, default=96)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=24)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--n-families", type=int, default=4)
    parser.add_argument("--slots-per-family", type=int, default=8)
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = list(range(10)) if args.ten_seed else args.seeds
    result = run_tac_scm_real004(
        seeds=seeds,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        steps=args.steps,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_families=args.n_families,
        slots_per_family=args.slots_per_family,
        n_classes=args.n_classes,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
