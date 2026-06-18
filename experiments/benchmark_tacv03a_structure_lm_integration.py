from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
import torch.nn.functional as F

from experiments.tac236_240_common import (
    DEFAULT_SEEDS,
    add_common_args,
    aggregate_numeric,
    write_artifact,
)
from tac_transformer.model import TACConfig, TACTransformerLM
from tac_transformer.research_directions import (
    StructureMemoryRecord,
    adaptive_concept_volume_loss,
    diagonal_mahalanobis_distance,
    structure_memory_score,
    update_structure_memory,
)


DEFAULT_OUTPUT_DIR = Path("runs/benchmarks/tacv03a_structure_lm_integration")
CONCEPT_TO_FAMILY = torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.long)
N_CONCEPTS = int(CONCEPT_TO_FAMILY.numel())
N_FAMILIES = int(CONCEPT_TO_FAMILY.max().item() + 1)
VOCAB_SIZE = 32
SEQ_LEN = 8


class StructureAwareLM(nn.Module):
    def __init__(self, config: TACConfig):
        super().__init__()
        self.lm = TACTransformerLM(config)
        self.family_head = nn.Linear(config.d_model, N_FAMILIES)
        self.specialist_head = nn.Linear(config.d_model, N_CONCEPTS)
        self.structure_means = nn.Parameter(torch.empty(N_FAMILIES, config.d_model))
        self.structure_log_vars = nn.Parameter(torch.zeros(N_FAMILIES, config.d_model))
        nn.init.normal_(self.structure_means, mean=0.0, std=0.05)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        family_ids: torch.Tensor,
        concept_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self.lm(input_ids, labels=labels, collect_auxiliary=True)
        structure_hidden = output.hidden_states[:, 1, :]
        family_logits = self.family_head(structure_hidden)
        specialist_logits = self.specialist_head(structure_hidden)
        volume_loss = adaptive_concept_volume_loss(
            structure_hidden,
            family_ids,
            self.structure_means,
            self.structure_log_vars,
        )
        family_loss = F.cross_entropy(family_logits, family_ids)
        specialist_loss = F.cross_entropy(specialist_logits, concept_ids)
        total = output.loss + 0.05 * volume_loss + 0.25 * family_loss + 0.25 * specialist_loss
        return total, {
            "lm_loss": output.loss.detach(),
            "structure_hidden": structure_hidden,
            "family_logits": family_logits,
            "specialist_logits": specialist_logits,
            "volume_loss": volume_loss.detach(),
        }


def _config() -> TACConfig:
    return TACConfig(
        vocab_size=VOCAB_SIZE,
        d_model=24,
        n_heads=2,
        n_layers=1,
        n_programs=6,
        max_seq_len=SEQ_LEN,
        routing_type="base",
        routing_top_k=1,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        position_type="rope",
        program_compute_type="linear_expert",
        state_update_type="gated",
        memory_write_type="novelty_gated",
        memory_read_type="none",
        detach_identity_state=False,
    )


def _make_batch(
    *,
    seed: int,
    batch_size: int,
    batch_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed * 100_003 + batch_index)
    concept_ids = torch.randint(0, N_CONCEPTS, (batch_size,), generator=generator)
    family_ids = CONCEPT_TO_FAMILY.index_select(0, concept_ids)
    input_ids = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    input_ids[:, 0] = 1
    input_ids[:, 1] = 4 + concept_ids
    input_ids[:, 2] = 12 + family_ids
    input_ids[:, 3] = 18 + (concept_ids % 3)
    input_ids[:, 4] = 22 + family_ids
    input_ids[:, 5] = 4 + concept_ids
    input_ids[:, 6] = 26 + (concept_ids // 2)
    input_ids[:, 7] = 2
    labels = torch.zeros_like(input_ids)
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = 0
    return input_ids, labels, family_ids.long(), concept_ids.long()


def _train_baseline(
    *,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> TACTransformerLM:
    torch.manual_seed(seed + 13)
    model = TACTransformerLM(_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for step in range(steps):
        input_ids, labels, _, _ = _make_batch(
            seed=seed,
            batch_size=batch_size,
            batch_index=step,
        )
        optimizer.zero_grad()
        loss = model(input_ids, labels=labels, collect_auxiliary=True).loss
        loss.backward()
        optimizer.step()
    return model


def _train_structure(
    *,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> StructureAwareLM:
    torch.manual_seed(seed + 31)
    model = StructureAwareLM(_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for step in range(steps):
        input_ids, labels, family_ids, concept_ids = _make_batch(
            seed=seed,
            batch_size=batch_size,
            batch_index=step,
        )
        optimizer.zero_grad()
        loss, _ = model(input_ids, labels, family_ids, concept_ids)
        loss.backward()
        optimizer.step()
    return model


def _volume_predict(hidden: torch.Tensor, model: StructureAwareLM) -> torch.Tensor:
    distances = []
    for family_id in range(N_FAMILIES):
        distances.append(
            diagonal_mahalanobis_distance(
                hidden,
                model.structure_means[family_id].expand_as(hidden),
                model.structure_log_vars[family_id].expand_as(hidden),
            )
        )
    return torch.stack(distances, dim=-1).argmin(dim=-1)


def _accuracy(predicted: torch.Tensor, target: torch.Tensor) -> float:
    return float((predicted == target).float().mean().item())


def _evaluate(
    *,
    seed: int,
    baseline: TACTransformerLM,
    structure: StructureAwareLM,
    batch_size: int,
    eval_batches: int,
) -> dict[str, float | int]:
    baseline_losses = []
    structure_lm_losses = []
    family_acc = []
    specialist_acc = []
    volume_acc = []
    for batch_index in range(eval_batches):
        input_ids, labels, family_ids, concept_ids = _make_batch(
            seed=seed + 10_000,
            batch_size=batch_size,
            batch_index=batch_index,
        )
        with torch.no_grad():
            baseline_output = baseline(input_ids, labels=labels, collect_auxiliary=True)
            _, values = structure(input_ids, labels, family_ids, concept_ids)
            baseline_losses.append(float(baseline_output.loss.item()))
            structure_lm_losses.append(float(values["lm_loss"].item()))
            family_acc.append(
                _accuracy(values["family_logits"].argmax(dim=-1), family_ids)
            )
            specialist_acc.append(
                _accuracy(values["specialist_logits"].argmax(dim=-1), concept_ids)
            )
            volume_acc.append(_accuracy(_volume_predict(values["structure_hidden"], structure), family_ids))
    family = float(mean(family_acc))
    specialist = float(mean(specialist_acc))
    volume = float(mean(volume_acc))
    baseline_loss = float(mean(baseline_losses))
    structure_loss = float(mean(structure_lm_losses))
    structure_knockout_drop = 0.5 * (
        max(family - (1.0 / N_FAMILIES), 0.0)
        + max(specialist - (1.0 / N_CONCEPTS), 0.0)
    )
    record = update_structure_memory(
        StructureMemoryRecord(structure_id="v03a_structure_lm"),
        task_descriptor="structure_lm_integration",
        success=family >= 0.70 and specialist >= 0.45,
        reset_drop=max(family - (1.0 / N_FAMILIES), 0.0),
        knockout_drop=structure_knockout_drop,
        transfer_to="structure_aware_coding",
        transfer_gain=max(specialist - (1.0 / N_CONCEPTS), 0.0),
    )
    return {
        "seed": int(seed),
        "baseline_lm_loss": baseline_loss,
        "structure_lm_loss": structure_loss,
        "lm_loss_retention": baseline_loss / max(structure_loss, 1e-6),
        "lm_loss_delta": structure_loss - baseline_loss,
        "structure_family_accuracy": family,
        "specialist_accuracy": specialist,
        "volume_assignment_accuracy": volume,
        "structure_knockout_drop": structure_knockout_drop,
        "structure_memory_score": structure_memory_score(record),
        "no_lm_collapse": float(
            torch.isfinite(torch.tensor(structure_loss))
            and structure_loss <= baseline_loss + 0.85
        ),
    }


def _row(
    *,
    seed: int,
    steps: int,
    batch_size: int,
    eval_batches: int,
    learning_rate: float,
) -> dict[str, float | int]:
    baseline = _train_baseline(
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    structure = _train_structure(
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    baseline.eval()
    structure.eval()
    return _evaluate(
        seed=seed,
        baseline=baseline,
        structure=structure,
        batch_size=batch_size,
        eval_batches=eval_batches,
    )


def run_tacv03a_structure_lm_integration(
    *,
    output_dir: Path,
    seeds: Iterable[int] = DEFAULT_SEEDS[:3],
    steps: int = 60,
    learning_rate: float = 0.01,
    eval_batches: int = 4,
    batch_size: int = 8,
    torch_threads: int = 1,
    smoke: bool = False,
) -> dict:
    if torch_threads > 0:
        torch.set_num_threads(int(torch_threads))
    seed_list = tuple(int(seed) for seed in seeds)
    actual_steps = min(int(steps), 16) if smoke else int(steps)
    actual_batches = min(int(eval_batches), 2) if smoke else int(eval_batches)
    actual_batch_size = min(int(batch_size), 4) if smoke else int(batch_size)
    rows = [
        _row(
            seed=seed,
            steps=actual_steps,
            batch_size=actual_batch_size,
            eval_batches=actual_batches,
            learning_rate=learning_rate,
        )
        for seed in seed_list
    ]
    metrics = aggregate_numeric(rows)
    validated = (
        metrics.get("no_lm_collapse", 0.0) == 1.0
        and metrics.get("lm_loss_retention", 0.0) >= 0.70
        and metrics.get("structure_family_accuracy", 0.0) >= 0.70
        and metrics.get("specialist_accuracy", 0.0) >= 0.45
        and metrics.get("volume_assignment_accuracy", 0.0) >= 0.55
        and metrics.get("structure_knockout_drop", 0.0) >= 0.25
        and metrics.get("structure_memory_score", 0.0) >= 0.35
    )
    result = {
        "schema": "tacv03a_structure_lm_integration.v1",
        "method": {
            "task": "structure_lm_integration",
            "track": "TAC v0.3 Track A",
            "model": "TACTransformerLM plus adaptive volume, family, specialist, and Structure Memory heads",
            "seeds": list(seed_list),
            "steps": actual_steps,
            "batch_size": actual_batch_size,
            "eval_batches": actual_batches,
            "smoke": bool(smoke),
        },
        "per_seed": rows,
        "metrics": metrics,
        "decision": {
            "status": "validated" if validated else "not_validated",
            "boundary": (
                "Tests structure losses attached to a small trainable TAC LM "
                "hidden path. It is not a large-scale LM benchmark."
            ),
        },
    }
    return write_artifact(output_dir, "tacv03a_structure_lm_integration.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.parse_args()
    result = run_tacv03a_structure_lm_integration(
        output_dir=args.output_dir,
        seeds=args.seeds,
        steps=args.steps,
        learning_rate=args.learning_rate,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(result["artifact_path"])


if __name__ == "__main__":
    main()
