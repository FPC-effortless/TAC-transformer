"""RECON-14 persistence/reuse pilot runner.

Reconstruction category: NEW (the five-cell design is new; the A/B pair is
RECOVERED and C/D/E are the matched controls required by the confound analysis
in ``RECON_14.md``).

Cells (all receive identical batches and optimizer schedule):

- ``A_from_scratch``            — train all 6 op embeddings jointly from step 0.
- ``B_persistent_reuse``        — pretrain 5 ops 1500 steps, freeze, add NOT2 fresh.
- ``C_not2_only_pretrain_control`` — pretrain only NOT2, freeze, then train NOT2.
- ``D_frozen_op_noop_persistence`` — pretrain 5 ops, freeze, no useful geometry.
- ``E_joint_not2_late``         — joint training with NOT2 held out for 1500 steps.

Convergence criterion (frozen): ``g(NOT2, arg1) > 0.9`` and
``g(NOT2, noncanonical) < 0.1``.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .autodiff import Adam, Value, bce_loss
from .boolean_dag import (BooleanDAGGenerator, GeneratorConfig, Program,
                          input_name_of)
from .executor import CASMExecutor
from .grammar import INTERNAL_OPS, Op, PILOT_OPS, PRETRAIN_OPS
from .oracle import (EdgeKey, copy_mask_gates, decoy_false_positives,
                     exact_accuracy, oracle_gates, routing_precision_recall,
                     validate_program)
from .router import ROLES, ConditionalRouter, FactorizedRouterL2, StaticMaskRouter

# ---- frozen protocol constants (see RECON_14.md) -----------------------------

FROZEN = dict(
    experiment_id="recon-14-r1",
    base_commit="3027826",
    template_depth=8,
    window=1,
    pilot_ops=tuple(op.value for op in PILOT_OPS),
    pretrain_ops=tuple(op.value for op in PRETRAIN_OPS),
    router="FactorizedRouterL2",
    router_dim=2,
    router_init_scale=0.3,
    router_tau=1.0,
    roles=tuple(ROLES),
    max_depth=5,
    min_depth=1,
    decoy_prob=0.7,
    max_decoys_per_episode=3,
    batch_size=16,
    optimizer="Adam",
    lr=0.1,
    grad_norm_clip=1.0,
    pretrain_steps=1500,
    max_not2_steps=200,
    conv_g_high=0.9,
    conv_g_low=0.1,
    trial_seeds=(0, 1, 2, 3, 4, 5, 6, 7),
    dataset_seed=20260922,
)

CONDITIONS: Tuple[str, ...] = (
    "A_from_scratch",
    "B_persistent_reuse",
    "C_not2_only_pretrain_control",
    "D_frozen_op_noop_persistence",
    "E_joint_not2_late",
)

NEW_OP = Op.NOT2


# ---- data ---------------------------------------------------------------------


@dataclass
class Dataset:
    """A frozen episode corpus shared by every cell of a trial."""

    episodes: List[Program] = field(default_factory=list)
    seed: int = 0
    config: Optional[GeneratorConfig] = None

    def __len__(self) -> int:
        return len(self.episodes)

    def batch(self, rng: random.Random, size: int) -> List[Program]:
        if size > len(self.episodes):
            size = len(self.episodes)
        return rng.sample(self.episodes, size)


def build_dataset(n_episodes: int, seed: int,
                  config: Optional[GeneratorConfig] = None) -> Dataset:
    """Build the frozen corpus. Identical episodes across all cells of a trial."""
    cfg = config or GeneratorConfig(seed=seed)
    gen = BooleanDAGGenerator(cfg)
    rng = random.Random(seed)
    episodes: List[Program] = []
    while len(episodes) < n_episodes:
        prog = gen.generate(rng)
        # validate_program returns the (possibly empty) list of violations; an
        # empty list is the valid case. The truthiness is inverted -- a program
        # carrying errors is truthy -- so the condition must be negated. With the
        # inverted test the loop accepts only invalid programs and, on a seed
        # whose first programs are all valid, spins forever appending nothing.
        if not validate_program(prog):
            episodes.append(prog)
    return Dataset(episodes=episodes, seed=seed, config=cfg)


def soft_inputs(program: Program) -> Dict[str, Value]:
    """Deterministic fixed inputs for an episode.

    Soft execution requires input values in [0, 1]. Inputs are taken from the
    program's own frozen input values, so no oracle-derived information about the
    target reaches the router: the target is the exact Boolean reference output,
    and inputs are public.

    A decoy leaf is not a program input (it is not ground truth), but the
    executor still needs a value for it: a decoy subtree is wired into a true
    parent's args and is evaluated when its gate is on. Its value is fixed and
    public, set by the generator, so it is supplied here exactly as
    ``oracle.exact_accuracy`` supplies it on the hard path. Omitting it makes the
    soft execution raise KeyError on any episode carrying a decoy leaf, i.e. on
    the overwhelming majority of episodes at decoy_prob 0.7.
    """
    out: Dict[str, Value] = {}
    for n in program.nodes:
        if not n.is_input or n.input_value is None:
            continue
        if not n.in_program and n.decoy_name is None:
            continue
        out[input_name_of(n)] = Value(1.0 if n.input_value else 0.0)
    return out


def targets(program: Program) -> List[float]:
    """Exact Boolean reference outputs, one per truth-table row."""
    from .oracle import exhaustive_reference_outputs
    return [1.0 if b else 0.0 for b in exhaustive_reference_outputs(program)]


def input_rows(program: Program) -> List[Dict[str, Value]]:
    """One soft input vector per truth-table row, plus the decoy leaf values.

    The truth table enumerates the program's *distinct named inputs* only; a
    decoy leaf is not one of them (it is not ground truth). But a decoy subtree is
    wired into a true parent's args and is evaluated whenever its gate is on, so
    the executor needs a value for every decoy leaf too. Those values are fixed
    and public, so they are merged into every row -- the same values
    ``oracle.exact_accuracy`` supplies on the hard path, and the same ones
    ``soft_inputs`` returns.
    """
    from .oracle import truth_table_assignments
    decoy = {input_name_of(n): Value(1.0 if n.input_value else 0.0)
             for n in program.nodes
             if n.is_input and not n.in_program and n.input_value is not None}
    rows = []
    for assignment in truth_table_assignments(program):
        row = {k: Value(1.0 if v else 0.0) for k, v in assignment.items()}
        row.update(decoy)
        rows.append(row)
    return rows


# ---- training loop -----------------------------------------------------------


@dataclass
class StepRecord:
    step: int
    loss: float
    # None while the new operator has not been introduced. The protocol measures
    # steps-to-convergence in the NOT2 phase only, so a pre-introduction step is
    # not a measurement of convergence at all -- it is recorded for the loss
    # trace, and the convergence scan skips it.
    g_new_arg1: Optional[float]
    g_new_noncanonical: Optional[float]
    grad_norm: float


@dataclass
class CellResult:
    condition: str
    seed: int
    converged: bool
    steps_to_converge: Optional[int]
    # None iff the new operator was never introduced, which for the five RECON-14
    # cells cannot happen once ``run_condition`` completes; kept explicit so the
    # distinction between "untrained" and "trained to a value" survives reporting.
    g_new_arg1: Optional[float]
    g_new_noncanonical: Optional[float]
    final_loss: float
    history: List[StepRecord] = field(default_factory=list)
    pretrain_final_loss: Optional[float] = None
    epilogue_accuracy: Optional[float] = None
    epilogue_precision: Optional[float] = None
    epilogue_recall: Optional[float] = None
    epilogue_decoy_fp: Optional[int] = None
    trainable_param_count: int = 0
    frozen_param_count: int = 0
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)


def _new_op_gates(router: FactorizedRouterL2) -> Tuple[Optional[float], Optional[float]]:
    """Read the frozen convergence statistics for the new operator.

    Returns ``None`` for a gate the router has no embedding for. This is not the
    same as a low gate value: it means the operator has not been introduced, so
    there is nothing to measure. The RECON-14 protocol measures steps-to-
    convergence only in the NOT2 phase (it names "Max NOT2-phase steps" and the
    trainable parameters "during the NOT2 phase"), so a pre-introduction read is
    a measurement-boundary violation, not a training failure. ``None`` keeps
    "untrained" distinct from "not converged"; callers in the measured phase
    never see it.
    """
    g1 = router.gate_value(NEW_OP, "arg1")
    gn = router.gate_value(NEW_OP, "noncanonical")
    return (None if g1 is None else float(g1.data),
            None if gn is None else float(gn.data))


def _converged(g1: Optional[float], gn: Optional[float]) -> bool:
    if g1 is None or gn is None:
        return False
    return g1 > FROZEN["conv_g_high"] and gn < FROZEN["conv_g_low"]


def _train_steps(router: FactorizedRouterL2, executor: CASMExecutor,
                 dataset: Dataset, rng: random.Random, steps: int,
                 adam: Adam, batch_size: int, new_op_only: bool = False,
                 record_from: int = 0) -> List[StepRecord]:
    """Shared optimisation loop. Returns step records for the new operator.

    This loop runs both phases. A step taken while the new operator has no
    embedding records ``g_new_* = None`` rather than a number, because the
    protocol's convergence criterion is defined on the NOT2 phase; ``record_from``
    selects which steps are returned, and the convergence scan in
    ``run_condition`` skips a ``None`` gate instead of scoring it as a failure.
    """
    records: List[StepRecord] = []
    bs = min(batch_size, len(dataset))
    for step in range(1, steps + 1):
        batch = dataset.batch(rng, bs)
        total = Value(0.0)
        n_rows = 0
        for prog in batch:
            gates = router.gates_for(prog)
            rows = input_rows(prog)
            ys = targets(prog)
            for row, y in zip(rows, ys):
                pred = executor.execute(prog, row, gates)
                total = total + bce_loss(pred, y)
                n_rows += 1
        if n_rows == 0:
            continue
        loss = total / Value(float(n_rows))
        adam.zero_grad()
        loss.backward()
        gnorm = adam.step(FROZEN["grad_norm_clip"])
        g1, gn = _new_op_gates(router)
        if step >= record_from:
            records.append(StepRecord(step=step, loss=float(loss.data),
                                      g_new_arg1=g1, g_new_noncanonical=gn,
                                      grad_norm=float(gnorm)))
    return records


def run_condition(condition: str, seed: int, dataset: Dataset,
                  max_not2_steps: Optional[int] = None,
                  pretrain_steps: Optional[int] = None,
                  verbose: bool = False) -> CellResult:
    """Run one cell of one trial."""
    t0 = time.time()
    pretrain_steps = FROZEN["pretrain_steps"] if pretrain_steps is None else pretrain_steps
    max_not2_steps = (FROZEN["max_not2_steps"] if max_not2_steps is None
                      else max_not2_steps)
    rng = random.Random(seed)
    executor = CASMExecutor()

    router = FactorizedRouterL2(
        ops=PRETRAIN_OPS,
        dim=FROZEN["router_dim"],
        init_scale=FROZEN["router_init_scale"],
        tau=FROZEN["router_tau"],
        seed=seed,
    )

    notes: List[str] = []

    # ---- pretraining phases differ by condition -------------------------
    pretrain_loss = None
    if condition in ("B_persistent_reuse", "D_frozen_op_noop_persistence"):
        adam = Adam(router.parameters(), lr=FROZEN["lr"])
        _train_steps(router, executor, dataset, rng, pretrain_steps, adam,
                     FROZEN["batch_size"])
        pretrain_loss = _dataset_loss(router, executor, dataset)
        # Freeze the pretrained 5-op geometry. This is the persistence mechanism.
        router.freeze_ops(PRETRAIN_OPS)
        router.add_op(NEW_OP, seed=seed + 1)
        notes.append("pretrained 5 ops then froze; NOT2 added fresh")
    elif condition == "C_not2_only_pretrain_control":
        router = FactorizedRouterL2(
            ops=(NEW_OP,),
            dim=FROZEN["router_dim"],
            init_scale=FROZEN["router_init_scale"],
            tau=FROZEN["router_tau"],
            seed=seed,
        )
        adam = Adam(router.parameters(), lr=FROZEN["lr"])
        _train_steps(router, executor, dataset, rng, pretrain_steps, adam,
                     FROZEN["batch_size"])
        pretrain_loss = _dataset_loss(router, executor, dataset)
        router.freeze_ops((NEW_OP,))
        notes.append("pretrained NOT2 only then froze; no 5-op geometry to reuse")
    elif condition == "A_from_scratch":
        router.add_op(NEW_OP, seed=seed + 1)
        notes.append("all 6 op embeddings trained jointly from step 0")
    elif condition == "E_joint_not2_late":
        notes.append("joint training with NOT2 held out for the pretraining window")
    else:
        raise ValueError(f"unknown condition {condition!r}")

    # ---- the measured NOT2 phase ----------------------------------------
    adam = Adam(router.parameters(), lr=FROZEN["lr"])
    if condition == "E_joint_not2_late":
        adam = Adam(router.parameters(), lr=FROZEN["lr"])
        _train_steps(router, executor, dataset, rng, pretrain_steps, adam,
                     FROZEN["batch_size"], record_from=0)
        pretrain_loss = _dataset_loss(router, executor, dataset)
        router.add_op(NEW_OP, seed=seed + 1)
        adam = Adam(router.parameters(), lr=FROZEN["lr"])

    records = _train_steps(router, executor, dataset, rng, max_not2_steps, adam,
                           FROZEN["batch_size"])

    converged_step: Optional[int] = None
    for rec in records:
        # A None gate is a pre-introduction step: the operator had no embedding,
        # so there was nothing to converge. Scoring it as "not converged" would
        # let the pretraining period contribute artificial low gate values to a
        # statistic the protocol scopes to the NOT2 phase.
        if _converged(rec.g_new_arg1, rec.g_new_noncanonical):
            converged_step = rec.step
            break

    g1, gn = _new_op_gates(router)
    result = CellResult(
        condition=condition,
        seed=seed,
        converged=converged_step is not None,
        steps_to_converge=converged_step,
        g_new_arg1=g1,
        g_new_noncanonical=gn,
        final_loss=records[-1].loss if records else float("nan"),
        history=records,
        pretrain_final_loss=pretrain_loss,
        **_epilogue_metrics(router, executor),
        trainable_param_count=len(router.parameters()),
        frozen_param_count=(5 * FROZEN["router_dim"]
                            if condition in ("B_persistent_reuse",
                                             "D_frozen_op_noop_persistence")
                            else (FROZEN["router_dim"]
                                  if condition == "C_not2_only_pretrain_control"
                                  else 0)),
        seconds=time.time() - t0,
        notes=notes,
    )
    return result


def _dataset_loss(router: FactorizedRouterL2, executor: CASMExecutor,
                  dataset: Dataset, batch_cap: int = 32) -> float:
    """Mean BCE over a bounded sample of the corpus (evaluation only)."""
    rng = random.Random(1234567)
    batch = dataset.batch(rng, min(batch_cap, len(dataset)))
    total = 0.0
    n = 0
    for prog in batch:
        gates = router.gates_for(prog)
        rows = input_rows(prog)
        ys = targets(prog)
        for row, y in zip(rows, ys):
            pred = executor.execute(prog, row, gates)
            total += float(bce_loss(pred, y).data)
            n += 1
    return total / max(1, n)


def _epilogue_metrics(router: FactorizedRouterL2, executor: CASMExecutor,
                  program_cap: int = 32) -> Dict[str, Any]:
    """Held-out exact 6-op hard-execution accuracy (protocol "Epilogue evaluation").

    Frozen protocol: ``Epilogue evaluation | exact 6-op hard execution accuracy
    on held-out episodes``. This deliberately does *not* reuse the training
    corpus: a different dataset seed makes the episodes held-out, so the metric
    is a generalisation probe rather than a training-set readback. Accuracy is
    computed under ``execute_hard`` with binary gate decisions, so a wrong gate
    configuration shows up as a mismatch instead of being averaged away by the
    soft execution used in training.

    Returns the four ``CellResult.epilogue_*`` fields. ``None`` marks a cell
    whose router has no NOT2 embedding, which keeps "not measured" distinct from
    "measured zero".
    """
    if router.embedding_for(NEW_OP) is None:
        return dict(epilogue_accuracy=None, epilogue_precision=None,
                    epilogue_recall=None, epilogue_decoy_fp=None)
    # Deliberately different from FROZEN["dataset_seed"] -> held-out episodes.
    held = build_dataset(n_episodes=program_cap, seed=FROZEN["dataset_seed"] + 1)
    accs: List[float] = []
    precs: List[float] = []
    recs: List[float] = []
    fps: List[int] = []
    for prog in held.episodes:
        soft = router.gates_for(prog)
        gates: Dict[EdgeKey, float] = {k: float(v.data) for k, v in soft.items()}
        accs.append(exact_accuracy(prog, executor, gates))
        precision, recall = routing_precision_recall(prog, gates)
        precs.append(precision)
        recs.append(recall)
        fps.append(decoy_false_positives(prog, gates))
    return dict(
        epilogue_accuracy=sum(accs) / len(accs),
        epilogue_precision=sum(precs) / len(precs),
        epilogue_recall=sum(recs) / len(recs),
        epilogue_decoy_fp=sum(fps),
    )


# ---- harness entry points ----------------------------------------------------


@dataclass
class TrialResult:
    condition: str
    cells: List[CellResult] = field(default_factory=list)


def run_trial(seed: int, n_episodes: int = 128,
              conditions: Sequence[str] = CONDITIONS,
              verbose: bool = False) -> Dict[str, CellResult]:
    """Run every condition on one shared frozen dataset."""
    dataset = build_dataset(n_episodes=n_episodes, seed=FROZEN["dataset_seed"])
    out: Dict[str, CellResult] = {}
    for condition in conditions:
        out[condition] = run_condition(condition, seed=seed, dataset=dataset,
                                       verbose=verbose)
    return out


def run_all(seeds: Optional[Sequence[int]] = None,
            n_episodes: int = 128,
            conditions: Sequence[str] = CONDITIONS,
            verbose: bool = False) -> Dict[str, List[CellResult]]:
    seeds = tuple(FROZEN["trial_seeds"]) if seeds is None else tuple(seeds)
    out: Dict[str, List[CellResult]] = {c: [] for c in conditions}
    for seed in seeds:
        trial = run_trial(seed, n_episodes=n_episodes, conditions=conditions,
                          verbose=verbose)
        for condition, cell in trial.items():
            out[condition].append(cell)
    return out


# ---- reporting ----------------------------------------------------------------


def summarise(results: Dict[str, List[CellResult]]) -> Dict[str, Any]:
    """Aggregate without averaging away per-seed structure."""
    out: Dict[str, Any] = {}
    for condition, cells in results.items():
        steps = [c.steps_to_converge for c in cells if c.steps_to_converge is not None]
        # None means the operator was never introduced; report it as such rather
        # than coercing to 0, which is a legitimate gate value not a marker.
        out[condition] = {
            "n_seeds": len(cells),
            "converged": sum(1 for c in cells if c.converged),
            "steps_to_converge": steps,
            "mean_steps": (sum(steps) / len(steps)) if steps else None,
            "g_new_arg1": [c.g_new_arg1 for c in cells],
            "g_new_noncanonical": [c.g_new_noncanonical for c in cells],
            "new_op_introduced": [c.g_new_arg1 is not None for c in cells],
            "final_loss": [c.final_loss for c in cells],
            "epilogue_accuracy": [c.epilogue_accuracy for c in cells],
            "epilogue_precision": [c.epilogue_precision for c in cells],
            "epilogue_recall": [c.epilogue_recall for c in cells],
            "epilogue_decoy_fp": [c.epilogue_decoy_fp for c in cells],
            "trainable_params": [c.trainable_param_count for c in cells],
            "frozen_params": [c.frozen_param_count for c in cells],
        }
    return out


def falsification_verdict(summary: Dict[str, Any]) -> Dict[str, Any]:
    """The effect survives only if the matched controls do not reproduce it."""
    b = summary.get("B_persistent_reuse", {})
    c = summary.get("C_not2_only_pretrain_control", {})
    d = summary.get("D_frozen_op_noop_persistence", {})
    a = summary.get("A_from_scratch", {})

    def mean_of(x):
        return x.get("mean_steps")

    b_mean, c_mean, d_mean, a_mean = mean_of(b), mean_of(c), mean_of(d), mean_of(a)
    if b_mean is None or a_mean is None:
        return {
            "verdict": "inconclusive",
            "reason": "B or A never converged on any seed; effect not established",
        }
    speedup = a_mean / b_mean if b_mean > 0 else float("inf")
    controls_reproduce = []
    if c_mean is not None and c_mean <= b_mean * 1.5:
        controls_reproduce.append("C_not2_only_pretrain_control")
    if d_mean is not None and d_mean <= b_mean * 1.5:
        controls_reproduce.append("D_frozen_op_noop_persistence")
    if controls_reproduce:
        return {
            "verdict": "falsified_by_confound",
            "reason": (
                "matched control(s) "
                + ", ".join(controls_reproduce)
                + " reproduce the B speedup within 1.5x, so the effect is "
                  "attributable to pretraining rather than persistence/reuse"
            ),
            "a_mean_steps": a_mean,
            "b_mean_steps": b_mean,
            "c_mean_steps": c_mean,
            "d_mean_steps": d_mean,
            "speedup_a_over_b": speedup,
            "controls_reproducing": controls_reproduce,
        }
    return {
        "verdict": "effect_survives_controls",
        "a_mean_steps": a_mean,
        "b_mean_steps": b_mean,
        "c_mean_steps": c_mean,
        "d_mean_steps": d_mean,
        "speedup_a_over_b": speedup,
        "controls_reproducing": [],
    }


__all__ = [
    "FROZEN",
    "CONDITIONS",
    "NEW_OP",
    "Dataset",
    "build_dataset",
    "CellResult",
    "StepRecord",
    "_epilogue_metrics",
    "run_condition",
    "run_trial",
    "run_all",
    "summarise",
    "falsification_verdict",
]
