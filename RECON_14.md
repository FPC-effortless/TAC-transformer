# RECON-14 — Persistence/Reuse Pilot Reconstruction

**Status:** PREREGISTERED. This document is written and frozen **before** any
implementation or training. No hyperparameter, seed, dataset size, threshold, or
condition definition in the frozen protocol table below is changed after results
are observed.

## Purpose

Independently reconstruct the historical CASM `phase1` "Experiment 14 —
Persistence & Reuse Pilot" (`experiments/exp_14_persistence_reuse_pilot.py`) as a
standalone RECON-14 experiment, so that the reported historical effect can be
either reproduced or falsified under an audited protocol.

The scientific question is exactly:

> Does freezing previously-learned operator embeddings — and routing a new
> operator through the resulting persistent geometry — causally reduce the number
> of optimization steps required to place a new operator's edges correctly,
> relative to training all operators jointly from scratch?

## Provenance categories (do not blur these)

This workspace adopts the same three categories as the established
`casm_v01/phase15a/MANIFEST.md` convention, because the historical situation is
structurally identical: the original artefacts were observed on a machine that is
no longer accessible.

### RECOVERED — directly present in the surviving material

From the historical experiment description supplied to this session (the only
surviving artefact):

- The effect under test: Condition A convergence steps `[35, 35, 41, 31]`
  (mean ~35.5); Condition B convergence steps `[1, 1, 1, 1]` (mean 1.0).
- Condition definitions: **A** trains all 6 op embeddings plus role embeddings
  jointly from random init at step 0; **B** pretrains the original 5 ops for
  1500 steps to convergence, freezes those parameters, adds `NOT2` with a fresh
  embedding, and trains `NOT2` in isolation.
- Pilot configuration: router is the L2-parameterized factorized router, `d=2`,
  initial scale `0.3`; `PILOT_OPS = (AND, OR, XOR, NOT, IDENTITY, NOT2)`;
  `MAX_DEPTH = 5`, `min_depth = 1`, `decoy_prob = 0.7`,
  `max_decoys_per_episode = 3`, `batch_size = 16`, `lr = 0.1`.
- Convergence criterion: `g(NOT2, arg1) > 0.9` AND `g(NOT2, noncanonical) < 0.1`.
- Behavioural identity of the new operator: `NOT2(x) = not x`, arity 1,
  behaviourally identical to `NOT`, and excluded from the default internal op set
  of the 5-op grammar (so its edges are not pre-populated by construction).
- Grammar: `ARITY = {AND:2, OR:2, XOR:2, NOT:1, IDENTITY:1, INPUT:0, NOT2:1}`;
  `INTERNAL_OPS = (AND, OR, XOR, NOT, IDENTITY)`; commutative ops are
  `{AND, OR, XOR}`.
- Benchmark ambiguity engineering: window decoys at non-canonical admissible
  slots, plus arity decoys attached to unused canonical slots of arity-1 ops.
  For the same physical slot the edge is valid for parent `AND/OR/XOR` and a
  decoy for parent `NOT/IDENTITY/NOT2`.
- Roles used by the router: `{arg1, arg2, noncanonical}` — a window-decoy edge
  takes the `noncanonical` role regardless of port; canonical edges take their
  port role. The router sees only public template information.
- L2 gate: `g = sigmoid((b - ||e_op - e_role||^2) / tau)`, with `tau = 1`.
- Autodiff: a hand-written reverse-mode autodiff engine with `Value`, `Adam`,
  and `bce_loss`, verified against finite differences. PyTorch is not used for
  the study router or the executor.

### CANONICAL SPECIFICATION — defined by the established benchmark description

Implemented exactly as specified, unchanged from the frozen CASM Phase-1
benchmark:

- Fixed binary-tree `SupersetTemplate`, slots addressed by `(level, index)`,
  level 0 = root, each occupied parent has exactly 2 canonical child positions.
- Admissible children = canonical children plus same-level neighbouring positions
  with `window = 1`.
- Template spans depth 1-8; the pilot samples active programs with
  `min_depth = 1` and `MAX_DEPTH = 5`.
- Execution is single-pass and bottom-up, from the deepest level to the root,
  over the topologically sorted DAG.
- Structural signature canonicalization for deduplication, scoped to
  deduplication only, never collapsing raw operand order.
- Oracle-derived values are used only for targets and evaluation, never as a
  router input.

### UNRECOVERABLE HISTORICAL DETAILS — must never be guessed or silently filled

- The original `experiments/exp_14_persistence_reuse_pilot.py` source code.
- The original `phase1/autodiff.py` engine source code.
- The original 145-test suite.
- The original checkpoints.
- The four seed values behind the reported step lists `[35, 35, 41, 31]` and
  `[1, 1, 1, 1]`.
- The original dataset generation seeds and the original episode-to-batch
  mapping.
- The original embedding initialisation scheme beyond the stated
  "initial scale = 0.3".
- The original early-stopping and convergence-detection policy in full.

The historical numbers are preserved **separately** as comparison targets. They
are **not** targets to optimise toward and **not** a definition of success. In
particular, Condition B's `[1, 1, 1, 1]` is treated as an unverified claim, not
as an expected value.

## Why the historical effect needs matched controls

The historical pilot reports a ~30-40x step reduction in Condition B relative to
Condition A. But A and B differ in **more than one** factor:

| Factor | Condition A | Condition B |
|---|---|---|
| Number of trainable parameters during the NOT2 phase | 6 op embs + roles (all fresh) | 1 op emb + roles (5 op embs frozen) |
| Prior training on the 5-op grammar | none | 1500 steps |
| Optimizer state carried into the NOT2 phase | none | inherited from pretraining |
| Dataset the NOT2 phase sees | 6-op episodes from step 0 | 6-op episodes on a converged router |
| NOT2 gradient flow | interferes with, and is interfered by, all other ops | isolated |

So B's advantage could be caused by **pretraining** (the 5-op geometry is already
a good prior) rather than by **persistence/reuse**. The two are separable, and
RECON-14 separates them. If the effect survives the controls below, it is
attributable to persistence/reuse; if it does not, the historical claim is
falsified by confound.

## Frozen protocol record (frozen BEFORE any training)

| Item | Value | Category |
|---|---|---|
| Experiment id | `recon-14-r1` | new |
| Branch | `research/recon-14-persistence-reuse` | new |
| Base commit | `3027826` on `feat/tac-osm-v0.1` | recovered |
| Template | `SupersetTemplate(max_depth=8, window=1)` | canonical |
| Pilot ops | `AND, OR, XOR, NOT, IDENTITY, NOT2` | recovered |
| Pretrain ops | `AND, OR, XOR, NOT, IDENTITY` | recovered |
| Router | `FactorizedRouterL2`, `tau = 1`, `d = 2`, init scale `0.3` | recovered |
| Roles | `{arg1, arg2, noncanonical}` | canonical |
| Executor | single-pass soft-logic over Boolean DAGs | canonical |
| Autodiff | hand-written reverse-mode (`Value`, `Adam`, `bce_loss`) | recovered |
| MAX_DEPTH (active program) | `5` | recovered |
| min_depth | `1` | recovered |
| decoy_prob | `0.7` | recovered |
| max_decoys_per_episode | `3` | recovered |
| batch_size | `16` | recovered |
| Optimizer | `Adam`, `lr = 0.1`, gradient-norm clip `1.0` | recovered |
| Pretrain steps (Condition B) | `1500` | recovered |
| Convergence criterion | `g(NOT2,arg1) > 0.9` and `g(NOT2,noncanonical) < 0.1` | recovered |
| Max NOT2-phase steps | `200` | new |
| Trial seeds | `{0, 1, 2, 3, 4, 5, 6, 7}` per condition | new |
| Dataset seed | `20260922` (frozen; identical episodes across all cells of a trial) | new |
| Pretrain dataset seed | `20260922` (identical corpus to the NOT2 phase) | new |
| Success metric | steps-to-convergence (primary); `g(NOT2,arg1)`, `g(NOT2,noncanonical)` | new |
| Epilogue evaluation | exact 6-op hard execution accuracy on held-out episodes | new |
| Statistic | per-seed steps; report full list, never mean-only | new |
| Falsifier | matched controls C and D must **not** reproduce the B speedup | new |

**These values are frozen. No hyperparameter, seed, dataset size, or threshold is
altered after results are observed.**

## Conditions (all cells receive identical batches and optimizer schedule)

Five cells. A and B reconstruct the historical pair. C, D, and E are the matched
controls that separate persistence/reuse from pretraining.

- **`A_from_scratch`** — historical Condition A. All 6 op embeddings plus role
  embeddings randomly initialised and trained jointly from step 0.
- **`B_persistent_reuse`** — historical Condition B. Pretrain 5 ops for 1500
  steps to convergence, freeze the pretrained parameters, add `NOT2` with a fresh
  embedding, train `NOT2` in isolation.
- **`C_not2_only_pretrain_control`** — the pretraining control. Pretrain **only
  `NOT2`** for the same 1500 steps on the same 6-op corpus, freeze it, then train
  `NOT2` alone. This has the same optimizer history and the same frozen-parameter
  budget as B, but no persistent 5-op geometry to reuse. If the speedup is
  pretraining, C reproduces it. If it is persistence/reuse, C does not.
- **`D_frozen_op_noop_persistence`** — the no-op persistence control. Pretrain
  the 5 ops for 1500 steps, then **freeze them and train nothing but the NOT2
  embedding**, with the frozen parameters held at their pretrained values and
  made to behave as a no-op toward `NOT2` — i.e. `NOT2` receives no gradient
  information through the frozen geometry beyond the loss on its own edges. This
  is the persistence-without-useful-reuse cell. If D converges as fast as B, the
  effect is "any frozen prior," not "reusable geometry."
- **`E_joint_not2_late`** — the timing control. Train all 6 ops jointly, but
  hold `NOT2` out of the corpus for the first 1500 steps, then introduce `NOT2`
  and continue joint training. This matches B's wall-clock pretraining budget
  while keeping joint training and no frozen parameters.

Cells must not be compared across different datasets, different batches, or
different optimizer states other than as explicitly defined above.

## What RECON-14 does and does not license

RECON-14 is a bounded reconstruction of one historical pilot. Success licenses
only the narrow claim that under this contract a frozen operator geometry
causally reduces steps-to-place a new operator's edges. It does **not** establish:

- that every persistent-state mechanism improves every benchmark;
- that this transfers to natural-language tasks or trained language checkpoints;
- that the mechanism survives distribution shift beyond the held-out episodes;
- any claim about TAC-OSM, PLM, or intelligence-compression work (see Out of
  scope).

Failure of A or B to replicate the historical numbers is a valid negative result
and is reported as such.

## Out of scope (mandatory exclusions)

Fusion with TAC-OSM, PLM, or the newer intelligence-compression work. Temperature
annealing, contrastive losses, arity supervision, new structural supervision,
auxiliary classifiers, new router architectures, recurrence, causal-access masks,
partition loss, intervention experiments, multi-seed scaling beyond the frozen 8
seeds, and any later CASM lifecycle/persistence work beyond this pilot. If a
result appears to invite one of these, stop and record it rather than expanding
scope.
