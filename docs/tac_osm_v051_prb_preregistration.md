# TAC-OSM v0.5.1 PR-B Preregistration

Date: 2026-09-25
Status: **FROZEN.** Referenced by PR-B and PR-C. Not modified after the first
PR-B execution.
Lineage parent: `3c13dff` (`83a9468` Stage-A execution repair).
Supersedes for diagnostic purposes: the factor-probe statistic in
`tac_osm/experiment_v05_factor_probe.py` (preserved unrevised as the
historical record on PR #23).

This document is a preregistration, not a results report. It commits to the
estimand, the probe hierarchy, the inferential quantity, the null calibration,
and the compute budget **before** any PR-B execution. It contains no results.

## 0. What this is and is not

PR-B asks exactly one question:

> Does a statistically meaningful compositional-representation diagnostic
> exist for this architecture, when the combinatorial grid is large enough
> to support one?

PR-B does **not** test the architecture. It does not test routing, causal
necessity, or substrate utility. Those are PR-C and the PNDS bridge, each of
which references this document rather than restating it. Three claims are kept
strictly separate, in this order:

```
information exists  !=  memory carries it  !=  routing uses it
```

The scientific escalation is gated: PR-C cannot begin because an intermediate
diagnostic looks promising. It begins only after the preregistered PR-B
criteria in §8 are satisfied, and not before.

## 1. Three defects in the v0.5 diagnostic, and why they block inference

### 1.1 The target is the model's own forecast

`collect_probe_data` builds its target from `model.candidate_forecast(...)` and
then asks whether the representations can reconstruct that same output. This
cannot establish that the representation contains the environment's
compositional factor. At most it establishes that one part of the model can
explain another part of the same model. The probe target must never be
derived from the model being probed.

### 1.2 The probe never touches the persistent state

```python
context_hidden, context_states = model.encode(t0, state, write=True)
context_repr = context_hidden.detach()        # <-- encoder hidden
```

`ControlledOperationalModel.encode` returns `(hidden, state)` where
`state = (slots, validity)` and `slots` has shape
`(batch, state_slots, structure_dim)`. The probe consumes `hidden` and
discards `state`. So the v0.5 diagnostic measures the pre-memory encoder
output, not the persistent structural state that is supposed to carry context
across the observation gap. The quantity the architecture is claimed to
provide was never measured.

### 1.3 Sample complexity: 8 observations against ~3*hidden features

The factor probe fits a ridge model with `3 * hidden` features on the 8
training cells. The reported absolute held-out MSE is then dominated by the
shrinkage prior rather than by learned structure, which is why the v0.5
numbers cannot be interpreted as evidence of compositionality.

## 2. Environmental confound: the held-out cells are unseen *actions*, not unseen combinations

This is the most consequential finding, and it is a property of the
environment, not of the model or the probe. It must be resolved in the v0.5.1
design before any diagnostic is meaningful.

The v0.5 rule is `mapped_action = context XOR action`, and the split is
`held_out_pair(c, a) = ((c + a) % 2) == 0`. The environment's action effect is
`effect[mapped_action]`, from a fixed 4x4 matrix `E` whose rows are linearly
independent (verified: rank 4).

Enumerating all 16 cells, the remapped action row `r = c ^ a` is:

| c | a | (c+a)%2 | held-out | r = c ^ a |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0 | yes | 0 |
| 0 | 1 | 1 | no | 1 |
| 0 | 2 | 0 | yes | 2 |
| 0 | 3 | 1 | no | 3 |
| 1 | 0 | 1 | no | 1 |
| 1 | 1 | 0 | yes | 0 |
| 1 | 2 | 1 | no | 3 |
| 1 | 3 | 0 | yes | 2 |
| 2 | 0 | 0 | yes | 2 |
| 2 | 1 | 1 | no | 3 |
| 2 | 2 | 0 | yes | 0 |
| 2 | 3 | 1 | no | 1 |
| 3 | 0 | 1 | no | 3 |
| 3 | 1 | 0 | yes | 2 |
| 3 | 2 | 1 | no | 1 |
| 3 | 3 | 0 | yes | 0 |

- Cells available at training time map to effect rows `{1, 3}`.
- Held-out cells map to effect rows `{0, 2}`.
- These sets are **disjoint**.

So the model never sees the effect vectors `E[0]` or `E[2]` at training time,
in any context. The held-out cells are not "unseen combinations of observed
factors" — they are **unseen factors**. A model that generalizes here would be
extrapolating to effect directions absent from its training data. That is a
different, much harder problem than compositional generalization, and the
distinction is exactly what the experiment claims to be about.

The intended reading of this benchmark is "every individual context and action
remains observed, only their combination is held out." The XOR remapping
violates that reading, because two different actions under two different
contexts can produce the same remapped row. The split is not factor-holding.

This is reported here rather than fixed silently, because it changes what a
held-out result would mean, and because it was present in the v0.5 evidence
record that PR #23 documents.

## 3. v0.5.1 environment design

### 3.1 Grid

`C = 16` contexts, `A = 4` actions, giving `64` context-action cells and a
pre-registered `32 train / 32 held-out` split. This is **not** to increase the
training set for the probe's convenience; it is to make the diagnostic
identifiable, per §2.

### 3.2 The split must be factor-holding

The split must satisfy

```
forall c: exists a_train, exists a_test
forall a: exists c_train, exists c_test
```

so that held-out cells are unseen *combinations*, not unseen factors.

This is a hard constraint on the v0.5.1 design, and it is the specific point
on which the XOR+checkerboard design fails. Any proposed split is validated by
an explicit unit test asserting both conditions, plus disjointness of the
remapped effect rows across partitions when the environment remaps actions.

### 3.3 C = 16 does not by itself fix the additive nullity

Verified: with `C=16, A=4` and a checkerboard split, the additive design
`(u_c, v_a)` has 20 unknowns and rank 18, so nullity 2 — the same nullity as
`C=4, A=4` (8 unknowns, rank 6). Scaling the grid does not remove the additive
model's degeneracy. It must be removed by split design, not by grid size.

This is recorded because the naive fix ("make the grid bigger") does not work,
and because a design that assumes otherwise would silently preserve the
confound at larger scale.

### 3.4 A context-size constraint that must be fixed in the environment

```python
context_index = context.argmax(-1).clamp_max(self.cfg.action_dim - 1)
```

`AntiLookupWorld.transition` clamps the context index to `action_dim - 1`.
With `action_dim = 4`, any `context_dim > 4` collapses to 4 distinct contexts.
Setting `C = 16` without changing this line would silently produce 12 dead
context labels, and the experiment would report 16 contexts while the
environment only distinguishes 4. This is a required change, recorded here so
it is not mistaken for an optional cleanup.

### 3.5 The semantic rule is not changed

`mapped_action = c XOR a` is retained. v0.5.1 changes statistical
identifiability only. It does not introduce a new causal rule, a richer
context code, or a harder mapping, so that a reviewer cannot attribute a
changed result to semantic novelty rather than to the split design.

**Interaction constraint:** §3.1-3.4 must land together. A larger grid with an
unfixed context clamp (§3.4) or an unfixed split (§3.2) is worse than the
current design, because it would report 16 contexts and 64 cells while
retaining the confound.

## 4. Estimand

The probe target is the **oracle** transition effect, never the model's own
prediction:

```
Y(c, a) = T(s, a, c) - D(s)
```

where `T` is the environment transition under context `c` and action `a`, and
`D` is the context- and action-independent state evolution, which for this
additive environment is available in closed form
(`state @ dynamics.T + bias`) rather than estimated.

Because the environment is additive in the action effect, `Y(c, a)` isolates
exactly the context-remapped action effect. `Y` is constructed without any
access to the model being probed, which closes §1.1.

The state-marginalized estimand is

```
Y(c, a) = E_{s ~ P_S} [ T(s, a, c) - D(s) ]
```

Sample `K` fixed evaluation states from a pre-registered RNG seed. Report both
the per-state errors `E(s_1), ..., E(s_K)` and the aggregate
`E_bar = (1/K) sum_k E(s_k)`. The per-state distribution is reported so that a
single lucky query state cannot carry a result.

## 5. Probe hierarchy: the full causal path

Four probes, one per stage of the gap-crossing chain, all against the same
oracle target `Y`:

| Representation | Extracts | Tests |
| --- | --- | --- |
| `H_t0` | `model.encode(t0, state, write=True)[0]` | context was encoded at all |
| `M_t0` | `... [1][0]` (the slots, written at t0) | context entered persistent state |
| `M_t1` | slots after the observation gap | context **survived** the gap |
| `R_t1` | retrieved structure from `Retrieve(M_t1, q_t1)` | the query actually retrieves it |

For `ControlledOperationalModel`, `M` is `state[0]` of shape
`(batch, state_slots, structure_dim)`; the retrieved structure is the
content-addressed read used inside `encode` / `candidate_forecast`, computed
from the same route query, not a new head.

This yields a failure taxonomy that a single MSE cannot produce:

- `H_t0` good, `M_t0` good, `M_t1` poor -> **persistence** fails, not representation.
- `M_t1` good, `R_t1` poor -> persistence works, **retrieval** fails.
- `R_t1` good but the model's prediction fails -> the problem moves downstream
  into computation/transition.

The v0.5 probe measured only `H_t0`, i.e. only the first row, which is why it
could not distinguish any of these.

### 5.1 Retrieval query protocol (fixed before running)

`R_t1` is undefined until the query is specified. For every held-out `(c, a)`,
construct `q_t1` from that cell's actual `t_1` observation, with the context
masked exactly as the environment specifies. Retrieval is then
`R_t1 = Retrieve(M_t1, q_t1)` under the real query distribution rather than an
artificial fixed query. This is recorded here so "retrieval works/fails" is
not revisable after the fact.

## 6. Inferential quantity: Delta, not absolute error

### 6.1 Capacity-matched probes

Project `M_c` and `e_a` into the same fixed dimension `d_p` in **both** arms:

```
phi_add  = [ M'_c ; e'_a ]
phi_fact = [ M'_c ; e'_a ; M'_c (x) e'_a ]
```

The two arms differ only in the presence of the interaction block, not in
total probe width. Without this, `Delta` partially measures extra probe
capacity rather than compositionality, and the permutation null becomes the
only calibration of that threat rather than a second layer over a fixed
baseline.

### 6.2 Delta

For each representation `X` and each ridge `lambda`:

```
E_X^add(lambda)  : additive probe held-out MSE
E_X^fact(lambda) : factorized probe held-out MSE

Delta_X(lambda) = E_X^add(lambda) - E_X^fact(lambda)
```

Larger positive `Delta` means the interaction-capable representation helps
more on held-out pairs.

### 6.3 The null hypothesis is about Delta

The primary inferential question is **not** whether either probe achieves low
error. It is whether the observed `Delta` exceeds its permutation null:

```
H_0 : Delta_X = 0
```

Explicitly, this is **not** `H_0 : E_X^fact = E_null`. That weaker null is
satisfied trivially by shrinkage and is the reason the v0.5 statistic carries
no information. The distinction is recorded here in code and evidence, not
just in discussion.

A `Delta` near zero is reported as "no detectable interaction information," not
as a negative result about the architecture.

## 7. Null calibration

### 7.1 Upstream permutation

For permutation replicate `b`, sample one permutation `pi_b` of context
identities and apply it **before** representation extraction, so that
`H_t0`, `M_t0`, `M_t1`, and `R_t1` are all extracted under the same shuffle.

One shuffle per (seed, replicate). If each representation were shuffled
independently, the §5 failure taxonomy would break: `M_t1` "good" under one
shuffle and `R_t1` "poor" under another is not interpretable as "persistence
works, retrieval fails." The four probes must remain commensurable.

### 7.2 Random-representation control

Replace `M_c` with an equal-dimensional random Gaussian representation, keep
the eight targets fixed, and rerun the identical probe. This answers a
different question than the permutation null: whether the probe architecture
itself can manufacture an apparent interaction advantage from an unrelated
representation.

Both controls are retained.

### 7.3 Permutation p-value

```
p = (1 + #{ b : Delta_b^perm >= Delta_obs }) / (B + 1)
```

The finite-sample correction is mandatory: without it the minimum attainable
p-value is 0, which is not a valid p-value for a permutation test.

### 7.4 Ridge surface

Pre-registered grid, not selected using held-out cells:

```
lambda in { 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1 }
```

For every `lambda`, report `E_add(lambda)`, `E_fact(lambda)`, and
`Delta(lambda)`. The result of interest is not a single low point; it is
`Delta(lambda) > 0` across a reasonable region of `lambda`, relative to the
permutation distribution.

## 8. Decision structure and stopping rule

### 8.1 Pre-registered compute budget

- Seeds: `n = 10`. **Fixed before the first run.** No additional seeds may be
  added because the observed effect is inconvenient or unstable.
- Permutation replicates: `B = 1000` per seed.
- This sets the resolution floor of the permutation p-value at ~1e-3.

Seed-level `Delta_obs` distribution is reported alongside the pooled
permutation null, so that "consistently positive" is distinguishable from
"positive on average, negative on a third of seeds." The latter would be a
genuine finding about write-gate instability, but it is only visible if the
seed count is committed up front.

### 8.2 Report tuple (per representation, per seed, per lambda)

```
( L_train, ||M||, ||Y||, ||beta||,
  E_train, E_held, E_null, E_shuffle, E_random,
  Delta, lambda )
```

plus the four-representation chain from §5.

### 8.3 Seed diagnostics instead of seed decisions

Collapse ratios are reported, not thresholded:

```
rho_Y = ||Y||_F / median_s ||Y_s||_F
rho_M = ||M||_F / median_s ||M_s||_F
```

Also recorded: `train_loss`, `held_loss`, `||M_t0||_F`, `||M_t1||_F`,
retrieval entropy, write-gate mean/std, probe coefficient norm, target norm,
additive and factorized train error, additive and factorized held-out error,
`Delta`, and `lambda`.

This makes seed 3 (the v0.5 outlier, with `train_loss` two orders of magnitude
above the others and effect magnitudes at ~3% of the median) diagnosable
without deleting it. Comparing `rho_Y` against `rho_M` distinguishes a
collapsed target scale from a collapsed persistent representation from an
unstable probe.

### 8.4 Stopping rule

> If the gap-crossing probe (`M_t1`) or the `Delta` permutation test fails,
> stop the architectural escalation and investigate the substrate before
> touching routing.

This is the only mechanism in the document that prevents architectural
escalation from outrunning evidence. It is the sentence that PR-C is bound by.

No positive threshold is designated as "success" in this preregistration. The
first step is to determine the distribution of `Delta` under null controls;
interpretation follows from that, not from a target.

## 9. Scope exclusions

PR-B contains **no architecture changes**. Specifically excluded:

- action-conditioned routing `C_t = Route(h_t, R_t, a)` — that is PR-C;
- `do(M = 0)`, `do(M = M_j)`, `do(M + eps)` interventions — PR-C;
- routing entropy `H(gamma | c)` and `H(gamma | c, a)` — PR-C;
- the donor-context semantic-flip signature — PR-C;
- any policy head, learned consumer, or objective change.

Including action-conditioned routing here would convert PR-B from "can we
measure the existing factor correctly" into "can a modified architecture
produce a stronger factor," which requires a separate evidence record.

PR-C references this document for its statistical commitments and does not
restate or modify them.

## 10. Lineage

```
83a9468   Stage A: execution repair only
   |
3c13dff   historical evidence: executable but statistically invalid probe
   |
PR-B  --> validated representation evidence
   |
PR-C  --> mechanism evidence
   |
PNDS --> cross-task substrate
```

Claim ownership, in order:

```
PR-B:  information  ->  persistence  ->  retrieval  ->  compositional representation
PR-C:  computation selection  ->  causal necessity / specificity
PNDS:  cross-task substrate
```

The PNDS bridge does not add a learned consumer. Its first form is a frozen
readout: `M_t -> utility/value prediction` with the world-model pathway
absent, then asking whether the same persistent representation supports both
`M -> counterfactual prediction` and `M -> task utility`. A learned head could
become an alternative shortcut and reduce use of the world-model pathway,
which would weaken rather than strengthen the substrate claim.

## 11. Environmental/API anchors used by this document

All claims above are grounded in the following, verified at `3c13dff`:

- `tac_osm/experiment_v04.py`, `ControlledOperationalModel.encode` returns
  `(hidden, state)`; `state = (slots, validity)`; `slots` has shape
  `(batch, state_slots, structure_dim)`. Retrieval inside
  `encode` / `candidate_forecast` is content-addressed via `route_query`.
- `tac_osm/experiment_v04.py`, `ExperimentConfig` defaults: `input_dim 8`,
  `hidden_dim 32`, `structure_dim 16`, `state_slots 8`, `action_dim 4`,
  `context_dim 4`.
- `tac_osm/experiment_v05_anti_lookup.py`, `held_out_pair(c, a) = ((c + a) % 2) == 0`.
- `tac_osm/experiment_v05_anti_lookup.py`, `AntiLookupWorld.transition`,
  `mapped_action = bitwise_xor(context_index, action_index)`, with the
  `clamp_max(action_dim - 1)` context constraint of §3.4.
- `tac_osm/environment.py`, `SyntheticOperationalWorld`: `effect` is a fixed
  `action_dim x state_dim` matrix; `dynamics` is
  `eye(state_dim) * scale`; `bias` is zero; `noise_std` is 0 in v0.5.

The three structural findings in §1.3, §2 and §3.3 were verified
computationally in pure Python before this document was written, using only
the environment definitions above, not the trained model.

## 12. Frozen

This preregistration is frozen as of the date above. Any deviation discovered
during PR-B execution is recorded as a deviation in the PR-B evidence
document, with the original commitment preserved. It is not edited into this
file retroactively, and the v0.5 numbers in PR #23 are not recomputed under
the new estimand.
