# TAC-OSM v0.5 Anti-Lookup Evidence Status

Date: 2026-09-25
PR: #22 (`research/tac-osm-v05-anti-lookup-main`)
Branch tip at time of writing: `8ee9711`
Companion code note: docstring of `tac_osm/experiment_v05_anti_lookup.py`

## 1. Purpose and scope

This is an **interpretation caveat**, not a correction of results.

- The v0.5 experiment is **unchanged**.
- Its original numbers are **not recomputed, corrected, or retracted**.
- Nothing in this document edits the PR-B preregistration
  (`docs/tac_osm_v051_prb_preregistration.md`), which stays frozen at its own
  commit.

The distinction being drawn is between a historical record and the current
claim status. Both are preserved.

## 2. What v0.5 measured

The benchmark ran as registered. In Fast CI run `36097914803` on branch
`research/tac-osm-v05-anti-lookup-main`, the `TAC-OSM v0.5 anti-lookup` job
concluded **SUCCESS**. The job runs

```
python -m tac_osm.experiment_v05_anti_lookup --seeds 0,1,2,3,4
```

so the registered experiment executes and produces its five-seed output. Those
outputs remain the historical record of the implemented benchmark.

## 3. The caveat: the split is not factor-holding under the XOR mapping

The design intent, stated in the experiment docstring and in PR #22's body, is
that "every individual context and action remains observed" and that only
their *combination* is withheld. That intent is correct as stated at the
`(c, a)` level, and it is why the existing test
`test_anti_lookup_split_exposes_all_individual_factors` **passes**.

The problem is one level down. The action effect actually applied at a cell is

```
effect[c ^ a]          # not  effect[a]
```

because `AntiLookupWorld.transition` computes
`mapped_action = bitwise_xor(context_index, action_index)`. The relevant
"factor" for whether held-out evaluation is a recombination test is therefore
the **remapped** index `c ^ a`, not the raw action index. Checking the split on
the remapped index, with `held_out_pair(c, a) = ((c + a) % 2) == 0`:

| partition | cells | remapped effect rows `c ^ a` |
| --- | --- | --- |
| training | `(c + a) % 2 == 1` (8 cells) | `{1, 3}` |
| held-out | `(c + a) % 2 == 0` (8 cells) | `{0, 2}` |

These sets are **disjoint**. `effect` is a fixed `4 x 8` matrix of rank 4, so
rows 0 and 2 are linearly independent physical effect directions that appear
**nowhere** in training, in any context.

## 4. Why this blocks the stronger interpretation

The benchmark's headline claim is compositional generalization: combine an
observed context with an observed action in a way never seen together. Two
distinct failure modes are being conflated:

1. the factors were observed, but not in this combination — a genuine
   compositional test;
2. the physical effect direction itself was never observed — extrapolation to
   unseen inputs.

The v0.5 split places held-out evaluation in case 2. A model that fails on the
held-out cells is consistent with "the effect rows are unseen," which is not
evidence about composition at all. A model that succeeds would be
extrapolating to effect directions absent from its training distribution.

So the result is real, but the inference it can support is narrower than the
"compositional generalization" label implies:

> The v0.5 result cannot by itself be interpreted as evidence of
> compositional generalization across unseen context-action combinations.
> It remains a valid historical result for the implemented benchmark, but its
> stronger compositional-generalization interpretation is not established.

Compactly:

```
v0.5 demonstrated behavior on its implemented split
  !=
v0.5 demonstrated compositional generalization
```

## 5. Why the existing test did not catch this

`test_anti_lookup_split_exposes_all_individual_factors` asserts

```
forall c: exists a seen, exists a held
forall a: exists c seen, exists c held
```

This is the correct *factor-level* coverage condition and it genuinely holds.
It is not wrong. It is **insufficient**: it is stated over the raw action
index, and so it does not see the XOR remapping that decides which physical
effect each cell exercises. The disjointness only becomes visible when the
same coverage condition is checked on `c ^ a`.

This is recorded because a passing test was providing false confidence, and
because the fix belongs in split validation rather than in the test's current
form.

## 6. What does and does not follow

Does **not** follow from this caveat:

- that the v0.5 numbers are wrong or should be deleted;
- that the v0.5 model failed or succeeded for any particular reason;
- that the architecture is or is not capable of compositional
  generalization;
- that the shortcut audit motivating PR #22 was mistaken.

Does follow:

- any compositional-generalization claim resting solely on the v0.5 split is
  **unsupported**;
- the v0.5 anti-lookup result should be cited as "behavior on the implemented
  split," not as compositional generalization;
- the claim is not closed, only unestablished. It becomes testable again under
  a split that is factor-holding on the remapped index.

## 7. Where the claim can be earned

PR-B (v0.5.1) is the first experiment from which a compositional-generalization
claim can potentially be earned. Its split is required to be factor-holding on
the remapped index, i.e. every physical effect direction must appear on both
sides of the split, in addition to every context and action. That requirement,
and the environment cardinality preflight that supports it, are specified in
the frozen preregistration and are not restated or modified here.

## 8. Provenance

The lineage is intended to remain auditable rather than to appear as though
the original experiment never happened:

```
PR #22 / v0.5
   |
   +-- historical benchmark, original split, original numbers
   +-- interpretation caveat added (this document + docstring note)
   |
   v
PR #23 / Stage A
   |
   +-- execution repair of the factor probe
   |
   v
PR-B / v0.5.1
   |
   +-- correct persistent-state measurement
   +-- factor-holding split (on remapped index)
   +-- corrected environment cardinality
   +-- capacity-matched probes
   +-- Delta estimand
   +-- gap-crossing probe hierarchy
   +-- preregistered inference
```

## 9. Verification

The disjointness claim in §3 was verified computationally in pure Python from
the environment definitions in `tac_osm/environment.py` and
`tac_osm/experiment_v05_anti_lookup.py` at `8ee9711`, enumerating all 16
`(c, a)` cells. No model was trained or evaluated for this document, and no
v0.5 number was recomputed.
