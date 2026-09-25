# TAC-OSM v0.5 Evidence Status

Recorded to keep the audit trail for the v0.5 anti-lookup line precise.
This file records what has and has not been measured, and is deliberately
separate from any claim about what the results mean.

## v0.5 anti-lookup benchmark

- Status: **executes; registered CI job passes.**
- In Fast CI run `36104680154` the `TAC-OSM v0.5 anti-lookup` job
  concluded `SUCCESS`. That job runs
  `python -m tac_osm.experiment_v05_anti_lookup --seeds 0,1,2,3,4`.
- The registered benchmark therefore produces a five-seed result.

## v0.5 factor-probe diagnostic

- Status: **no scientific result.**
- Fast CI run `36104680154` (2026-09-25T06:50Z) **did execute** the
  factor-probe diagnostic on `push`, because `.github/workflows/ci.yml`
  lists `research/tac-osm-v05-factor-probe` in the `push` trigger and the
  job runs
  `python -m tac_osm.experiment_v05_factor_probe --seeds 0,1,2,3,4 --steps 1000 --batch 128`.
- The `TAC-OSM v0.5 factor probe` job concluded `FAILURE` at the unit-test
  step with:
  ```
  IndexError: index 4 is out of bounds for dimension 0 with size 4
  tac_osm/experiment_v05_factor_probe.py:56: IndexError
  ```
  in `_fit_factor_probe`.
- The experiment therefore **crashed before emitting any JSON**. No
  diagnostic metric was produced for any seed.

This is materially different from two adjacent statements, both of which
were wrong:

| Statement | Accuracy |
| --- | --- |
| "CI did not run the diagnostic" | Wrong. CI ran it; the `push` trigger was configured for this branch. |
| "The diagnostic produced a negative empirical result" | Wrong. It produced no result at all. |
| "Run 36104680154 executed the factor-probe diagnostic but produced no scientific result because the implementation crashed before emitting JSON." | Correct. |

## Root cause and scope of the execution repair

`_fit_factor_probe` derived its grid dimensions from a representation
tensor:

```python
c, a = context_repr.shape   # (n_context, hidden) -> 4, 6
```

The context representation carries a hidden width that need not equal the
action count. With `hidden=6` and four actions, the loop reached `ai=4`
and indexed past the end of `action_repr` / `seen_mask`. The probe never
reached the model.

The repair derives the grid from the authoritative effect matrix instead:

```python
n_context, n_action = effects.shape[:2]
```

and retains `context_repr[ci]` / `action_repr[ai]` indexing. No reported
statistic, metric, or interpretation was changed.

## Known failing jobs on the v0.5 branches

The following are **pre-existing** failures on the base branch
(`research/tac-osm-v05-anti-lookup-main`, runs `36096453406` /
`36096643852`) and are not caused by the factor probe:

- `Python smoke tests`: `ModuleNotFoundError: No module named
  'tac_transformer.structure_types'`
- `Frontend smoke`: fails on the base branch as well.

They remain separately attributable.

## Open item: probe statistical validity

The registered probe fits a bilinear model with `3 * hidden` parameters
on 8 observations and reports an absolute held-out ridge MSE with a fixed
`ridge=1e-3`. The p/n ratio and the fixed ridge mean the absolute
held-out MSE does not currently support the distinction the diagnostic is
designed to make (representation failure vs execution/composition
failure). That is a metric-design issue, not an execution issue, and is
handled as a separate change rather than bundled with the crash repair.
