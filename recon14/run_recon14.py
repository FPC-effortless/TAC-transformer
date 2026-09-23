#!/usr/bin/env python3
"""RECON-14 protocol driver.

Runs the frozen five-cell RECON-14 protocol (see ``RECON_14.md``) and writes a
machine-readable result bundle. This is the entry point the
``.github/workflows/recon-14.yml`` dispatch lane calls; it exists because the
protocol needs a CLI surface and ``recon14.experiment`` previously exposed only
library functions.

Frozen protocol values are read from ``recon14.experiment.FROZEN`` and are never
overridden from the command line. The CLI exposes only *what* to run (seeds,
conditions, episode count) and *where* to write, so a dispatch cannot silently
change a frozen hyperparameter, seed, threshold, or dataset size.

RECON-14 does not use PyTorch: differentiation is the hand-written reverse-mode
engine in ``recon14/autodiff.py``. This module imports only the standard library
plus ``recon14``, so the runner does not need ``torch``.

Usage::

    python -m recon14.run_recon14 --output outputs/recon14/recon14_results.json
    python -m recon14.run_recon14 --quick --output /tmp/recon14_smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from recon14.experiment import (CONDITIONS, CellResult, FROZEN, build_dataset,
                                falsification_verdict, run_condition,
                                run_trial, summarise)

# Quick settings are a *lane* convenience for a smoke check only. They reduce
# seeds and episode counts; they do not touch any frozen hyperparameter,
# threshold, or dataset seed. A --quick run is not a protocol run and its output
# is labelled as such.
QUICK_SEEDS = (0, 1)
QUICK_EPISODES = 16


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="recon14.run_recon14",
        description="Run the frozen RECON-14 persistence/reuse protocol.",
    )
    p.add_argument(
        "--output", "-o",
        required=True,
        help="Path to write the JSON result bundle.",
    )
    p.add_argument(
        "--seeds",
        default="",
        help=("Comma-separated trial seeds. Defaults to the frozen set "
              f"{FROZEN['trial_seeds']}. Overriding is a deviation and is "
              "recorded in the bundle."),
    )
    p.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
        help="Comma-separated subset of conditions to run (default: all five).",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=128,
        help="Episodes per trial corpus (default: 128).",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help=("Reduced smoke settings (fewer seeds/episodes). NOT a protocol "
              "run; the bundle is labelled non-protocol."),
    )
    return p.parse_args(argv)


def _cell_to_dict(cell: CellResult) -> Dict[str, Any]:
    """Serialise one cell. The full per-step history is kept because the frozen
    protocol reports the full per-seed step list, never a mean only."""
    d = {k: v for k, v in cell.__dict__.items()}
    d["history"] = [rec.__dict__ for rec in cell.history]
    return d


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    t0 = time.time()

    quick = bool(args.quick)
    seeds: Sequence[int]
    if args.seeds:
        seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    else:
        seeds = tuple(FROZEN["trial_seeds"]) if not quick else QUICK_SEEDS
    conditions = tuple(c for c in args.conditions.split(",") if c.strip())
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        print(f"unknown condition(s): {unknown}", file=sys.stderr)
        return 2
    n_episodes = QUICK_EPISODES if quick else int(args.episodes)

    print("=" * 72, flush=True)
    print(f"RECON-14  {FROZEN['experiment_id']}", flush=True)
    print(f"  seeds       : {seeds}", flush=True)
    print(f"  conditions  : {conditions}", flush=True)
    print(f"  episodes    : {n_episodes}", flush=True)
    print(f"  quick       : {quick}", flush=True)
    print(f"  dataset_seed: {FROZEN['dataset_seed']} (frozen)", flush=True)
    print("=" * 72, flush=True)

    # The corpus is built once per trial and shared by every cell, so all cells
    # in a trial see identical batches (protocol: "all cells receive identical
    # batches and optimizer schedule").
    results: Dict[str, List[CellResult]] = {c: [] for c in conditions}
    for si, seed in enumerate(seeds, start=1):
        t_trial = time.time()
        print(f"[trial {si}/{len(seeds)}] seed={seed} building corpus...",
              flush=True)
        dataset = build_dataset(n_episodes=n_episodes,
                                seed=FROZEN["dataset_seed"])
        for cond in conditions:
            t_cell = time.time()
            cell = run_condition(cond, seed=seed, dataset=dataset, verbose=False)
            results[cond].append(cell)
            print(f"    {cond:<32} conv={cell.converged!s:<5} "
                  f"steps={cell.steps_to_converge} "
                  f"acc={cell.epilogue_accuracy} "
                  f"({cell.seconds:.1f}s)", flush=True)
        print(f"[trial {si}/{len(seeds)}] done in {time.time() - t_trial:.1f}s",
              flush=True)

    summary = summarise(results)
    verdict = falsification_verdict(summary)

    bundle: Dict[str, Any] = {
        "experiment_id": FROZEN["experiment_id"],
        "protocol": "RECON_14.md",
        "branch": "research/recon-14-persistence-reuse",
        "base_commit": FROZEN["base_commit"],
        "frozen": {k: list(v) if isinstance(v, tuple) else v
                   for k, v in FROZEN.items()},
        "seeds": list(seeds),
        "conditions": list(conditions),
        "n_episodes": n_episodes,
        "quick_smoke": quick,
        "deviations": [],
        "summary": summary,
        "verdict": verdict,
        "cells": {cond: [_cell_to_dict(c) for c in cells]
                  for cond, cells in results.items()},
        "wall_time_seconds": time.time() - t0,
    }
    if list(seeds) != list(FROZEN["trial_seeds"]):
        bundle["deviations"].append(
            f"seeds={list(seeds)} differ from frozen {list(FROZEN['trial_seeds'])}"
        )
    if n_episodes != 128:
        bundle["deviations"].append(
            f"n_episodes={n_episodes} differs from the protocol corpus size 128"
        )
    if quick:
        bundle["deviations"].append(
            "quick smoke settings: reduced seeds and episode count; "
            "not a protocol run"
        )
    bundle["deviations"] = sorted(set(bundle["deviations"]))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, sort_keys=True),
                        encoding="utf-8")

    print("=" * 72, flush=True)
    print(f"verdict: {verdict.get('verdict')}", flush=True)
    if verdict.get("reason"):
        print(f"  {verdict['reason']}", flush=True)
    for cond in conditions:
        steps = summary[cond]["steps_to_converge"]
        print(f"  {cond:<32} mean_steps={summary[cond]['mean_steps']} "
              f"converged={summary[cond]['converged']}/{summary[cond]['n_seeds']} "
              f"steps={steps}", flush=True)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)", flush=True)
    print(f"total wall time: {time.time() - t0:.1f}s", flush=True)
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
