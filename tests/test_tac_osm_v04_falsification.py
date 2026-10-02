"""Small falsification run for TAC-OSM v0.4.

This is deliberately a smoke-scale empirical gate, not the publication run.
It exercises all four matched cells for one seed and prints the held-out
forecast/planning/intervention metrics for inspection.
"""
import json
import math

import torch

from tac_osm.experiment_v04 import ExperimentConfig, train_one


def test_v04_small_falsification_run(capsys):
    cfg = ExperimentConfig(
        train_steps=30,
        batch_size=64,
        eval_batch=256,
    )
    device = torch.device("cpu")

    results = []
    for persistent in (False, True):
        for action_conditioned in (False, True):
            metrics = train_one(
                seed=0,
                persistent=persistent,
                action_conditioned=action_conditioned,
                cfg=cfg,
                device=device,
            )
            results.append(metrics)

    assert len({r["parameters"] for r in results}) == 1
    for r in results:
        for key in (
            "forecast_mse_held_out_all_actions",
            "planning_accuracy_held_out",
            "achieved_objective",
            "oracle_objective",
            "objective_regret",
            "action_forecast_delta",
            "state_reset_delta",
            "state_shuffle_delta",
            "state_corrupt_delta",
        ):
            assert math.isfinite(r[key]), (r["persistent"], r["action_conditioned"], key, r[key])

    by_cell = {
        (r["persistent"], r["action_conditioned"]): r for r in results
    }

    # Mechanistic factor-off controls must be exact:
    # without action conditioning, candidate action cannot change prediction;
    # without persistence, resetting the hidden state cannot change prediction.
    assert by_cell[(False, False)]["action_forecast_delta"] == 0.0
    assert by_cell[(False, True)]["action_forecast_delta"] > 0.0
    assert by_cell[(False, False)]["state_reset_delta"] == 0.0
    assert by_cell[(False, True)]["state_reset_delta"] == 0.0

    print(json.dumps({
        "experiment": "TAC-OSM-v0.4-small-falsification",
        "seed": 0,
        "steps": cfg.train_steps,
        "cells": results,
    }, indent=2))
