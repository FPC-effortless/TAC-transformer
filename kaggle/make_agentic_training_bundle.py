from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "kaggle_agentic_training_bundle"
DEFAULT_DATASET_METADATA = {
    "id": "jeffkolo/tac-ats-transfer-code-2026-06-05",
    "title": "TAC ATS Transfer Code 2026-06-05",
    "licenses": [{"name": "CC0-1.0"}],
}


FILES = [
    "README.md",
    "research.md",
    "docs/best_tac_architecture.md",
    "docs/agentic_architecture_research_recommendation.md",
    "docs/hybrid_mixer_experiments.md",
    "docs/hard_agentic_corpus.md",
    "docs/kaggle_synthesis_training_preflight.md",
    "docs/kaggle_identity_first_run3_preflight.md",
    "docs/program_specialization_analysis.md",
    "docs/usef_authority_reporting_transfer.md",
    "docs/capability_sanity_gate.md",
    "docs/run5_pathfinder_research.md",
    "docs/run5_failure_protocol.md",
    "docs/tac_control_v1_research_contract.md",
    "docs/tac_optimizer.md",
    "docs/tac_agentic_rl_mathematical_contract.md",
    "docs/tac_serving_and_architecture.md",
    "experiments/benchmark_capability_sanity.py",
    "experiments/benchmark_run5_pathfinder.py",
    "experiments/benchmark_routing_pressure_phase.py",
    "experiments/evaluate_external_run5b_validation.py",
    "experiments/advance_tac_research_plan.py",
    "experiments/aggregate_phase_b_seed_results.py",
    "experiments/aggregate_phase_c_identity_stability.py",
    "experiments/aggregate_phase_d_benchmarks.py",
    "experiments/stage_phase_d_benchmark_suite.py",
    "experiments/stage_phase_d_suite_dataset.py",
    "experiments/score_phase_d_predictions.py",
    "experiments/run_phase_d_checkpoint_predictions.py",
    "experiments/run_phase_d_benchmark_matrix.py",
    "experiments/benchmark_agentic_controller_learning.py",
    "experiments/benchmark_live_agentic_policy_adapter.py",
    "experiments/benchmark_live_agentic_policy_training.py",
    "experiments/benchmark_agentic_scratchpad_state.py",
    "experiments/benchmark_phase_d_scratchpad_state_execution.py",
    "experiments/benchmark_scratchpad_autoregressive_decoding.py",
    "experiments/benchmark_joint_tac_controller_training.py",
    "experiments/benchmark_agentic_trajectory_records.py",
    "experiments/benchmark_agentic_verifier_rewards.py",
    "experiments/benchmark_group_relative_trajectory_training.py",
    "experiments/benchmark_dynamic_sampling_cost_shaping.py",
    "experiments/benchmark_sequence_process_value_support.py",
    "experiments/benchmark_identity_coalition_math_upgrade.py",
    "experiments/benchmark_live_phase_d_scratchpad_policy.py",
    "experiments/benchmark_parallel_program_trajectories.py",
    "experiments/benchmark_full_parallel_program_architecture.py",
    "experiments/benchmark_persistent_computational_identity.py",
    "experiments/benchmark_persistent_identity_broader_tasks.py",
    "experiments/benchmark_live_persistent_identity_state_bridge.py",
    "experiments/benchmark_trained_identity_collapse_recovery.py",
    "experiments/benchmark_identity_interference_stress.py",
    "experiments/benchmark_relaxed_identity_routing_memory.py",
    "experiments/benchmark_phase_boundary_quantification.py",
    "experiments/benchmark_memory_advantage_model_version.py",
    "experiments/benchmark_long_horizon_memory_advantage.py",
    "experiments/benchmark_kaggle_tac_training_speed_profile.py",
    "experiments/benchmark_local_tac_efficiency_matrix.py",
    "experiments/benchmark_cpu_research_tac_version.py",
    "experiments/benchmark_identity_attention_selectivity.py",
    "experiments/benchmark_ats_transfer_suite.py",
    "experiments/stage_ats_transfer_corpus.py",
    "experiments/run_ats_checkpoint_predictions.py",
    "experiments/aggregate_ats_checkpoint_runs.py",
    "experiments/benchmark_ats_answer_copy_training.py",
    "kaggle/README.md",
    "kaggle/__init__.py",
    "kaggle/train_best_tac_agentic.py",
    "kaggle/train_vanilla_baseline.py",
    "kaggle/analyze_program_specialization.py",
    "kaggle/analyze_routing_collapse.py",
    "kaggle/inspect_identity_memory.py",
    "kaggle/evaluate_checkpoint_harder_matrix.py",
    "kaggle/prepare_tac_corpus.py",
    "kaggle/profile_kaggle_memory.py",
    "scripts/prepare_tac_tokenized_corpus.py",
    "scripts/tac_generate.py",
    "scripts/tac_gradio_gui.py",
    "tac_transformer/__init__.py",
    "tac_transformer/model.py",
    "tac_transformer/training.py",
    "tac_transformer/serving.py",
    "tac_transformer/presets.py",
    "tac_transformer/optimization.py",
    "tac_transformer/data.py",
    "tac_transformer/distillation_datasets.py",
    "tac_transformer/evaluation.py",
    "tac_transformer/agentic.py",
    "tac_transformer/agentic_rl_math.py",
    "tac_transformer/agentic_controller.py",
    "tac_transformer/knowledge_work.py",
    "tac_transformer/authority.py",
    "tac_transformer/capability.py",
    "tac_transformer/research_plan.py",
    "tac_transformer/phase_d_benchmarks.py",
    "tac_transformer/ats_transfer.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small code-only Kaggle bundle for best TAC agentic training."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for relative in FILES:
        source = ROOT / relative
        target = args.output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    (args.output_dir / "RUN_ON_KAGGLE.md").write_text(_instructions(), encoding="utf-8")
    (args.output_dir / "dataset-metadata.json").write_text(
        json.dumps(DEFAULT_DATASET_METADATA, indent=2) + "\n",
        encoding="utf-8",
    )
    zip_path = args.output_dir / "best-tac-agentic-training-bundle.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in args.output_dir.rglob("*"):
            if path == zip_path:
                continue
            archive.write(path, path.relative_to(args.output_dir))
    print(zip_path)


def _instructions() -> str:
    return """# Best TAC Agentic Training Bundle

Upload this zip as a Kaggle Dataset. Upload `runs/prepared_corpus_agentic_hard_upload.zip`
as a second Kaggle Dataset named `tac-hard-agentic-corpus`, enable GPU, then run:

```python
from pathlib import Path
from zipfile import ZipFile
import shutil

work = Path("/kaggle/working/best_tac_agentic_code")
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)

bundle_zip = next(Path("/kaggle/input").glob("**/best-tac-agentic-training-bundle.zip"), None)
if bundle_zip is not None:
    ZipFile(bundle_zip).extractall(work)
else:
    train_script = next(Path("/kaggle/input").glob("**/kaggle/train_best_tac_agentic.py"))
    source_root = train_script.parents[1]
    for item in source_root.iterdir():
        target = work / item.name
        if item.is_dir():
            shutil.copytree(item, target, ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"))
        else:
            shutil.copy2(item, target)

%cd /kaggle/working/best_tac_agentic_code

!python kaggle/train_best_tac_agentic.py \\
  --scale base \\
  --steps 20000 \\
  --warmup-steps 500 \\
  --batch-size 12 \\
  --grad-accum-steps 3 \\
  --eval-every 500 \\
  --eval-batches 8 \\
  --checkpoint-every 250 \\
  --output-dir /kaggle/working/best_tac_agentic_run4_semantic_selected_mi \\
  --device auto \\
  --precision fp32 \\
  --min-healthy-gradient-norm 1e-12 \\
  --fail-on-unhealthy-optimization \\
  --max-seconds 30600 \\
  --stop-buffer-seconds 1200 \\
  --routing-type base_semantic \\
  --routing-top-k 2 \\
  --routing-load-balance-weight 0.05 \\
  --category-route-weight 0.5 \\
  --category-route-objective selected_mi \\
  --specialization-checkpoints 2000 5000 10000 20000 \\
  --specialization-checkpoint-max-records-per-category 16 \\
  --analyze-specialization-at-end \\
  --specialization-max-records-per-category 64 \\
  --specialization-device cpu \\
  --skip-end-specialization-on-time-stop
```

For both Kaggle T4 GPUs:

```python
!torchrun --standalone --nproc_per_node=2 kaggle/train_best_tac_agentic.py \\
  --scale base \\
  --steps 20000 \\
  --warmup-steps 500 \\
  --batch-size 12 \\
  --grad-accum-steps 3 \\
  --eval-every 500 \\
  --eval-batches 8 \\
  --checkpoint-every 250 \\
  --output-dir /kaggle/working/best_tac_agentic_run4_semantic_selected_mi \\
  --device auto \\
  --precision fp32 \\
  --min-healthy-gradient-norm 1e-12 \\
  --fail-on-unhealthy-optimization \\
  --max-seconds 30600 \\
  --stop-buffer-seconds 1200 \\
  --routing-type base_semantic \\
  --routing-top-k 2 \\
  --routing-load-balance-weight 0.05 \\
  --category-route-weight 0.5 \\
  --category-route-objective selected_mi \\
  --specialization-checkpoints 2000 5000 10000 20000 \\
  --specialization-checkpoint-max-records-per-category 16 \\
  --analyze-specialization-at-end \\
  --specialization-max-records-per-category 64 \\
  --specialization-device cpu \\
  --skip-end-specialization-on-time-stop
```

If TAC training is much slower than the vanilla LLM job, first run the opt-in
speed profile. It keeps semantic selected-MI TAC training but limits
content-read queries and uses local causal attention:

```python
!python experiments/benchmark_kaggle_tac_training_speed_profile.py \\
  --device cuda \\
  --seq-len 176 \\
  --batch-size 4 \\
  --iters 5 \\
  --output-dir /kaggle/working/kaggle_tac_training_speed_profile

!python experiments/benchmark_local_tac_efficiency_matrix.py \\
  --device cpu \\
  --seq-len 64 \\
  --batch-size 2 \\
  --torch-threads 1 \\
  --interop-threads 1 \\
  --iters 3 \\
  --output-dir /kaggle/working/local_tac_efficiency_matrix

!python experiments/benchmark_cpu_research_tac_version.py \\
  --device cpu \\
  --seq-len 64 \\
  --batch-size 2 \\
  --torch-threads 1 \\
  --interop-threads 1 \\
  --iters 10 \\
  --output-dir /kaggle/working/cpu_research_tac_version

!torchrun --standalone --nproc_per_node=2 kaggle/train_best_tac_agentic.py \\
  --preset kaggle_fast_tac \\
  --scale base \\
  --seq-len 176 \\
  --steps 5000 \\
  --batch-size 12 \\
  --grad-accum-steps 3 \\
  --eval-every 1000 \\
  --eval-batches 4 \\
  --checkpoint-every 500 \\
  --output-dir /kaggle/working/best_tac_agentic_fast_selected_mi \\
  --device auto \\
  --max-seconds 30600 \\
  --stop-buffer-seconds 1200 \\
  --specialization-checkpoint-max-records-per-category 8 \\
  --skip-end-specialization-on-time-stop
```

For the strongest current Run 5B capability launch, use the integrated preset.
It combines the TAC-188 memory-advantage stack, the TAC-169 cue-chain readout,
Run 5B fp32/fail-fast optimizer health, selected-route MI, and the implemented
training-speed cadence:

```python
!torchrun --standalone --nproc_per_node=2 kaggle/train_best_tac_agentic.py \\
  --preset run5b_best_capability_fast \\
  --scale base \\
  --seq-len 176 \\
  --steps 20000 \\
  --batch-size 12 \\
  --grad-accum-steps 3 \\
  --eval-every 1000 \\
  --eval-batches 4 \\
  --checkpoint-every 500 \\
  --aux-loss-cadence 4 \\
  --output-dir /kaggle/working/run5b_best_capability_fast \\
  --device auto \\
  --max-seconds 30600 \\
  --stop-buffer-seconds 1200 \\
  --specialization-checkpoints 2000 5000 10000 20000 \\
  --specialization-checkpoint-max-records-per-category 16 \\
  --analyze-specialization-at-end \\
  --specialization-max-records-per-category 64 \\
  --specialization-device cpu \\
  --skip-end-specialization-on-time-stop
```

If the local matrix reports `opt_in_aux_cadence_candidate`, add
`--aux-loss-cadence 4` to the TAC command for a speed run, then validate the
checkpoint against the same capability gate before treating it as equivalent.
For a stronger CPU-only research branch that does not modify the main TAC
architecture, use `--preset cpu_research_tac`; it is intentionally opt-in and
must pass downstream capability gates before any idea is promoted.

If Kaggle chooses a different dataset folder, locate the files with:

```python
!find /kaggle/input -name "train.prepared.jsonl" -o -name "eval.prepared.jsonl"
```

The trainer also searches `/kaggle/input` recursively, so explicit `--train-jsonl`
and `--eval-jsonl` paths are optional unless multiple prepared corpora are attached.

The promoted Run 4 model is identity-first TAC with hard semantic routing,
content-addressed memory, two content reads, the synthesis gate, anti-collapse
losses, load-balancing pressure, and a category-program MI objective. The trainer writes `last.pt`,
`best.pt`, `metrics.jsonl`, `run_manifest.json`, and `final_summary.json`.
With `--specialization-checkpoints`, it writes lightweight checkpoint
attribution snapshots under `specialization_checkpoints/step_*`. With
`--analyze-specialization-at-end`, the same run also writes
`specialization/program_specialization.json`,
`specialization/program_attribution.csv`, and a `specialization_analysis`
summary in `final_summary.json`.
The faster profile keeps the original effective batch size:
`12 batch * 3 grad_accum * 2 GPUs = 72 sequences`. If Kaggle runs out of memory,
fall back to `--batch-size 8 --grad-accum-steps 4`.
Do not attach Run 3 checkpoint datasets when starting this semantic-MI run
fresh. The useful-family route policy is a Run 3 checkpoint patch; Run 4 should
learn semantic routing without fixed program IDs.
The final specialization pass still runs when the target step is reached; the
skip flag only avoids spending hours on end-of-run analysis after an
intermediate Kaggle time stop.

After training, inspect the best checkpoint:

```python
!python kaggle/inspect_identity_memory.py \\
  --checkpoint /kaggle/working/best_tac_agentic_run4_semantic_selected_mi/best.pt \\
  --prompt "Use calculator, verify result, then answer." \\
  --output /kaggle/working/best_tac_agentic_run4_semantic_selected_mi/memory_inspection.json
```

The Run 4 command above already analyzes functional specialization from
`best.pt`. To rerun it manually with a different sample size:

```python
!python kaggle/analyze_program_specialization.py \\
  --checkpoint /kaggle/working/best_tac_agentic_run4_semantic_selected_mi/best.pt \\
  --jsonl /kaggle/input/tac-hard-agentic-corpus/hard_agentic_eval.generated.jsonl \\
  --max-records-per-category 64 \\
  --output /kaggle/working/best_tac_agentic_run4_semantic_selected_mi/program_specialization.json \\
  --csv-output /kaggle/working/best_tac_agentic_run4_semantic_selected_mi/program_attribution.csv
```

Run checkpoint-only harder probes:

```python
!python kaggle/evaluate_checkpoint_harder_matrix.py \\
  --checkpoint /kaggle/working/best_tac_agentic_run4_semantic_selected_mi/best.pt \\
  --seeds 11 23 37 \\
  --eval-batches 8 \\
  --eval-batch-size 32 \\
  --output /kaggle/working/best_tac_agentic_run4_semantic_selected_mi/checkpoint_harder_matrix.json
```

Run a Phase D benchmark prediction pass from a staged task suite:

```python
!python experiments/run_phase_d_checkpoint_predictions.py \\
  --tasks-jsonl /kaggle/input/tac-phase-d-suite/seed_11/tasks.jsonl \\
  --checkpoint /kaggle/working/best_tac_agentic_run4_semantic_selected_mi/best.pt \\
  --control-id tac_control_v1_seed_11 \\
  --seed 11 \\
  --output-jsonl /kaggle/working/best_tac_agentic_run4_semantic_selected_mi/phase_d_seed_11_predictions.jsonl \\
  --score-output-json /kaggle/working/best_tac_agentic_run4_semantic_selected_mi/phase_d_seed_11_score.json \\
  --score-output-jsonl /kaggle/working/best_tac_agentic_run4_semantic_selected_mi/phase_d_seed_11_score.jsonl \\
  --device auto \\
  --precision fp16
```

Run the full local Phase D matrix once Phase B seed outputs have been pulled:

```python
!python experiments/stage_phase_d_benchmark_suite.py \\
  --output-dir /kaggle/working/tac_phase_d_suite

!python experiments/run_phase_d_benchmark_matrix.py \\
  --suite-dir /kaggle/working/tac_phase_d_suite \\
  --phase-b-dir /kaggle/input/tac-control-v1-phase-b-results \\
  --vanilla-checkpoint /kaggle/input/tac-vanilla-run5b-parameter-matched/vanilla_run5b_parameter_matched/best.pt \\
  --output-dir /kaggle/working/tac_control_v1_phase_d_predictions \\
  --device auto \\
  --precision fp16
```

The matrix runner prefers `specialization_checkpoints/step_010000/checkpoint.pt`
for each TAC seed before falling back to `best.pt` or `last.pt`, then writes a
combined `phase_d_benchmark_rows.jsonl` that
`experiments/aggregate_phase_d_benchmarks.py` discovers automatically from the
default predictions directory.
Instead of generating the suite in `/kaggle/working`, you can attach the
prepared private dataset `jeffkolo/tac-control-v1-phase-d-suite-2026-06-04` and
set `--suite-dir` to its Kaggle input path.
"""


if __name__ == "__main__":
    main()
