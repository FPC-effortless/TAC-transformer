# TAC v0.2 Kaggle Mechanism Reproduction

Date: 2026-06-15

Purpose: rerun TAC-235, TAC-242, and TAC-272 on Kaggle/GPU before starting
real-data LM pilots.

## Runs

Initial full run:

- Kernel: `jeffkolo/tac-v02-mechanism-reproduction-mfull1`
- Artifact root: `runs/kaggle_v02_mechanism_full_mfull1_outputs`
- Result: TAC-242 and TAC-272 validated; TAC-235 did not validate because the
  multi-variant sweep selected `input_program_bottleneck`.

Targeted rerun:

- Kernel: `jeffkolo/tac-v02-mechanism-reproduction-mfull2`
- Artifact root: `runs/kaggle_v02_mechanism_full_mfull2_outputs`
- TAC-235 command used the validated variant explicitly:
  `--variants slot_conditioned_program_bottleneck`.

## Result

| Benchmark | Kaggle Status | Note |
|---|---|---|
| TAC-235 | not_validated | Carry/state survive, but route/program causality misses gates. |
| TAC-242 | validated | Algorithm distillation transfer reproduced. |
| TAC-272 | validated | Causal fix disambiguation reproduced. |

## TAC-235 Local vs Kaggle

Same slot-conditioned command, same aggregate seeds:

| Metric | Local | Kaggle |
|---|---:|---:|
| carry_accuracy | 0.9329 | 0.9259 |
| state_advantage | 0.7315 | 0.6944 |
| internal_route_role_accuracy | 0.8843 | 0.6829 |
| program_knockout_selectivity_gap | 0.5671 | 0.3310 |
| state_slot_knockout_drop | 0.3519 | 0.3148 |

Interpretation:

- TAC-235's persistent-state and answer-accuracy effects reproduce on Kaggle.
- TAC-235's native route/program causal gates do not reproduce under the current
  Kaggle run.
- This is not a clean CPU artifact dismissal. The failure is narrower:
  state/carry survives, but native program coordination is unstable remotely.

Decision:

Do not claim full TAC-235 external reproduction yet. Before treating 112M LM
results as interpretable, either harden TAC-235 remote determinism or explicitly
carry this as a known pre-scaling risk in the stage-gate report.
