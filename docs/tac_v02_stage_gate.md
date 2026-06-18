# TAC v0.2 Stage Gate

Current assessment:

TAC is no longer just an architecture idea. It is an experimental
proof-of-concept with validated mechanisms and unproven scaling behavior. The
best concise stage label is `mechanism validated, scaling unproven` or TRL 3-4.

Core question:

> When TAC is scaled to about 112M parameters and trained on real data, do
> persistent state, repair planning, and compression advantages still exist?

Required comparison:

| Metric | Transformer 50M | TAC 50M |
|---|---:|---:|
| Best LM eval loss | 1.0611 | 1.4999 |
| Final LM eval loss | 1.4505 | 1.6570 |
| Final LM perplexity | 4.2651 | 5.2435 |
| Runtime seconds | 1,757.34 | 24,932.19 |
| TAC-280 mechanism family wins | baseline | 3 / 4 |
| TAC-280 carry-positive families | n/a | 2 / 4 |
| TAC-280 bottleneck knockout delta | n/a | 5.4773 |
| Compression / structure reuse loss | 2.3750 | 2.2062 |
| Repair trace reuse loss | 2.2357 | 1.9367 |
| Persistent-state carry loss | 5.3479 | 6.9881 |

112M comparison remains pending. The attempted Kaggle kernel
`jeffkolo/tac-v02-lm-112m-pilot-lm112a` is reported by Kaggle as
`KernelWorkerStatus.CANCEL_ACKNOWLEDGED`, so it is not evidence for or against
112M mechanism survival. The 50M pilot is negative for plain LM efficiency and
positive-but-uneven for checkpoint mechanism retention.

TAC-281 checkpoint variants:

| Variant | Best Eval Loss | LM Gap Shrink | Speed Penalty | Mechanism Wins | Carry Families | Knockout Delta | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `small_adapter` | 1.082974 | 95.01% | 4.79x | 3 / 4 | 3 / 4 | 0.000021 | fail |
| `late_bottleneck` | 1.534723 | -7.94% | 6.96x | 3 / 4 | 3 / 4 | 5.7775 | fail |
| `auxiliary_mechanism` | 1.507470 | -1.73% | 13.22x | 3 / 4 | 2 / 4 | 5.5232 | fail |

TAC-281 completed with `do_not_scale_yet`: no 30M-50M variant passed all checks
needed to justify another 112M run.

Continue only if:

- TAC training does not diverge.
- routing entropy does not collapse to a single expert.
- program utilization remains non-trivial.
- memory utilization and carry metrics remain positive.
- TAC preserves a measurable advantage on persistent-state, repair, or 20x compression holdouts.

Stop scaling if:

- TAC loses the matched LM comparison and the mechanism holdouts.
- routing or state collapse explains the loss.
- optimization interference prevents a clean matched-token comparison.

Execution order:

1. Integrate TAC-235 so the default LM path forces generation through
   state-conditioned native program computation.
2. Reproduce TAC-235, TAC-242, and TAC-272 on Kaggle/GPU to rule out local CPU
   artifacts.
3. Run the 30M-50M real autoregressive LM pilot.
4. Run TAC-280 checkpoint mechanism retests over the trained transformer and
   TAC checkpoints.
5. Review TAC-280 robustness before launching any bigger 112M run.
6. Run TAC-281 mechanism sharpening at 30M-50M: late bottleneck, small adapter,
   and auxiliary-mechanism TAC.
7. Do not run another 112M transformer/TAC pilot until a 30M-50M variant passes
   the TAC-281 gate.
8. Write the scaling-retention report around the question: do persistent
   computational states survive scale?

Do not add new benchmark families until these gates are answered.

V02-006 implementation:

- `TACConfig.lm_readout_type` now supports
  `slot_conditioned_program_bottleneck`.
- `configs/tac_v02_50m.py` and `configs/tac_v02_112m.py` opt into that readout.
- For opted-in TAC models, final LM logits are produced from selected native
  program expert outputs conditioned on each program memory slot, then passed
  through the normal `lm_head`.
- The previous hidden-state readout remains available as the `hidden` ablation.

V02-007 status:

- TAC-242 and TAC-272 reproduced on Kaggle/GPU.
- TAC-235 is only partially reproduced: carry accuracy and state advantage
  survive, but internal route-role accuracy and native program knockout
  selectivity miss the validation gates on Kaggle.
- See `docs/tac_v02_kaggle_mechanism_reproduction.md`.

V02-008 status:

- The 30M-50M real-data LM pilot completed on Kaggle as
  `jeffkolo/tac-v02-lm-50m-pilot-lm50a`.
- Transformer and TAC both trained for 2,000 steps on the same v0.2 train/eval
  files.
- The matched transformer beat TAC on best eval loss and final perplexity.
- TAC used the native program bottleneck readout during the run.
- See `docs/tac_v02_lm50_pilot_report.md`.

V02-011 / TAC-280 status:

- Checkpoint mechanism retests completed on Kaggle as
  `jeffkolo/tac-v02-checkpoint-retests-tac280a`.
- The decision status is `mechanism_advantage`: TAC wins 3 of 4 mechanism
  families by loss against the transformer full-context baseline, carry-state
  is positive overall, and native program bottleneck knockout sharply degrades
  TAC.
- The result is positive but uneven: TAC loses aggregate average loss because
  persistent-state carry remains weaker than transformer full context, and
  carry-vs-reset is positive in only 2 of 4 families.
- See `docs/tac_v02_tac280_checkpoint_retests.md`.

V02-012 / TAC-281 status:

- Mechanism-sharpening infrastructure is implemented for three 30M-50M
  variants: `tac_50m_late_bottleneck`, `tac_50m_small_adapter`, and
  `tac_50m_auxiliary_mechanism`.
- TAC-281 is complete. The combined decision is `do_not_scale_yet` with zero
  scale-ready variants.
- `small_adapter` completed on Kaggle as
  `jeffkolo/tac-v02-tac281-small-adapter-tac281a-small` and failed the scale
  gate because bottleneck knockout delta was effectively zero.
- `late_bottleneck` first failed on Kaggle as
  `jeffkolo/tac-v02-tac281-late-bottleneck-tac281a-late` due an inactive
  identity-context shape bug, so that run is not a scientific variant result.
  The fixed rerun completed as
  `jeffkolo/tac-v02-tac281-late-bottleneck-tac281b-latefix` and failed because
  its LM gap worsened despite mechanism retention.
- `auxiliary_mechanism` completed on Kaggle as
  `jeffkolo/tac-v02-tac281-auxiliary-mechanism-tac281a` and failed because its
  LM gap did not shrink enough despite mechanism retention.
- See `docs/tac_v02_tac281_mechanism_sharpening.md` and
  `docs/tac_v02_tac281_final_gate.md`.
