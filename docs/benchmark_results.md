# TAC vs Vanilla Baseline Benchmarks

Date: 2026-05-26

Environment:

- Device: CPU
- PyTorch: local environment
- Task: `SyntheticProgramBatcher`
- TAC config: `vocab_size=64`, `d_model=64`, `n_heads=4`, `n_layers=2`, `n_programs=16`, `seq_len=16`
- Matched vanilla config: `d_model=68`, `n_heads=4`, `n_layers=2`

## 120-Step Same-Backbone Run

Output: `runs/benchmarks/synthetic_same_backbone_120.json`

| Model | Params | Final Loss | Final PPL | Accuracy | Train tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| TAC | 128,032 | 3.3187 | 27.6249 | 0.3003 | 8,193.9 |
| Vanilla | 109,312 | 3.2649 | 26.1781 | 0.3101 | 820.9 |

Same-backbone comparison gives TAC extra identity-field parameters, but vanilla still has lower final loss and higher accuracy on this run.

## 120-Step Parameter-Matched Runs

Outputs:

- `runs/benchmarks/synthetic_matched_120_seed11.json`
- `runs/benchmarks/synthetic_matched_120.json`
- `runs/benchmarks/synthetic_matched_120_seed37.json`

| Seed | TAC Loss | Vanilla Loss | TAC Accuracy | Vanilla Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 11 | 3.2467 | 3.0940 | 0.3279 | 0.3750 |
| 23 | 3.3187 | 3.0701 | 0.3003 | 0.3733 |
| 37 | 3.3823 | 3.0985 | 0.2527 | 0.3757 |
| Average | 3.3159 | 3.0875 | 0.2936 | 0.3747 |

Average perplexity:

- TAC: 27.5898
- Vanilla: 21.9247

Average training throughput:

- TAC: 9,544.7 tokens/sec
- Vanilla: 12,724.6 tokens/sec

## Readout

Current TAC is trainable and causally correct, but on this synthetic task it does not yet beat a simpler vanilla transformer. The parameter-matched vanilla baseline wins consistently across three seeds on final loss, perplexity, and accuracy.

The next architectural work should focus on proving that the identity field adds a capability the baseline lacks, not only adding capacity. Good candidates:

- sequence-level tasks where persistent `identity_states` are carried across chunks;
- longer-horizon synthetic tasks with repeated latent programs;
- ablations that remove program context, coherence bias, or state persistence one at a time;
- optimizing the causal identity path before scaling.
