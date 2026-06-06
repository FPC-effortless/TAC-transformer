# Distillation Dataset Builder

Date: 2026-06-01

`kaggle/build_distillation_datasets.py` builds the dataset family needed for coding KD, agentic trajectory cloning, execution-repair training, knowledge-work synthesis, DPO, and curriculum sampling.

Current generated artifact:

```bash
python kaggle/build_distillation_datasets.py \
  --output-dir runs/distillation_datasets \
  --train-records-per-stream 1000 \
  --eval-records-per-stream 150 \
  --seed 2026
```

Outputs:

- `runs/distillation_datasets/train.prepared.jsonl`: 7,000 records.
- `runs/distillation_datasets/eval.prepared.jsonl`: 1,050 records.
- `runs/distillation_datasets/preference_pairs.train.jsonl`: 1,000 DPO pairs.
- `runs/distillation_datasets/preference_pairs.eval.jsonl`: 150 DPO pairs.
- `runs/distillation_datasets/*.raw.jsonl`: schema-rich raw records with payloads and training views.
- `runs/distillation_datasets/*.prepared.jsonl`: per-stream prepared text files.
- `runs/distillation_datasets/manifest.json`: counts, stream paths, approximate token counts, and difficulty tiers.

Large no-API artifact:

```bash
python kaggle/build_distillation_datasets.py \
  --output-dir runs/distillation_datasets_70k \
  --train-records-per-stream 10000 \
  --eval-records-per-stream 1000 \
  --seed 2026
```

Outputs:

- `runs/distillation_datasets_70k/train.prepared.jsonl`: 70,000 records.
- `runs/distillation_datasets_70k/eval.prepared.jsonl`: 7,000 records.
- `runs/distillation_datasets_70k/preference_pairs.train.jsonl`: 10,000 DPO pairs.
- `runs/distillation_datasets_70k/preference_pairs.eval.jsonl`: 1,000 DPO pairs.
- Approximate train tokens: 37,159,810.
- Approximate eval tokens: 3,710,484.

Streams:

| Stream | Purpose |
| --- | --- |
| `coding_evol_instruct` | Evol-Instruct-style coding tasks with mutation operators, repo trees, tests, quality gates, Pass@k, and difficulty tiers. |
| `coding_oss_instruct` | OSS-Instruct-style concept extraction from seed code, with generated instructions, solution files, tests, and license-safety notes. |
| `agentic_trajectory` | Public ReAct-style trajectories with Thought, Action, Observation, Reflection, and final success state. |
| `execution_repair` | `[Buggy Code -> Runtime Error -> Root Cause -> Patched Code -> Validation]` tuples. |
| `knowledge_synthesis` | Chunk-grounded multi-turn synthesis records with citations and faithfulness checks. |
| `preference_pair` | Chosen/rejected contrastive pairs for DPO or reward-model training. |
| `curriculum_metadata` | Pass@k-derived difficulty records with structural features and curriculum schedule logits. |

The generator is deterministic and local, so it does not require teacher API credentials. In this project, Codex-authored dataset logic is the teacher path: expand or refine the local payload builders, regenerate the JSONL files, and keep the same raw/prepared file contracts.

Larger local build command:

```bash
python kaggle/build_distillation_datasets.py \
  --output-dir runs/distillation_datasets_70k \
  --train-records-per-stream 10000 \
  --eval-records-per-stream 1000 \
  --seed 2026
```

Training integration:

```bash
python kaggle/train_best_tac_agentic.py \
  --train-jsonl runs/distillation_datasets_70k/train.prepared.jsonl \
  --eval-jsonl runs/distillation_datasets_70k/eval.prepared.jsonl
```

DPO consumers should read:

```text
runs/distillation_datasets_70k/preference_pairs.train.jsonl
runs/distillation_datasets_70k/preference_pairs.eval.jsonl
```
