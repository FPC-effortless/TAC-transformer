# Kaggle Ready Synthesis Handoff

Date: 2026-05-30

Superseded on 2026-05-31 by the identity-first Run 3 handoff in `docs/kaggle_identity_first_run3_preflight.md`. Keep this note only as historical context for the synthesis-gated launch.

## Upload Files

The prepared Kaggle upload folder is:

```text
runs/kaggle_ready_synthesis_2026_05_30
```

It contains:

```text
best-tac-agentic-training-bundle.zip
prepared_corpus_agentic_hard_upload.zip
train_best_tac_agentic_synthesis.ipynb
```

Upload `best-tac-agentic-training-bundle.zip` as a Kaggle Dataset for the code.
Upload `prepared_corpus_agentic_hard_upload.zip` as a second Kaggle Dataset named
`tac-hard-agentic-corpus`.
Import `train_best_tac_agentic_synthesis.ipynb` as the Kaggle notebook.

## Model

The bundle uses the promoted `content_synthesis_k1` preset:

```text
BASE routing
+ content-addressed cue/value memory
+ content_read_steps = 2
+ content_read_gate_type = synthesis
+ gated residual memory injection
```

## Training Command

In a Kaggle notebook with two T4 GPUs enabled:

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

!torchrun --standalone --nproc_per_node=2 kaggle/train_best_tac_agentic.py \
  --scale base \
  --steps 6000 \
  --warmup-steps 300 \
  --batch-size 12 \
  --grad-accum-steps 3 \
  --eval-every 500 \
  --eval-batches 8 \
  --checkpoint-every 250 \
  --output-dir /kaggle/working/best_tac_agentic_anticollapse \
  --device auto \
  --precision fp16 \
  --max-seconds 30600 \
  --stop-buffer-seconds 1200 \
  --auto-resume
```

This keeps the original effective batch size of 72 sequences:

```text
12 per GPU * 3 grad accumulation * 2 GPUs = 72
```

The previous run reached about 93% eval accuracy by step 5000, so `--steps 6000`
is the default target for the anti-collapse run. Start with this faster profile.
If Kaggle runs out of memory, use `--batch-size 8 --grad-accum-steps 4`.

If Kaggle changes the dataset folder name, locate the paths with:

```python
!find /kaggle/input -name "train.prepared.jsonl" -o -name "eval.prepared.jsonl"
```

The trainer searches `/kaggle/input` recursively, so the launch command does not
need hardcoded `--train-jsonl` or `--eval-jsonl` paths unless you attach multiple
prepared corpora.

## Output Files

Download these from Kaggle output:

```text
/kaggle/working/best_tac_agentic_anticollapse/last.pt
/kaggle/working/best_tac_agentic_anticollapse/best.pt
/kaggle/working/best_tac_agentic_anticollapse/metrics.jsonl
/kaggle/working/best_tac_agentic_anticollapse/run_manifest.json
/kaggle/working/best_tac_agentic_anticollapse/final_summary.json
```

## Diagnostics Tracked

The trainer now logs the diagnostics needed for repair decisions, not just loss:

```text
loss / next_token_loss / aux_loss / eval.loss / eval.accuracy / eval.perplexity
aux_loss_* and weighted_aux_loss_* for every auxiliary objective
metric_* for every scalar IdentityState metric emitted by the model
program_memory_cosine
content_addressed_hit
content_synthesis_gate
content_gate_entropy
content_cue_cosine
content_reconsolidation_gate
metric_memory_allocation_dead_rate
metric_memory_allocation_load_std
metric_memory_allocation_write_frequency
metric_routing_load_std
metric_active_expert_fraction
gradient_norm / grad_scaler_scale
cuda_memory_allocated_mib / cuda_memory_reserved_mib
cuda_max_memory_allocated_mib / cuda_max_memory_reserved_mib
tokens_seen / sequences_seen / epoch_equivalent
tokens_per_second / elapsed_seconds
```

The critical anti-collapse checks during the next run are:

```text
program_memory_cosine should move down from the old collapsed ~0.967 level
content_synthesis_gate should not pin near 1.0 for the whole run
content_gate_entropy should stay meaningfully above zero
content_addressed_hit should not steadily decay as the model trains
cuda_max_memory_reserved_mib should stay comfortably below T4 capacity
gradient_norm should not spike persistently against the clip threshold
grad_scaler_scale should not repeatedly collapse during fp16 training
```

## Resume

The launch command includes `--auto-resume`.

- Same Kaggle session: rerun the command with the same `--output-dir`; it resumes from `/kaggle/working/best_tac_agentic_anticollapse/last.pt`.
- New Kaggle session: save the prior notebook output, attach that previous output as an input dataset, then rerun the same command. The trainer searches attached inputs for `last.pt`, then `best.pt`.
- Do not attach the old collapsed `best_tac_agentic` output when starting this anti-collapse run fresh.
- Explicit resume path is also supported:

```python
--resume /kaggle/input/previous-best-tac-agentic-output/last.pt
```

## Local Verification

The rebuilt code bundle was extracted locally and smoke-trained against the hard corpus eval split.
The run completed, wrote `last.pt` and `best.pt`, and confirmed the manifest uses:

```text
memory_read_type = content_addressed
content_store_size = 8
content_read_steps = 2
content_read_gate_type = synthesis
routing_type = base
```
