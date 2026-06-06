# Kaggle Benchmark Runbook

The Kaggle kernel bundle lives at:

```text
runs/kaggle_benchmark_kernel
```

It contains:

- `benchmark_kernel.py`
- `kernel-metadata.json`
- a local copy of `tac_transformer/`
- `MANUAL_UPLOAD.md`

A ready-to-upload zip is generated at:

```text
runs/kaggle_benchmark_kernel/tac-transformer-kaggle-bundle.zip
```

## Manual Website Run

1. Go to Kaggle and create a new Dataset.
2. Upload `runs/kaggle_benchmark_kernel/tac-transformer-kaggle-bundle.zip`.
3. Create a new Kaggle Notebook.
4. Enable GPU in Notebook settings.
5. Add your uploaded dataset to the notebook.
6. Run:

```python
import os

os.environ["TAC_BENCHMARK_STEPS"] = "300"
os.environ["TAC_BENCHMARK_BATCH_SIZE"] = "64"
os.environ["TAC_BENCHMARK_EVAL_BATCHES"] = "16"
os.environ["TAC_BENCHMARK_SEEDS"] = "11,23,37"

!rm -rf /kaggle/working/tac_benchmark
!mkdir -p /kaggle/working/tac_benchmark
!unzip -q /kaggle/input/tac-transformer-kaggle-bundle/tac-transformer-kaggle-bundle.zip -d /kaggle/working/tac_benchmark
%cd /kaggle/working/tac_benchmark
!python benchmark_kernel.py
```

If Kaggle gives your dataset a different folder name under `/kaggle/input`, adjust that path in the `unzip` command.

Download this file from the Kaggle output panel:

```text
/kaggle/working/tac_vs_baseline_benchmark.json
```

## Credential Status

Local Kaggle authentication was tested on 2026-05-26. The CLI reached Kaggle, but Kaggle returned `401 Unauthorized` because `~/.kaggle/kaggle.json` has an empty `key` value.

Generate a new Kaggle API token from Kaggle account settings, then replace:

```text
C:\Users\warit\.kaggle\kaggle.json
```

The file must contain both `username` and a non-empty `key`.

## Push And Run

Use the installed Kaggle CLI path:

```powershell
& "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\Scripts\kaggle.exe" kernels push -p runs/kaggle_benchmark_kernel --accelerator gpu
```

The kernel id is:

```text
jeffwilliamsr/tac-transformer-benchmark
```

## Check Status

```powershell
& "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\Scripts\kaggle.exe" kernels status jeffwilliamsr/tac-transformer-benchmark
```

## Download Results

```powershell
New-Item -ItemType Directory -Force -Path runs/kaggle_results | Out-Null
& "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\Scripts\kaggle.exe" kernels output jeffwilliamsr/tac-transformer-benchmark -p runs/kaggle_results
```

Expected output file:

```text
tac_vs_baseline_benchmark.json
```

## Benchmark Defaults

The Kaggle script runs a parameter-matched TAC vs vanilla comparison on seeds `11,23,37`.

Defaults:

- `TAC_BENCHMARK_STEPS=300`
- `TAC_BENCHMARK_BATCH_SIZE=64`
- `TAC_BENCHMARK_EVAL_BATCHES=16`
- `TAC_BENCHMARK_SEEDS=11,23,37`

These can be overridden with Kaggle environment variables if needed.
