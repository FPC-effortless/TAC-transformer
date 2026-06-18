# TAC v0.2 Dataset Plan

Goal: train TAC and a matched transformer on the same real-data token budget,
then test persistent state, repair, and compression on held-out tasks.

The dataset is for a real autoregressive language-model run, not another
synthetic benchmark family.

Training streams:

| Stream | Source | Purpose |
|---|---|---|
| Language | `HuggingFaceFW/fineweb-edu` | high-quality LM signal |
| Language | `DKYoon/SlimPajama-6B` | broad web/text mixture |
| Code | `code-search-net/code_search_net` | software repair signal after license audit |
| Long-horizon | generated planning/repair/execution traces | persistent-state pressure |

Validation holdouts:

- persistent-state tasks
- repair tasks
- compression tasks

Rules:

- Never pass `validation_holdout.v02.jsonl` as a training input.
- Run the baseline transformer first.
- Use identical train/eval files, token count, optimizer budget, and checkpoint cadence for TAC.
- Treat CodeSearchNet as pending license review before any public trained checkpoint is released.
- Run a 30M-50M pilot before the full 112M pilot.
- Keep the paper framing on persistent computation survival, not broad claims
  about coding, math, planning, or reasoning.

Builder:

```bash
python scripts/build_v02_datasets.py --output-dir runs/v02_dataset --per-source-limit 100000 --long-horizon-count 50000
```

Offline smoke test:

```bash
python scripts/build_v02_datasets.py --offline-synthetic-only --output-dir runs/v02_dataset_smoke
```
