# WikiText-2 exploratory TAC-derived language comparison

This experiment trains an independent local-window program model and the upstream nanoGPT implementation from scratch. It was run with the TrainLab runner and imported into Research Lab OS. The study is exploratory, single-seed and unqualified. The Veritas operational-world verifier does not evaluate this language-model task.

## Results

The official WikiText-2 raw train and validation files were byte-tokenized after joining source rows with newline. Each model received 2,560,000 sampled training bytes in 2,500 updates, 16 sequences per batch, 64 bytes per sequence. The validation metric is average next-byte cross-entropy in nats, calculated across 4,485 fixed 64-byte windows with a stride of 256 bytes.

| Model | Parameters | Validation loss | training seconds |
|---|---:|---:|---:|
| Local-window program model | 156,064 | 1.802640 | 34.38 |
| nanoGPT, two layers | 149,472 | 1.872436 | 96.77 |

The program model uses a five-byte local window and 32 learned program vectors. It has 4.4% more parameters. Both models trained on the same random window sequence, optimizer and number of updates. GPU and isolated latency measurements were not performed. Timing is not comparable because the nanoGPT run overlapped another fit and verification on a two-CPU quota. The loss is measured on validation used for research decisions. Full test and multiple seeds are outstanding.

The original single-seed standalone program fit overlapped the nanoGPT fit. A second TrainLab program fit produced the same checkpoint hash and loss. The recorded program duration above is the latter fit. A separate process reloaded each checkpoint and recomputed the metric using the same scorer semantics. This replay is not external independent review. Research Lab imported both verified TrainLab records. Both records explicitly carry NOT_QUALIFIED, UNASSESSED research adequacy, and unverified source rights.

## Reproduce

Use Python 3.12, torch 2.15.0+cpu and pyarrow 26.0.1. Install them in an isolated environment. Run from this directory:

    python fetch_data.py
    export PYTHONPATH=/path/to/veritas/integrations/trainlab/src
    python -m trainlab run gpt-recipe.json --runs runs
    python -m trainlab run program-recipe.json --runs runs
    python verify.py runs/RUN_ID/output/checkpoint.pt

The data files are fetched from Salesforce/wikitext revision b08601e04326c79dfdd32d625aee71d232d685c3, and checked by SHA-256. The source metadata lists CC-BY-SA-3.0 and GFDL. Confirm the upstream license and attribution for your use. Trained checkpoints and optimizer states are currently retained in the local workspace only. Recreate them using the recipes until artifact upload is available. Use trusted checkpoints only.

The code under tac_model.py is taken from TAC-transformer main at 76a7900e93dab282c0788f43bc3e89ae486c0360. nanogpt_model.py is from karpathy/nanoGPT, with the upstream license included. The Veritas ecosystem source used was main at eec439492407b0aa26922c23b5ae40d95289f4eb. The copied TAC model file supports program_lm.py's unused state-mixer class and is included so imports reproduce.

The run receipts have no Git commit because the runs were performed outside a Git checkout. Their source/input SHA-256 values identify the files. The imported Research Lab evidence snapshot is included with the receipts. The lab database remains in the local workspace. These artifacts do not independently validate data rights, benchmark fairness or scientific claims.

## Continuation

The next study should pin multiple seeds and learning-rate sweeps for both models, match compute as well as tokens, use comparable context lengths and parameter counts, and record a new development protocol before using an untouched test split. Evaluate long-context dependencies separately: the selected model's receptive field is five bytes. Replay and independent review should precede any general architecture superiority claim.
