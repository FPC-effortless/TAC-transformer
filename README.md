# TAC-Transformer Identity Field Lab

## TAC v0.1 Public Research Package

TAC is an experimental persistent-state architecture for long-horizon AI agents,
with validated mechanisms for memory, compression, control, repair, and causal
fix selection in bounded benchmarks.

This repository does **not** claim that TAC beats transformers. The current
evidence supports a narrower claim: persistent identity state and routed program
modules can become causally useful in controlled local-CPU benchmarks, including
context compression, multi-session continuity, repair control, no-restore
multi-file sandbox repair, and causal fix disambiguation under injected
ambiguity.

Start here:

- [LIMITATIONS.md](LIMITATIONS.md) defines what TAC v0.1 does not prove.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) gives the local and Kaggle validation commands.
- [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) summarizes the TAC-235 through TAC-272 evidence.
- [runs/benchmarks/benchmark_summary_tac235_tac272.md](runs/benchmarks/benchmark_summary_tac235_tac272.md) is the compact benchmark table.

Core validation pack:

```bash
python experiments/kaggle_validate_tac_core.py --benchmarks tac251,tac252,tac267,tac270,tac272 --seeds 5 --cases 50 --output runs/kaggle_validation/tac_core_validation.json
```

Current local validation-pack result:

| Benchmark | CPU Metric | Local Measured | Gate | Tolerance | Decision |
|---|---:|---:|---:|---:|---|
| TAC-251 | 20.0000 | 20.0000 | 20.0000 | 0.0000 | PASS |
| TAC-252 | 20.0000 | 20.0000 | 20.0000 | 0.0000 | PASS |
| TAC-267 | 0.6767 | 0.6756 | 0.6000 | 0.0800 | PASS |
| TAC-270 | 0.9635 | 0.9639 | 0.8500 | 0.0800 | PASS |
| TAC-272 | 0.8417 | 0.8000 | 0.6500 | 0.1000 | PASS |

The local run writes `runs/kaggle_validation/tac_core_validation.json`.

Kaggle validation:

- Kernel: https://www.kaggle.com/code/jeffkolo/tac-v0-1-core-validation-2026-06-13
- Pulled artifact: `runs/kaggle_tac_core_validation_2026_06_13_output/runs/kaggle_validation/tac_core_validation.json`
- Decision: PASS
- `execution_environment`: `kaggle`
- `validated_on_kaggle`: true for TAC-251, TAC-252, TAC-267, TAC-270, and TAC-272

Expected public claim:

> TAC is an experimental persistent-state architecture for long-horizon AI
> agents, with validated mechanisms for memory, compression, control, repair,
> and causal fix selection in bounded benchmarks.

Interactive Vite/React prototype for testing an Identity Field Layer beside attention.

The current prototype is deterministic and inspectable rather than trained. It demonstrates:

- scaled dot-product baseline attention
- coherence-modulated attention: `softmax(QK^T / sqrt(d) + beta * C)`
- executable program embeddings with stability and energy cost
- sparse program routing under an energy budget
- compressed identity memory traces
- sleep-phase consolidation that merges stable repeated traces and prunes unstable ones

The repository also includes a real trainable PyTorch implementation in `tac_transformer/`. See `docs/tac_transformer_architecture.md` for the module layout and loss wiring.

## Parameter Counts

The default Kaggle training config in `kaggle/train_tac_synthetic.py` has:

- total parameters: `150,002,688`
- trainable parameters: `150,002,688`
- identity-field parameters: `10,630,656`

The default config is `vocab_size=10624`, `d_model=768`, `n_heads=12`, `n_layers=16`, `n_programs=96`, and `max_seq_len=256`.

A smaller local smoke config can be passed explicitly:

```bash
python kaggle/train_tac_synthetic.py --steps 2 --batch-size 4 --d-model 32 --n-layers 1 --n-heads 4 --n-programs 8 --vocab-size 64 --seq-len 17
```

To compare TAC against a vanilla transformer with the same backbone dimensions:

```bash
python kaggle/benchmark_tac_vs_baseline.py --steps 120
```

Add `--match-baseline-parameters` to widen the vanilla baseline to the nearest
parameter count found by the local estimator.

To run the causal effectiveness scorecard with carry/reset and shuffled-state probes:

```bash
python kaggle/benchmark_effectiveness.py --steps 1 --batch-size 2 --eval-batches 1 --eval-batch-size 2 --probe-batches 2
```

See `docs/effectiveness_benchmark_runbook.md` for the full protocol.

Backbone modernization is opt-in so legacy results stay comparable:

```bash
python kaggle/benchmark_effectiveness.py --norm-type rmsnorm --mlp-type swiglu --position-type rope --n-kv-heads 2 --program-compute-type linear_expert --state-update-type gated --steps 120
```

To test explicit cross-chunk memory instead of only short next-token behavior:

```bash
python kaggle/benchmark_chunked_memory.py --norm-type rmsnorm --mlp-type swiglu --position-type rope --n-kv-heads 2 --program-compute-type linear_expert --state-update-type gated --memory-write-type novelty_gated --identity-attention-type compressed_memory --value-loss-weight 3.0 --memory-read-type program_memory --memory-read-loss-weight 3.0 --memory-injection-weight 1.0 --steps 120
```

The current best clean architecture from the 2026-05-28 research matrix is available as a preset and one-command benchmark:

```bash
python kaggle/benchmark_best_tac.py --steps 120
```

In code:

```python
from tac_transformer import best_tac_config

config = best_tac_config(vocab_size=64, d_model=64, n_heads=4, n_layers=2)
```

That preset uses RMSNorm, SwiGLU, RoPE, grouped-query attention, linear program experts, BASE-style balanced routing, gated state updates, novelty-gated memory writes, program-memory readout, and the gated residual memory adapter. It intentionally leaves hierarchical memory, product-key memory, sink programs, compressed identity attention, dual streams, CREB allocation, and multi-token prediction as ablations because the harder matrix showed the all-features stack and CREB variants underperformed the simpler BASE-routed TAC.

See `docs/best_tac_architecture.md` for task context, metric definitions, seed statistics, routing justification, and current limitations.

Agentic action/world/reward/reflection experiments are tracked in `docs/agentic_objective_experiments.md`. Current result: direct memory-conditioned action policy is the best tested agentic adapter, but it is not reliable enough to promote into the default architecture yet.

For model-native memory integration, use the residual adapter instead of direct logit injection:

```bash
python kaggle/benchmark_chunked_memory.py --norm-type rmsnorm --mlp-type swiglu --position-type rope --n-kv-heads 2 --program-compute-type linear_expert --state-update-type gated --memory-write-type novelty_gated --identity-attention-type compressed_memory --value-loss-weight 3.0 --memory-read-type program_memory --memory-read-loss-weight 3.0 --memory-adapter-type residual --memory-adapter-weight 1.0 --steps 120
```

The higher-capacity gated adapter is the stronger clean path found so far:

```bash
python kaggle/benchmark_chunked_memory.py --norm-type rmsnorm --mlp-type swiglu --position-type rope --n-kv-heads 2 --program-compute-type linear_expert --state-update-type gated --memory-write-type novelty_gated --identity-attention-type compressed_memory --value-loss-weight 3.0 --memory-read-type program_memory --memory-read-loss-weight 3.0 --memory-adapter-type gated_residual --memory-adapter-weight 4.0 --steps 120
```

To measure data and compute-proxy efficiency across multiple training budgets:

```bash
python kaggle/benchmark_efficiency.py --budgets 20 60 120 --norm-type rmsnorm --mlp-type swiglu --position-type rope --n-kv-heads 2 --program-compute-type linear_expert --state-update-type gated --memory-write-type novelty_gated --value-loss-weight 3.0 --memory-read-type program_memory --memory-read-loss-weight 3.0 --memory-adapter-type gated_residual --memory-adapter-weight 6.0 --match-baseline-parameters
```

Use `--n-sink-programs` as a StreamingLLM-style ablation switch; the first sink sweep kept all runs effective but did not beat the no-sink reference.

A larger small-LLM-style config with `vocab_size=32000`, `d_model=512`, `n_heads=8`, `n_layers=8`, `n_programs=64`, and `max_seq_len=512` has:

- total parameters: `62,715,392`
- trainable parameters: `62,715,392`
- identity-field parameters: `2,363,904`

To count any custom config, use:

```bash
python kaggle/train_tac_synthetic.py --steps 0
```

or import `count_parameters` from `tac_transformer.training`.

## Kaggle Training

The Kaggle-ready trainer is:

```bash
python kaggle/train_tac_synthetic.py --device auto --steps 1000
```

It writes a checkpoint to `/kaggle/working/tac_transformer.pt` when running on Kaggle, or `runs/tac_transformer.pt` locally.

Dataset preparation notes live in `docs/dataset_preparation.md`. The prep loader drops missing answers, caps duplicates, serializes rows into training text, and supports both JSONL and JSON-array datasets.

## Serving And Tokenized Corpus

TAC checkpoint serving uses the existing byte-token training contract rather than GPT-2 BPE, so current checkpoints remain compatible. The reusable serving API is in `tac_transformer/serving.py`.

Build optimized tokenized train/valid memmaps:

```bash
python scripts/prepare_tac_tokenized_corpus.py --train-jsonl runs/prepared_corpus/train.prepared.jsonl --valid-jsonl runs/prepared_corpus/eval.prepared.jsonl --output-dir tokenized --vocab-size 512
```

Generate from a checkpoint:

```bash
python scripts/tac_generate.py --checkpoint runs/TAC-seed\ 37/best.pt --prompt "The quick brown fox" --max-new-tokens 80 --temperature 0.7 --top-k 50 --top-p 0.9
```

Launch the optional Gradio GUI:

```bash
pip install gradio
python scripts/tac_gradio_gui.py --checkpoint runs/TAC-seed\ 37/best.pt
```

See `docs/tac_serving_and_architecture.md` for the TAC layer arrangement and the GPT-style backbone comparison.

## Development

```bash
npm install
npm run dev
```

## Checks

```bash
npm test
npm run test:python
npm run test:js
npm run lint
npm run build
```

The browser prototype core lives in `src/lib/identityField.js`; the trainable PyTorch implementation lives in `tac_transformer/model.py`.
