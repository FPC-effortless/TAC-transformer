# TAC Effectiveness Benchmark Runbook

Use this harness before accepting major architecture changes from the research roadmap. A change is only interesting if it improves the scorecard for the reason it was designed to improve it.

## Quick Smoke

```bash
python kaggle/benchmark_effectiveness.py --steps 1 --batch-size 2 --eval-batches 1 --eval-batch-size 2 --probe-batches 2
```

The command prints a JSON scorecard with:

- TAC and vanilla parameter counts.
- short-context language-model metrics.
- carry-vs-reset identity-state probe.
- correct-state-vs-shuffled-state probe.
- routing metrics: used energy, active programs, and routing entropy.
- an explicit `decision.status`: `effective` or `inconclusive`.

## Research Run

```bash
python kaggle/benchmark_effectiveness.py \
  --steps 120 \
  --batch-size 32 \
  --eval-batches 8 \
  --eval-batch-size 32 \
  --probe-batches 8 \
  --match-baseline-parameters \
  --output runs/benchmarks/effectiveness_seed7.json
```

Repeat with at least three seeds before treating a result as meaningful.

To test the opt-in modern backbone slice:

```bash
python kaggle/benchmark_effectiveness.py \
  --steps 120 \
  --batch-size 32 \
  --eval-batches 8 \
  --eval-batch-size 32 \
  --probe-batches 8 \
  --norm-type rmsnorm \
  --mlp-type swiglu \
  --position-type rope \
  --n-kv-heads 2 \
  --program-compute-type linear_expert \
  --state-update-type gated \
  --match-baseline-parameters \
  --output runs/benchmarks/effectiveness_modern_seed7.json
```

Keep a legacy run beside the modern run. If both TAC and vanilla improve by the same amount, the gain belongs to the backbone, not the identity field.

## Chunked Memory Run

```bash
python kaggle/benchmark_chunked_memory.py \
  --steps 120 \
  --batch-size 32 \
  --eval-batches 8 \
  --eval-batch-size 32 \
  --norm-type rmsnorm \
  --mlp-type swiglu \
  --position-type rope \
  --n-kv-heads 2 \
  --program-compute-type linear_expert \
  --state-update-type gated \
  --memory-write-type novelty_gated \
  --identity-attention-type compressed_memory \
  --value-loss-weight 3.0 \
  --memory-read-type program_memory \
  --memory-read-loss-weight 3.0 \
  --memory-injection-weight 1.0 \
  --match-baseline-parameters \
  --output runs/benchmarks/chunked_memory_seed7.json
```

To test the model-native residual adapter instead of direct logit injection, replace `--memory-injection-weight 1.0` with:

```bash
--memory-adapter-type residual --memory-adapter-weight 1.0
```

To test the stronger gated residual adapter, use:

```bash
--memory-adapter-type gated_residual --memory-adapter-weight 4.0
```

This task creates paired context/query chunks. The context introduces a key/value pair; the query asks the model to predict the value after seeing the same key. The value-token target is scored separately from ordinary language-model loss.

`--identity-attention-type compressed_memory` is the TAC-native compressed-attention path. It exposes carried identity `program_memory` as compressed key/value slots beside the normal causal token keys, borrowing the long-range compressed-attention idea while keeping TAC state as the source of memory.

`--memory-write-type novelty_gated` is the Titans-inspired write path. It adds a trainable gate over candidate and previous program memory so the state update can learn when a memory write is worth preserving.

## How To Read The Scorecard

- `short_loss_ratio`: TAC short-context loss divided by vanilla loss. Lower is better. A value above the configured tolerance means TAC is hurting ordinary language modeling.
- `memory_carry_delta`: reset loss minus carry loss. Positive means carried identity state helped.
- `state_shuffle_penalty`: shuffled-state loss minus carry loss. Positive means the correct memory helped more than another sample's memory.
- `routing_entropy`: whether program activations are distributed or collapsing.
- `active_programs`: average selected programs per sample under the energy budget.

The default `effective` decision requires `memory_carry_delta` and `state_shuffle_penalty` to exceed `1e-4`, which keeps tiny numerical noise from counting as evidence.

## Acceptance Bar

Treat a result as promising only when:

- TAC does not regress short-context loss beyond tolerance.
- carry beats reset.
- correct state beats shuffled state.
- the result holds across multiple seeds.
- active compute or memory improves when sparse/compressed architecture features are added.

An `inconclusive` decision is not failure. It means the architecture has not yet proven the specific mechanism under test.
