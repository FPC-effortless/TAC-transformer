# Legacy README Reference

This file preserves the pre-cleanup README content so the public README can become reviewer-oriented without losing prior commands and notes.

---

# TAC-Transformer Identity Field Lab

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

Add `--match-baseline-parameters` to widen the vanilla baseline to the nearest parameter count found by the local estimator.

To run the causal effectiveness scorecard with carry/reset and shuffled-state probes:

```bash
python kaggle/benchmark_effectiveness.py --steps 1 --batch-size 2 --eval-batches 1 --eval-batch-size 2 --probe-batches 2
```

See `docs/effectiveness_benchmark_runbook.md` for the full protocol.

The current best clean architecture from the 2026-05-28 research matrix is available as a preset and one-command benchmark:

```bash
python kaggle/benchmark_best_tac.py --steps 120
```

In code:

```python
from tac_transformer import best_tac_config

config = best_tac_config(vocab_size=64, d_model=64, n_heads=4, n_layers=2)
```

For full historical commands, see the Git history before the reviewer-ready README rewrite.
