# TAC v0.2 100M+ Kaggle Feasibility

Decision: `feasible_for_kaggle_pilot`

Recommended pilot shape:

| Field | Value |
|---|---:|
| TAC params | 112173136 |
| Vocab size | 8192 |
| d_model | 512 |
| n_heads | 8 |
| n_layers | 8 |
| n_programs | 24 |
| seq_len | 512 |
| attention_window_size | 128 |

Launch command:

```python
!python kaggle/train_best_tac_agentic.py \
  --preset kaggle_fast_tac \
  --scale smoke \
  --vocab-size 8192 \
  --d-model 512 \
  --n-heads 8 \
  --n-layers 8 \
  --n-programs 24 \
  --seq-len 512 \
  --attention-window-size 128 \
  --steps 1000 \
  --batch-size 1 \
  --grad-accum-steps 32 \
  --eval-every 250 \
  --eval-batches 2 \
  --checkpoint-every 250 \
  --precision fp16 \
  --device auto \
  --max-seconds 21600 \
  --stop-buffer-seconds 1200 \
  --skip-end-specialization-on-time-stop \
  --output-dir /kaggle/working/tac_v02_100m_pilot
```

Boundary:

- This is a feasibility and launch-shape estimate.
- It does not validate scale survival.
- TAC v0.2 still requires carried-state, reset-state, vanilla, and knockout controls.
