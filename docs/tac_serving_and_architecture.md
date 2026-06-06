# TAC Serving And Layer Arrangement

## Serving Pipeline

TAC now has a checkpoint-serving path that keeps the current architecture contract:

- tokenizer: UTF-8 byte tokens offset by 4, with EOS token 3
- checkpoint loader: `tac_transformer.serving.load_tac_checkpoint_for_generation`
- generation controls: temperature, top-k, top-p, max new tokens, seed
- CLI generation: `scripts/tac_generate.py`
- optional Gradio GUI: `scripts/tac_gradio_gui.py`
- tokenized corpus memmaps: `scripts/prepare_tac_tokenized_corpus.py`

The tokenizer is intentionally not GPT-2 BPE yet. Current TAC experiments and
checkpoints are trained on byte-level IDs, so GPT-2 BPE would make old
checkpoints incompatible. The memmap builder writes the optimized equivalent of
`train.bin` and `valid.bin` as:

```text
tokenized/
  manifest.json
  train/
    tokens.uint16.bin
    record_offsets.int64.npy
    record_lengths.int32.npy
    category_ids.int16.npy
    categories.json
    manifest.json
  valid/
    tokens.uint16.bin
    record_offsets.int64.npy
    record_lengths.int32.npy
    category_ids.int16.npy
    categories.json
    manifest.json
```

Build tokenized files from prepared JSONL:

```bash
python scripts/prepare_tac_tokenized_corpus.py \
  --train-jsonl runs/prepared_corpus/train.prepared.jsonl \
  --valid-jsonl runs/prepared_corpus/eval.prepared.jsonl \
  --output-dir tokenized \
  --vocab-size 512
```

Generate from a checkpoint:

```bash
python scripts/tac_generate.py \
  --checkpoint runs/TAC-seed\ 37/best.pt \
  --prompt "The quick brown fox" \
  --max-new-tokens 80 \
  --temperature 0.7 \
  --top-k 50 \
  --top-p 0.9
```

Launch the GUI after installing the optional dependency:

```bash
pip install gradio
python scripts/tac_gradio_gui.py --checkpoint runs/TAC-seed\ 37/best.pt
```

## Is TAC GPT-Style?

The transformer backbone is GPT-style in the important architectural sense:

- autoregressive next-token language model
- token embedding plus learned positions or RoPE
- causal self-attention
- residual transformer blocks
- MLP/feed-forward block
- final norm and linear language-model head
- cross-entropy next-token loss when labels are provided

TAC is not a plain GPT block because every TAC block adds an Identity Field Layer
beside the attention path. The identity field routes program embeddings,
maintains per-layer persistent identity state, produces coherence signals, and
can bias or augment attention and hidden states.

## TAC Layer Order

At the language-model level:

```text
input token IDs
  -> token embedding
  -> learned position embedding or RoPE-aware attention positions
  -> TAC block 1
  -> TAC block 2
  -> ...
  -> TAC block N
  -> final norm
  -> lm_head logits
```

Inside one TAC block:

```text
hidden
  -> norm_attention(hidden)
  -> IdentityFieldLayer(normalized, previous_identity_state)
       produces coherence, selected programs, program context, next identity state
  -> causal self-attention(normalized, identity coherence/context/masks)
  -> optional state mixer path, depending on sequence_mixer_type
  -> project identity program context into program_bias
  -> residual update:
       single stream: hidden + attention/state update + program_bias
       dual stream: gated content update + gated identity update
  -> norm_mlp
  -> GELU or SwiGLU feed-forward
  -> residual MLP update
```

The vanilla baseline uses the same token embedding, position handling, causal
attention, MLP, final norm, and LM head, but removes the Identity Field Layer,
program residual path, and persistent identity states.
