# TAC Long-Context and Subword Experiment Report

Date: 2026-06-06

## Question

Can TAC's memory make a small context window behave like long context, and what is
the efficient path toward context large enough for agentic and knowledge-work
conversations?

## Short Answer

TAC memory should help long context, but the current checkpoint does not prove that
it can replace a real context window. In the local screens, carried memory was weak
and sometimes inconclusive. The best path is:

1. Move training data to tokenized memmaps.
2. Add a real subword tokenizer.
3. Train context length progressively.
4. Use local attention plus TAC memory for 4k+.
5. Add retrieval and conversation compaction for knowledge work.

## Public Frontier Technique Map

Publicly visible frontier long-context systems combine several methods:

- Subword tokenization: more language per token than byte tokens.
- RoPE or position-scaling variants: Position Interpolation, YaRN, LongRoPE, or
  internal variants.
- Continuation training on long sequences: config-only extension is not enough.
- Efficient attention/runtime: FlashAttention, local/global attention, GQA/MQA,
  distributed attention, or sparse attention.
- Recurrence/memory: Transformer-XL-style recurrence, Infini-attention-style
  compression, retrieval memory, and tool/session state.
- Context engineering: prompt caching, summaries, retrieval, file search, durable
  notes, and compaction.

Sources:

- Gemini 1.5 technical report: https://arxiv.org/abs/2403.05530
- Gemini long-context docs: https://ai.google.dev/gemini-api/docs/long-context
- OpenAI GPT-4.1 docs: https://platform.openai.com/docs/models/gpt-4.1
- OpenAI GPT-4.1 prompting guide: https://cookbook.openai.com/examples/gpt4-1_prompting_guide
- Anthropic context windows: https://docs.anthropic.com/en/docs/build-with-claude/context-windows
- Anthropic context engineering: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Llama 3 Herd paper: https://arxiv.org/abs/2407.21783
- Position Interpolation: https://arxiv.org/abs/2306.15595
- YaRN: https://arxiv.org/abs/2309.00071
- LongRoPE: https://arxiv.org/abs/2402.13753
- FlashAttention: https://arxiv.org/abs/2205.14135
- Longformer: https://arxiv.org/abs/2004.05150
- Transformer-XL: https://arxiv.org/abs/1901.02860
- Infini-attention: https://arxiv.org/abs/2404.07143
- SentencePiece: https://arxiv.org/abs/1808.06226

## Local Results

### 1. Config-only context extension failed

Output:

`runs/TAC-seed 20/context_extension_rope_screen.json`

| Setting | Byte accuracy | NLL | Perplexity |
| --- | ---: | ---: | ---: |
| trained 256 / none | 0.7952 | 0.9011 | 2.46 |
| 512 / no scaling | 0.5247 | 2.4538 | 11.63 |
| 512 / linear RoPE | 0.2332 | 5.0934 | 162.95 |
| 512 / yarn-like RoPE | 0.2242 | 5.0498 | 155.98 |
| 1024 / no scaling | 0.3886 | 3.8864 | 48.74 |
| 1024 / linear RoPE | 0.1689 | 5.8341 | 341.76 |
| 1024 / yarn-like RoPE | 0.1540 | 5.7030 | 299.76 |

Interpretation: do not just change `max_seq_len`. The current 256-byte checkpoint
needs continuation training before it can use 512 or 1024 safely.

### 2. Tokenized memmap helps data throughput

Output:

`runs/TAC-seed 20/long_context_efficiency_screen_2026_06_06/long_context_efficiency_local.json`

Memmap speedup over online JSONL byte encoding:

| Seq len | Speedup |
| ---: | ---: |
| 256 | 3.93x |
| 512 | 12.38x |
| 1024 | 47.56x |

Interpretation: long-context training should use pretokenized memmaps, not online
JSONL parsing and byte encoding.

### 3. Local attention reduces theoretical attention cost

With window 128:

| Seq len | Useful local attention / dense |
| ---: | ---: |
| 256 | 0.500 |
| 512 | 0.250 |
| 1024 | 0.125 |

Interpretation: local attention is necessary for 4k+, but the current CPU path does
not always translate the theoretical reduction into wall-clock speed. It is still
the correct architecture direction for GPU/Kaggle scaling.

### 4. TAC memory carry is not strong enough yet

Output:

`runs/TAC-seed 20/memory_carry_long_context_screen.json`

Small delayed-query memory screen:

| Variant | Carry | Reset | Shuffled | Baseline | Carry - Reset |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense 128 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| local 128 w64 | 0.000 | 0.062 | 0.000 | 0.000 | -0.062 |
| dense 256 | 0.062 | 0.000 | 0.000 | 0.000 | 0.062 |
| local 256 w128 | 0.062 | 0.000 | 0.000 | 0.062 | 0.062 |

Interpretation: TAC memory is promising but not validated as a long-context
replacement. It must be trained and gated with carry > reset and carry > shuffled.

### 5. Subword tokenization gives the biggest context multiplier

Output:

`runs/TAC-seed 20/subword_tokenizer_migration_screen.json`

Eval corpus compression relative to byte tokens:

| Tokenization proxy | Tokens / byte tokens |
| --- | ---: |
| regex subword proxy | 0.329 |
| chars/4 BPE estimate | 0.251 |
| word-like tokens only | 0.104 |

Record fit fraction:

| Context | Byte tokens | Regex subword proxy | Chars/4 BPE estimate |
| ---: | ---: | ---: | ---: |
| 1024 | 0.644 | 0.995 | 0.999 |
| 2048 | 0.990 | 0.999 | 0.999 |
| 4096 | 0.999 | 1.000 | 1.000 |

Parameter cost at `d_model=256`:

| Vocab | Estimated total params if only vocab changes | Embedding+head fp16 |
| ---: | ---: | ---: |
| 512 | 20.99M | 0.5 MiB |
| 4096 | 22.83M | 4.0 MiB |
| 8192 | 24.93M | 8.0 MiB |
| 16384 | 29.12M | 16.0 MiB |
| 32768 | 37.51M | 32.0 MiB |
| 65536 | 54.29M | 64.0 MiB |

Interpretation: a 16k vocab is the best first serious TAC subword target. It adds
about 8.1M parameters versus the current 512-byte vocabulary, but roughly triples
or quadruples useful language per context slot.

## Recommended Next Training Ladder

### Phase A: Byte-token long-context bridge

Goal: prove TAC can train beyond 256 without quality collapse.

- Start from current architecture, not necessarily current weights.
- Train at mixed lengths: 128, 256, 512, 1024.
- Use tokenized memmaps.
- Evaluate short-context and long-prefix final-answer continuation.
- Gate: 1024-byte model must beat 256-truncation on long-prefix tasks without
  regressing short-context accuracy by more than 2 points.

### Phase B: Subword TAC v1

Goal: make context size meaningful for language.

- Add a tokenizer abstraction: `TacTokenizer.encode`, `decode`, `vocab_size`,
  `special_token_ids`.
- Train a 16k BPE or Unigram tokenizer on the TAC corpus.
- Store tokenized memmaps with `tokenizer.json` or SentencePiece model metadata.
- Train a fresh TAC model at 1024 and 2048 subword tokens.
- Gate: compare against byte-token TAC at equivalent compute and equivalent text
  coverage.

### Phase C: Efficient long context

Goal: scale to 4096-8192 subword tokens.

- Use `attention_window_size` for local attention.
- Keep a small number of global/identity memory paths.
- Train segment recurrence: 512/1024-token segments with carried `IdentityState`.
- Gate: carry > reset and carry > shuffled on multi-segment tasks.

### Phase D: Agentic/knowledge-work context

Goal: useful long conversations, not just long LM loss.

- Build evals with conversation history, plans, tool results, retrieved docs,
  summaries, stale memory, and final answers.
- Add retrieval-backed content memory.
- Add compaction/summarization records to training data.
- Gate: answers improve with relevant retrieval and ignore irrelevant retrieval.

## Decision

Do not rely on TAC memory alone. Use TAC memory as the long-range state mechanism,
but pair it with:

```text
 subword tokenizer
 staged context training
 tokenized memmaps
 local attention
 recurrence gates
 retrieval and compaction
```

The next optimal experiment is a 1024-byte bridge model only as a training-method
test. The next strategically important model is a 16k-vocab subword TAC at 1024
then 2048 subword tokens.
