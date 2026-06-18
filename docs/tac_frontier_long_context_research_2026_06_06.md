# TAC Frontier Long-Context Research

Date: 2026-06-06

## Goal

Build a TAC context strategy large enough for agentic and knowledge-work conversations.
For the current byte-token checkpoint, useful context should mean at least 4k-16k
byte tokens locally, then a path toward subword-token contexts where 8k-32k tokens
can hold real project conversations, tool traces, retrieved documents, and notes.

## What Frontier Systems Publicly Show

Frontier labs do not fully disclose their long-context training recipes, but the public
record points to the same stack:

- Large native context windows plus long-context training and evaluation. Gemini 1.5
  reports recall and reasoning across millions of tokens, with retrieval tests up to
  at least 10M tokens.
- RoPE extension methods for existing checkpoints. Position Interpolation, YaRN, and
  LongRoPE all modify positional scaling and then use continuation training to recover
  quality.
- Attention/runtime work. FlashAttention improves exact attention efficiency by reducing
  memory traffic; local/global attention methods like Longformer reduce the quadratic
  cost pattern for long documents.
- Recurrence and memory. Transformer-XL, Infini-attention, and related approaches avoid
  relying only on one huge dense context by carrying compressed state across segments.
- Context engineering. Long agent workflows use compaction, structured notes, retrieval,
  and sub-agents because bigger context alone still suffers from relevance and attention
  dilution.

References:

- Gemini 1.5 technical report: https://arxiv.org/abs/2403.05530
- Gemini long-context docs: https://ai.google.dev/gemini-api/docs/long-context
- OpenAI GPT-4.1 model docs: https://platform.openai.com/docs/models/gpt-4.1
- OpenAI GPT-4.1 prompting guide: https://cookbook.openai.com/examples/gpt4-1_prompting_guide
- Anthropic context windows: https://docs.anthropic.com/en/docs/build-with-claude/context-windows
- Anthropic effective context engineering: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Llama 3 Herd paper: https://arxiv.org/abs/2407.21783
- Position Interpolation: https://arxiv.org/abs/2306.15595
- YaRN: https://arxiv.org/abs/2309.00071
- LongRoPE: https://arxiv.org/abs/2402.13753
- FlashAttention: https://arxiv.org/abs/2205.14135
- Longformer: https://arxiv.org/abs/2004.05150
- Transformer-XL: https://arxiv.org/abs/1901.02860
- Infini-attention: https://arxiv.org/abs/2404.07143

## Current TAC Result

I ran a no-finetune context-extension screen on `runs/TAC-seed 20/best.pt`.
The checkpoint was reloaded with larger `max_seq_len` values and RoPE modes, then
evaluated on held-out final-answer continuation.

Output:

`runs/TAC-seed 20/context_extension_rope_screen.json`

| Setting | Context bytes | Byte accuracy | NLL | Perplexity |
| --- | ---: | ---: | ---: | ---: |
| trained 256 / none | 256 | 0.7952 | 0.9011 | 2.46 |
| 512 / no scaling | 512 | 0.5247 | 2.4538 | 11.63 |
| 512 / linear | 512 | 0.2332 | 5.0934 | 162.95 |
| 512 / yarn | 512 | 0.2242 | 5.0498 | 155.98 |
| 1024 / no scaling | 806.6 mean | 0.3886 | 3.8864 | 48.74 |
| 1024 / linear | 806.6 mean | 0.1689 | 5.8341 | 341.76 |
| 1024 / yarn | 806.6 mean | 0.1540 | 5.7030 | 299.76 |

Conclusion: changing context length without continuation training is not viable for
this checkpoint. The model was trained around 256 byte-token position behavior, and
larger windows degrade immediately.

## Recommended TAC Path

1. Train a 512-byte TAC continuation model from `best.pt`.
   Use mixed sequence lengths: 128, 256, 384, 512. Keep a short-context eval gate.

2. If 512 passes, extend to 1024 bytes.
   Compare RoPE `none`, `linear`, and `yarn`, but judge them only after continuation
   training. No-finetune scaling failed.

3. Add local attention for 1024+.
   Dense 1024 is still cheap enough for small local experiments, but 4096+ needs
   `attention_window_size` plus memory/retrieval, otherwise compute grows badly.

4. Add segment recurrence.
   Train records as 512- or 1024-token segments with carried `IdentityState`.
   Gate: carried state must beat reset and shuffled state.

5. Move from byte tokens to subword tokens.
   A 4096 byte-token window is still only around 700-900 English words. A real
   agentic/knowledge-work model needs a tokenizer where 4096-16384 tokens can cover
   many paragraphs, tool calls, and retrieved chunks.

6. Add context engineering for product behavior.
   Use compact conversation state, durable notes, retrieval, and tool-result clearing.
   Do not aim to hold every raw tool output forever in attention.

## Gates

- `512` gate: target eval byte accuracy at least equal to the 256 checkpoint within
  2 percentage points, with no short-context regression.
- `1024` gate: target eval byte accuracy at least 90% of the 512 model, plus better
  long-prefix continuation than truncating to 256.
- `segment memory` gate: carry > reset and carry > shuffled on multi-segment tasks.
- `agentic usefulness` gate: conversation/task evals must include tool traces,
  retrieved documents, summaries, and final answers, not only byte-level LM loss.
