# TAC Long-Context Efficiency Research

Date: 2026-06-02

## Current Problem

Current Run 4 TAC uses `max_seq_len=256` with byte-level tokens. That is not a
usable context length for agentic or knowledge-work tasks. It is roughly 256
UTF-8 bytes, not 256 BPE-style LLM tokens.

The current architecture also has an inference tax:

- current promoted TAC carried-query throughput vs parameter-matched vanilla:
  `0.09618`
- current promoted TAC decode throughput vs parameter-matched vanilla:
  `0.12263`
- practical result: about `8x` to `10x` slower than vanilla decode

So the long-context plan must solve both problems together:

1. Increase usable context.
2. Avoid making the already-expensive TAC path even slower.

Simply raising `max_seq_len` from 256 to 4096 with dense attention is the wrong
move. Dense attention scales quadratically. Going from 256 to 4096 tokens makes
the attention matrix `256x` larger per sequence.

## External Research Anchors

Relevant prior work:

- FlashAttention shows that exact attention can be made faster and more memory
  efficient by reducing GPU memory traffic with tiled IO-aware kernels.
- Longformer shows a practical local-window plus global-token pattern for
  long documents, reducing attention from quadratic to roughly linear in
  sequence length.
- Transformer-XL and Compressive Transformer establish the core idea that
  long context should use segment recurrence and compressed memories rather
  than one huge dense attention span.
- YaRN shows that RoPE-based models can extend context more efficiently by
  modifying positional scaling and using far less continuation training than
  earlier approaches.
- RAG and RETRO show that knowledge work should not rely only on parametric
  context. Retrieval gives access to much larger knowledge stores without
  increasing the direct attention window.

The TAC-specific synthesis should be:

```text
 subword/persistent tokenized data
+ RoPE context scaling
+ local/global attention
+ segment recurrence
+ compressed TAC identity memory
+ retrieval-backed content memory
+ KV-cache/state-cache decode
```

## Design Principle

TAC should not become a standard dense long-context transformer. That would
remove the main reason TAC exists.

The right design is:

```text
short dense local workspace
+ persistent identity/program state
+ compressed segment summaries
+ retrieval into external memory
+ sparse/global task tokens
```

For agentic work, the model does not need every previous byte in full attention.
It needs:

- current instruction and local working set
- active plan
- tool results
- retrieved documents
- durable facts
- verification state
- failure/repair state

These map naturally onto TAC programs and identity memory.

## Proposed Long-Context TAC Stack

### 1. Replace byte-level training data with tokenized records

Current byte tokens make context length look larger than it is. A usable system
should train on pretokenized sequences.

Options:

- near-term: keep byte tokens but raise context only for controlled tests
- serious path: introduce a subword tokenizer with a 16k-64k vocabulary
- package data as memmap token arrays with record offsets and category IDs

Recommended format:

```text
train.tokens.uint16.bin or train.tokens.uint32.bin
train.record_offsets.int64.npy
train.record_lengths.int32.npy
train.category_ids.int16.npy
categories.json
```

This removes JSON parsing and online byte encoding, but more importantly it
makes `4096` mean something closer to a real LLM context.

Gate:

```text
tokenized batcher throughput >= 5x JSONL byte batcher
no training regression at 256-token equivalence
```

### 2. Add RoPE scaling for context extension

TAC already uses RoPE in the current best preset. Add explicit RoPE scale
controls:

```text
rope_base
rope_scale
rope_scaling_type = none | linear | yarn
original_context_length
target_context_length
```

This lets us test context extension without immediately retraining from scratch.

Gate:

```text
256 -> 1024 context extension does not regress short-context eval by >1%
```

### 3. Use local attention plus TAC global memory

Dense attention should remain available for short sequences, but long TAC should
switch to local/global attention:

```text
attention_window_size = 512 or 1024
global tokens = instruction / plan / memory / retrieval delimiters
identity memory slots remain globally visible
```

This follows the Longformer pattern, but TAC's global tokens should be semantic:

- task header
- current plan
- active tool output
- retrieved document header
- verifier state
- failure marker

Gate:

```text
4096 context train step fits on T4/A100-class GPU
long-context loss improves over 256-context truncation
```

### 4. Segment recurrence with TAC identity carry

Instead of training one giant 16k sequence, split records into segments:

```text
segment length: 512 or 1024
carry between segments:
  - IdentityState
  - content cue/value memory
  - compressed segment summaries
  - selected program state
```

This is the TAC-native equivalent of Transformer-XL style recurrence.

Important: gradients should not initially backpropagate through every previous
segment. Start with stop-gradient recurrence, then test truncated BPTT only if
needed.

Gate:

```text
multi-segment carry > reset
multi-segment carry > shuffled-state
```

### 5. Add compressed segment memory

For every segment, produce a compact memory record:

```text
segment_summary_vector
category/program distribution
salient key/value cues
tool/result flags
verification/failure flags
```

This becomes a fixed-size memory bank. TAC should retrieve from this bank rather
than attend over every old token.

Compression targets:

```text
512-token segment -> 4 to 16 memory vectors
4096-token record -> 32 to 128 memory vectors
```

Gate:

```text
compressed-memory retrieval beats last-window-only baseline
```

### 6. Retrieval-backed content memory for knowledge work

Agentic and knowledge tasks need external retrieval. TAC's content-addressed
memory should be split into:

```text
local content memory: current sequence/session
episodic content memory: previous segments
external document memory: retrieved chunks
```

The model should learn when to use each. Retrieval is not optional for knowledge
work; otherwise context length becomes a brute-force substitute for search.

Gate:

```text
retrieved-doc attribution improves answer accuracy
irrelevant retrieval is ignored or downweighted
```

### 7. Gate expensive memory reads

Current TAC pays memory-read cost broadly. Long-context TAC needs cheap routing:

```text
read memory only for:
  - query tokens
  - plan/update tokens
  - tool-result boundary tokens
  - verification tokens
  - high-uncertainty logits
```

Gate:

```text
memory read rate <=30%
loss regression <=0.5%
decode speed improves
```

### 8. KV-cache and TAC state-cache

Long context will not be usable without incremental decode.

Cache:

- standard attention K/V
- identity/program state per layer
- content memory state
- compressed segment memory
- route decisions for prior tokens

Decode should update only the new token plus small state structures. It should
not recompute full sequence identity routing every step.

Gate:

```text
decode throughput >=0.5x parameter-matched vanilla
```

## Experimental Roadmap

### Phase 0: Measurement

Build a decomposition benchmark:

```text
vanilla dense
TAC dense no memory reads
TAC local attention
TAC local + identity carry
TAC local + compressed memory
TAC local + retrieval memory
TAC with KV/state cache
```

Measure:

- train tokens/sec
- decode tokens/sec
- GPU memory
- eval loss
- carry/reset/shuffled
- specialization MI
- category-conditioned knockout selectivity

### Local automated pass: 2026-06-02

This pass was run locally only. It did not call Kaggle APIs, push kernels, pull
outputs, kill processes, or alter the running Run 4 kernel.

Implemented:

- `TokenizedMemmapBatcher` plus `build_tokenized_memmap_from_jsonl(...)` using
  the planned token-array/offset/length/category file contract.
- RoPE controls on TAC and vanilla attention:
  `rope_base`, `rope_scale`, `rope_scaling_type`, `original_context_length`,
  and `target_context_length`.
- Local long-context decomposition driver:
  `experiments/benchmark_long_context_efficiency.py`.
- Local/inference benchmark plumbing for RoPE scaling and local-attention
  variants.

Artifacts:

- `runs/benchmarks/long_context_efficiency_local_2026_06_02/RESULTS.md`
- `runs/benchmarks/long_context_efficiency_local_2026_06_02/long_context_efficiency_local.json`
- `runs/benchmarks/rope_scaling_local_2026_06_02/none.json`
- `runs/benchmarks/rope_scaling_local_2026_06_02/linear.json`
- `runs/benchmarks/rope_scaling_local_2026_06_02/yarn.json`

Local setup:

```text
device: CPU
model profile: d_model=32, n_layers=1, n_heads=4, n_programs=8
profile seq_lens: 256 and 1024
attention window: 128
batcher corpus: runs/prepared_corpus_agentic_hard/eval.prepared.jsonl
tokenized records: 5,441
tokenized byte-level tokens: 5,102,534
```

Batcher result:

| Seq len | JSONL byte online tok/s | tokenized memmap tok/s | Speedup |
| ---: | ---: | ---: | ---: |
| 256 | 943,592.64 | 2,249,035.68 | 2.38x |
| 1024 | 2,164,720.73 | 3,013,291.43 | 1.39x |

Decision:

- The file contract works and avoids online JSON parsing/tokenization.
- The strict `>=5x` batcher gate is not passed yet. This first implementation is
  still byte-token memmap, not a real subword tokenizer plus fully optimized
  sampler.
- Next batcher work should use true subword IDs and reduce Python list/tensor
  conversion overhead.

Model decomposition result:

| Variant | Seq len | Window | Train tok/s | Decode tok/s | Useful/dense attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| vanilla dense | 256 | dense | 26,154.21 | 2,913.34 | 1.000 |
| vanilla local | 256 | 128 | 25,110.10 | 2,102.91 | 0.500 |
| TAC dense no memory | 256 | dense | 9,235.91 | 504.16 | 1.000 |
| TAC local no memory | 256 | 128 | 8,537.34 | 448.09 | 0.500 |
| TAC local content memory | 256 | 128 | 7,806.15 | 377.52 | 0.500 |
| vanilla dense | 1024 | dense | 16,221.00 | 2,869.00 | 1.000 |
| vanilla local | 1024 | 128 | 12,160.24 | 2,219.94 | 0.125 |
| TAC dense no memory | 1024 | dense | 6,338.05 | 486.68 | 1.000 |
| TAC local no memory | 1024 | 128 | 5,904.19 | 512.94 | 0.125 |
| TAC local content memory | 1024 | 128 | 5,929.00 | 359.91 | 0.125 |

Decision:

- Local attention masking reduces the useful attention-edge proxy from dense
  `1.0` to `0.125` at 1024 with a 128-token window.
- It does **not** produce wall-clock speedup in the current implementation
  because the code still materializes dense attention logits before applying the
  mask.
- Therefore local/global attention is architecturally correct but not yet an
  efficiency solution. The next implementation needs block-sparse/sliding-window
  kernels or a gather-based attention path.
- The 1024 CPU profile confirms the present TAC path still has a large decode
  tax: local content-memory TAC decode was about `0.125x` vanilla dense decode
  in this tiny profile.

Segment-carry screen:

| Variant | Seq len | Carry | Reset | Shuffled | Baseline | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense identity-first | 64 | 0.0625 | 0.0000 | 0.0000 | 0.0000 | effective |
| local identity-first | 64 | 0.0625 | 0.0000 | 0.0000 | 0.0000 | effective |

Decision:

- Segment carry still behaves correctly in the tiny delayed-query screen.
- This is only a smoke-level result, not evidence that 1024/4096 long-context
  training works.
- The next real gate is multi-seed `seq_len=256/512/1024` segment-carry
  training with reset/shuffle controls.

RoPE scaling smoke:

| Scaling | TAC train tok/s | Carry | Reset | Carry loss | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| none | 15,107.58 | 0.0000 | 0.0000 | 4.332681 | inconclusive |
| linear | 15,497.07 | 0.0000 | 0.0000 | 4.332927 | inconclusive |
| yarn-lite | 15,746.07 | 0.0000 | 0.0000 | 4.332912 | inconclusive |

Decision:

- The new RoPE controls run and preserve parameter count.
- This 8-step CPU smoke is too weak to choose `linear` vs `yarn`.
- Treat `linear` as the default context-extension control until a real
  continuation-training screen shows a benefit for `yarn`.

### Local solution candidate: 2026-06-02 continuation

After the initial pass failed to produce a wall-clock local-attention win at
1024, the local implementation continued with three concrete fixes:

1. Replace dense-masked local attention with compact causal sliding-window
   attention. Local windows now materialize `B * H * L * W` logits instead of
   `B * H * L * L` logits when there are no compressed memory slots.
2. Replace Python list construction in the memmap batcher with preallocated
   NumPy arrays and CPU zero-copy `torch.from_numpy(...)`.
3. Add serving-style inference controls:
   - `collect_auxiliary=False` to skip diagnostic losses/metrics during serving;
   - `update_content_memory=False` for one-token decode so content memory writes
     are gated to prefill/query/boundary phases instead of every generated token.

Artifacts:

- `runs/benchmarks/long_context_efficiency_solution_attempt_2026_06_02/RESULTS.md`
- `runs/benchmarks/long_context_large_seq_inference_2026_06_02/RESULTS.md`
- `runs/benchmarks/long_context_decode_gate_fast_decode_2026_06_02/RESULTS.md`

Improved batcher result:

| Seq len | JSONL byte online tok/s | optimized memmap tok/s | Speedup | Gate |
| ---: | ---: | ---: | ---: | --- |
| 256 | 1,652,212.50 | 36,773,560.03 | 22.26x | pass |
| 1024 | 2,295,462.52 | 128,031,257.62 | 55.78x | pass |

Decision:

- The tokenized/memmap input path now passes the `>=5x` gate locally.
- This is still byte-tokenized, but the storage/sampling mechanism is no longer
  the bottleneck. The same path can carry future subword IDs.

Large-context inference result:

| Variant | Seq len | Prefill tok/s | Query tok/s | Decode tok/s | Query vs vanilla | Decode vs vanilla |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vanilla matched | 4096 | 11,369.70 | 11,420.75 | 1,314.88 | 1.0000 | 1.0000 |
| current_best dense | 4096 | 6,070.26 | 6,466.76 | 303.91 | 0.5662 | 0.2311 |
| current_best local | 4096 | 15,741.53 | 15,026.80 | 331.26 | 1.3157 | 0.2519 |
| TAC no-memory local | 4096 | 16,237.45 | 14,730.39 | 328.25 | 1.2898 | 0.2496 |

Decision:

- Compact local attention is not a CPU training win at 1024 in the tiny profile,
  but it becomes useful for 4096-token inference.
- The first local solution candidate is:

```text
4096-token TAC
+ optimized tokenized/memmap batcher
+ RoPE linear scaling control
+ compact causal sliding-window attention, window=128
+ content-addressed memory read
+ collect_auxiliary=False for serving
+ update_content_memory=False during one-token decode
```

Gate status:

```text
4096 prefill/query throughput: pass locally
  current_best_local query = 1.3157x vanilla

first decode gate: pass narrowly locally
  current_best_local decode = 0.2519x vanilla

tokenized batcher gate: pass locally
  22x-56x over online JSONL byte batching
```

Limits:

- This is a local CPU/inference-profile solution, not a full training proof.
- The best 4096 result used `content_read_steps=1` in the decode-gate profile,
  not the heavier synthesis two-step read.
- The capability gate still needs multi-seed long-context segment-carry training
  and agentic trace evaluation.
- GPU kernels may change the ratios, so the next GPU run should profile the
  exact serving configuration before making commercial claims.

## Full Matrix Check: 2026-06-02

The requested local full matrix was run without Kaggle API/status/push/pull
commands.

Artifact:

- `runs/benchmarks/long_context_solution_matrix_256_solution_k1_full_2026_06_02/RESULTS.md`

Matrix:

```text
seq_len: 256
attention window: 128
tasks: longer_single_key, multi_key, delayed_query, noisy_key, multi_hop
seeds: 11, 23, 37
steps: 120
batch_size: 4
eval: 4 batches x 8 samples
```

Overall result:

| Variant | Effective | Mean carry | Carry-reset | Gap vs matched vanilla | Train TPS ratio | Query TPS ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| solution local w128 k1 | 12/15 | 0.1812 | 0.1688 | 0.1646 | 0.7663 | 0.9373 |
| dense current-best synthesis | 10/15 | 0.1667 | 0.1417 | 0.1500 | 0.3937 | 0.4217 |
| solution local w128 synthesis | 10/15 | 0.1667 | 0.1417 | 0.1500 | 0.7449 | 1.0139 |

By-task failure pattern:

| Task | Best local effective | Note |
| --- | ---: | --- |
| longer_single_key | 3/3 | pass |
| multi_key | 3/3 | pass |
| delayed_query | 3/3 | pass |
| noisy_key | 2/3 | partial |
| multi_hop | 1/3 | fail |

Decision:

- The strict full-matrix capability gate is not passed. The exact fast-decode
  k1 variant has positive aggregate carry and beats reset/shuffled/matched
  vanilla on average, but it is not all-effective.
- Local window attention itself is not the source of the capability failure:
  local-synthesis exactly matched dense current-best carry at this budget while
  being much faster.
- The working result should therefore be described as an efficiency solution
  with a capability gap, not as a complete long-context TAC solution.
- Next repair target: noisy-key and multi-hop, probably with semantic global
  tokens, retrieval-graph traversal, or verifier/task-gated heavier reads while
  keeping compact local attention for the active workspace.

### Phase 1: 1024-token TAC

Target:

```text
usable 1024-token context
local attention optional
RoPE scaling enabled
segment carry enabled
```

Tasks:

- long recall
- multi-key recall
- delayed query
- tool result after delay
- stale-memory rejection

Gate:

```text
1024 TAC carry > 256 truncation
1024 TAC carry > reset/shuffled
throughput >=0.25x vanilla
```

### Phase 2: 4096-token agentic TAC

Target:

```text
4096 tokenized-context window
local/global attention
segment memory
gated content reads
```

Tasks:

- multi-document synthesis
- long tool traces
- plan/execute/repair sequences
- verification after long delay
- retrieved evidence grounding

Gate:

```text
agentic task accuracy improves over truncation
decode >=0.25x vanilla
```

### Phase 3: 8192-16384-token knowledge TAC

Target:

```text
8192 direct token context
+ retrieval-backed memory for larger corpora
```

This is where TAC becomes useful for real knowledge work. The direct context is
large enough for several documents, while external retrieval handles larger
knowledge bases.

Gate:

```text
lost-in-middle robustness
multi-hop citation accuracy
retrieval attribution
decode >=0.5x vanilla after cache work
```

## Recommended First Implementation

Do not jump straight to 16k.

The 2026-06-02 local continuation produced a first usable 4096-token inference
candidate. The next implementation should now be:

```text
1. keep optimized memmap sampling as the default input path
2. replace byte IDs with true subword-token IDs
3. add semantic global tokens on top of compact local attention
4. validate content_read_steps=1 vs synthesis k2 at 4096
5. run multi-seed 256/512/1024/4096 segment-carry training
6. run long agentic trace evaluation with decode write-gating enabled
7. profile the same serving path on GPU
```

This gives a clean answer to the central question:

```text
Does TAC memory let a 1024/4096-token model behave like a longer-context agent
without paying dense long-context cost?
```

If yes, continue to 4k/8k. If no, the long-context path needs stronger
retrieval and compression before longer windows.

## Success Criteria

Minimum viable agentic TAC:

```text
context: 4096 real tokens
training: >=0.25x vanilla throughput
decode: >=0.25x vanilla throughput
carry > reset/shuffled on long agentic tasks
retrieval grounding works
category-conditioned program knockout remains selective
```

Research milestone:

```text
8192 real tokens
decode >=0.5x vanilla
functional program specialization survives long-context training
compressed memory improves long tasks over truncation and vanilla
```

## Decision

The most promising fused direction is:

```text
TAC as a long-context working-memory architecture,
not TAC as a dense 16k transformer.
```

The architecture should use dense local attention for the current workspace,
TAC identity memory for persistent computational state, compressed segment
memory for prior context, and retrieval for external knowledge.

This keeps TAC's research advantage while attacking the current efficiency
failure directly.
