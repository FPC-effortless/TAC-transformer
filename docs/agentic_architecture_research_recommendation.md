# Agentic TAC Architecture Research Recommendation

Date: 2026-05-28

Goal: decide how to apply the requested agentic layers to TAC so the result is commercially viable for a small lab: useful, data-efficient, energy-conscious, and not overloaded with unproven machinery.

## Bottom Line

Do not put the full agent loop directly into the base model.

The best path is:

```text
lean TAC base model
+ memory-aware tool/action adapter
+ external agent runtime for planning, tools, reflection, memory, and orchestration
+ high-quality execution-trace training data
+ strict eval gates before promoting anything into the model
```

In other words:

```text
TAC should become the memory-aware cognitive core.
The agent platform should own planning, tool execution, reflection, and orchestration.
```

This matches our experiments: the all-agentic model stack did not beat the simpler `memory_policy` adapter, and none of the agentic additions passed carry/reset/shuffle validation. It also matches the broader literature: ReAct, Toolformer, ToolLLM, Reflexion, Voyager, MemGPT, Generative Agents, Tree of Thoughts, and LATS mostly succeed through loops, tools, memory, search, and verification around the model, not by forcing every layer into one forward pass.

## What Goes Inside TAC

### 1. Keep The Current Base

Default base should remain:

```text
best_tac_config
= RMSNorm + SwiGLU + RoPE + GQA
+ hash-routed identity programs
+ novelty-gated memory writes
+ program-memory readout
+ gated residual memory adapter
```

Reason: this is the only architecture we have that passed the no-leak chunked recall carry/reset/shuffle test.

### 2. Add Only A Narrow Agent Adapter

The only agentic model-side addition worth continuing is:

```text
memory-conditioned action/tool policy
```

But it should stay as an adapter, not the default model path, until it passes:

```text
carry action accuracy > reset
carry action accuracy > shuffled
carry action accuracy > vanilla
carry action accuracy > recurrent baseline
```

Current status: not passed.

### 3. Train TAC State Directly

The next model-side improvement should not be a planner or multi-agent block. It should be better state supervision:

```text
same memory helps
wrong memory hurts
stale memory is ignored
tool outcome updates memory
verified repair updates procedural memory
```

Concretely:

- memory-action contrastive loss with harder negatives;
- memory readout loss over real tool traces;
- state carry/reset/shuffle loss on long-horizon tool trajectories;
- procedural skill retrieval loss;
- memory write/no-write labels from execution traces.

Our first contrastive loss was too weak because shuffled state remained competitive. The fix is not "add planner now"; the fix is harder and more realistic memory supervision.

## What Stays Outside The Base Model

### Planning

Use planning as runtime search, not a permanent model block.

Tree of Thoughts and LATS support this direction: generate candidate plans/actions, score them, backtrack or search, then execute. This is expensive but controllable. For a small lab, make planning adaptive:

```text
simple task -> one ReAct loop
uncertain task -> small beam/search
high-risk task -> planner + verifier + sandbox
```

Do not spend planner compute on every token.

### Tool Use

Tool use should be trained from tool traces and executed by the platform.

Toolformer and ToolLLM point to the winning recipe:

```text
tool schemas
tool-call demonstrations
API argument supervision
execution result integration
repair after tool error
```

The model should learn when/which/how to call tools, but actual execution, retries, permissions, sandboxing, and audit logs belong in the runtime.

### Memory

Use two memory systems:

```text
TAC identity state = fast neural working/procedural memory
external memory OS = durable episodic/semantic/procedural memory
```

MemGPT, MemoryBank, Generative Agents, and Titans all point to tiered memory with write filters, consolidation, reflection, and retrieval. TAC should not try to store all durable memory in `program_memory`; it should produce and consume compact memory handles.

Recommended platform memory:

```text
working: current task scratchpad and active constraints
episodic: task/event traces with timestamps and outcomes
semantic: distilled facts and user/project knowledge
procedural: successful tool plans, code patches, workflows, skills
```

### Reflection

Reflection should be a runtime operation that writes lessons into memory, not an always-on model head.

Reflexion and Voyager show the useful version:

```text
observe failure
summarize cause
store repair rule / skill
retry with the lesson available
```

Our reflection head did not help the base model. Keep reflection textual, sparse, outcome-gated, and externally auditable.

### Multi-Agent Orchestration

Do not put multi-agent orchestration inside the base model.

Use it as a platform scheduler:

```text
planner
executor
critic/verifier
memory manager
tool specialist
```

Only spawn extra agents when expected value exceeds cost. Multi-agent systems add latency, coordination errors, and token cost; they are commercially useful when they reduce expensive failures, not as a default path.

## Architecture For A Small Lab

Recommended commercial stack:

```text
1. TAC Base Model
   - best_tac_config
   - memory-aware hidden state
   - trained on language + memory + tool traces

2. Agent Adapter
   - optional memory-conditioned tool/action policy
   - not default until state-content tests pass

3. Runtime Controller
   - ReAct-style loop
   - tool execution
   - repair loop
   - uncertainty gate

4. Memory OS
   - working / episodic / semantic / procedural stores
   - write filters
   - trust and decay
   - consolidation jobs

5. Planner
   - cheap direct mode by default
   - tree/search mode only under uncertainty or high value

6. Verifier
   - execution checks
   - unit tests / API validation / policy checks
   - reward and success scoring

7. Orchestrator
   - single-agent default
   - spawn critic/researcher/executor only when useful
```

## Training Recipe

### Stage 1: Keep Base Efficient

Train TAC on:

- normal next-token data;
- chunked recall/state-carry tasks;
- tool-call syntax and schema following;
- memory read/write synthetic tasks.

Do not yet train full planner/orchestration heads.

### Stage 2: Tool Trace Supervision

Collect or synthesize trajectories:

```text
goal
observation
memory retrieved
tool selected
arguments
tool result
repair if failed
final success
memory writes
```

Train:

```text
L_total =
  L_token
+ L_tool_choice
+ L_argument_schema
+ L_result_integration
+ L_memory_read
+ L_memory_write
+ L_state_contrast
+ L_verifier
```

### Stage 3: Distill Runtime Search

Use the runtime planner/search to produce better trajectories, then distill them back into the model:

```text
L_distill =
  action imitation
+ plan-quality ranking
+ verifier score prediction
+ repair action prediction
```

This is more practical than making the small model invent planning from scratch.

### Stage 4: Promote Only Proven Adapters

An adapter is promotable only if it beats:

- base TAC;
- vanilla transformer;
- small recurrent baseline;
- reset and shuffled state;
- same or acceptable throughput budget.

## Evaluation Gates

Synthetic gates:

- chunked recall carry/reset/shuffle;
- action/tool selection carry/reset/shuffle;
- counterfactual memory: correct memory helps, wrong memory hurts;
- stale-memory rejection;
- memory write precision/recall.

Agent gates:

- ToolBench / ToolLLM-style API-call success;
- tau-bench-style multi-turn tool-user tasks;
- WebArena-style realistic web tasks;
- coding-agent tasks such as SWE-bench-style patch/test loops;
- internal commercial tasks from the target platform.

Efficiency gates:

- tokens/sec;
- active expert fraction;
- tool-call count;
- planner expansions per task;
- memory retrieval latency;
- pass rate per dollar.

## Recommended Next Implementation

Build a real agent runtime harness before adding more model blocks:

```text
TACAgentRuntime
  retrieve_memory()
  run_react_step()
  execute_tool()
  verify_result()
  reflect_if_failed()
  write_memory()
```

Then create a trace dataset from that runtime and train TAC on the traces.

The next model experiment should be:

```text
ToolTraceBatcher
  context: goal + retrieved memory + tool schema
  target: tool choice + arguments + success-conditioned memory write
  interventions: carry / reset / shuffled / wrong-memory
```

This is more likely to make TAC agentic than adding another internal head.

## Source Index

- ReAct: https://arxiv.org/abs/2210.03629
- Toolformer: https://arxiv.org/abs/2302.04761
- ToolLLM / ToolBench: https://arxiv.org/abs/2307.16789
- Reflexion: https://arxiv.org/abs/2303.11366
- Voyager: https://arxiv.org/abs/2305.16291
- Tree of Thoughts: https://arxiv.org/abs/2305.10601
- LATS: https://arxiv.org/abs/2310.04406
- MemGPT: https://arxiv.org/abs/2310.08560
- MemoryBank: https://arxiv.org/abs/2305.10250
- Generative Agents: https://arxiv.org/abs/2304.03442
- Titans: https://arxiv.org/abs/2501.00663
- Mamba: https://arxiv.org/abs/2312.00752
- RWKV: https://arxiv.org/abs/2305.13048
- xLSTM: https://arxiv.org/abs/2405.04517
- AgentBench: https://arxiv.org/abs/2308.03688
- WebArena: https://arxiv.org/abs/2307.13854
- tau-bench: https://arxiv.org/abs/2406.12045
