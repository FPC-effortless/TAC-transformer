# TAC Agentic RL Efficiency Review

Date: 2026-06-04

This document reviews the current TAC-Control / TAC-Agent-RL direction for
data efficiency and compute efficiency. It focuses on the model we have been
defining mathematically:

```text
TAC-Agent-RL =
    (Backbone, IdentityField, Router, Memory, Scratchpad,
     Policy, WorldModel, Planner, Verifier, CommitGate,
     Reward, Value, CostGate)
```

## Executive Findings

1. The current TAC stack is capability-oriented, not yet cost-optimal.
   The empirical notes show strong carry/memory behavior, but the model is
   slower than vanilla in the harder inference profile. The design should keep
   TAC as an adaptive control layer, not as work that every token always pays
   for.

2. The largest immediate compute risk is decode-time recomputation.
   The current notes explicitly state there is no real transformer KV cache yet.
   Without KV cache plus TAC state cache, any agentic scratchpad or simulation
   feature will multiply an already expensive decode path.

3. Sparse routing is not automatically hardware-efficient.
   In `linear_expert` mode the implementation computes all expert outputs before
   weighting them. In the experimental agent controller, modular cognition also
   computes all experts before applying a top-2 sparse mixture. This is
   mathematically sparse but not fully compute sparse.

4. Content-addressed memory should stay cheap by default.
   The best current serving-oriented setting is one content read over a small
   store. Two-step or synthesis reads are useful for quality, but they should be
   uncertainty-gated instead of applied globally.

5. Scratchpad and future simulation are useful, but only with budgets.
   They should be bounded latent workspaces with verifier-gated commit, not
   unlimited generated chain-of-thought tokens. Otherwise RL training cost grows
   with group size, rollout horizon, scratchpad length, and planner branches.

6. RL should optimize reward per unit cost, not reward alone.
   Group-relative training, dynamic sampling, exact/verifier rewards, and
   process rewards can improve data efficiency. PPO-style value-heavy training
   should be delayed until the cheaper verifier-scored setup is stable.

7. Input/data pipeline efficiency has already improved substantially.
   The memmap/tokenized path reported large throughput gains over JSONL byte
   loading. The next data-efficiency step is true subword tokenization plus
   packed examples, not more architectural complexity.

## Current Efficiency State

The current research notes indicate:

- content-addressed memory with `content_store_size=8` and `content_read_steps=1`
  is the best default for serving-oriented memory;
- `content_read_steps=2` or synthesis reads can improve some carry behavior but
  add extra memory lookup and MLP cost;
- local attention improves long-context throughput compared with dense TAC
  attention, but capability gates are not fully passed yet;
- optimized memmap data loading is a major win and should become the default;
- current TAC decode profiles are still slower than a vanilla transformer,
  especially before real KV-cache serving is implemented.

The practical conclusion is:

```text
TAC should spend extra compute only when identity, memory, planning, or
verification is expected to change the answer.
```

## Cost Model

Let:

```text
B = batch size
T = sequence length
L = transformer layers
d = hidden width
P = number of identity programs
k = active routed programs per token
K = content-memory slots
W = local attention window
G = RL group size per prompt
H = simulation/planning horizon
J = planner branches
X = scratchpad item count
```

Approximate dense training cost:

```text
C_dense_attn = O(L * B * T^2 * d)
```

Approximate local-attention training cost:

```text
C_local_attn = O(L * B * T * W * d)
```

Identity coordinate cost:

```text
C_identity = O(B * T * P * d)
```

Dense coherence cost:

```text
C_coherence = O(B * T^2 * P)
```

Local coherence cost:

```text
C_local_coherence = O(B * T * W * P)
```

Dense all-program expert cost:

```text
C_dense_expert = O(B * T * P * d^2)
```

Actually sparse top-k expert cost:

```text
C_sparse_expert = O(B * T * k * d^2)
```

Content-addressed read cost:

```text
C_content_read = O(B * Q * K * d)
```

where `Q` is the number of query positions that read memory. The current
implementation often lets every token read, so `Q = T`. The efficiency target is
to make `Q << T` through read gating.

Agentic rollout cost:

```text
C_RL = O(G * C_decode(answer + scratchpad + tool_calls))
```

Planning/simulation cost:

```text
C_plan = O(J * H * C_world_step)
```

The improved model objective should therefore optimize:

```text
maximize E[Reward] - lambda * E[Cost]
```

where:

```text
Cost =
    c_tok * generated_tokens
  + c_id * identity_updates
  + c_route * routed_programs
  + c_mem_read * memory_reads
  + c_mem_write * memory_writes
  + c_plan * simulated_steps
  + c_verify * verifier_calls
  + c_scratch * scratchpad_items
```

This makes efficiency part of the behavior we train, not an afterthought.

## Main Bottlenecks

### 1. No Real Decode Cache Yet

The inference notes say the current decode profile is not a true KV-cache
serving path. This is the highest-priority compute issue because every future
agentic feature depends on repeated generation.

Required serving state:

```text
K_t = (
    transformer_kv_cache,
    identity_state_cache,
    content_memory_cache,
    scratchpad_cache,
    verifier_cache
)
```

Decode should be:

```text
h_t, K_t = EncOneToken(o_t, K_{t-1})
z_t      = CheapIdentityCoordinate(h_t, I_{t-1})
```

and only then:

```text
if g_control_t = 1:
    update route, memory read, scratchpad, verifier, or planner
```

### 2. Coherence Is Quadratic Unless Localized

The identity layer builds:

```text
Kappa = token_program_weights @ token_program_weights^T
```

This is useful, but dense coherence is `O(T^2 * P)` and becomes expensive at
long context. For long-context agentic use, coherence should be one of:

```text
local window coherence:      O(T * W * P)
low-rank summary coherence:  O(T * r * P), r << T
event-token coherence:       O(T * E * P), E << T
```

Recommended default:

```text
coherence_mode = local_window + semantic_global_tokens
```

Dense coherence should be a research/eval mode, not the default long-context
serving mode.

### 3. Program Expert Sparsity Must Be Real

The current `linear_expert` path computes all program experts before weighting:

```text
expert_outputs = all P experts(hidden)
program_context = sum_i routed_weight_i * expert_output_i
```

That cost is:

```text
O(B * T * P * d^2)
```

The efficient target is true top-k dispatch:

```text
selected = TopK(RouteScore, k)
program_context = sum_{i in selected} weight_i * Expert_i(hidden)
```

with cost:

```text
O(B * T * k * d^2)
```

Implementation caveat: the current sparse path uses the maximum active count in
the batch. If one example activates many programs, the whole batch can pay that
larger `k`. Keep route budgets fixed and small, and batch examples by route
budget when profiling.

### 4. Memory Reads Need Query Gating

Current content-addressed read cost is:

```text
scores = query_hidden @ cues^T
read   = softmax(scores) @ values
```

If every token reads memory:

```text
Q = T
```

The efficient target is:

```text
Q = number of uncertain/event/action tokens
Q << T
```

Read gate:

```text
g_read_t = 1[
    u_uncertainty_t
  + u_memory_need_t
  + u_entity_reference_t
  + u_action_need_t
  > tau_read
]
```

Then:

```text
m_t =
    Read(h_t, S_t)  if g_read_t = 1
    0              otherwise
```

This also improves data efficiency because memory reads become interpretable
events that can receive direct reward or verifier pressure.

### 5. Memory Writes Need Verification and Novelty

A persistent memory write is more expensive and riskier than a scratchpad write.
It should happen only when the information is:

```text
novel, useful, stable, and verified
```

Commit gate:

```text
g_commit_t = 1[
    novelty_t
  * utility_t
  * verification_t
  * stability_t
  > tau_commit
]
```

Persistent update:

```text
S_{t+1}^real =
    Commit(S_t^real, X_t)  if g_commit_t = 1
    S_t^real              otherwise
```

This prevents imagined futures, failed plans, or speculative assumptions from
polluting real memory.

### 6. Scratchpad Cost Can Explode

A scratchpad is beneficial for future simulation and teaching the model how to
think, but only if it is a bounded working state:

```text
X_t = {x_j}_{j=1..N_t}, N_t <= B_scratch
```

Scratchpad item:

```text
x_j = (type, payload, source, confidence, expiry, verified, cost)
```

Update:

```text
X_{t+1} = TopB(
    X_t union Draft(b_t, a_t, o_t),
    score = utility + verification - risk - age - cost
)
```

The model should learn when to use scratchpad:

```text
g_think_t = 1[
    uncertainty_t
  + planning_need_t
  + tool_need_t
  + contradiction_t
  > tau_think
]
```

No scratchpad for easy next-token prediction:

```text
if g_think_t = 0:
    emit directly
```

This keeps scratchpad as an adaptive computation mechanism rather than a default
token sink.

### 7. Future Simulation Must Be Selective

Simulation should estimate future states, not generate unrestricted text.

World step:

```text
S_{t,b,k+1}^imag = WorldModel(S_{t,b,k}^imag, a_{t,b,k})
```

Branch score:

```text
score(b) =
    E[reward_b]
  - lambda_plan * H_b
  - lambda_risk * risk_b
  - lambda_uncertainty * model_uncertainty_b
```

Simulation gate:

```text
g_sim_t = 1[
    consequence_t * uncertainty_t * irreversibility_t > tau_sim
]
```

This means simulation is used for high-consequence or ambiguous decisions, not
for every token.

## Data Efficiency Review

### Current Strengths

The current direction is data-efficient in the right places:

- persistent identity state lets the model reuse carried information;
- content-addressed memory lets the model retrieve rather than relearn;
- carry/reset/shuffled evaluations create strong causal tests;
- verifier rewards and exact task rewards can avoid expensive human labels;
- group-relative RL can reuse multiple samples from the same prompt.

### Current Risks

1. Synthetic memory tasks can overfit routing behavior.
   The model may learn benchmark-specific memory tricks rather than general
   agentic thinking.

2. Auxiliary heads can consume data without improving final behavior.
   World/reward/reflection/planner heads should be trained only when their
   ablations prove they improve action accuracy, calibration, or cost.

3. RL can waste samples on prompts that are too easy or impossible.
   All-correct and all-wrong groups produce weak learning signal.

4. Learned reward models can become a second data bottleneck.
   Prefer exact verifiers, executable checks, and implicit process rewards
   before training a separate reward model.

### Data-Efficient Training Policy

Use dynamic prompt filtering:

```text
Keep(x) = 1[
    eps < mean_i r_i(x) < 1 - eps
    and Var_i(r_i(x)) > sigma_min
]
```

Use group-relative advantages:

```text
A_i = (r_i - mean_j r_j) / (std_j r_j + eps)
```

Train only on informative groups:

```text
L_policy = -mean_i log pi_theta(y_i | x) * stopgrad(A_i)
```

Add cost-aware reward:

```text
r_i =
    r_task
  + alpha_verify * r_verify
  + alpha_process * r_process
  + alpha_carry * r_carry
  - lambda_cost * Cost_i
```

Use replay selectively:

```text
Replay = high-reward traces + high-regret failures + verifier-approved repairs
```

Do not replay low-information traces:

```text
drop all-correct easy traces
drop all-wrong impossible traces
drop high-cost traces with no reward gain
```

## Compute-Efficient Mathematical Representation

The previous mathematical contract should be amended so cost is first-class.

Target model:

```text
TAC-Agent-RL-Efficient =
    (Backbone, IdentityField, Router, Memory, Scratchpad,
     Policy, WorldModel, Planner, Verifier, CommitGate,
     Reward, Value, CostModel, BudgetPolicy)
```

Decision state:

```text
b_t = B_theta(h_t, z_t, r_t, m_t, X_t, A_t, cost_t)
```

Budget policy:

```text
beta_t = BudgetPolicy_theta(b_t)
```

where:

```text
beta_t = (
    route_budget_t,
    memory_read_budget_t,
    scratchpad_budget_t,
    simulation_budget_t,
    verifier_budget_t
)
```

Action policy:

```text
a_t ~ pi_theta(a | b_t, beta_t)
```

Training objective:

```text
J(theta) =
    E[
        R_task
      + R_process
      + R_verify
      + R_carry
      - lambda_cost * C
      - lambda_kl * KL(pi_theta || pi_ref)
    ]
```

This makes the model learn:

```text
when to think,
when to retrieve,
when to simulate,
when to verify,
when to commit,
and when to answer directly.
```

## Recommended Efficiency Changes

### Priority 1: Serving Cache

Add true decode cache:

```text
transformer KV cache
identity state cache
content-memory cache
scratchpad cache
```

Serving should call the identity layer in a cheap one-token mode, not reprocess
the whole prompt on each decode step.

Gate:

```text
decode_tps_TAC_cached >= 0.75 * decode_tps_BASE_cached
```

for short context, and:

```text
decode_tps_TAC_cached >= 0.25 * decode_tps_vanilla_cached
```

for long-context agentic mode.

### Priority 2: Default Cheap Memory

Default:

```text
content_store_size = 8
content_read_steps = 1
memory_read_type = content_addressed
```

Only enable synthesis/two-step read when:

```text
g_deep_read_t = 1[
    memory_conflict_t
  + low_first_read_confidence_t
  + verifier_requires_evidence_t
  > tau_deep_read
]
```

### Priority 3: True Sparse Experts

Avoid all-expert computation in default modes.

Target:

```text
program_compute_type = sparse_linear_expert
k <= 2 or 3
```

Replace any sparse-after-dense path with dispatch-before-compute.

For the agent controller, `_apply_modular_cognition` should not compute all
cognitive experts if only top-2 are used. Either:

```text
use a single shared low-rank adapter
```

or:

```text
route first, then compute selected experts only
```

### Priority 4: Local/Global Attention

Long context should use:

```text
local attention window W
semantic global tokens E
compressed TAC memory
retrieval-backed content memory
```

Dense all-token identity coherence should be reserved for short-context tests.

### Priority 5: Event-Driven Identity Updates

Not every token needs a full identity-state write. Use:

```text
g_identity_update_t = 1[
    boundary_t
  + action_t
  + tool_result_t
  + memory_event_t
  + large_identity_delta_t
  > tau_identity
]
```

Then:

```text
I_{t+1} =
    UpdateIdentity(I_t, h_t)  if g_identity_update_t = 1
    I_t                      otherwise
```

Every token can still receive identity coordinates, but persistent state updates
should be sparse.

### Priority 6: Scratchpad Budget and Compression

Use typed scratchpad entries instead of raw unlimited text.

Budgets:

```text
B_scratch_items <= 32
B_scratch_tokens <= task_budget
B_sim_branches <= 4
B_sim_horizon <= 3 to 8 for early experiments
```

Compression:

```text
X_t = Compress(X_t) when N_t > B_scratch
```

Verifier rule:

```text
Only verified or high-confidence scratchpad entries can be committed to memory.
```

### Priority 7: RL With Cost-Aware Dynamic Sampling

Immediate RL target:

```text
GRPO/RLOO + DAPO-style filtering + verifier rewards + cost penalty
```

Use small group sizes first:

```text
G = 4 to 8
```

Skip:

```text
all-correct groups
all-wrong groups
groups with no reward variance
groups whose reward gain does not justify cost
```

Delay:

```text
large PPO
large value model
deep multi-branch planning
learned reward model
```

until the cheap policy loop proves measurable improvement.

## Proposed Efficient Execution Loop

```text
for each input token/event o_t:
    h_t, K_t = EncOneTokenOrChunk(o_t, K_{t-1})

    z_t = IdentityCoordinate(h_t, I_t)

    beta_t = BudgetPolicy(h_t, z_t, X_t, cost_so_far)

    if beta_t.route_budget > 0:
        R_t = Route(h_t, z_t, beta_t.route_budget)
    else:
        R_t = default_route

    if ShouldRead(h_t, z_t, X_t, beta_t):
        m_t = ReadMemory(h_t, R_t, S_t)
    else:
        m_t = 0

    b_t = Belief(h_t, z_t, R_t, m_t, X_t, cost_so_far)

    if ShouldThink(b_t, beta_t):
        X_t = ScratchUpdate(X_t, b_t)

    if ShouldSimulate(b_t, beta_t):
        X_t = PlanWithWorldModel(X_t, b_t, beta_t.simulation_budget)

    if ShouldVerify(b_t, X_t, beta_t):
        v_t = Verify(b_t, X_t)
    else:
        v_t = null

    if ShouldCommit(X_t, v_t, beta_t):
        S_t = CommitVerifiedScratchpad(S_t, X_t, v_t)

    if ShouldUpdateIdentity(o_t, b_t):
        I_t, M_t = UpdateIdentityAndMemory(I_t, M_t, b_t)

    y_t = EmitOrAct(b_t, X_t, S_t)
```

The key property is adaptive computation:

```text
easy token -> encode + cheap identity coordinate + emit
hard event -> route + read + scratchpad + simulate + verify + commit
```

## Efficiency Gates

A change should not be accepted as the new default unless it passes both quality
and cost gates.

### Data Gates

```text
informative_group_rate >= 0.30
verifier_coverage >= 0.80 on target RL tasks
scratchpad_success_lift > direct_success_lift
carry > reset
carry > shuffled
```

### Training Compute Gates

```text
train_tps_TAC >= 0.25 * train_tps_vanilla
RL_tokens_per_rewarded_success decreases over baseline
mean_cost_successful_trace <= configured budget
```

### Inference Gates

```text
cached_decode_tps_TAC >= 0.75 * cached_decode_tps_BASE
long_context_query_tps_TAC >= 1.0 * vanilla_query_tps where memory is required
memory_reads_per_output_token <= configured budget
scratchpad_tokens_per_answer <= configured budget
simulation_steps_per_answer <= configured budget
```

### Ablation Gates

Each component must survive knockout:

```text
remove scratchpad  -> hard planning accuracy drops
remove verifier    -> false commit rate rises
remove memory      -> carry accuracy drops
remove simulation  -> future-state tasks degrade
remove cost penalty -> same reward costs more compute
```

If a component does not show a selective ablation effect, it should not be in
the default path.

## Implementation Order

1. Add profiling counters for:

```text
identity_updates
memory_reads
memory_writes
active_programs
scratchpad_items
simulation_steps
verifier_calls
generated_tokens
```

2. Add a cost model and log:

```text
cost_per_batch
cost_per_success
cost_per_answer
```

3. Implement cached decode before expanding agentic rollout depth.

4. Make content read query-gated.

5. Make scratchpad first-class but bounded.

6. Add verifier-gated commit.

7. Start RL with:

```text
GRPO/RLOO + dynamic filtering + cost-aware verifier reward
```

8. Only then test larger planning horizons, value heads, or sequence-level
GSPO/VAPO-style objectives.

## Initial Implementation Experiment

The first implementation experiment adds opt-in content-read query gating:

```text
TACConfig.content_read_query_top_k: Optional[int]
```

Default behavior remains unchanged:

```text
content_read_query_top_k = None
```

When set, content-addressed memory reads are applied only to the top-k token
positions per batch according to the existing learned memory gate. Skipped
positions receive a zero content-memory read. The model now reports:

```text
content_read_queries
content_read_query_fraction
content_read_skipped_fraction
```

Reusable profile script:

```text
python experiments/profile_content_read_query_gating.py
```

Smoke artifact:

```text
runs/benchmarks/content_read_query_gating_smoke_2026_06_04
```

CPU smoke result, with `seq_len=128`, `batch_size=4`, one layer, and
`content_store_size=8`:

```text
full_read: 512 read query positions per forward, read fraction 1.0000
top_k_1:     4 read query positions per forward, read fraction 0.0078
top_k_2:     8 read query positions per forward, read fraction 0.0156
top_k_4:    16 read query positions per forward, read fraction 0.0312
top_k_8:    32 read query positions per forward, read fraction 0.0625
```

In that small CPU profile, `top_k_8` reached about `1.31x` full-read throughput.
The smaller top-k settings reduced read work more aggressively but did not always
improve wall-clock speed because gather/scatter overhead and CPU noise dominate
at this small scale. The main validated result is that the implementation makes
`Q << T` measurable and opt-in, matching the cost model:

```text
C_content_read = O(B * Q * K * d)
```

Capability-preservation follow-up:

```text
python experiments/benchmark_content_read_query_gating_capability.py
```

Artifact:

```text
runs/benchmarks/content_read_query_gating_capability_2026_06_04
```

This benchmark compares full content reads against top-k query-gated reads on
the existing carry/reset/shuffled chunked-recall task. It only reports
`preserved` when:

```text
full-read carry >= 0.20
full-read state utility >= 0.05
gated carry drop <= 0.02
gated state-utility drop <= 0.02
gated read fraction <= 0.50
```

Three-seed local result on `single_key`, `seq_len=8`, `steps=120`:

```text
full_read: effective 3/3, carry 0.6927, state utility 0.6406, read fraction 1.0000
top_k_2:   effective 3/3, carry 0.6927, state utility 0.6458, read fraction 0.2500
```

Decision:

```text
top_k_2 capability preserved on the small direct-recall gate.
```

The benchmark now also reports per-task preservation decisions. This matters
because mixed harder-task averages can hide whether a result is preserved,
regressed, or simply blocked by weak full-read capability on one task.

Long-context preservation follow-up artifacts:

```text
runs/benchmarks/content_read_query_gating_long_context_single_key_64_2026_06_04
runs/benchmarks/content_read_query_gating_long_context_delayed_query_64_2026_06_04
```

Three-seed local result on `single_key`, `seq_len=64`, `steps=120`,
`top_k_8`:

```text
full_read: effective 3/3, carry 0.4688, state utility 0.3958, read fraction 1.0000
top_k_8:   effective 3/3, carry 0.4688, state utility 0.3958, read fraction 0.1250
decision:  preserved
```

Three-seed local result on `delayed_query`, `seq_len=64`, `steps=120`,
`top_k_8`:

```text
full_read: effective 3/3, carry 0.3750, state utility 0.3438, read fraction 1.0000
top_k_8:   effective 3/3, carry 0.3750, state utility 0.3542, read fraction 0.1250
decision:  preserved
```

Decision:

```text
top_k_8 capability is preserved on the tested long-context direct-recall and
delayed-query gates while skipping 87.5% of content-read query positions.
```

Harder-key follow-up artifact:

```text
runs/benchmarks/content_read_query_gating_harder_capability_2026_06_04
```

Three-seed local result at `seq_len=8`, `steps=120`, `top_k_2`:

```text
noisy_key:
  full_read carry 0.1250, state utility 0.0833, read fraction 1.0000
  top_k_2   carry 0.1302, state utility 0.0885, read fraction 0.2500
  decision  blocked_by_full_read_capability

multi_hop:
  full_read carry 0.0573, state utility 0.0260, read fraction 1.0000
  top_k_2   carry 0.0521, state utility 0.0156, read fraction 0.2500
  decision  blocked_by_full_read_capability
```

Noisy-key was retested with longer 400-step and 600-step sweeps:

```text
runs/benchmarks/content_read_query_gating_noisy_key_capability_400_2026_06_04
runs/benchmarks/content_read_query_gating_noisy_key_capability_600_2026_06_04
```

The 600-step result preserves the measured behavior but still misses the
configured full-read capability threshold:

```text
full_read: carry 0.1927, state utility 0.1458, read fraction 1.0000
top_k_2:   carry 0.1823, state utility 0.1406, read fraction 0.2500
decision:  blocked_by_full_read_capability
```

Interpretation:

```text
The gate is not shown to regress noisy-key or multi-hop behavior, but those are
not formal preservation proofs yet because full-read itself is below the current
proof threshold. Multi-hop especially needs a stronger baseline task setup,
longer training recipe, or a dedicated chain-retrieval objective before the
query gate can be judged there.
```

## Bottom Line

The most efficient version of this model is not "TAC everywhere." It is:

```text
cheap transformer prediction by default
+ cheap identity coordinates on every token
+ sparse identity/memory updates on important events
+ bounded scratchpad for hard reasoning
+ selective future simulation
+ verifier-gated memory commit
+ cost-aware RL
```

This preserves the core goal: a model with persistent identity, teachable
thinking behavior, memory, and future-state simulation. It also prevents those
features from becoming an uncontrolled multiplier on tokens, rollouts, and
memory bandwidth.
