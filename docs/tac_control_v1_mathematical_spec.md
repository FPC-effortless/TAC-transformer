# TAC-Control-v1 Mathematical Specification

Date: 2026-06-04

This document converts the current TAC work into a mathematical target. It is
not a claim that the architecture is already perfect. It is the cleanest formal
description of what the model is trying to become, how the implemented model
approximates it, and which gates must pass before we can call the behavior real.

## Current State

The repository now contains three layers of work:

- A deterministic browser lab in `src/lib/identityField.js`.
- A trainable PyTorch TAC transformer in `tac_transformer/model.py`.
- Research automation for validation, seed replication, identity stability, and
  Phase D memory/agentic benchmarks.

The current frozen research candidate is TAC-Control-v1. The Phase A reference
is the Run 5B program-conditioned CREB-k6 checkpoint at step 10000 and
183,600,000 tokens. It has:

- next-token eval accuracy: `0.9414`
- same-backbone vanilla eval accuracy: `0.9530`
- parameter-matched vanilla eval accuracy: `0.9549`
- program-memory cosine: `0.0031`
- selected-route MI: `0.2821` bits
- max knockout loss delta: `0.3401`
- max knockout selectivity span: `0.1868`

Phase A is `freeze_ready`. Phase B is not cleared yet: seed 11 failed the route
MI gate, seed 23 is pending knockout evidence, and seed 37 plus the seed 23
resume are still external-state dependent. Phase C identity stability and Phase
D benchmark claims remain blocked by Phase B.

## One-Sentence Model

TAC-Control-v1 is a causal transformer with a persistent latent program field:

```text
next token = transformer(content stream, identity program stream, persistent memory)
```

Mathematically, each layer is a recurrent controlled system:

```text
(H_l^+, S_l^+) = F_l(H_l, S_l; theta_l)
```

where `H_l` is the token hidden state and `S_l` is the layer's identity state.
The intended identity state is:

```text
S_l = (s_l, M_l, A_l, F_l, C_l, V_l, mask_l)
```

with:

- `s_l in R^P`: per-program stability.
- `M_l in R^{P x d}`: per-program memory.
- `A_l in R^P`: per-program age since last write.
- `F_l in R^P`: per-program write frequency.
- `C_l, V_l in R^{K x d}`: content-addressed cue/value memory.
- `P`: number of identity programs.
- `K`: content store size.
- `d`: hidden width.

The stronger representation is not a fixed program ID table. It is a set of
learned roles, stable only up to permutation:

```text
IdentityRoleSet = {(p_i, s_i, M_i, route_i, knockout_i)}_{i=1..P} / Sym(P)
```

That quotient matters. Phase C should align program roles across seeds by
memory vectors, selected-route distributions, and knockout profiles rather than
assuming program `i` means the same thing in every seed.

## Token-To-Program Field

The identity field is shared. Individual tokens do not own independent fields;
they receive coordinates in the shared field. The distinction is:

```text
shared field:        P = {p_1, ..., p_P}
token coordinate:    w_t in Delta^{P-1}
persistent state:    S_t = (s_t, M_t, C_t, V_t, ...)
```

So the target is:

```text
token-specific identity expression + shared persistent identity state
```

not:

```text
one private identity field per token
```

For a normalized hidden vector `h_t in R^d` and learned program embeddings
`p_i in R^d`:

```text
z_{t,i} = sqrt(d) * <normalize(h_t), normalize(p_i)>
a_{t,i} = sigmoid(z_{t,i})
```

For causal processing, stability is updated token by token:

```text
g^s_{t,i} = sigmoid(W_s h_t)_i
s_{t,i} = (1 - g^s_{t,i}) s_{t-1,i} + g^s_{t,i} a_{t,i}
```

The token's soft identity distribution is:

```text
w_{t,i} = softmax_i(z_{t,i} + log(max(s_{t,i}, eps)))
```

The identity vector carried by the token is:

```text
u_t = sum_i w_{t,i} p_i
```

The pairwise identity coherence matrix is:

```text
Kappa_{t,u} = clip(sum_i w_{t,i} w_{u,i}, 0, 1)
```

Interpretation:

- `a_t` says which programs recognize the token.
- `s_t` says which programs remain stable over time.
- `w_t` says which program identity the token expresses.
- `Kappa` says which tokens share identity structure.

The important design rule is:

```text
every token gets w_t
only useful tokens materially update S_t
```

In other words, identity coordinates are dense enough to support attention and
coherence, but identity state writes are gated by novelty, relevance, and cost.

## Identity State Computation Policy

The model should not recompute the full identity field for all previous tokens
at every generated output token. The ideal inference policy is:

```text
prefill:
    compute token coordinates w_1...w_T in parallel/chunked form
    build cached identity state S_T

decode step T+1:
    read cached S_T
    compute only the new token coordinate w_{T+1}
    route sparse programs if useful
    apply a gated update Delta S_{T+1}
    cache S_{T+1}
```

Mathematically:

```text
w_{t} = Coordinate(h_t, S_{t-1})
g_t = sigmoid(novelty(h_t, S_{t-1}) + relevance(h_t) - cost(h_t))
S_t = S_{t-1} + g_t * DeltaS(h_t, w_t, S_{t-1})
```

where `g_t` may be near zero for punctuation, local filler, or predictable
tokens, and high for entities, goals, contradictions, tool decisions, evidence,
or memory-worthy reasoning steps. The model should therefore pay a cheap
per-token identity-read cost and a sparse state-write cost only when the token
changes the persistent identity state.

## Identity-First Attention

TAC-Control-v1 uses identity-first attention. Query vectors come from token
content, but key/value vectors are conditioned on both token content and token
identity:

```text
q_t = W_Q h_t
(k_t, v_t) = W_KV_id([h_t; u_t])
```

For attention head dimension `d_h`, attention logits are:

```text
ell_{t,u} =
    <q_t, k_u> / sqrt(d_h)
    + beta * lambda_coh * Kappa_{t,u}
    + mask_{causal}(t,u)
```

and:

```text
Attn_t = sum_u softmax_u(ell_{t,u}) v_u
```

This is the core "identity beside attention" idea in mathematical form:
attention still follows content, but content is biased by shared program
identity and K/V projections are identity-conditioned.

## Sparse Program Routing

Each program has a learned positive cost:

```text
e_i = softplus(r_i) + 0.05
```

TAC-Control-v1 uses `base_semantic` routing with top-k 2. The first route is a
BASE-style balanced assignment:

```text
i_base(row) = n_sink + (row mod (P - n_sink))
```

The extra semantic route is chosen by activation-per-cost:

```text
rho_{t,i} = a_{t,i} / e_i
```

The final hard route mask `R_{t,i} in {0,1}` is:

```text
R_t = BudgetTrim({i_base(t)} union TopK_semantic(rho_t, k - 1))
sum_i R_{t,i} e_i <= B
```

where `B` is the route energy budget. Training uses a straight-through route so
the hard selected programs remain compatible with gradient learning.

Selected program weights are normalized activations:

```text
alpha_{t,i} = R_{t,i} a_{t,i} / max(sum_j R_{t,j} a_{t,j}, eps)
```

The load-balance pressure is:

```text
L_load = mean_i((load_i / sum_j load_j) - 1/P)^2
```

## Program Expert Context

For the promoted TAC family, selected programs are linear experts:

```text
E_i(h_t) = W_i h_t + b_i
```

The routed expert context is:

```text
G_t = sum_i alpha_{t,i} E_i(h_t)
```

TAC then adds an active memory read:

```text
PContext_t = G_t + ReadMemory(h_t, S_t)
```

and writes it back to the residual stream:

```text
h_t' = h_t + Attn_t + lambda_program W_P PContext_t
h_t^+ = h_t' + MLP(Norm(h_t'))
```

The intended role of `PContext_t` is not just extra capacity. It is a sparse,
recurrent, identity-conditioned computation path.

## Program Memory Update

TAC-Control-v1 uses program-conditioned memory updates. Let:

```text
bar_h = mean_t h_t
candidate_i = U([bar_h; p_i]) * s_i
```

The write gate is a learned memory gate multiplied by a novelty gate:

```text
g^m_i = sigmoid(W_m bar_h)_i
nu_i = sigmoid(w_n [candidate_i; M_i])
eta_i = g^m_i * nu_i
```

CREB allocation chooses which programs may write:

```text
creb_i =
    alpha_c * (1 - s_i)
    + beta_c * a_i
    - gamma_c * A_i
    - delta_c * F_i
```

For TAC-Control-v1:

```text
write_mask_i = 1[i in TopK(creb, 6)]
eta_i^* = eta_i * write_mask_i
```

The memory update is:

```text
M_i^+ = (1 - eta_i^*) M_i + eta_i^* candidate_i
```

Age and write frequency evolve as:

```text
A_i^+ = (A_i + 1) * 1[eta_i^* <= eps]
F_i^+ = decay * F_i + (1 - decay) * 1[eta_i^* > eps]
```

Program-memory collapse is discouraged by:

```text
L_mem_sep = mean_{i != j} cos(M_i, M_j)^2
memory_cosine = mean_{i != j} |cos(M_i, M_j)|
```

The Phase B gate currently requires:

```text
memory_cosine <= 0.25
```

## Content-Addressed Memory

TAC-Control-v1 also keeps cue/value hidden-state pairs:

```text
C_k = h_t
V_k = h_{t+1}
```

Given a query `q`, one content read is:

```text
score_k(q) = <normalize(q), normalize(C_k)>
pi_k(q) = softmax_k(score_k(q) + mask_k)
Read(q) = sum_k pi_k(q) V_k
```

The current model uses a two-step synthesis read:

```text
r_1 = Read(h_t)
r_2 = Read(r_1)
phi = [h_t; r_1; r_2; r_1 - r_2; r_1 * r_2]
synth = W_syn phi
g_syn = sigmoid(w_syn phi)
r = (1 - g_syn) r_1 + g_syn synth
```

The memory read is gated before entering the program context:

```text
ReadMemory(h_t, S_t) = sigmoid(W_read h_t) * r
```

Cue collapse is discouraged by:

```text
L_cue_sep = mean_{i != j} mask_i mask_j cos(C_i, C_j)^2
```

## Output And Training Objective

The model predicts:

```text
logits_t = W_vocab Norm(h_t^+)
```

The core language loss is:

```text
L_lm = CE(logits, y)
```

The full implemented objective is:

```text
L =
    L_lm
    + lambda_coherence L_coherence
    + lambda_reuse L_reuse
    + lambda_energy L_energy
    + lambda_mem_sep L_mem_sep
    + lambda_cue_sep L_cue_sep
    + lambda_gate_entropy L_gate_entropy
    + lambda_load L_load
    + lambda_route L_route
```

with:

```text
L_coherence = mean((1 - Kappa)^2)
L_reuse = mean(1 - a_i)
L_energy = mean(sum_i R_i e_i) / B
L_gate_entropy = mean(1 - H(g_syn) / log(2))
```

For the TAC-Control-v1 route objective, category labels `c` are used to
encourage program specialization by maximizing mutual information:

```text
L_route = -I(c; p_selected)
I(c; p) = sum_{c,p} P(c,p) log(P(c,p) / (P(c) P(p)))
```

The current frozen route pressure is:

```text
lambda_route = 0.1
```

## Behavioral Target

The model we actually want is not just a lower-loss model. It is a constrained
optimizer:

```text
theta^* = argmin_theta E[L_lm + L_aux]
```

subject to:

```text
Capability(theta) >= Capability(vanilla) - delta_cap
StateUtility(theta) > 0
Specialization(theta) >= tau_mi
NonCollapse(theta) <= tau_cos
KnockoutUtility(theta) >= tau_knockout
SeedStability(theta) >= tau_align
AuthorityRisk(theta) <= tau_authority
CostNormalizedGain(theta) > 0
```

Where the gates are:

```text
StateUtility = Score(carry) - max(Score(reset), Score(shuffled))
Specialization = I(category; selected_program)
NonCollapse = mean_{i != j} |cos(M_i, M_j)|
KnockoutUtility = max_program_ablation_delta
SeedStability = max-matching role similarity across seeds
AuthorityRisk = false_authority + cross_domain_authority_violations
CostNormalizedGain = (Score_TAC - Score_vanilla) / Cost_TAC
```

Current Phase B/C/D thresholds encode part of this:

```text
eval_accuracy >= 0.93
selected_route_mi >= 0.15
program_memory_cosine <= 0.25
max_knockout_loss_delta >= 0.05
Phase C alignment similarity >= 0.80 across at least 2 passing seeds
Phase D TAC score > parameter-matched vanilla score on required task families
```

## Perfect Representation

The clean mathematical representation of the desired model is:

```text
TAC = (Backbone, ProgramField, SparseRouter, ProgramMemory,
       ContentMemory, AuthorityMonitor, CostGate)
```

with dynamics:

```text
H_{t+1}, S_{t+1}, y_t =
    TAC_theta(x_t, H_{<=t}, S_t)
```

and identity:

```text
S_t / Sym(P)
```

not raw program IDs.

In words:

1. The backbone models local linguistic and task structure.
2. The program field gives every token a soft identity over learned roles.
3. The router chooses a small number of active roles under an energy budget.
4. Program memory stores persistent role-specific state.
5. Content memory stores cue/value facts for direct retrieval.
6. Identity-first attention lets content attend through program identity.
7. Authority monitoring checks whether the model trusted the right evidence.
8. The behavioral gates reject models whose memory looks useful only by
   aggregate loss but fails reset, shuffle, specialization, or knockout tests.

## Improved Agentic Representation

To support reinforcement learning, process supervision, tool use, and simulated
future states, the base TAC representation should be extended from a language
model into a state-action model:

```text
TAC-Agent =
    (Backbone, ProgramField, SparseRouter, ProgramMemory, ContentMemory,
     Scratchpad, Policy, Value, WorldModel, Planner, Verifier, CommitGate,
     CostGate)
```

The core recurrent state must distinguish real memory from imagined memory:

```text
S_t^real = (I_t, M_t, C_t, A_t, K_t)
S_t^imag = (I_t~, M_t~, C_t~, A_t~, K_t~)
X_t = bounded working scratchpad
```

where:

- `I_t`: identity/program state.
- `M_t`: program memory.
- `C_t`: content-addressed memory.
- `A_t`: authority/confidence state.
- `K_t`: KV/cache and decode-local computation state.
- `X_t`: short-lived deliberation, branch, tool, and verifier workspace.

The real state is updated only from observed or verified information. The
imagined state is used for planning and must not automatically contaminate real
memory. The scratchpad is neither real memory nor imagined memory. It is a
bounded working area for intermediate computation.

The improved top-level dynamics are:

```text
b_t = Belief(o_{\le t}, S_t^real)
a_t ~ pi_theta(a | b_t, S_t^real, X_t)
y_t ~ p_theta(y | b_t, S_t^real, X_t, a_t)
X_{t+1} = ScratchUpdate(X_t, b_t, a_t, y_t, verifier_t)
S_{t+1}^real = RealUpdate(S_t^real, o_t, a_t, y_t, X_{t+1}, verifier_t)
```

For language-only decoding, `a_t` can be the implicit next-token action. For
agentic operation, `a_t` belongs to a structured action set:

```text
a_t in {
    emit_token,
    internal_thought,
    retrieve_memory,
    route_program,
    call_tool,
    verify,
    simulate,
    write_memory,
    stop
}
```

This representation supports "teaching the model how to think" because internal
actions become trainable and measurable rather than hidden side effects of next
token prediction.

## Core Functions

The improved model should expose these mathematical functions.

### 1. Belief Function

```text
b_t = B_theta(o_t, h_t, S_t^real)
```

`b_t` is the compact decision state used by policy, value, verifier, and world
model heads. It should include content hidden state, identity coordinates,
memory reads, route information, and authority features.

### 2. Identity Coordinate Function

```text
w_t = Q_theta(h_t, S_t^real)
u_t = sum_i w_{t,i} p_i
```

This keeps the earlier rule: every token/action gets identity coordinates in a
shared field, not a private identity field.

### 3. Sparse Route Function

```text
R_t = Route_theta(b_t, S_t^real)
sum_i R_{t,i} e_i <= B
```

The route selects the active reasoning/program modules under an energy budget.

### 4. Memory Read Function

```text
m_t = Read_theta(b_t, R_t, S_t^real)
```

`m_t` combines program memory and content-addressed memory reads. It is a
decision input, not just a hidden-state decoration.

### 5. Thought/Computation Function

```text
c_t = Think_theta(b_t, m_t, R_t, X_t)
```

`c_t` is the model's internal computation state. In a pure language model it is
absorbed into hidden states. In an agentic model it can be supervised by process
traces, tool traces, verifier traces, or RL rollouts.

### 6. Scratchpad Function

```text
X_{t+1} = ScratchUpdate_theta(X_t, c_t, a_t, o_t, v_t)
```

`X_t` is a bounded working memory for intermediate reasoning. It should support:

- candidate plans and branches;
- simulated futures and their scores;
- tool call arguments and returned observations;
- verifier notes and uncertainty flags;
- short process traces used for supervision.

The scratchpad should be compressed or pruned:

```text
size(X_t) <= B_scratch
X_t = PruneOrSummarize(X_t, utility, recency, risk)
```

Scratchpad writes do not imply persistent memory writes. Persistent memory
requires the commit gate.

### 7. Policy Function

```text
pi_theta(a_t | c_t, S_t^real, X_t)
```

This chooses whether to emit, think, retrieve, route, verify, call a tool,
simulate, write memory, or stop.

### 8. Value Function

```text
V_theta(c_t, S_t^real, X_t) = E[sum_{k>=0} gamma^k r_{t+k}]
```

This estimates expected outcome quality and is required for RL, planning, and
cost-aware stopping.

### 9. World / Future Simulation Function

```text
S_{t+k+1}^imag, X_{t+k+1}^imag, o_{t+k+1}~, r_{t+k}~ =
    W_theta(S_{t+k}^imag, X_{t+k}^imag, a_{t+k})
```

The world model predicts possible future observations, rewards, and imagined
state transitions. The scratchpad is the natural place to hold the imagined
trajectory, branch score, assumptions, and verifier status while planning.

### 10. Planner Function

```text
tau^* =
argmax_{a_{t:t+K}}
    E_W[sum_{k=0}^{K} gamma^k r_{t+k}~
        - lambda_cost Cost(a_{t+k})
        - lambda_risk Risk(S_{t+k}^imag)]
```

The planner searches over imagined action trajectories. It can be implemented
as sampling, beam search, tree search, learned latent planning, or a small
deliberation loop. The planner should write branches into `X_t`, not directly
into `S_t^real`.

### 11. Verifier / Authority Function

```text
v_t = Verify_theta(c_t, a_t, y_t, evidence_t, S_t^real, X_t)
```

The verifier estimates whether the current output/action is supported by the
right evidence source. It should detect false authority, cross-domain
contamination, weak retrieval, failed tool execution, and unsupported imagined
states.

### 12. Commit Gate

```text
k_t = Commit_theta(S_t^real, S_t^imag, X_t, v_t)
S_{t+1}^real =
    RealUpdate(S_t^real, observed_delta_t)
    + k_t * VerifiedDelta(S_t^imag, X_t, S_t^real)
```

The commit gate is the guardrail between imagination and memory. Simulated
states can influence decisions, but they should enter persistent memory only
when verified or explicitly marked as hypothetical.

## Improved Training Objective

The improved objective is:

```text
L_total =
    L_lm
    + L_identity
    + L_memory
    + L_route
    + L_authority
    + L_policy
    + L_value
    + L_world
    + L_process
    + L_cost
```

with:

```text
L_policy = -log pi_theta(a_t^* | c_t, S_t)
L_value = (V_theta(c_t, S_t) - G_t)^2
L_world = CE(o_{t+1}~, o_{t+1}) + (r_t~ - r_t)^2
L_process = distance(thought_trace_pred, thought_trace_target)
L_authority = false_authority + cross_domain_contamination + verifier_error
L_cost = compute_cost + memory_write_cost + tool_cost + planning_cost
```

For reinforcement learning, the policy can be optimized by:

```text
J(theta) = E[
    sum_t gamma^t
    (r_t - lambda_cost Cost_t - lambda_risk Risk_t)
]
```

with an advantage estimator:

```text
A_t = G_t - V_theta(c_t, S_t)
L_RL = -log pi_theta(a_t | c_t, S_t) * stopgrad(A_t)
       + beta_v L_value
       - beta_H H(pi_theta)
```

The key constraint is:

```text
imagined_success != verified_success
```

World-model rollouts may improve action choice, but persistent memory should
prefer observed facts, verified tool outputs, and explicitly labeled
hypotheses.

## Improved Behavioral Gates

The final model should pass these behavior gates:

```text
CarryScore > ResetScore
CarryScore > ShuffledStateScore
VerifiedPlanningScore > NoPlanningScore
WorldPredictionError <= tau_world
FalseAuthorityRate <= tau_false_authority
HypothesisContaminationRate <= tau_contamination
CostAdjustedReward > baseline
RoleStability >= tau_role
KnockoutUtility >= tau_knockout
```

This makes the model's "thinking" measurable:

- If planning helps, verified planning must beat no-planning.
- If memory helps, carried state must beat reset and shuffled state.
- If simulation helps, imagined futures must predict real outcomes.
- If authority works, trusted evidence must be correct and domain-appropriate.
- If identity is real, program-role knockouts must change behavior selectively.

## RL Post-Training Roadmap

The attached RL survey suggests that TAC should not start with full PPO. PPO
requires a trainable critic, reference model, reward model, and active policy,
which is a poor first fit for TAC because the identity field already adds
memory and routing cost. The practical path is staged.

### Stage 1: Group-Relative Scratchpad Training

Use GRPO/RLOO-style group-relative optimization over multiple candidate
scratchpad/action trajectories for the same prompt:

```text
{tau_i}_{i=1..G} ~ pi_old(. | prompt, S^real)
r_i = VerifierReward(tau_i)
A_i = (r_i - mean_j r_j) / std_j(r_j)
```

Then optimize:

```text
L_group =
    -mean_i A_i log pi_theta(tau_i | prompt, S^real)
    + beta_KL KL(pi_theta || pi_ref)
```

This fits TAC because:

- it avoids a large critic at the start;
- it can score full scratchpad trajectories with existing verifiers;
- it naturally compares different routes, memory reads, and tool decisions for
  the same prompt;
- it works with warm-started models rather than requiring open-ended
  exploration from scratch.

### Stage 2: DAPO-Style Stability Features

Add the stability features before scaling:

```text
dynamic_sampling(prompt) = keep only if group rewards are not all equal
clip_low = 0.20
clip_high ~= 0.28
L_length = penalty(overlong scratchpad or response)
```

Dynamic sampling is especially useful for TAC because many prompts will be
uninformative: all candidate trajectories may either solve the task or fail it.
Those prompts should not drive route, memory, or scratchpad updates.

The overlong penalty is also important. Without it, the scratchpad may learn to
spend excess computation instead of learning better routing or verification.

### Stage 3: GSPO-Style Sequence-Level Updates

TAC has sparse program routing, so token-level importance ratios can be noisy.
For route-heavy or scratchpad-heavy rollouts, prefer sequence-level clipping:

```text
s_i(theta) =
    exp((1 / |tau_i|) *
        sum_t log pi_theta(a_{i,t} | state_{i,t})
        - log pi_old(a_{i,t} | state_{i,t}))
```

```text
L_sequence =
    -mean_i min(
        s_i A_i,
        clip(s_i, 1 - eps, 1 + eps) A_i
    )
```

This aligns the optimization unit with the reward unit: the whole reasoning,
tool, memory, and output trajectory.

### Stage 4: PRIME-Style Implicit Process Rewards

Once exact outcome verifiers exist, use implicit process rewards to train the
scratchpad without hand-labeled process traces:

```text
r_process(t) = beta * log(pi_theta(step_t) / pi_ref(step_t))
```

combined with verified outcome reward:

```text
r_total = r_outcome + lambda_process * discounted(r_process)
```

For TAC, the process reward should be attached to:

- memory reads that improve later verification;
- route choices that improve knockout-sensitive behavior;
- scratchpad steps that reduce verifier uncertainty;
- simulation branches that predict observed outcomes.

This is a better fit than manually annotating every reasoning step.

### Stage 5: VAPO-Style Value Head For Hard Exploration

After scratchpad and verifier gates pass, add a lightweight value head over
`(c_t, S_t^real, X_t)`:

```text
V_theta(c_t, S_t^real, X_t)
```

Use value warmup, auxiliary LM loss, and length-adaptive GAE:

```text
lambda_policy(length) = 1 - 1 / (alpha * length)
A_t = GAE_length_adaptive(r_t, V_t, lambda_policy)
```

This should be delayed until needed. The survey's main warning is that
critic-free methods refine warm-started policies efficiently, but complex
exploration benefits from a value model. TAC's multi-hop planning and tool
simulation are likely in that second category.

### Stage 6: Filtration And Replay

Add filtration for noisy reward or off-policy data:

```text
keep(sample) =
    verifier_confident(sample)
    and not reward_outlier(sample)
    and not authority_violation(sample)
```

Rejected examples should still be logged as authority events, but they should
not necessarily update the active policy.

## RL Recommendation

The immediate TAC post-training target should be:

```text
GRPO/RLOO + DAPO filtering + scratchpad trajectories + verifier rewards
```

not:

```text
full PPO with a large critic
```

The later target should be:

```text
GSPO/VAPO hybrid:
    sequence-level route/scratchpad optimization
    + lightweight value head
    + length-adaptive advantages
    + verified commit gate
```

This staged path matches the current evidence: the base identity/memory model is
promising, but planner/world/reward heads are not ready to promote until memory
use, scratchpad behavior, and verifier outcomes pass carry/reset/shuffle and
authority gates.

## What Is Still Mathematically Unsettled

The formal target exposes the remaining open questions:

- Seed stability is not proven. Program roles must be stable up to permutation,
  not just present in one seed.
- Multi-hop retrieval is not solved. Two-step content synthesis helps direct
  memory, but Phase D must test multi-hop chain retrieval and long context.
- Decode economics are unresolved. TAC is data-efficient in memory tasks, but
  the identity paths still carry a wall-clock cost.
- Route MI can fail by seed. The objective may need a stronger role-alignment
  or anti-collapse term if Phase B failures persist.
- Agentic behavior should stay mostly outside the base model until carried
  memory beats reset and shuffled state on action/tool tasks.

## Recommended Next Math Step

The next clean mathematical refinement is to make role stability explicit:

```text
L_role =
    -I(category; selected_program)
    + lambda_perm * min_{pi in Sym(P)} d(RoleSeedA, pi(RoleSeedB))
    + lambda_knockout * max(0, tau_knockout - KnockoutUtility)
```

In practice, this means Phase C should not merely report stability. It should
become a training or selection pressure once enough seed profiles exist. That is
the path from "we found a promising identity field" to "the model has stable,
interpretable, reusable program identities."
