# TAC Agentic RL Mathematical Contract

Date: 2026-06-04

This contract reviews the RL/post-training survey against the current TAC model
and defines the improved mathematical representation we should target next.

## Summary Verdict

The current TAC-Control-v1 model is a strong identity/memory transformer, but
it is not yet a full agentic RL model. It has:

- persistent identity state;
- content-addressed memory;
- sparse program routing;
- identity-first attention;
- experimental action/world/reward/reflection/planner heads.

It does not yet have:

- explicit scratchpad state;
- explicit imagined future state;
- verifier-gated commit from scratchpad/simulation into real memory;
- group-relative trajectory optimization;
- sequence-level route/scratchpad RL;
- value-guided planning for hard exploration.

The improved target is:

```text
TAC-Agent-RL =
    (Backbone, IdentityField, Router, Memory, Scratchpad,
     Policy, WorldModel, Planner, Verifier, CommitGate,
     Reward, Value, CostGate)
```

The immediate post-training target should be:

```text
GRPO/RLOO + DAPO filtering + verifier-scored scratchpad trajectories
```

The later target should be:

```text
GSPO/VAPO hybrid for sequence-level route/scratchpad optimization
```

## Current Model Mapping

| Contract Component | Current State | Gap |
| --- | --- | --- |
| `Backbone` | Implemented in `TACTransformerLM`; joint TAC/controller local gate proved | Needs external-scale rollout training evidence. |
| `IdentityField` | Implemented through token-program activations/coherence; identity persistence math added | Needs Phase B identity-persistence logging across seeds. |
| `Router` | Implemented: energy, BASE, base_semantic, authority_gated, and coalition variants | Needs capability-gated coalition promotion evidence. |
| `Memory` | Implemented: program memory, content-addressed memory, and local overlap-link math | Needs live memory-link retrieval paths in Phase C/D. |
| `Scratchpad` | Implemented as bounded `AgenticScratchpadState` | Needs live Phase D scratchpad advantage. |
| `Policy` | Implemented through `AgenticPolicyController` heads and trajectory objectives | Needs live rollout RL on external tasks. |
| `WorldModel` | Partial: next-observation/simulation primitives | Needs learned multi-step imagined transition. |
| `Planner` | Implemented locally through reward-risk-cost branch selection | Needs live search over imagined trajectories. |
| `Verifier` | Implemented through authority reports, verifier rewards, and gated scratchpad commit | Needs model/runtime verifier integration at scale. |
| `CommitGate` | Implemented locally for verified scratchpad state | Needs persistent memory write integration. |
| `Value` | Implemented as controller value head and value regression loss | Needs VAPO-style warmup on live hard-exploration tasks. |
| `RL objective` | Implemented locally: GRPO/RLOO, DAPO, GSPO, PRIME, value support | Needs end-to-end empirical capability advantage. |

## State Spaces

Let:

```text
o_t        = observed token, tool result, verifier result, or environment event
h_t        = backbone hidden state
P          = number of identity programs
d          = hidden width
K          = content-memory slots
B_scratch  = scratchpad budget
```

The real persistent state is:

```text
S_t^real = (I_t, M_t, C_t, V_t, A_t, K_t, G_t, P_t)
```

where:

```text
I_t in R^P              identity stability / program state
M_t in R^{P x d}        program memory
C_t in R^{K x d}        content memory cues
V_t in R^{K x d}        content memory values
A_t                    authority/confidence state
K_t                    decode/cache state
G_t                    memory-overlap / coalition-link graph
P_t                    identity-persistence statistics
```

The imagined state is:

```text
S_{t,b,k}^imag
```

for planning branch `b` and rollout step `k`.

The scratchpad is:

```text
X_t = {x_j}_{j=1..N_t},  N_t <= B_scratch
```

Each scratchpad item is:

```text
x_j = (type_j, payload_j, source_j, confidence_j, expiry_j, verified_j)
```

The scratchpad is working memory. It is not persistent memory. It can hold
candidate plans, tool arguments, tool outputs, simulated states, verifier notes,
temporary assumptions, and process traces.

Program identity is still role-based up to permutation:

```text
RoleSet_t = {(p_i, I_{t,i}, M_{t,i}, route_i, knockout_i)}_{i=1..P} / Sym(P)
```

Identity persistence is the preferred post-specialization metric:

```text
IPS(p) =
    (retrieval_recurrence(p)
   + memory_survival_rate(p)
   + reuse_frequency(p)) / 3
```

Memory linkage is induced by identity overlap plus temporal/coactivation
proximity:

```text
overlap_ij = cosine(M_i, M_j)
link_ij = 1[overlap_ij > tau_link] * 1[coactivation_window_ij <= tau_time]
```

Basal/apical belief decomposition separates current evidence from top-down
identity proposals:

```text
b_t^basal  = f(h_t, retrieved_memory_t)
b_t^apical = g(I_t, X_t, route_context_t)
agreement_t = cosine(b_t^basal, b_t^apical)
b_t = integrate(b_t^basal, b_t^apical, agreement_t)
```

## Core Functions

### 1. Encode

```text
h_t = Enc_theta(o_{\le t}, K_{t-1})
```

Produces local content representation and updates decode-local cache.

### 2. Identity Coordinate

```text
z_{t,i} = sqrt(d) * <normalize(h_t), normalize(p_i)>
a_{t,i} = sigmoid(z_{t,i})
w_{t,i} = softmax_i(z_{t,i} + log(max(I_{t-1,i}, eps)))
u_t = sum_i w_{t,i} p_i
```

Every token/action receives identity coordinates in the shared field.

### 3. Coherence

```text
Kappa_{t,u} = sum_i w_{t,i} w_{u,i}
```

Attention can use:

```text
Attn = softmax(QK^T / sqrt(d_h) + beta * Kappa + causal_mask) V
```

### 4. Route

```text
R_t = Route_theta(h_t, w_t, I_{t-1}, X_t)
sum_i R_{t,i} cost_i <= B_route
```

For the current TAC-Control family, this corresponds to base-semantic routing:

```text
R_t = BASE_anchor(t) union TopK_semantic(a_t / cost, k - 1)
```

### 5. Memory Read

```text
m_t = Read_theta(h_t, R_t, S_t^real)
```

Read combines:

```text
m_t = program_read_t + content_read_t
```

For content-addressed memory:

```text
score_k = <normalize(h_t), normalize(C_{t,k})>
pi_k = softmax_k(score_k + mask_k)
content_read_t = sum_k pi_k V_{t,k}
```

### 6. Belief

```text
b_t = B_theta(h_t, u_t, R_t, m_t, X_t, A_t)
```

`b_t` is the compact decision state used by policy, value, verifier, and world
model functions.

### 7. Scratchpad Update

```text
X_{t+1} = ScratchUpdate_theta(X_t, b_t, a_t, o_t, v_t)
```

with pruning:

```text
N_{t+1} <= B_scratch
X_{t+1} = Prune(X_{t+1}, utility, recency, risk, verified)
```

Scratchpad writes are cheap and reversible. They do not imply persistent memory
writes.

### 8. Policy

```text
pi_theta(a_t | b_t, X_t, S_t^real)
```

The action space should include:

```text
emit_token
internal_thought
retrieve_memory
route_program
simulate
call_tool
verify
write_scratchpad
commit_memory
stop
```

### 9. World Model

```text
S_{t,b,k+1}^imag, X_{t,b,k+1}^imag, o_{t,b,k+1}~, r_{t,b,k}~ =
    W_theta(S_{t,b,k}^imag, X_{t,b,k}^imag, a_{t,b,k})
```

The world model should predict:

```text
next observation
reward
uncertainty
state delta
authority risk
```

Imagined rollouts remain hypothetical until verified.

### 10. Planner

```text
tau_b = (a_{t,b,0}, ..., a_{t,b,K})
```

```text
Plan_theta =
argmax_b E_W[
    sum_{k=0}^{K} gamma^k r_{t,b,k}~
    - lambda_cost Cost(tau_b)
    - lambda_risk Risk(S_{t,b,k}^imag, X_{t,b,k}^imag)
]
```

Planner outputs should be written to scratchpad:

```text
X_t <- X_t union {branch_summary, score, assumptions, verifier_status}
```

not directly to persistent memory.

### 11. Verifier / Authority

```text
v_t = Verify_theta(b_t, a_t, y_t, evidence_t, X_t, S_t^real)
```

Verifier output:

```text
v_t = (supported, confidence, source_domain, target_domain, false_authority_risk)
```

It should reject:

```text
unsupported memory
unsupported tool output
unverified simulated state
cross-domain contamination
low-confidence route decisions
```

### 12. Commit Gate

```text
k_t = Commit_theta(S_t^real, S_t^imag, X_t, v_t)
```

Persistent update:

```text
S_{t+1}^real =
    RealUpdate(S_t^real, observed_delta_t)
    + k_t * VerifiedDelta(S_t^imag, X_t, S_t^real)
```

The key invariant is:

```text
imagined_state cannot become persistent_state without verifier support
```

### 13. Value

```text
V_theta(b_t, X_t, S_t^real) =
    E[sum_{k>=0} gamma^k r_{t+k}]
```

This should be added after scratchpad/verifier training is stable.

### 14. Cost

```text
Cost_t =
    c_route * active_programs_t
    + c_memory * memory_reads_t
    + c_scratch * size(X_t)
    + c_tool * tool_calls_t
    + c_sim * imagined_steps_t
    + c_decode * generated_tokens_t
```

The policy should maximize verified utility per cost.

## Execution Loop

```text
for each step t:
    h_t = Enc(o_{\le t}, K_{t-1})
    w_t, u_t = IdentityCoordinate(h_t, S_t^real)
    R_t = Route(h_t, w_t, S_t^real, X_t)
    m_t = Read(h_t, R_t, S_t^real)
    b_t = Belief(h_t, u_t, R_t, m_t, X_t, A_t)
    a_t ~ Policy(b_t, X_t, S_t^real)

    if a_t == simulate:
        write imagined branches into X_t
    if a_t == call_tool:
        observe tool output and write result into X_t
    if a_t == verify:
        v_t = Verify(...)
    if a_t == commit_memory:
        S_t^real = Commit(...)
    if a_t == emit_token:
        y_t ~ p_theta(. | b_t, X_t, S_t^real)

    X_{t+1} = ScratchUpdate(...)
    S_{t+1}^real = gated real-state update
```

## Training Objectives

### Supervised Warm Start

```text
L_sft =
    L_lm
    + lambda_action L_action
    + lambda_process L_process
    + lambda_verify L_verify
    + lambda_commit L_commit
```

This teaches formatting, basic tool use, scratchpad structure, and verifier
semantics before RL.

### Group-Relative Policy Optimization

For prompt `q`, sample `G` trajectories:

```text
tau_i ~ pi_old(. | q, S^real), i=1..G
r_i = R(tau_i)
A_i = (r_i - mean_j r_j) / (std_j(r_j) + eps)
```

Objective:

```text
L_grpo =
    -mean_i A_i log pi_theta(tau_i | q)
    + beta_KL KL(pi_theta || pi_ref)
```

This should be the first RL method because it avoids a large critic.

### DAPO-Style Filtering

Use only informative groups:

```text
keep(q) = not all_equal({r_i}_{i=1..G})
```

Use asymmetric clipping:

```text
clip_low = 1 - eps_low
clip_high = 1 + eps_high
eps_high > eps_low
```

Add length/cost penalty:

```text
r_i' = r_i - lambda_len max(0, len(tau_i) - len_budget)
```

### GSPO-Style Sequence Objective

Use sequence-level ratios for route-heavy and scratchpad-heavy trajectories:

```text
s_i(theta) =
    exp((1 / |tau_i|)
        sum_t [log pi_theta(a_{i,t}|state_{i,t})
             - log pi_old(a_{i,t}|state_{i,t})])
```

```text
L_gspo =
    -mean_i min(
        s_i A_i,
        clip(s_i, 1 - eps, 1 + eps) A_i
    )
```

This is better aligned with TAC because reward is usually assigned to the whole
route/memory/scratchpad/output trajectory, not isolated tokens.

### PRIME-Style Implicit Process Reward

When exact outcome verifiers exist:

```text
r_process(t) = beta * log(pi_theta(step_t) / pi_ref(step_t))
```

Total reward:

```text
r_total =
    r_outcome
    + lambda_process discounted(r_process)
    - lambda_authority false_authority
    - lambda_cost Cost
```

Attach process rewards to:

```text
useful memory reads
useful route choices
verified scratchpad steps
simulation branches that predict observed outcomes
tool calls that improve final verification
```

### VAPO-Style Value Head

After GRPO/DAPO/GSPO scratchpad training passes gates, add:

```text
V_theta(b_t, X_t, S_t^real)
```

Use value warmup:

```text
for first N_value_warmup steps:
    train V only, keep policy objective conservative
```

Use length-adaptive advantages:

```text
lambda_policy(L) = 1 - 1 / (alpha * L)
A_t = GAE(r_t, V_t, lambda_policy(L))
```

This is for hard exploration: multi-hop planning, tool selection, and long
scratchpad trajectories.

## Reward Function

The reward should be explicit:

```text
R(tau) =
    w_task * TaskSuccess(tau)
    + w_verify * VerificationPass(tau)
    + w_state * StateUtility(tau)
    + w_route * RouteUtility(tau)
    + w_ips * IdentityPersistence(tau)
    + w_link * MemoryLinkUtility(tau)
    + w_coalition * CoalitionUtility(tau)
    + w_world * WorldPredictionAccuracy(tau)
    - w_false * FalseAuthority(tau)
    - w_contam * HypothesisContamination(tau)
    - w_cost * Cost(tau)
```

Where:

```text
StateUtility = Score(carry) - max(Score(reset), Score(shuffled))
RouteUtility = knockout_delta or route-specific contribution
IdentityPersistence = mean IPS over programs used by tau
MemoryLinkUtility = utility gained by linked-neighbor retrieval over direct recall
CoalitionUtility = verified gain from co-active program network use
HypothesisContamination = imagined facts committed as real without support
```

## Behavioral Gates

Do not promote the agentic/RL path unless:

```text
CarryScore > ResetScore
CarryScore > ShuffledScore
ScratchpadPolicy > NoScratchpadPolicy
VerifiedPlanning > NoPlanning
WorldPredictionError <= tau_world
FalseAuthorityRate <= tau_false_authority
HypothesisContaminationRate <= tau_contamination
CostAdjustedReward > baseline
RoleStability >= tau_role
KnockoutUtility >= tau_knockout
```

## Executable Proof Status

Local proof artifact:

```text
runs/benchmarks/agentic_rl_math_proof_2026_06_04
```

Implemented proof primitives:

```text
tac_transformer.agentic_rl_math
```

The proof covers the mathematical support layer, not default model promotion.
It verifies:

```text
cost-adjusted rewards penalize expensive rollouts
group-relative advantages are zero-mean within a sampled group
policy-gradient loss increases positive-advantage actions and suppresses negative-advantage actions
scratchpad updates are budget-bounded
imagined scratchpad items cannot commit without verifier support
future simulation selects the best reward-risk-cost branch
verified process-teaching loss prefers correct verified steps
```

Result:

```text
mathematical_support = proved
empirical_promotion = blocked
```

The empirical block uses the existing all-agentic benchmark:

```text
runs/benchmarks/agentic_full_layers_2026_05_28/aggregate_agentic_full_layers.json
```

The selected `all_agentic` row remains blocked:

```text
carry_score = 0.2695
reset_score = 0.2773
shuffled_score = 0.2734
baseline_score = 0.2604
cost_adjusted_reward = 0.1511
baseline_cost_adjusted_reward = 0.2604
```

Failed promotion checks:

```text
carry_beats_reset = false
carry_beats_shuffled = false
scratchpad_beats_no_scratchpad = false
simulation_beats_no_simulation = false
teaching_beats_no_teaching = false
world_error_bounded = false
cost_adjusted_reward_beats_baseline = false
```

Interpretation:

```text
The broader RL/scratchpad/simulation/teaching functions are now executable and
mathematically coherent, but the current model-side all-agentic stack is not
promotable. The next proof step must add first-class scratchpad and simulation
benchmarks with explicit no-scratchpad/no-simulation/no-teaching controls.
```

## Scratchpad / Simulation Mechanism Gate

Local mechanism artifact:

```text
runs/benchmarks/scratchpad_simulation_proof_2026_06_04
```

Benchmark implementation:

```text
experiments/benchmark_scratchpad_simulation_proof.py
```

This benchmark creates synthetic tasks with explicit controls:

```text
scratchpad       vs no_scratchpad
simulation       vs no_simulation
process teaching vs no_teaching
verified commit  vs unverified imagined commit
```

Result on 64 examples:

```text
scratchpad_score = 1.0000
no_scratchpad_score = 0.0000
scratchpad_gain = 1.0000

simulation_score = 1.0000
no_simulation_score = 0.0000
simulation_gain = 1.0000

teaching_score = 1.0000
no_teaching_score = 0.2500
teaching_gain = 0.7500

hypothesis_contamination_rate = 0.0000
```

Decision:

```text
mechanisms_proved
```

Scope:

```text
This proves the control mechanics and evaluation gate. It does not prove that
TAC-Control-v1 has learned scratchpad use, future simulation, or process
teaching end to end. The next development step is to connect this gate to a
trainable controller/model path and require it to beat the same controls.
```

## Trainable Controller Gate

Local controller-learning artifact:

```text
runs/benchmarks/agentic_controller_learning_2026_06_04
```

Benchmark implementation:

```text
experiments/benchmark_agentic_controller_learning.py
```

Controller implementation:

```text
tac_transformer/agentic_controller.py
```

This benchmark trains explicit policy heads over verifier-labeled traces:

```text
scratchpad commit logits
future-branch selection logits
process-step teaching logits
```

Result on 64 examples:

```text
initial_loss = 3.134920
final_loss = 0.003444
loss_reduction = 3.131476

scratchpad_policy_score = 1.0000
no_scratchpad_score = 0.0000

simulation_policy_score = 1.0000
no_simulation_score = 0.0000

teaching_policy_score = 1.0000
no_teaching_score = 0.2500

hypothesis_contamination_rate = 0.0000
```

Decision:

```text
policy_learned
```

Scope:

```text
This proves a trainable policy controller can learn the scratchpad, simulation, and
process-teaching policies from verified traces. It is still not an end-to-end
TAC language-model result because the controller is trained on synthetic
trace features rather than on live hidden states emitted by TAC-Control-v1.
The next development step is to wire these policy heads to TAC hidden/state
features and require the learned controller to preserve Phase B/D capability
while improving internal action quality.
```

## Live TAC Feature Adapter Gate

Local live-adapter artifact:

```text
runs/benchmarks/live_agentic_policy_adapter_2026_06_04
```

Benchmark implementation:

```text
experiments/benchmark_live_agentic_policy_adapter.py
```

Public adapter functions:

```text
build_agentic_policy_features_from_tac_output
run_agentic_policy_controller_from_tac_output
```

This gate runs a real tiny `TACTransformerLM` forward pass, converts its
`TACOutput` into policy-controller features, runs `AgenticPolicyController`,
and backpropagates from policy logits into the TAC token embeddings.

Result:

```text
scratchpad_features = [2, 3, 8]
simulation_features = [2, 3, 5]
context_features = [2, 4]

scratchpad_logits = [2, 3]
simulation_logits = [2, 3]
process_logits = [2, 4, 4]

token_embedding_grad_abs_sum = 0.026591
```

Decision:

```text
live_features_connected
```

Scope:

```text
This proves the policy controller can consume live TAC outputs and
backpropagate through TAC hidden-state features. It still does not prove
end-to-end scratchpad/simulation capability gains on Phase B or Phase D tasks.
The next gate must train these live-connected heads against verified action
targets while preserving the existing TAC capability scorecards.
```

## Frozen TAC Live Policy Training Gate

Local frozen-live training artifact:

```text
runs/benchmarks/live_agentic_policy_training_2026_06_04
```

Benchmark implementation:

```text
experiments/benchmark_live_agentic_policy_training.py
```

This gate trains `AgenticPolicyController` on live TAC-derived feature tensors
while freezing the TAC backbone. It then reruns the same TAC input batch and
compares logits and evaluation loss before versus after controller training.

Result:

```text
initial_loss = 3.169181
final_loss = 0.000436
loss_reduction = 3.168745

scratchpad_policy_score = 1.0000
simulation_policy_score = 1.0000
teaching_policy_score = 1.0000

before_eval_loss = 3.580766
after_eval_loss = 3.580766
eval_loss_drift = 0.000000
max_logit_drift = 0.000000
```

Decision:

```text
live_policy_trained_capability_preserved
```

Scope:

```text
This proves the live-connected policy heads can learn verified action targets
without degrading a frozen TAC backbone on the local gate. It does not yet
prove joint TAC+controller training, scratchpad state integration, or Phase B/D
task improvement.
```

## First-Class Scratchpad State Gate

Local scratchpad-state artifact:

```text
runs/benchmarks/agentic_scratchpad_state_2026_06_04
```

Benchmark implementation:

```text
experiments/benchmark_agentic_scratchpad_state.py
```

Public state/transition functions:

```text
AgenticScratchpadState
apply_agentic_scratchpad_transition
```

This gate adds first-class scratchpad state `X_t` and applies controller commit
logits through a verifier-gated transition. Verification is applied before the
budgeted update, so an unverified imagined hypothesis cannot consume scratchpad
capacity ahead of supported observations.

Result:

```text
initial_state = []
selected_ids = [left, right, imagined]
committed_ids = [left, right]
rejected_ids = [imagined]
next_state = [left, right]
next_step = 1
hypothesis_contamination_rate = 0.0000
```

Decision:

```text
scratchpad_state_verified
```

Scope:

```text
This proves first-class scratchpad state can consume controller commit logits,
require verifier support before budgeted state update, and block unverified
imagined hypotheses. It does not yet attach scratchpad state to autoregressive
decoding or Phase B/D task execution.
```

## Phase D Scratchpad State Execution Gate

Local Phase D scratchpad execution artifact:

```text
runs/benchmarks/phase_d_scratchpad_state_execution_2026_06_04
```

Benchmark implementation:

```text
experiments/benchmark_phase_d_scratchpad_state_execution.py
```

Public Phase D scratchpad functions:

```text
format_agentic_scratchpad_context
augment_phase_d_prompt_with_scratchpad
run_phase_d_scratchpad_state_predictions
```

This gate attaches verified `AgenticScratchpadState` to Phase D prompts and
emits standard prediction rows accepted by `score_phase_d_predictions`.
Unverified imagined scratchpad payloads are carried in state for auditability
but are excluded from the augmented task prompt and answer path.

Result:

```text
example_count = 5
prediction_count = 5
mean_primary_score = 1.0000
unverified_prompt_leaks = 0

multi_hop_chain_retrieval = 1.0000
long_context_retrieval_4096 = 1.0000
episodic_fact_update = 1.0000
tool_selection = 1.0000
delayed_goal_binding = 1.0000
```

Decision:

```text
phase_d_scratchpad_state_execution_verified
```

Scope:

```text
This proves verified scratchpad state can enter the Phase D task execution and
scoring contract while excluding unverified imagined payloads. It does not yet
prove learned autoregressive decoding from scratchpad context or end-to-end
joint TAC+controller training.
```

## Learned Scratchpad Autoregressive Decoding Gate

Local learned-decoding artifact:

```text
runs/benchmarks/scratchpad_autoregressive_decoding_2026_06_04
```

Benchmark implementation:

```text
experiments/benchmark_scratchpad_autoregressive_decoding.py
```

This gate trains a real `TACTransformerLM` decoder on verified
scratchpad-augmented prompts using masked next-token loss. Prompt tokens are
ignored by the loss; only the completion bytes after the prompt are supervised.
The trained model is then evaluated with greedy byte-level autoregressive
generation through `generate_phase_d_completion`.

Result:

```text
train_examples = 80
train_steps = 50
initial_loss = 5.438906
final_loss = 0.111742

scratchpad_score = 1.0000
counterfactual_score = 1.0000
no_scratchpad_score = 0.1000
scratchpad_control_margin = 0.9000
```

Decision:

```text
scratchpad_autoregressive_decoding_proved
```

Scope:

```text
This proves learned autoregressive decoding from verified scratchpad context on
a compact local digit-copy gate. The counterfactual score shows the generated
answer follows the scratchpad value, while the no-scratchpad control stays low.
It does not yet prove full Phase D task solving or joint TAC+controller
optimization.
```

## Joint TAC + Controller Training Gate

Local joint-training artifact:

```text
runs/benchmarks/joint_tac_controller_training_2026_06_04
```

Benchmark implementation:

```text
experiments/benchmark_joint_tac_controller_training.py
```

This gate jointly optimizes `TACTransformerLM` and `AgenticPolicyController`
with a single optimizer. The objective is masked decoder next-token loss plus
supervised controller action loss. Unlike the frozen-live policy gate, TAC
parameters are trainable, controller parameters are trainable, and the benchmark
explicitly measures policy-loss gradient flow into TAC.

Result:

```text
train_examples = 80
train_steps = 50

initial_decoder_loss = 5.907524
final_decoder_loss = 0.141864
initial_controller_loss = 3.234594
final_controller_loss = 0.150132

scratchpad_score = 1.0000
no_scratchpad_score = 0.1000
controller_scratchpad_score = 1.0000
controller_simulation_score = 1.0000
controller_teaching_score = 1.0000

tac_policy_grad_abs_sum = 3.262380
tac_max_abs_delta = 0.439958
controller_max_abs_delta = 0.566976
```

Decision:

```text
joint_tac_controller_training_proved
```

Scope:

```text
This proves local joint TAC+controller optimization: decoder loss improves,
controller loss improves, both parameter sets update, policy loss reaches TAC
gradients, scratchpad-conditioned autoregressive decoding remains successful,
and controller scratchpad/simulation/teaching policies are learned. It does not
prove large-scale RL rollout training or external Phase B/D benchmark
improvement.
```

## Agentic Trajectory Record Gate

Local trajectory-record artifact:

```text
runs/benchmarks/agentic_trajectory_records_2026_06_04
```

Benchmark implementation:

```text
experiments/benchmark_agentic_trajectory_records.py
```

Public trajectory functions:

```text
AgenticTrajectoryStep
AgenticTrajectory
build_agentic_trajectory
trajectory_to_training_record
```

This gate adds first-class rollout records for later RL objectives. Each
trajectory preserves internal actions, action log probabilities, route IDs,
memory-read IDs, scratchpad state IDs, verifier scores, step costs, final
reward, and cost-adjusted reward.

Result:

```text
actions = [read_memory, write_scratchpad, answer]
action_logprob_sum = -0.6500
route_ids = [program_memory, scratchpad_writer, decoder]
total_cost = 0.3000
final_reward = 1.0000
cost_adjusted_reward = 0.8500
verifier_mean = 0.9667
```

Decision:

```text
trajectory_records_proved
```

Scope:

```text
This proves first-class trajectory records can preserve the data needed for
later RL objectives and audit trails. It does not yet implement verifier reward
shaping or group-relative trajectory training.
```

## Authority-Report Verifier Reward Gate

Local verifier-reward artifact:

```text
runs/benchmarks/agentic_verifier_rewards_2026_06_04
```

Benchmark implementation:

```text
experiments/benchmark_agentic_verifier_rewards.py
```

Public reward function:

```text
verifier_reward_from_authority_report
```

This gate consumes the existing `AuthorityReport.to_manifest()` contract and
converts trusted verification evidence into reward shaping. Clean trusted
verification receives a bonus; false authority and cross-domain authority
violations receive explicit penalties.

Result:

```text
clean_verifier_reward = 1.2500
contaminated_verifier_reward = -0.2500
false_authority_rate = 0.3333
trusted_accuracy = 0.6667
```

Decision:

```text
verifier_rewards_proved
```

Scope:

```text
This proves verifier reward shaping can use existing authority-report evidence
and penalize false authority plus cross-domain contamination. It does not yet
apply shaped rewards inside group-relative policy training.
```

## Group-Relative Trajectory Training Gate

Local group-relative trajectory artifact:

```text
runs/benchmarks/group_relative_trajectory_training_2026_06_04
```

Benchmark implementation:

```text
experiments/benchmark_group_relative_trajectory_training.py
```

Public trajectory policy function:

```text
group_relative_trajectory_policy_loss
```

This gate applies GRPO/RLOO-style normalization to complete trajectory rewards
within a sampled prompt group, then uses the resulting advantages in the
existing policy-gradient loss.

Result:

```text
good_action_probability = 0.5000 -> 0.9568
bad_action_probability = 0.5000 -> 0.0000
rewards = [0.9900, -0.0500]
advantages = [0.999998, -0.999998]
```

Decision:

```text
group_relative_trajectory_training_proved
```

Scope:

```text
This proves cost-adjusted trajectory rewards can be normalized within a prompt
group and train a policy toward the better trajectory while suppressing the
bad trajectory. It does not yet implement sequence-level GSPO.
```

## Dynamic Sampling And Cost/Length Shaping Gate

Local dynamic sampling artifact:

```text
runs/benchmarks/dynamic_sampling_cost_shaping_2026_06_04
```

Benchmark implementation:

```text
experiments/benchmark_dynamic_sampling_cost_shaping.py
```

Public reward/sampling functions:

```text
shaped_trajectory_rewards
dapo_dynamic_sampling_filter
```

This gate applies explicit compute-cost and length penalties to complete
trajectory rewards, then keeps only prompt groups with useful success/failure
contrast for group-relative training.

Result:

```text
short_success shaped reward = 0.9000
long_success shaped reward = 0.4000
mixed_failure shaped reward = -0.1000
failed_attempt shaped reward = -0.3000
selected groups = ["mixed"]
dropped groups = ["failed", "solved"]
selected fraction = 0.5000
```

Decision:

```text
dynamic_sampling_cost_shaping_proved
```

Scope:

```text
This proves DAPO-style dynamic sampling can keep only prompt groups with useful
success/failure contrast while cost and length shaping prefer the shorter
successful rollout over a longer equally successful rollout. It does not yet
implement sequence-level GSPO, process rewards, or a value head.
```

## Sequence, Process, And Value Support Gate

Local sequence/process/value artifact:

```text
runs/benchmarks/sequence_process_value_support_2026_06_04
```

Benchmark implementation:

```text
experiments/benchmark_sequence_process_value_support.py
```

Public sequence/process/value functions:

```text
gspo_sequence_policy_loss
implicit_process_rewards
value_prediction_loss
```

Controller support:

```text
AgenticPolicyController.value_head
agentic_controller_supervised_loss(..., value_targets=...)
```

This gate adds the remaining objective support needed after trajectory records,
verifier rewards, group-relative training, and dynamic sampling:

```text
sequence-level clipped policy ratios for whole action trajectories
implicit process rewards from policy/reference ratios plus verifier signal
masked value regression for trajectory/process returns
lightweight controller value head over TAC-derived context features
```

Result:

```text
sequence ratios = [1.0000, 1.0000] -> [1.0525, 0.9373]
verified process reward = 1.1000
unsupported process reward = -1.0500
masked process reward = 0.0000
value loss = 0.843310 -> 0.000000147
value-head max absolute delta = 0.600793
```

Decision:

```text
sequence_process_value_support_proved
```

Scope:

```text
This proves local support for GSPO-style sequence-level trajectory updates,
PRIME-style implicit process rewards, and a trainable lightweight controller
value head. It is a mechanism and objective gate, not a claim that external
long-horizon tasks are solved without running those task benchmarks.
```

## Identity And Coalition Math Upgrade Gate

Local identity/coalition artifact:

```text
runs/benchmarks/identity_coalition_math_upgrade_2026_06_05
```

Benchmark implementation:

```text
experiments/benchmark_identity_coalition_math_upgrade.py
```

Public identity/coalition functions:

```text
identity_persistence_score
memory_overlap_graph
memory_link_utility
coalition_participation_metrics
basal_apical_belief_state
phase_d_agentic_reward
```

This gate absorbs the identity/coalition research upgrade into the mathematical
contract. It adds:

```text
identity persistence scoring for stable invariant reuse
memory-overlap graph links for multi-hop retrieval paths
coalition participation/coactivation metrics
basal/apical belief disagreement as a useful signal
Phase D rewards with identity, memory-link, and coalition terms
cost-aware DAPO success classification from shaped rewards
```

Result:

```text
identity_persistence_scores = [0.8000, 0.3000]
memory_link direct_utility = 0.0000
memory_link linked_utility = 0.9939
memory_link link_gain = 0.9939
basal/apical disagreement = [0.0000, 2.0000]
Phase D reward = 3.7439
```

Decision:

```text
identity_coalition_math_upgrade_proved
```

Scope:

```text
This proves local mathematical support for identity persistence, memory-overlap
links, coalition participation, basal/apical disagreement, shaped DAPO success
classification, and Phase D agentic rewards with identity/link/coalition terms.
It does not prove external Phase B/D capability advantage.
```

## Live Phase D Scratchpad Policy Gate

Local Phase D live-policy artifact:

```text
runs/benchmarks/live_phase_d_scratchpad_policy_2026_06_05
```

Benchmark implementation:

```text
experiments/benchmark_live_phase_d_scratchpad_policy.py
```

This gate connects live TAC-derived Phase D prompt/candidate features to
`AgenticPolicyController` scratchpad logits, turns those logits into
verifier-gated `AgenticScratchpadState`, and scores the resulting predictions
with the standard Phase D scorer against an empty-scratchpad control.

Result:

```text
scratchpad_mean_score = 1.0000
no_scratchpad_mean_score = 0.0000
score_margin = 1.0000
scratchpad_selection_score = 1.0000
live_tac_token_embedding_grad_abs_sum = 0.000683
hypothesis_contamination_rate = 0.0000
unverified_prompt_leak_count = 0
```

Decision:

```text
live_phase_d_scratchpad_policy_proved
```

Scope:

```text
This proves the live policy heads can be wired into Phase D scratchpad state and
produce a local scratchpad-vs-no-scratchpad score lift without unverified
payload leakage. It does not prove external-scale rollout RL, held-out OOD
capability advantage, or end-to-end verifier/tool integration.
```

## ATS Transfer Benchmark Gate

Local ATS/OOD benchmark artifact:

```text
runs/benchmarks/ats_transfer_suite_2026_06_05
```

Benchmark implementation:

```text
tac_transformer/ats_transfer.py
experiments/benchmark_ats_transfer_suite.py
```

This gate defines the held-out transfer measurement required by Phase D. The
suite uses disjoint train/test domains and includes:

```text
cross_domain_identity_transfer
two_program_sequential
```

The controls are not model claims. `identity_oracle` proves the benchmark has a
solvable invariant target, while `surface_baseline` proves that train-domain
surface memorization fails on held-out domains.

Result:

```text
example_count = 32
train_domains = [navigation, inventory]
test_domains = [lab_protocol, incident_response]
identity_oracle_test_score = 1.0000
surface_baseline_train_score = 1.0000
surface_baseline_test_score = 0.0000
oracle_test_advantage = 1.0000
```

Decision:

```text
ats_transfer_benchmark_valid
```

Scope:

```text
This validates the OOD multi-step / ATS transfer benchmark and scoring
contract. It does not prove TAC capability advantage until TAC and
parameter-matched vanilla checkpoints are run against the suite.
```

## ATS Checkpoint Validation

Checkpoint runner:

```text
run_ats_checkpoint_predictions
experiments/run_ats_checkpoint_predictions.py
```

After the first checkpoint pass, the ATS prompt contract was tightened so every
prompt fits inside the 256-token checkpoint context window:

```text
max_prompt_bytes <= 220
```

Available local TAC checkpoints were then scored against the compact fixed ATS
suite with a 24-token generation budget:

```text
runs/benchmarks/ats_checkpoint_tac_seed11_compact_24tok_fixedprompt_2026_06_05
runs/benchmarks/ats_checkpoint_tac_seed37_compact_24tok_fixedprompt_2026_06_05
```

Result:

```text
tac_control_v1_seed_11 train score = 0.0000
tac_control_v1_seed_11 test score = 0.0000
tac_control_v1_seed_37 train score = 0.0000
tac_control_v1_seed_37 test score = 0.0000
```

Diagnosis:

```text
The runner works and prompt truncation is fixed. Raw completions are generic
hard-corpus-style prose/JSON fragments rather than ATS target tokens. Prompt
format probes did not elicit exact token copying. No nonzero local vanilla .pt
checkpoint is currently available for the required parameter-matched comparison.
```

Decision:

```text
ats_checkpoint_validation_failed_model_behavior
```

Next repair:

```text
Stage ATS transfer examples as a training/evaluation corpus, launch or restore
both TAC and parameter-matched vanilla checkpoints for that corpus, then rerun
the ATS checkpoint scorer.
```

## ATS Supervised Corpus Staging

Staging implementation:

```text
stage_ats_transfer_training_corpus
experiments/stage_ats_transfer_corpus.py
```

The ATS suite can now be materialized into the existing prepared JSONL trainer
contract:

```text
text = prompt + answer + "\n"
```

Artifact:

```text
runs/benchmarks/ats_transfer_training_corpus_2026_06_05
```

Result:

```text
train_records = 512
eval_records = 512
train_domains = [navigation, inventory]
eval_domains = [lab_protocol, incident_response]
max_text_bytes = 173
test_domain_rows_in_train = 0
train_domain_rows_in_eval = 0
duplicate_record_ids = 0
```

Decision:

```text
ats_transfer_training_corpus_staged
```

Scope:

```text
This stages the supervised TAC/vanilla comparison input. It does not prove ATS
capability recovery until both checkpoint families are trained on this corpus
and scored with the ATS checkpoint runner.
```

Local diagnostics:

```text
generic_jsonl_tac_smoke_20_score = 0.0000 / 0.0000
generic_jsonl_vanilla_smoke_20_score = 0.0000 / 0.0000
generic_jsonl_tac_smoke_200_score = 0.0000 / 0.0000
generic_jsonl_vanilla_smoke_200_score = 0.0000 / 0.0000
generic_jsonl_tac_smoke_500_seq176_score = 0.0000 / 0.0000
generic_jsonl_vanilla_smoke_500_seq176_score = 0.0000 / 0.0000
```

Answer-only diagnostic:

```text
experiments/benchmark_ats_answer_copy_training.py
runs/benchmarks/ats_answer_copy_training_2026_06_05
```

Result:

```text
tac_answer_only_train_score = 1.0000
tac_answer_only_test_score = 0.0000
vanilla_answer_only_train_score = 1.0000
vanilla_answer_only_test_score = 0.0000
```

Diagnosis:

```text
The scorer and greedy byte-level generation path are valid because both models
can emit exact train-domain answers under masked answer-only supervision. The
remaining failure is held-out domain transfer: small local controls memorize
train-domain answer templates but do not copy instance-specific lab/incident
response answers.
```

External base-scale runs:

```text
datasets:
  jeffkolo/tac-ats-transfer-code-2026-06-05
  jeffkolo/tac-ats-transfer-corpus-2026-06-05
kernels:
  jeffkolo/tac-ats-transfer-tac-base-5k-2026-06-05
  jeffkolo/tac-ats-transfer-vanilla-base-5k-2026-06-05
```

These kernels train base-scale `seq_len=176` TAC and parameter-matched vanilla
checkpoints for `5000` steps and then run the ATS checkpoint scorer on `best.pt`.
TAC-175 remains open until those outputs are available or a concrete Kaggle
failure is diagnosed and repaired.

## Implementation Priority

All local mechanism/objective proof gates requested in this sequence are now
implemented, including the identity/coalition math upgrade and the local live
Phase D scratchpad-policy wiring gate. The OOD multi-step / ATS transfer
benchmark contract is also now implemented, and the supervised ATS corpus is
staged. The next development priority is external-scale validation: rerun
long-context, noisy-key, multi-hop, Phase B identity-persistence logging, Phase
B2 coalition ablations, Phase D evaluations, and ATS transfer scoring with TAC
and parameter-matched vanilla checkpoints. The immediate ATS repair is to
retrieve or repair the launched base-scale Kaggle ATS transfer runs.

The main architectural correction is that future simulation should not write
into `IdentityState` directly. It should write into `X_t` and only reach
`S_t^real` through `Verify` and `CommitGate`.
