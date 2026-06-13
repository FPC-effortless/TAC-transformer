# TAC v0.1 Research Outreach List

Goal: get technical feedback and discussion. Do not pitch TAC v0.1 as a product
or as a transformer replacement.

Recommended positioning:

> TAC v0.1 is a reproducible research asset for persistent-state agent
> architecture. The public package includes bounded benchmark evidence,
> negative results, a recovery benchmark, and Kaggle replication.

## Outreach Themes

### World Models And Predictive State

Why relevant: TAC has persistent state and repair planning, but does not yet
prove world-model behavior. Feedback from this area should focus on whether TAC
state could become predictive rather than merely historical.

Target profile:

- JEPA and hierarchical predictive world models.
- Action planning with learned state representations.
- Spatial/world-model startups and research labs.

Useful public references:

- Yann LeCun, "A Path Towards Autonomous Machine Intelligence":
  https://openreview.net/pdf?id=BZ5a1r-kVsf
- JEPA planning example:
  https://arxiv.org/abs/2601.00844

### Persistent Memory And Agent Operating Systems

Why relevant: TAC's strongest public evidence includes persistent state,
compression, continuity, and repair control.

Target profile:

- MemGPT / Letta-style agent memory.
- Context management and virtual context.
- Long-term memory for agents.

Useful public references:

- MemGPT:
  https://arxiv.org/abs/2310.08560
- Memory-for-agents survey list:
  https://github.com/TsinghuaC3I/Awesome-Memory-for-Agents

### Agentic Systems And Software-Engineering Agents

Why relevant: TAC v0.1 now has bounded repair planning, no-restore multi-file
repair, causal fix disambiguation, and interaction-aware repair planning.

Target profile:

- Researchers building software-engineering agents.
- Benchmark authors for coding-agent evaluation.
- Engineers working on long-running tool-use and verification loops.

Feedback questions:

- What would count as a serious real-repository repair corpus?
- Which tests would make TAC v0.2 credible to agent researchers?
- How should TAC compare against tool-using prompt agents and coding agents?

### Program Synthesis And Repair

Why relevant: TAC repair benchmarks increasingly resemble program-repair
problems: root-cause selection, candidate repair scoring, patch ordering, and
regression avoidance.

Target profile:

- Program-repair researchers.
- Program-synthesis researchers.
- Static-analysis and test-generation researchers.

Feedback questions:

- Are the TAC-270 through TAC-274 controls enough to distinguish repair from
  restoration?
- What would be a stronger benchmark for causal patch selection?
- How should multi-bug repair chains be evaluated outside bounded simulations?

### Long-Horizon Planning

Why relevant: TAC has evidence for memory and repair planning, but explicit
planning remains an open frontier from earlier failed stages.

Target profile:

- Long-horizon agent researchers.
- Planning and hierarchical-control researchers.
- Researchers separating memory from executive control.

Feedback questions:

- What long-horizon tasks isolate planning from memory?
- How should state-continuity and plan-correctness be measured separately?
- What is the smallest scaled model that would make the result meaningful?

### ARC-Style Generalization And Reasoning

Why relevant: ARC-style tasks emphasize adaptation, generalization, and learning
from sparse experience. TAC v0.1 does not solve this, but the persistent-state
mechanism may be relevant to future work.

Target profile:

- ARC-style reasoning benchmark builders.
- Researchers studying generalization from sparse demonstrations.
- Researchers interested in skill reuse and compositional adaptation.

Useful public references:

- ARC Prize research page:
  https://arcprize.org/research
- ARC-AGI-3 interactive reasoning:
  https://arcprize.org/arc-agi/3

## Suggested Outreach Message

Subject: TAC v0.1: reproducible persistent-state agent architecture benchmark

Hi,

I am looking for technical feedback on TAC v0.1, a public research package for a
persistent-state agent architecture. The claim is intentionally narrow: TAC
shows bounded benchmark evidence for memory, compression, repair control,
causal fix selection, and interaction-aware repair planning. It does not claim
to beat transformers or solve open-ended autonomy.

The package includes:

- a technical report
- limitations
- reproducibility instructions
- a one-page architecture diagram
- a Kaggle-replicated validation pack
- a negative result in TAC-273 and a targeted recovery in TAC-274

I would value feedback on whether the mechanisms and controls are meaningful,
and what would make a TAC v0.2 scaling experiment credible.

Repository branch:

https://github.com/FPC-effortless/TAC-transformer/tree/tac-v0.1-public

Kaggle validation:

https://www.kaggle.com/code/jeffkolo/tac-v0-1-core-validation-2026-06-13

## Outreach Tracking Table

| Target Area | Person Or Group | Link | Contacted | Response | Follow-Up |
|---|---|---|---|---|---|
| World models |  |  |  |  |  |
| Persistent memory |  |  |  |  |  |
| Software agents |  |  |  |  |  |
| Program repair |  |  |  |  |  |
| Long-horizon planning |  |  |  |  |  |
| ARC-style reasoning |  |  |  |  |  |
