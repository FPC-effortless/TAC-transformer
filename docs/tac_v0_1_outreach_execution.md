# TAC v0.1 Outreach Execution

Goal: open the technical feedback loop before starting TAC v0.2 scaling.

Do not pitch TAC as a product or a transformer replacement. The ask is feedback
on whether the mechanisms and controls are meaningful.

## Week 1 Outreach Batch

Send 5-10 messages across these groups:

- Persistent memory and agent memory researchers.
- Software-engineering agent researchers.
- Program repair and program synthesis researchers.
- ARC-style reasoning benchmark builders.
- World-model and predictive-state researchers.

Use the message template in `docs/tac_v0_1_research_outreach.md`.

## Feedback Questions

Ask each person:

1. What result would convince you TAC is more than benchmark scaffolding?
2. Which TAC v0.1 control is weakest?
3. What should the TAC v0.2 scaling gate require?
4. Which external benchmark would be credible but not unfairly broad?
5. What failure mode should we try hardest to expose?

## Feedback Intake Schema

Record feedback in this shape:

| Date | Contact | Area | Main Critique | Suggested Gate | Follow-Up |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

## Decision Rule Before TAC v0.2

Do not begin the 100M+ training run until at least three serious external
critiques have been summarized and mapped to the Stage-4 gates.

Acceptable critique sources:

- Researcher email or DM.
- GitHub issue or discussion.
- Review from an engineer with agent or ML systems experience.
- Written feedback from an accelerator, grant reviewer, or lab contact.

## Public Update Text

```text
TAC v0.1 is now public as a reproducible research package. The next question is
not another benchmark: it is whether the persistent-state mechanisms survive
scale. I am looking for technical feedback on the benchmark controls and TAC
v0.2 scaling gates.
```

