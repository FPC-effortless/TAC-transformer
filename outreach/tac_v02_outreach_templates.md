# TAC v0.2 Outreach Templates

## Researcher

Subject: TAC v0.2 external review request: persistent state at 112M scale

Hi [Name],

I am preparing TAC v0.2, a matched 112M-parameter scaling test of an
experimental persistent-state transformer variant. v0.1 validated bounded
mechanisms for persistent state, 20x context compression, repair control, and
causal fix selection, but it does not prove those mechanisms survive real-data
scaling.

The v0.2 question is narrow: when TAC and a matched transformer are trained on
the same language/code data and token budget, do persistent state, repair, and
compression advantages still exist?

Would you be willing to review the protocol or point out the strongest failure
modes to test before results are public?

## Founder

Subject: TAC v0.2 scaling test and infrastructure feedback

Hi [Name],

I am moving TAC from bounded v0.1 mechanism validation to a 112M-parameter
matched scaling test. The aim is to learn whether persistent state, repair
planning, and 20x compression survive real language/code training against a
same-budget transformer baseline.

I am looking for blunt feedback on whether this would matter for agent
infrastructure or developer tooling if the advantages survive, and what result
would make it credible enough for deeper collaboration or angel conversations.

Would a short review of the v0.2 run plan be useful?

