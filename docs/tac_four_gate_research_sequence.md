# TAC Four-Gate Research Sequence

Date: 2026-06-16

Status: planning contract implemented; experiment runs pending.

## Purpose

The next TAC work should reduce the four largest remaining uncertainties before
adding more TAC-numbered variants:

| Gate | Risk Tested | Primary Role |
| --- | --- | --- |
| PSM-007 | Benchmark artifact | Credibility |
| ID001 | Identity carry value | Architecture |
| TAC-281 | LM efficiency penalty | Efficiency |
| 112M Pilot | Scale survival | Scaling |

## Order

1. PSM-007
2. ID001
3. TAC-281
4. 112M Pilot

The 112M pilot remains blocked until TAC-281 passes. TAC-281 itself should be
interpreted only after PSM-007 and ID001 answer whether the benchmark and
identity-carry claims are credible.

## Gate Contracts

### PSM-007

Question: Does TAC work on problems it did not design?

Inputs:

- real GitHub bugs
- SWE-bench-lite
- human-written repair tasks

Constraints:

- run TAC exactly as-is
- no redesign
- no retuning
- no metric changes

Success means TAC advantage survives outside TAC-created benchmarks.

### ID001

Question: Are structures and procedures better when carried by persistent
identities?

Controls:

- carried identity
- reset identity
- shuffled identity
- identity knockout

Success means carried beats reset, carried beats shuffled, and knockout hurts.

### TAC-281

Question: Can TAC keep its mechanism while becoming a better language model?

Variants:

- late bottleneck
- small adapter
- auxiliary mechanism

Success means carry advantage remains, mechanism wins remain at least 3 of 4
families, bottleneck knockout delta stays positive, the LM loss gap shrinks, and
the speed penalty drops.

### 112M Pilot

Question: Does any of this survive scale?

Run a 100M+ TAC model using the improved architecture against a matched
transformer baseline on real language/code data.

Success means structure memory, procedural memory, and identity carry effects
survive real training.

## Machine-Readable Contract

The plain-data contract lives in `tac_transformer/four_gate_plan.py` and is
covered by `tests_py/test_four_gate_research_sequence.py`.
