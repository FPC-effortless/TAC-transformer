# TAC-235 Through TAC-272 Benchmark Summary

| Stage | Result | Key Metric | Interpretation |
|---|---:|---:|---|
| TAC-235 | validated | correct program knockout drop 0.5718 | Native program computation can become causally necessary. |
| TAC-236 | validated | 9/9 reproduction cells | TAC-235 reproduced across seeds, sizes, and task families. |
| TAC-242 | validated | transfer advantage +0.1226 | Programs show reusable algorithmic specialization. |
| TAC-245 | validated | 10x compression | TAC matched/exceeded a longer-context transformer proxy in bounded compression. |
| TAC-248 | validated | 20x compression | Compression scaling holds to 20x and fails beyond that boundary. |
| TAC-251 | validated | 20x realistic workload compression | Compression extends to coding/research/document proxy workloads. |
| TAC-252 | validated | validated ROI ratio 20x | Compression maps to a cost/value proxy. |
| TAC-261 | validated | task-state retention 0.6940 | Persistent agent state survives across sessions. |
| TAC-266 | not_validated | architecture score 0.6402 | Repository continuity is strong, but completion/localization missed gates. |
| TAC-267 | validated | executive control score 0.6767 | Verification failures can drive responsible-program repair control. |
| TAC-268 | validated | autonomous editing score 0.9370 | Generated workspace repair loop works under bounded conditions. |
| TAC-269 | validated | real repository repair score 0.9700 | Copied real-file sandbox repair works, but restoration remained a concern. |
| TAC-270 | validated | no-restore repair score 0.9635 | Multi-file sandbox repair works without full-file restoration. |
| TAC-271 | not_validated | first-pass disambiguation 0.5583 | Ambiguity breaks first-pass causal fix choice. |
| TAC-272 | validated | first-pass disambiguation 0.8417 | Causal-fix scoring improves ambiguous repair selection. |

Public v0.1 claim:

> TAC is an experimental persistent-state architecture for long-horizon AI agents, with validated mechanisms for memory, compression, control, repair, and causal fix selection in bounded benchmarks.

Boundary:

TAC v0.1 does not claim transformer superiority, open-ended coding ability, or large-scale pretrained-model validation.
