# TAC Limitations

TAC is a research architecture, not a production agent system.

The supported claim is narrow:

> TAC is an experimental persistent-state and structure-centric architecture for long-horizon AI agents, with validated mechanisms for memory, compression, control, repair, structure routing, and causal structure-to-behavior use in bounded benchmarks.

Unsupported claims:

- TAC does not currently prove superiority over transformers.
- TAC does not currently prove open-ended autonomous software engineering.
- TAC does not currently prove robust long-horizon planning in unrestricted real repositories.
- TAC does not currently prove world-model behavior.
- TAC does not currently prove unrestricted repository repair.
- TAC does not currently prove that the same effects survive large-scale pretraining.
- TAC-SCM does not yet prove autonomous open-ended structure discovery at scale.

Main benchmark boundaries:

- Most stages are local-CPU bounded experiments.
- Repair benchmarks use copied sandbox files, not live repository mutation.
- TAC-270 removes full-file restoration, but the bug classes remain bounded.
- TAC-272 improves causal fix choice under injected ambiguity.
- TAC-273 exposes interacting repair-chain completion as a hard frontier.
- TAC-274 improves bounded interaction-aware repair planning, but does not yet test unrestricted live-repository chains.
- TAC-SCM REAL004/005/006/011 validate controlled structure-centric behavior and benchmark redesign, but remain synthetic or realistic controlled tasks.
- Kaggle validation is a reproducibility check for benchmark metrics, not an external product evaluation.

Current open frontier:

> long-horizon repair planning and structure discovery in real repositories at larger scale.
