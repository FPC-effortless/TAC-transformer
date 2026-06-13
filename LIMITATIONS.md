# TAC v0.1 Limitations

TAC v0.1 is a research package, not a production agent system.

The supported claim is narrow:

> TAC is an experimental persistent-state architecture for long-horizon AI agents, with validated mechanisms for memory, compression, control, repair, and causal fix selection in bounded benchmarks.

Unsupported claims:

- TAC does not currently prove superiority over transformers.
- TAC does not currently prove open-ended autonomous software engineering.
- TAC does not currently prove robust long-horizon planning.
- TAC does not currently prove world-model behavior.
- TAC does not currently prove unrestricted repository repair.
- TAC does not currently prove that the same effects survive large-scale pretraining.

Main benchmark boundaries:

- Most stages are local-CPU bounded experiments.
- Repair benchmarks use copied sandbox files, not live repository mutation.
- TAC-270 removes full-file restoration, but the bug classes remain bounded.
- TAC-272 improves causal fix choice under injected ambiguity, but does not yet test simultaneous independent bugs or long repair chains.
- Kaggle validation is a reproducibility check for benchmark metrics, not an external product evaluation.

Current open frontier:

> causal judgment under ambiguity at larger scale.

