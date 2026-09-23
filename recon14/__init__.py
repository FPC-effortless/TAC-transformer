"""RECON-14: Persistence/Reuse Pilot reconstruction.

Standalone reconstruction of the historical CASM ``phase1`` Experiment 14
(persistence/reuse pilot). See ``RECON_14.md`` for the frozen protocol and
provenance categories.

The historical ``phase1/`` package is unrecoverable (observed on an inaccessible
machine), so this package is a from-specification reimplementation of the
canonical CASM Phase-1 benchmark components plus the pilot. It deliberately does
not depend on PyTorch: differentiation is driven by the hand-written reverse-mode
engine in ``autodiff.py``.
"""

__all__ = [
    "autodiff",
    "grammar",
    "superset",
    "boolean_dag",
    "oracle",
    "canonicalize",
    "executor",
    "baselines",
    "router",
]
