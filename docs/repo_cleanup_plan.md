# Repository Cleanup Plan

This document tracks the safe cleanup path for TAC-transformer.

## Goals

- Preserve all existing research branches.
- Avoid force-merging diverged branches into `main`.
- Promote clean, reviewable changes through pull requests.
- Add packaging, CI, and reviewer documentation without deleting prior work.
- Present the repo as a structure-centric intelligence research program, not only an Identity Field prototype.

## Branch policy

- `main`: stable public/default branch.
- `feature/tac-scm-real003`: merged into `main` through PR #1.
- `tac-v0.1-public`: older public-release/docs branch; preserved as PR #2 and cherry-picked for reviewer docs.
- `develop/tac-v0.2`: large development lane; preserved as PR #3 and should be split into focused PRs.
- `research/tac-sie-mvp002-exp009`: experimental TAC-SIE lane; preserved as PR #4 and should stay separate until it passes its own validation gate.

## Completed actions

1. Opened draft preservation PRs for active branches.
2. Merged the clean TAC-SCM branch into `main`.
3. Added conservative public docs: `LIMITATIONS.md`, `REPRODUCIBILITY.md`, `TECHNICAL_REPORT.md`, and `RESULTS_SUMMARY.md`.
4. Added `docs/structure_centric_intelligence_research_program.md`.
5. Added `docs/tac_sie_research_lane.md`.
6. Added Python packaging metadata and requirements.
7. Added GitHub Actions CI smoke checks.
8. Added stable import facades for core, memory, and routing modules.
9. Preserved the legacy command-heavy README in `docs/legacy_readme_reference.md`.

## Remaining safe actions

1. Delete `__tmp_branch_probe_do_not_use` manually from GitHub. It was confirmed identical to `main`.
2. Keep PR #2, PR #3, and PR #4 open as preserved research/history lanes until each is split or reviewed.
3. Create a focused TAC-SIE validation PR only after EXP009C is implemented and tested.
4. Avoid merging `develop/tac-v0.2` directly; split it into docs, configs, benchmarks, data, and demo PRs.
