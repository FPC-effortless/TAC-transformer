# Repository Cleanup Plan

This document tracks the safe cleanup path for TAC-transformer.

## Goals

- Preserve all existing research branches.
- Avoid force-merging diverged branches into `main`.
- Promote clean, reviewable changes through pull requests.
- Add packaging, CI, and reviewer documentation without deleting prior work.

## Branch policy

- `main`: stable public/default branch.
- `feature/tac-scm-real003`: clean TAC-SCM integration candidate.
- `tac-v0.1-public`: older public-release/docs branch; cherry-pick useful docs rather than merge blindly.
- `develop/tac-v0.2`: large development lane; split into focused PRs.
- `research/tac-sie-mvp002-exp009`: experimental TAC-SIE lane; keep separate until it has its own validation gate.

## Immediate safe actions

1. Open draft PRs for each active lane.
2. Merge only the clean TAC-SCM branch after review/tests.
3. Port public-facing docs from `tac-v0.1-public` into a clean docs PR.
4. Add CI and Python packaging metadata.
5. Delete `__tmp_branch_probe_do_not_use` after confirming it is identical to `main`.
