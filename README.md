# TAC-Transformer

TAC-Transformer is an experimental persistent-state and structure-centric transformer research program. It tests whether reusable computational structures can be preserved, routed, compressed, transferred, bridged into behavior, and eventually recovered as executable objects.

The repository contains a real trainable PyTorch implementation in `tac_transformer/`, benchmark harnesses in `kaggle/` and `experiments/`, tests in `tests_py/`, and a browser visualization prototype in `src/`. The browser prototype is secondary; the core research artifact is the PyTorch architecture and its validation suite.

## Research thesis

The working thesis is:

> Intelligence is structure acquisition and structure use.

More specifically:

```text
Intelligence = discover reusable computational structure
             + compress it
             + preserve it
             + retrieve it
             + bind it
             + execute it
             + compose it
             + refine it
             + evolve it
```

The repo is organized as:

```text
Theory: Structure-Centric Intelligence
  ↓
Model science: TAC-SCM
  ↓
Engine decomposition: TAC-SIE
  ↓
Validation: REAL / EXP benchmarks
```

See `docs/structure_centric_intelligence_research_program.md` for the full research map.

## Current claim

The narrow claim is not that TAC beats large language models today. The current claim is:

> TAC provides a controlled research platform for testing whether persistent state and routed reusable computational structures can causally improve memory, compression, repair control, and structure-to-behavior transfer in bounded benchmarks.

## What is implemented on `main`

- Trainable PyTorch TAC model and configuration surface.
- Persistent `IdentityState` and identity-field routing.
- Program routing, memory read/write options, and state carry/reset/shuffle probes.
- TAC-SCM structure-centric components: concept volumes, structure memory, structure slots, structure bridge, structure lifecycle, procedural memory, and repair controller.
- Benchmark suites for TAC memory, compression, repair control, and TAC-SCM REAL004/005/006/011.
- Stable import facades under `tac_transformer/core`, `tac_transformer/memory`, and `tac_transformer/routing`.
- Kaggle-oriented training and validation scripts.
- Optional serving and Gradio generation utilities.
- Browser visualization prototype for identity-field intuition.

## TAC-SCM and TAC-SIE lanes

TAC-SCM is the model-science lane. It tests whether structure-centric models can preserve, route, reuse, transfer, compress, bridge, and recover structures. The clean TAC-SCM branch has been merged into `main`.

TAC-SIE is the engine-decomposition lane. It tests whether the same thesis can be decomposed into clean modules for preservation, retrieval, binding, execution, refinement, and evolution. TAC-SIE is preserved as PR #4 and is not merged into `main` yet because it needs its own validation gate.

See `docs/tac_sie_research_lane.md` for the TAC-SIE status and merge criteria.

## What has been validated so far

The strongest current evidence is bounded and controlled:

- persistent identity state can be carried across segments;
- carry/reset/shuffle probes can detect whether state is causally useful;
- content-addressed and program-conditioned memory can improve controlled recall tasks;
- context-compression benchmarks show useful behavior around the 10x-20x regime;
- repair-control benchmarks validate bounded localization, targeted repair, and causal fix disambiguation;
- TAC-SCM REAL004/005/006 validate controlled structure-to-behavior use, bridge stability, and realistic structure-transfer slices;
- TAC-SCM REAL011 improves the executable-structure benchmark so future failures are more attributable to model limitations than benchmark flaws.

See `TECHNICAL_REPORT.md` and `RESULTS_SUMMARY.md` for the compact evidence map.

## What is not yet validated

TAC does not yet prove:

- general language-modeling superiority;
- coding, math, or planning superiority over strong baselines;
- reliable open-ended autonomous agents;
- robust arbitrary binding;
- faithful executable recovery;
- autonomous open-ended structure discovery at scale;
- wall-clock efficiency over transformers;
- large-scale pretraining survival;
- replacement of current LLM architectures.

See `LIMITATIONS.md` for the full claim boundary.

## Quickstart

Install Python dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install frontend dependencies only if you want the browser prototype:

```bash
npm install
```

Run Python smoke tests:

```bash
python -m unittest tests_py.test_tac_transformer tests_py.test_tac_serving
```

Run the current best TAC benchmark smoke command:

```bash
python kaggle/benchmark_best_tac.py --steps 120
```

Run TAC-SCM structure-centric tests:

```bash
python -m unittest tests_py.test_tac_scm_real004 tests_py.test_tac_scm_real005 tests_py.test_tac_scm_real006 tests_py.test_tac_scm_real011
```

## Main reproducibility commands

Core TAC validation:

```bash
python experiments/kaggle_validate_tac_core.py --benchmarks tac251,tac252,tac267,tac270,tac272 --seeds 5 --cases 50 --output runs/kaggle_validation/tac_core_validation.json
```

TAC-SCM benchmark validity validation:

```bash
python kaggle/benchmark_tac_scm_real011.py --output_dir outputs/tac_scm_real011 --seeds 0 1 2 3 4 5 6 7 8 9 --train_samples 256 --eval_samples 256
```

See `REPRODUCIBILITY.md` for additional commands and interpretation.

## Repository map

```text
tac_transformer/       Core TAC implementation and research modules
tac_transformer/core/  Stable public import facade for the core model
tac_transformer/memory/ Stable memory import facade
tac_transformer/routing/ Stable routing and structure-bridge import facade
kaggle/                Kaggle-ready benchmark and training scripts
experiments/           Research benchmark scripts and roadmap experiments
tests_py/              Python test suite
docs/                  Architecture notes, runbooks, and research reports
src/                   Browser identity-field visualization prototype
```

The repository still contains active research lanes. Anything outside the stable facades should be treated as experimental unless the relevant doc marks it as promoted.

## Development checks

```bash
npm test
npm run test:python
npm run test:js
npm run lint
npm run build
```

GitHub Actions runs basic Python and frontend smoke checks on pushes and PRs to `main`.

## License

This repository is licensed under Apache-2.0. See `LICENSE`.

## Historical README

The previous command-heavy README is preserved in `docs/legacy_readme_reference.md` so older notes are not lost.
