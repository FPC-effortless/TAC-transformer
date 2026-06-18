# Dataset Quality Report

Status: **fixed_with_balanced_training_view**

## Counts

- CPT: `510066`
- SFT: `35007`
- Reasoning: `33500`
- Preference: `3340`
- Eval: `25330`

## SFT Checks

- Duplicate IDs: `0`
- Duplicate user prompts: `0`
- Assistant `<think>` targets: `0`
- Records preserving newlines: `27624`
- Structured multimodal records: `17`
- Stringified multimodal records: `0`

## Split Leakage

- ID overlap: `0`
- Full-message overlap: `0`
- User-prompt overlap: `0`

## Balanced TAC Train View

- Path: `splits_balanced/train/`
- Total records: `150902`
- Per-source cap: `35000`
- Max source share: `0.232`

Use `TRAINING_RECIPE_TAC_BALANCED.md` for TAC experiments.

## Residual Notes

- Reasoning traces still contain think-like text in a separate trace file. Do not use that file as ordinary assistant SFT.
- Full train files are retained for inspection and alternate sampling; use the balanced view when source dominance matters.
