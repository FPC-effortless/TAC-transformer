# Dataset Fix Report

Status: `fixed`
Generated: `2026-06-06T17:13:41.420557+00:00`

## What Changed
- Preserved internal message whitespace and newlines instead of collapsing formatting.
- Removed <think> blocks from SFT assistant targets while keeping reasoning traces separate.
- Preserved structured multimodal content arrays instead of stringifying them.
- Fixed ATS/component double-emission and duplicate IDs.
- Applied prompt-level SFT dedupe to prevent train/validation prompt leakage.
- Skipped mirrored parquet/jsonl enriched files and roleplay identity-drift sources.
- Added a balanced TAC train view with a 35,000-record per-source cap.

## Full Dataset Counts
- `cpt`: `510066`
- `sft`: `35007`
- `reasoning`: `33500`
- `preference`: `3340`
- `eval`: `25330`

## SFT Quality
- Records: `35007`
- Duplicate IDs: `0`
- Assistant `<think>` blocks: `0`
- Any `<think>` blocks: `0`
- Records preserving newlines: `27624`
- Structured multimodal records/messages: `17`
- Stringified multimodal records/messages: `0`
- Duplicate user prompts: `0`
- Max same user prompt count: `1`

## Split Leakage
- Train SFT records: `33191`
- Validation SFT records: `1816`
- ID overlap: `0`
- Full-message overlap: `0`
- User-prompt overlap: `0`

## Balanced TAC View
- Use `splits_balanced/train/*` for train files.
- Use `splits/validation/*` for validation files.
- Total balanced train records: `150902`
- Per-source cap: `35000`
- Max dataset share: `0.232`

## Balanced Integrity
- `unified_cpt.jsonl`: count `82702`, duplicate IDs `0`, duplicate content `0`
- `unified_sft_messages.jsonl`: count `33191`, duplicate IDs `0`, duplicate content `0`
- `unified_reasoning_traces.jsonl`: count `31838`, duplicate IDs `0`, duplicate content `0`
- `unified_preference_pairs.jsonl`: count `3171`, duplicate IDs `0`, duplicate content `0`

## Balanced vs Validation Overlap
- `unified_cpt.jsonl`: id_overlap `0`, content_overlap `0`
- `unified_sft_messages.jsonl`: id_overlap `0`, content_overlap `0`, user_prompt_overlap `0`
- `unified_reasoning_traces.jsonl`: id_overlap `0`, content_overlap `0`
- `unified_preference_pairs.jsonl`: id_overlap `0`, content_overlap `0`

## Notes
- Reasoning traces still contain trace-like material by design; do not mix them into normal assistant SFT unless deliberately training a reasoning-trace objective.
- The full unified train files are retained for inspection and alternate samplers, but TAC training should use splits_balanced/train to avoid source dominance.
- Raw sources were extracted from Training data.rar for the rebuild and then removed to recover disk space; the archive itself was retained.
