# Unified Training Curriculum

This folder contains a unified LLM training dataset built from the local `Training data` folder.

Main files:

- `unified_cpt.jsonl`: foundation continued-pretraining text records.
- `unified_sft_messages.jsonl`: OpenAI-style conversational SFT records.
- `unified_reasoning_traces.jsonl`: state/action/next_state reasoning traces.
- `unified_preference_pairs.jsonl`: DPO-style prompt/chosen/rejected records.
- `unified_eval.jsonl`: held-out source eval/test records.
- `splits/train/*` and `splits/validation/*`: deterministic split files.
- `curriculum_plan.json`: recommended stage order.
- `source_inventory.json`: source file actions and skip reasons.

Summary:

```json
{
  "output_counts": {
    "cpt": 510066,
    "sft": 35007,
    "reasoning": 33500,
    "preference": 3340,
    "eval": 25330
  },
  "split_counts": {
    "train": {
      "cpt": 484188,
      "sft": 33191,
      "reasoning": 31838,
      "preference": 3171,
      "eval": 0
    },
    "validation": {
      "cpt": 25878,
      "sft": 1816,
      "reasoning": 1662,
      "preference": 169,
      "eval": 0
    }
  }
}
```
