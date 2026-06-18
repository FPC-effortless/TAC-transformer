# Training Recipe

Use the split files for actual training:

1. Continued pretraining / knowledge transfer:
   - Train: `splits/train/unified_cpt.jsonl`
   - Validation: `splits/validation/unified_cpt.jsonl`
   - Format: `{ "text": ... }`

2. Supervised fine-tuning:
   - Train: `splits/train/unified_sft_messages.jsonl`
   - Validation: `splits/validation/unified_sft_messages.jsonl`
   - Format: OpenAI-style `{ "messages": [...] }`

3. Reasoning traces:
   - Train: `splits/train/unified_reasoning_traces.jsonl`
   - Validation: `splits/validation/unified_reasoning_traces.jsonl`
   - Format: `{ "state", "actions_json", "next_state", "reward" }`
   - Use directly with a custom objective, or convert to chat SFT for a trace-prediction task.

4. Preference alignment:
   - Train: `splits/train/unified_preference_pairs.jsonl`
   - Validation: `splits/validation/unified_preference_pairs.jsonl`
   - Format: TRL DPO-style `{ "prompt", "chosen", "rejected" }`

5. Evaluation:
   - `unified_eval.jsonl`

Important: the reasoning-oriented sources include a mix of evidence-grounded traces, synthetic traces, and reconstructed traces. Treat them as behavior supervision, not guaranteed hidden chain-of-thought truth.
