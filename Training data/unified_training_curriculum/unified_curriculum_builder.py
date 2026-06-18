import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


OUTPUT_FILES = {
    "cpt": "unified_cpt.jsonl",
    "sft": "unified_sft_messages.jsonl",
    "reasoning": "unified_reasoning_traces.jsonl",
    "preference": "unified_preference_pairs.jsonl",
    "eval": "unified_eval.jsonl",
}

SKIP_NAMES = {
    ".gitattributes",
    "manifest.json",
    "schema.json",
    "examples.json",
    "validation_report.json",
    "download_manifest.json",
}


def normalize_text(text):
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def strip_think_blocks(text):
    text = str(text or "")
    text = re.sub(r"(?is)<think>.*?</think>\s*", "", text)
    text = re.sub(r"(?is)<think>.*", "", text)
    return normalize_text(text)


def normalize_message_content(content, role=None):
    if isinstance(content, list):
        normalized = []
        for part in content:
            if isinstance(part, dict):
                item = {}
                for key, value in part.items():
                    item[key] = normalize_text(value) if isinstance(value, str) else value
                if item:
                    normalized.append(item)
            elif isinstance(part, str):
                text = normalize_text(part)
                if text:
                    normalized.append({"type": "text", "text": text})
        return normalized

    text = strip_think_blocks(content) if role == "assistant" else normalize_text(content)
    return text


def content_is_empty(content):
    if isinstance(content, list):
        return len(content) == 0
    return not bool(content)


def stable_hash(value):
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def assign_split(record_id, validation_ratio=0.05):
    bucket = int(stable_hash(record_id)[:8], 16) / 0xFFFFFFFF
    return "validation" if bucket < validation_ratio else "train"


def is_eval_path(path):
    lowered = str(path).replace("\\", "/").lower()
    name = path.name.lower()
    return (
        ".eval." in name
        or "/eval." in lowered
        or "/test-" in lowered
        or "\\test-" in str(path).lower()
        or name.startswith("eval.")
    )


def classify_jsonl_file(path):
    lowered = str(path).replace("\\", "/").lower()
    name = path.name.lower()
    if "/splits/" in lowered and ("enriched_" in lowered):
        return "skip_enriched_split_duplicate"
    if "llm-distillation-spec/expanded-dataset/" in lowered and name in {
        "agentic_trajectories.jsonl",
        "ats_transfer_curriculum.jsonl",
        "coding_samples.jsonl",
        "handwritten_behavior_multimodal.jsonl",
    }:
        return "skip_component_duplicate_in_combined_sft"
    if name.endswith(".raw.jsonl") and ("distillation_datasets" in lowered or "prepared_corpus" in lowered):
        return "skip_raw_has_prepared"
    if name in {"examples.jsonl"}:
        return "skip_metadata"
    if "roleplay_train" in name:
        return "skip_roleplay_identity_drift"
    if name.endswith("_no_reasoning.jsonl"):
        return "use"
    if name in {
        "code_train.jsonl",
        "full_train.jsonl",
        "instruct_train.jsonl",
    } and "angrygiraffe__claude-opus-4.6-4.7-reasoning-8.7k" in lowered:
        return "skip_reasoning_variant_has_no_reasoning_peer"
    return "use"


def classify_file(path, root=None):
    rel = path.relative_to(root) if root else path
    lowered = str(rel).replace("\\", "/").lower()
    name = path.name.lower()
    if "unified_training_curriculum/" in lowered:
        return "skip_output_artifact"
    if name in SKIP_NAMES:
        return "skip_metadata"
    if "wikimedia__structured-wikipedia_metadata_only" in lowered:
        return "skip_metadata_only"
    if lowered == "transcripts.jsonl":
        return "skip_raw_transcript_enriched_available"
    if lowered.startswith("new folder/"):
        return "skip_raw_transcript_enriched_available"
    if path.suffix.lower() == ".jsonl":
        return classify_jsonl_file(rel)
    if path.suffix.lower() == ".parquet" and (
        "/enriched_" in lowered or lowered.startswith("enriched_")
    ) and "enriched_sudoku_dataset" not in lowered:
        return "skip_parquet_mirror_jsonl_available"
    if path.suffix.lower() in {".md", ".py", ".yaml", ".yml"}:
        if "readme.md" == name and ("__" in lowered or "metadata_only" in lowered):
            return "skip_dataset_readme_metadata"
        return "use"
    if path.suffix.lower() in {".json", ".parquet"}:
        return "use"
    return "skip_unsupported"


def source_info(path, root=None, extra=None):
    rel = str(path.relative_to(root)) if root else str(path)
    parts = Path(rel).parts
    dataset = parts[0] if parts else rel
    info = {
        "path": rel,
        "dataset": dataset,
    }
    if extra:
        info.update(extra)
    return info


def _role(role):
    value = str(role or "").lower()
    if value in {"human", "user"}:
        return "user"
    if value in {"gpt", "assistant", "model", "bot"}:
        return "assistant"
    if value == "system":
        return "system"
    return "user"


def normalize_messages(messages):
    normalized = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role", message.get("from", "user"))
        content = message.get("content", message.get("value", ""))
        role = _role(role)
        content = normalize_message_content(content, role)
        if not content_is_empty(content):
            normalized.append({"role": role, "content": content})
    return normalized


def as_prompt_messages(value):
    if isinstance(value, list):
        messages = normalize_messages(value)
        return messages or [{"role": "user", "content": normalize_text(value)}]
    if isinstance(value, dict) and "messages" in value:
        return normalize_messages(value["messages"])
    return [{"role": "user", "content": normalize_text(value)}]


def as_assistant_messages(value):
    if isinstance(value, list):
        messages = normalize_messages(value)
        return messages or [{"role": "assistant", "content": normalize_text(value)}]
    if isinstance(value, dict) and "messages" in value:
        return normalize_messages(value["messages"])
    return [{"role": "assistant", "content": normalize_text(value)}]


def output_id(source_path, row_index, kind, seed):
    return f"{kind}_{stable_hash(f'{source_path}|{row_index}|{seed}')[:16]}"


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def message_content_text(content):
    if isinstance(content, list):
        return canonical_json(content)
    return str(content or "")


def split_basis(kind, payload):
    if kind == "sft":
        messages = payload.get("messages", [])
        user_parts = [
            message_content_text(message.get("content"))
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        return "\n".join(user_parts) or canonical_json(messages)
    if kind == "preference":
        return canonical_json(payload.get("prompt", ""))
    if kind == "reasoning":
        return normalize_text(payload.get("state", "")) or canonical_json(payload)
    if kind == "cpt":
        return normalize_text(payload.get("text", ""))[:2000] or canonical_json(payload)
    return canonical_json(payload)


def dedupe_basis(kind, payload):
    if kind == "sft":
        return {"prompt": split_basis(kind, payload)}
    if kind == "preference":
        return {"prompt": split_basis(kind, payload)}
    if kind == "reasoning":
        return {
            "state": payload.get("state", ""),
            "actions_json": payload.get("actions_json", ""),
            "next_state": payload.get("next_state", ""),
            "reward": payload.get("reward", 1),
        }
    if kind == "cpt":
        return {"text": payload.get("text", "")}
    return payload


def _sft_record(messages, source, record_id):
    messages = normalize_messages(messages)
    if not messages:
        return None
    return {
        "kind": "sft",
        "payload": {
            "id": record_id,
            "messages": messages,
            "source": source,
        },
    }


def _cpt_record(text, source, record_id):
    text = normalize_text(text)
    if not text:
        return None
    return {
        "kind": "cpt",
        "payload": {
            "id": record_id,
            "text": text,
            "source": source,
        },
    }


def _preference_record(prompt, chosen, rejected, source, record_id):
    prompt_messages = as_prompt_messages(prompt)
    chosen_messages = as_assistant_messages(chosen)
    rejected_messages = as_assistant_messages(rejected)
    if not prompt_messages or not chosen_messages or not rejected_messages:
        return None
    return {
        "kind": "preference",
        "payload": {
            "id": record_id,
            "prompt": prompt_messages,
            "chosen": chosen_messages,
            "rejected": rejected_messages,
            "source": source,
        },
    }


def _reasoning_record(state, actions, next_state, source, record_id, reward=1):
    state = normalize_text(state)
    if isinstance(actions, str):
        actions_json = actions
    else:
        actions_json = json.dumps(actions, ensure_ascii=False, default=str)
    if isinstance(next_state, str):
        next_state_text = normalize_text(next_state)
    else:
        next_state_text = json.dumps(next_state, ensure_ascii=False, default=str)
    if not state and not actions_json and not next_state_text:
        return None
    return {
        "kind": "reasoning",
        "payload": {
            "id": record_id,
            "state": state,
            "actions_json": actions_json,
            "next_state": next_state_text,
            "reward": reward,
            "source": source,
        },
    }


def normalize_record(record, source_path, row_index, source=None):
    if not isinstance(record, dict):
        return []
    src = dict(source or {"path": source_path})
    rows = []
    base = record.get("id") or record.get("record_id") or record.get("chunk_id") or row_index

    if "prompt" in record and "chosen" in record and "rejected" in record:
        row = _preference_record(
            record["prompt"],
            record["chosen"],
            record["rejected"],
            src,
            output_id(source_path, row_index, "preference", base),
        )
        if row:
            rows.append(row)

    if "query" in record and "chosen" in record and "rejected" in record and "prompt" not in record:
        row = _preference_record(
            record["query"],
            record["chosen"],
            record["rejected"],
            src,
            output_id(source_path, row_index, "preference", base),
        )
        if row:
            rows.append(row)

    if "messages" in record and isinstance(record["messages"], list):
        row = _sft_record(record["messages"], src, output_id(source_path, row_index, "sft", base))
        if row:
            rows.append(row)

    if "conversations" in record and isinstance(record["conversations"], list) and not rows:
        row = _sft_record(record["conversations"], src, output_id(source_path, row_index, "sft", base))
        if row:
            rows.append(row)

    if "question" in record and "answer" in record:
        messages = [
            {"role": "user", "content": normalize_text(record["question"])},
            {"role": "assistant", "content": normalize_text(record["answer"])},
        ]
        row = _sft_record(messages, src, output_id(source_path, row_index, "sft", base))
        if row:
            rows.append(row)

    if "input" in record and "output" in record and "messages" not in record and "inverted_reasoning" not in record:
        messages = [
            {"role": "user", "content": normalize_text(record["input"])},
            {"role": "assistant", "content": normalize_text(record["output"])},
        ]
        row = _sft_record(messages, src, output_id(source_path, row_index, "sft", base))
        if row:
            rows.append(row)

    if "visible_prompt" in record and "expected_answer" in record and not rows:
        messages = [
            {"role": "user", "content": normalize_text(record["visible_prompt"])},
            {"role": "assistant", "content": normalize_text(record["expected_answer"])},
        ]
        row = _sft_record(messages, src, output_id(source_path, row_index, "sft", base))
        if row:
            rows.append(row)

    if {"state", "actions_json", "next_state"}.issubset(record):
        row = _reasoning_record(
            record["state"],
            record["actions_json"],
            record["next_state"],
            src,
            output_id(source_path, row_index, "reasoning", base),
            record.get("reward", 1),
        )
        if row:
            rows.append(row)

    if {"state", "action_json", "next_state"}.issubset(record):
        row = _reasoning_record(
            record["state"],
            record["action_json"],
            record["next_state"],
            src,
            output_id(source_path, row_index, "reasoning", base),
            record.get("reward", 1),
        )
        if row:
            rows.append(row)

    if "inverted_reasoning" in record and "input" in record:
        row = _reasoning_record(
            record.get("input", ""),
            [{"type": "synthetic_trace_inversion", "content": record.get("inverted_reasoning", "")}],
            record.get("output", ""),
            src,
            output_id(source_path, row_index, "reasoning", base),
            1,
        )
        if row:
            rows.append(row)

    if "text" in record:
        row = _cpt_record(record["text"], src, output_id(source_path, row_index, "cpt", base))
        if row:
            rows.append(row)

    if "summary" in record and "key_points" in record:
        key_points = record.get("key_points")
        if isinstance(key_points, list):
            key_points_text = "\n".join(f"- {normalize_text(point)}" for point in key_points)
        else:
            key_points_text = normalize_text(key_points)
        text = f"Summary: {record.get('summary', '')}\nKey points:\n{key_points_text}"
        row = _cpt_record(text, src, output_id(source_path, row_index, "cpt", base))
        if row:
            rows.append(row)

    if "transcript" in record and not rows:
        row = _cpt_record(record["transcript"], src, output_id(source_path, row_index, "cpt", base))
        if row:
            rows.append(row)

    return rows


def iter_jsonl(path):
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            yield index, json.loads(line)


def iter_json(path):
    payload = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    if isinstance(payload, dict):
        yield 0, payload
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            if isinstance(item, dict):
                yield index, item


def iter_parquet(path):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    for index, row in enumerate(table.to_pylist()):
        yield index, row


def normalize_parquet_record(record, source_path, row_index, source=None):
    src = dict(source or {"path": source_path})
    columns = set(record)
    base = record.get("id") or row_index
    rows = []
    if {"question", "answer"}.issubset(columns):
        return normalize_record(record, source_path, row_index, src)
    if {"puzzle", "solution"}.issubset(columns):
        prompt = (
            f"Solve this Sudoku puzzle. Use 0 as blank cells.\n"
            f"Puzzle: {record['puzzle']}\n"
            f"Difficulty: {record.get('difficulty', 'unknown')}"
        )
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"Solution: {record['solution']}"},
        ]
        row = _sft_record(messages, src, output_id(source_path, row_index, "sft", base))
        if row:
            rows.append(row)
        return rows
    if {"state", "action_json", "next_state"}.issubset(columns):
        row = _reasoning_record(
            record["state"],
            record["action_json"],
            record["next_state"],
            src,
            output_id(source_path, row_index, "reasoning", base),
            record.get("reward", 1),
        )
        return [row] if row else []
    if {"question", "choices", "answer"}.issubset(columns):
        choices = record["choices"]
        if isinstance(choices, list):
            choice_text = "\n".join(f"{i + 1}. {choice}" for i, choice in enumerate(choices))
        else:
            choice_text = normalize_text(choices)
        prompt = f"Answer the multiple-choice spatial reasoning question.\nQuestion: {record['question']}\nChoices:\n{choice_text}"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": normalize_text(record["answer"])},
        ]
        row = _sft_record(messages, src, output_id(source_path, row_index, "sft", base))
        return [row] if row else []
    return normalize_record(record, source_path, row_index, src)


class OutputWriter:
    def __init__(self, output_dir, validation_ratio=0.05):
        self.output_dir = Path(output_dir)
        self.validation_ratio = validation_ratio
        self.handles = {}
        self.counts = {kind: 0 for kind in OUTPUT_FILES}
        self.split_counts = {
            "train": {kind: 0 for kind in OUTPUT_FILES},
            "validation": {kind: 0 for kind in OUTPUT_FILES},
        }
        self.seen = {kind: set() for kind in OUTPUT_FILES if kind != "eval"}
        self.used_ids = {kind: set() for kind in OUTPUT_FILES}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for kind, filename in OUTPUT_FILES.items():
            self._handle(filename)
            if kind != "eval":
                self._handle(f"splits/train/{filename}")
                self._handle(f"splits/validation/{filename}")

    def _handle(self, relative):
        path = self.output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative not in self.handles:
            self.handles[relative] = path.open("w", encoding="utf-8", newline="\n")
        return self.handles[relative]

    def write(self, kind, payload, force_eval=False):
        output_kind = "eval" if force_eval else kind
        if output_kind not in OUTPUT_FILES:
            return False
        key = stable_hash(dedupe_basis(output_kind, payload))
        if output_kind != "eval":
            if key in self.seen[output_kind]:
                return False
            self.seen[output_kind].add(key)
        payload = dict(payload)
        base_id = str(payload.get("id") or f"{output_kind}_{key[:16]}")
        record_id = base_id
        suffix = 1
        while record_id in self.used_ids[output_kind]:
            record_id = f"{base_id}_{suffix}"
            suffix += 1
        payload["id"] = record_id
        self.used_ids[output_kind].add(record_id)
        line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
        self._handle(OUTPUT_FILES[output_kind]).write(line)
        self.counts[output_kind] += 1
        if output_kind != "eval":
            split = assign_split(stable_hash(split_basis(output_kind, payload)), self.validation_ratio)
            self._handle(f"splits/{split}/{OUTPUT_FILES[output_kind]}").write(line)
            self.split_counts[split][output_kind] += 1
        return True

    def close(self):
        for handle in self.handles.values():
            handle.close()


def process_text_file(path, root, writer):
    text = path.read_text(encoding="utf-8", errors="replace")
    src = source_info(path, root)
    src["format"] = path.suffix.lower().lstrip(".")
    row = _cpt_record(text, src, output_id(str(path.relative_to(root)), 0, "cpt", path.name))
    written = 0
    if row and writer.write("cpt", row["payload"], force_eval=is_eval_path(path)):
        written += 1
    return {"records_read": 1, "written": {"cpt": written}}


def process_json_records(path, root, writer, iterator, parquet=False):
    rel = str(path.relative_to(root))
    written = {kind: 0 for kind in OUTPUT_FILES}
    read = 0
    for row_index, record in iterator(path):
        read += 1
        src = source_info(path, root, {"format": path.suffix.lower().lstrip(".")})
        normalized = (
            normalize_parquet_record(record, rel, row_index, src)
            if parquet
            else normalize_record(record, rel, row_index, src)
        )
        for item in normalized:
            kind = item["kind"]
            payload = item["payload"]
            if writer.write(kind, payload, force_eval=is_eval_path(path)):
                written["eval" if is_eval_path(path) else kind] += 1
    return {"records_read": read, "written": written}


def build_curriculum_plan(output_dir, manifest):
    plan = {
        "version": 1,
        "objective": "Train a model from broad source knowledge through instruction behavior, reasoning drills, domain-specific tasks, and preference alignment.",
        "notes": [
            "Run stages in order unless you already have an instruction-tuned base model.",
            "Use JSONL files under splits/train and splits/validation for training and evaluation.",
            "The preference stage is DPO-style and expects prompt/chosen/rejected records.",
        ],
        "stages": [
            {
                "id": "01_foundation_cpt",
                "name": "Foundation Continued Pretraining",
                "objective": "Transfer broad source knowledge, docs, code, transcripts, and prepared corpora.",
                "train_files": ["splits/train/unified_cpt.jsonl"],
                "validation_files": ["splits/validation/unified_cpt.jsonl"],
                "trainer": "TRL SFTTrainer or causal LM trainer with text field",
                "recommended_epochs": 1,
            },
            {
                "id": "02_instruction_sft",
                "name": "Instruction and Assistant Behavior SFT",
                "objective": "Teach instruction following, coding, math, summarization, and task response formats.",
                "train_files": ["splits/train/unified_sft_messages.jsonl"],
                "validation_files": ["splits/validation/unified_sft_messages.jsonl"],
                "trainer": "TRL SFTTrainer with conversational messages",
                "recommended_epochs": 1,
            },
            {
                "id": "03_reasoning_traces",
                "name": "Evidence-Grounded and Procedural Reasoning",
                "objective": "Teach explicit state/action/next_state style traces from transcript, Sudoku, math, and synthetic reasoning sources.",
                "train_files": ["splits/train/unified_reasoning_traces.jsonl"],
                "validation_files": ["splits/validation/unified_reasoning_traces.jsonl"],
                "trainer": "Convert traces to chat SFT or use a custom trace objective",
                "recommended_epochs": 1,
            },
            {
                "id": "04_preference_alignment",
                "name": "Preference Alignment",
                "objective": "Prefer grounded or higher-quality completions over rejected alternatives.",
                "train_files": ["splits/train/unified_preference_pairs.jsonl"],
                "validation_files": ["splits/validation/unified_preference_pairs.jsonl"],
                "trainer": "TRL DPOTrainer",
                "recommended_epochs": 1,
            },
            {
                "id": "05_eval",
                "name": "Held-Out Evaluation",
                "objective": "Evaluate on held-out/test/eval rows from source datasets.",
                "train_files": [],
                "validation_files": ["unified_eval.jsonl"],
                "trainer": "Evaluation only",
                "recommended_epochs": 0,
            },
        ],
    }
    path = Path(output_dir) / "curriculum_plan.json"
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return plan


def write_readme(output_dir, manifest, plan):
    text = f"""# Unified Training Curriculum

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
{json.dumps({'output_counts': manifest['output_counts'], 'split_counts': manifest['split_counts']}, indent=2)}
```
"""
    (Path(output_dir) / "README.md").write_text(text, encoding="utf-8")


def build_unified_dataset(source_root, output_dir, validation_ratio=0.05):
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    if output_dir.exists():
        # Remove only the intended output directory contents.
        resolved = output_dir.resolve()
        if resolved.name != "unified_training_curriculum":
            raise ValueError(f"Refusing to clear unexpected output directory: {resolved}")
        shutil.rmtree(resolved)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = OutputWriter(output_dir, validation_ratio=validation_ratio)
    inventory = []

    try:
        for path in sorted(p for p in source_root.rglob("*") if p.is_file()):
            action = classify_file(path, source_root)
            item = {
                "path": str(path.relative_to(source_root)),
                "bytes": path.stat().st_size,
                "action": action,
                "records_read": 0,
                "written": {kind: 0 for kind in OUTPUT_FILES},
            }
            try:
                if action != "use":
                    inventory.append(item)
                    continue
                suffix = path.suffix.lower()
                if suffix == ".jsonl":
                    result = process_json_records(path, source_root, writer, iter_jsonl)
                elif suffix == ".json":
                    result = process_json_records(path, source_root, writer, iter_json)
                elif suffix == ".parquet":
                    result = process_json_records(path, source_root, writer, iter_parquet, parquet=True)
                elif suffix in {".md", ".py", ".yaml", ".yml"}:
                    result = process_text_file(path, source_root, writer)
                else:
                    result = {"records_read": 0, "written": {}}
                item["records_read"] = result["records_read"]
                item["written"].update(result["written"])
            except Exception as exc:
                item["action"] = "error"
                item["error"] = str(exc)
            inventory.append(item)
    finally:
        writer.close()

    manifest = {
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "validation_ratio": validation_ratio,
        "source_file_count": len(inventory),
        "source_bytes": sum(item["bytes"] for item in inventory),
        "output_counts": writer.counts,
        "split_counts": writer.split_counts,
        "skipped_counts": {},
        "errored_files": [item for item in inventory if item["action"] == "error"],
        "output_files": list(OUTPUT_FILES.values())
        + [f"splits/train/{name}" for name in OUTPUT_FILES.values() if name != OUTPUT_FILES["eval"]]
        + [f"splits/validation/{name}" for name in OUTPUT_FILES.values() if name != OUTPUT_FILES["eval"]]
        + [
            "curriculum_plan.json",
            "source_inventory.json",
            "manifest.json",
            "README.md",
            "TRAINING_RECIPE.md",
            "unified_curriculum_builder.py",
        ],
    }
    for item in inventory:
        if item["action"].startswith("skip"):
            manifest["skipped_counts"][item["action"]] = manifest["skipped_counts"].get(item["action"], 0) + 1

    (output_dir / "source_inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plan = build_curriculum_plan(output_dir, manifest)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_readme(output_dir, manifest, plan)
    write_training_recipe(output_dir)
    shutil.copyfile(Path(__file__).resolve(), output_dir / "unified_curriculum_builder.py")
    return manifest


def write_training_recipe(output_dir):
    text = """# Training Recipe

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
"""
    (Path(output_dir) / "TRAINING_RECIPE.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Unify local LLM datasets into curriculum shards.")
    parser.add_argument(
        "--source-root",
        default=r"C:\Users\warit\OneDrive\Documents\My Programs\identity transformer\Training data",
    )
    parser.add_argument("--output-dir", default="outputs/unified_training_curriculum")
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    args = parser.parse_args()
    manifest = build_unified_dataset(args.source_root, args.output_dir, args.validation_ratio)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
