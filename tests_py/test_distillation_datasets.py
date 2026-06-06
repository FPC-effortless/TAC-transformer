import json
import tempfile
import unittest
from pathlib import Path

from kaggle import build_distillation_datasets
from tac_transformer.distillation_datasets import (
    STREAMS,
    generate_distillation_records,
    record_to_jsonl,
)


class DistillationDatasetTest(unittest.TestCase):
    def test_generator_emits_all_distillation_streams(self):
        generator = generate_distillation_records(seed=11, split="train")
        records = [next(generator) for _ in range(36)]
        domains = {record.domain for record in records}
        payloads = {record.domain: record.payload for record in records}

        self.assertTrue(set(STREAMS).issubset(domains))
        self.assertIn("mutation_operators", payloads["coding_evol_instruct"])
        self.assertIn("seed_code", payloads["coding_oss_instruct"])
        self.assertIn("steps", payloads["agentic_trajectory"])
        self.assertIn("compiler_or_runtime_error", payloads["execution_repair"])
        self.assertIn("chosen", payloads["preference_pair"])
        self.assertIn("difficulty", payloads["curriculum_metadata"])

        row = json.loads(record_to_jsonl(records[0]))
        self.assertEqual(row["source"], "distillation_curriculum")
        self.assertIn("<record", row["text"])
        self.assertGreater(len(row["text"]), 200)

    def test_build_distillation_datasets_writes_manifest_splits_and_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "distillation"
            build_distillation_datasets.main(
                [
                    "--output-dir",
                    str(output_dir),
                    "--train-records-per-stream",
                    "3",
                    "--eval-records-per-stream",
                    "2",
                    "--seed",
                    "19",
                ]
            )

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue((output_dir / "train.prepared.jsonl").exists())
            self.assertTrue((output_dir / "eval.prepared.jsonl").exists())
            self.assertTrue((output_dir / "preference_pairs.train.jsonl").exists())
            self.assertTrue((output_dir / "coding_evol_instruct.train.raw.jsonl").exists())
            self.assertEqual(manifest["train_records"], len(STREAMS) * 3)
            self.assertEqual(manifest["eval_records"], len(STREAMS) * 2)
            self.assertEqual(manifest["parts"]["preference_pair_train"]["stats"]["written"], 3)
            self.assertIn("difficulty_tiers", manifest)


if __name__ == "__main__":
    unittest.main()
