import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from experiments import benchmark_p8_p24_identity_robustness as bench
from tac_transformer import TACConfig, TACTransformerLM


class P8P24IdentityRobustnessTests(unittest.TestCase):
    def test_case_builder_spreads_interference_levels(self):
        cases = bench.build_retrieval_cases(
            case_count=6,
            interference_levels=[0, 4, 8],
        )

        self.assertEqual(len(cases), 6)
        self.assertEqual(
            [case.interference_pairs for case in cases[:3]],
            [0, 4, 8],
        )
        self.assertIn(cases[0].target_value, cases[0].alternatives)

    def test_benchmark_writes_schema_for_tiny_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p8 = root / "p8.pt"
            p24 = root / "p24.pt"
            config = TACConfig(
                vocab_size=512,
                d_model=16,
                n_heads=4,
                n_layers=1,
                n_programs=4,
                max_seq_len=64,
                memory_read_type="content_addressed",
                content_store_size=4,
                content_read_steps=1,
                identity_attention_type="identity_first",
                memory_adapter_type="gated_residual",
            )
            torch.manual_seed(7)
            torch.save(
                {
                    "step": 5,
                    "best_eval_loss": 1.0,
                    "model_state_dict": TACTransformerLM(config).state_dict(),
                    "config": asdict(config),
                    "parameter_counts": {"total": 1, "identity_field": 1},
                },
                p8,
            )
            torch.manual_seed(8)
            torch.save(
                {
                    "step": 5,
                    "best_eval_loss": 1.1,
                    "model_state_dict": TACTransformerLM(config).state_dict(),
                    "config": asdict(config),
                    "parameter_counts": {"total": 1, "identity_field": 1},
                },
                p24,
            )

            report = bench.run_p8_p24_identity_robustness(
                p8_checkpoint=p8,
                p24_checkpoint=p24,
                output_dir=root / "out",
                case_count=4,
                knockout_case_count=2,
                interference_levels=[0, 4],
                active_context_budgets=[4, 8],
            )

            self.assertEqual(report["schema"], "p8_p24_identity_robustness.v1")
            self.assertIn("long_context_carry", report["variants"]["p8"])
            self.assertEqual(
                report["variants"]["p8"]["program_knockout_robustness"]["programs_tested"],
                4,
            )
            saved = json.loads((root / "out" / "identity_robustness.json").read_text())
            self.assertEqual(saved["schema"], report["schema"])
            self.assertTrue((root / "out" / "RESULTS.md").exists())


if __name__ == "__main__":
    unittest.main()
