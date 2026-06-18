import json
import tempfile
import unittest
from pathlib import Path

from experiments import stage_ratio_controlled_identity_kaggle as staging


class RatioControlledIdentityKaggleStagingTests(unittest.TestCase):
    def test_staging_writes_low_rank_ratio_controlled_kernels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle_dir = root / "bundle"
            bundle_dir.mkdir()
            (bundle_dir / "best-tac-agentic-training-bundle.zip").write_bytes(b"zip")

            manifest = staging.stage_ratio_controlled_identity_kaggle(
                output_root=root,
                bundle_dir=bundle_dir,
            )

            self.assertEqual(
                manifest["code_dataset_id"],
                "jeffkolo/tac-identity-ratio-rc-code-2026-06-07",
            )
            self.assertEqual(len(manifest["kernels"]), 2)
            rows = {row["label"]: row for row in manifest["kernels"]}
            self.assertEqual(rows["p16"]["program_expert_rank"], 63)
            self.assertEqual(rows["p24"]["program_expert_rank"], 41)

            p16_script = Path(rows["p16"]["kernel_dir"]) / rows["p16"]["script_name"]
            script_text = p16_script.read_text(encoding="utf-8")
            self.assertIn("--program-compute-type", script_text)
            self.assertIn("low_rank_linear_expert", script_text)
            self.assertIn("--program-expert-rank", script_text)
            self.assertIn("EXPECTED_IDENTITY_TO_TRANSFORMER_RATIO", script_text)

            metadata = json.loads(
                (Path(rows["p24"]["kernel_dir"]) / "kernel-metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                metadata["id"],
                "jeffkolo/tac-identity-ratio-p24-rc-5k-2026-06-07",
            )


if __name__ == "__main__":
    unittest.main()
