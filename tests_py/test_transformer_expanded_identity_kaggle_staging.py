import json
import tempfile
import unittest
from pathlib import Path

from experiments import stage_transformer_expanded_identity_kaggle as staging


class TransformerExpandedIdentityKaggleStagingTests(unittest.TestCase):
    def test_staging_writes_full_rank_transformer_expanded_kernels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle_dir = root / "bundle"
            bundle_dir.mkdir()
            (bundle_dir / "best-tac-agentic-training-bundle.zip").write_bytes(b"zip")

            manifest = staging.stage_transformer_expanded_identity_kaggle(
                output_root=root,
                bundle_dir=bundle_dir,
            )

            self.assertEqual(
                manifest["code_dataset_id"],
                "eweewee2/tac-identity-ratio-tx-code-2026-06-07",
            )
            self.assertEqual(len(manifest["kernels"]), 2)
            rows = {row["label"]: row for row in manifest["kernels"]}
            self.assertEqual(rows["p16"]["mlp_ratio"], 7)
            self.assertEqual(rows["p24"]["mlp_ratio"], 10)
            self.assertEqual(rows["p16"]["program_compute_type"], "linear_expert")
            self.assertEqual(rows["p24"]["program_compute_type"], "linear_expert")

            p24_script = Path(rows["p24"]["kernel_dir"]) / rows["p24"]["script_name"]
            script_text = p24_script.read_text(encoding="utf-8")
            self.assertIn("--mlp-ratio", script_text)
            self.assertIn("PROGRAM_COMPUTE_TYPE = \"linear_expert\"", script_text)
            self.assertIn("EXPECTED_IDENTITY_TO_TRANSFORMER_RATIO", script_text)
            self.assertNotIn("--program-expert-rank", script_text)
            self.assertNotIn("low_rank_linear_expert", script_text)

            metadata = json.loads(
                (Path(rows["p16"]["kernel_dir"]) / "kernel-metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                metadata["id"],
                "eweewee2/tac-identity-ratio-p16-tx-5k-2026-06-07",
            )


if __name__ == "__main__":
    unittest.main()
