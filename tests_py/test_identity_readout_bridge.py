import json
import tempfile
import unittest
from pathlib import Path

from experiments import benchmark_identity_readout_bridge as bench


class IdentityReadoutBridgeTests(unittest.TestCase):
    def test_smoke_run_writes_bridge_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "bridge"

            report = bench.run_identity_readout_bridge(
                output_dir=output_dir,
                base_steps=2,
                probe_steps=2,
                bridge_steps=2,
                batch_size=4,
                eval_batches=1,
                n_pairs=2,
                seeds=[7],
                torch_threads=1,
            )

            self.assertEqual(report["schema"], "identity_readout_bridge.v1")
            self.assertIn("oracle_probe_accuracy", report["aggregate"])
            self.assertIn("logit_bridge_accuracy", report["aggregate"])
            self.assertIn("bridge_minus_base_carry", report["aggregate"])
            self.assertTrue((output_dir / "identity_readout_bridge.json").exists())
            saved = json.loads(
                (output_dir / "identity_readout_bridge.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(saved["schema"], report["schema"])
            self.assertTrue((output_dir / "RESULTS.md").exists())


if __name__ == "__main__":
    unittest.main()
